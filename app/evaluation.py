from collections.abc import Callable, Sequence
from io import BytesIO
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from app.xai import generate_xai_visualization


TOP_PIXEL_FRACTION = 0.20


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def binary_top_mask(cam: np.ndarray, top_fraction: float = TOP_PIXEL_FRACTION) -> np.ndarray:
    """Keep only the most activated pixels so IoU compares the strongest focus regions."""
    flat = cam.ravel()
    top_count = max(1, int(np.ceil(flat.size * top_fraction)))
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[top_indices] = True
    return mask.reshape(cam.shape)


def pearson_correlation(first_cam: np.ndarray, second_cam: np.ndarray) -> float:
    first = first_cam.ravel()
    second = second_cam.ravel()

    if np.isclose(first.std(), 0.0) or np.isclose(second.std(), 0.0):
        return 0.0

    return float(np.corrcoef(first, second)[0, 1])


def mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def focus_area_ratio(cam: np.ndarray, top_fraction: float = TOP_PIXEL_FRACTION) -> float:
    """Report how much of the image belongs to the high-activation explanation region."""
    return float(binary_top_mask(cam, top_fraction).mean())


def agreement_score(correlation: float, iou: float) -> float:
    # Correlation can be negative; clipping maps it onto the same 0-1 range as IoU.
    normalized_correlation = max(0.0, min(1.0, correlation))
    return float((normalized_correlation + iou) / 2.0)


def mask_important_region(image: Image.Image, cam: np.ndarray) -> Image.Image:
    """Delete the XAI-selected region by replacing the top pixels with neutral mean intensity."""
    mask = binary_top_mask(cam)
    image_array = np.asarray(image.convert("RGB")).copy()
    replacement_color = image_array.mean(axis=(0, 1)).astype(np.uint8)
    image_array[mask] = replacement_color
    return Image.fromarray(image_array)


def slightly_perturb_image(image: Image.Image) -> Image.Image:
    """Use a mild non-geometric perturbation so the explanation should remain spatially comparable."""
    perturbed = ImageEnhance.Brightness(image.convert("RGB")).enhance(1.05)
    perturbed = ImageEnhance.Contrast(perturbed).enhance(1.05)
    return perturbed


def build_xai_metrics(results: Sequence) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    masks = {result.method: binary_top_mask(result.cam) for result in results}

    focus_rows = [
        {
            "Method": result.method,
            "Focus area ratio": focus_area_ratio(result.cam),
        }
        for result in results
    ]

    pair_rows = []
    agreement_scores = []
    method_names = [result.method for result in results]
    cams = {result.method: result.cam for result in results}
    similarity_matrix = pd.DataFrame(
        np.eye(len(method_names)),
        index=method_names,
        columns=method_names,
    )

    for first_method, second_method in combinations(method_names, 2):
        correlation = pearson_correlation(cams[first_method], cams[second_method])
        iou = mask_iou(masks[first_method], masks[second_method])
        score = agreement_score(correlation, iou)
        agreement_scores.append(score)

        pair_rows.append(
            {
                "Comparison": f"{first_method} vs {second_method}",
                "Pearson correlation": correlation,
                "Top 20% IoU": iou,
                "Agreement score": score,
            }
        )
        similarity_matrix.loc[first_method, second_method] = correlation
        similarity_matrix.loc[second_method, first_method] = correlation

    average_score = float(np.mean(agreement_scores)) if agreement_scores else 1.0
    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(focus_rows),
        similarity_matrix,
        average_score,
    )


def interpret_agreement(score: float) -> str:
    if score >= 0.70:
        return "The XAI methods show strong agreement, so they are highlighting broadly similar knee regions."
    if score >= 0.45:
        return "The XAI methods show moderate agreement, suggesting some shared focus with method-specific differences."
    return "The XAI methods show low agreement, so the explanations may be emphasizing different image regions."


def build_deletion_metrics(
    results: Sequence,
    prediction: dict,
    model_name: str,
    predict_fn: Callable[[bytes, str], dict],
) -> tuple[pd.DataFrame, float]:
    predicted_grade = int(prediction["grade"])
    original_confidence = float(prediction["probabilities"][f"KL Grade {predicted_grade}"])
    rows = []

    for result in results:
        masked_image = mask_important_region(result.original, result.cam)
        masked_prediction = predict_fn(pil_to_png_bytes(masked_image), model_name)
        masked_confidence = float(masked_prediction["probabilities"][f"KL Grade {predicted_grade}"])
        confidence_drop = original_confidence - masked_confidence

        rows.append(
            {
                "Method": result.method,
                "Original class confidence": original_confidence,
                "After deletion confidence": masked_confidence,
                "Confidence drop": confidence_drop,
                "Deleted prediction": f"Grade {masked_prediction['grade']}",
            }
        )

    average_drop = float(np.mean([row["Confidence drop"] for row in rows])) if rows else 0.0
    return pd.DataFrame(rows), average_drop


def build_stability_metrics(
    model,
    image: Image.Image,
    results: Sequence,
    predicted_grade: int,
    model_name: str,
) -> tuple[pd.DataFrame, float]:
    perturbed_image = slightly_perturb_image(image)
    rows = []

    for result in results:
        # Stability compares the original explanation with the explanation after a small input change.
        perturbed_result = generate_xai_visualization(
            model=model,
            image=perturbed_image,
            method=result.method,
            class_idx=predicted_grade,
            model_name=model_name,
        )
        correlation = pearson_correlation(result.cam, perturbed_result.cam)
        iou = mask_iou(binary_top_mask(result.cam), binary_top_mask(perturbed_result.cam))
        score = agreement_score(correlation, iou)

        rows.append(
            {
                "Method": result.method,
                "Perturbation": "brightness +5%, contrast +5%",
                "Pearson correlation": correlation,
                "Top 20% IoU": iou,
                "Stability score": score,
            }
        )

    average_stability = float(np.mean([row["Stability score"] for row in rows])) if rows else 0.0
    return pd.DataFrame(rows), average_stability


def interpret_deletion(average_drop: float) -> str:
    if average_drop >= 0.30:
        return "Deletion Check shows a strong confidence drop, so the highlighted regions are highly influential for the model decision."
    if average_drop >= 0.10:
        return "Deletion Check shows a moderate confidence drop, so the highlighted regions contain useful decision evidence."
    return "Deletion Check shows a small confidence drop, so the highlighted regions may not fully capture the model evidence."


def interpret_stability(score: float) -> str:
    if score >= 0.70:
        return "Stability Check is high: small brightness and contrast changes keep the explanation mostly consistent."
    if score >= 0.45:
        return "Stability Check is moderate: the explanation keeps some structure but changes in noticeable areas."
    return "Stability Check is low: small input changes produce substantially different explanations."
