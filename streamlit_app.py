from io import BytesIO
from itertools import combinations

import numpy as np
import requests
import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance

from app.config import KL_LABELS, MODEL_CONFIGS
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.xai import generate_xai_visualization, load_xai_model_by_name


GRADE_MEANINGS = {
    0: "No radiographic osteoarthritis.",
    1: "Doubtful osteoarthritis with possible osteophytes or minimal narrowing.",
    2: "Mild osteoarthritis with definite osteophytes and possible joint-space narrowing.",
    3: "Moderate osteoarthritis with multiple osteophytes and definite joint-space narrowing.",
    4: "Severe osteoarthritis with marked narrowing, sclerosis, or bone deformity.",
}

DISCLAIMER = (
    "This system is developed for educational and research purposes only. "
    "It is not intended for clinical diagnosis or medical decision-making."
)

XAI_METHODS = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Integrated Gradients", "SmoothGrad"]
DEFAULT_XAI_METHODS = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM"]
TOP_PIXEL_FRACTION = 0.20


@st.cache_resource
def get_local_classifier(model_name: str) -> ONNXKneeOsteoarthritisClassifier:
    return ONNXKneeOsteoarthritisClassifier(MODEL_CONFIGS[model_name]["onnx_path"])


@st.cache_resource
def get_xai_model(model_name: str):
    return load_xai_model_by_name(model_name)


def predict_locally(image_bytes: bytes, model_name: str = "vit") -> dict:
    prediction = get_local_classifier(model_name).predict(image_bytes)
    return {
        "grade": prediction.grade,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "latency_ms": prediction.latency_ms,
    }


def predict_with_api(image_bytes: bytes, filename: str, api_url: str, model_name: str) -> dict:
    endpoint = api_url.rstrip("/") + "/predict"
    response = requests.post(
        endpoint,
        files={"file": (filename, image_bytes, "image/png")},
        data={"model_name": model_name},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def build_xai_metrics(results) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
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


def build_deletion_metrics(results, prediction: dict, model_name: str) -> tuple[pd.DataFrame, float]:
    predicted_grade = int(prediction["grade"])
    original_confidence = float(prediction["probabilities"][f"KL Grade {predicted_grade}"])
    rows = []

    for result in results:
        masked_image = mask_important_region(result.original, result.cam)
        masked_prediction = predict_locally(pil_to_png_bytes(masked_image), model_name=model_name)
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
    results,
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


def show_prediction(prediction: dict) -> None:
    grade = int(prediction["grade"])
    confidence = float(prediction["confidence"])

    st.subheader(f"Predicted class: Grade {grade}")
    st.write(f"**Meaning:** {GRADE_MEANINGS.get(grade, KL_LABELS.get(grade, 'Unknown grade'))}")

    metric_cols = st.columns(2)
    metric_cols[0].metric("Confidence", format_percent(confidence))
    metric_cols[1].metric("Latency", f"{float(prediction.get('latency_ms', 0.0)):.1f} ms")

    probabilities = prediction["probabilities"]
    chart_data = pd.DataFrame(
        {
            "Grade": [label.replace("KL Grade ", "Grade ") for label in probabilities],
            "Probability": [float(probability) for probability in probabilities.values()],
        }
    )
    st.bar_chart(chart_data, x="Grade", y="Probability", height=280)


def show_xai(
    image: Image.Image,
    prediction: dict,
    methods: list[str],
    alpha: float,
    model_name: str,
) -> None:
    if not methods:
        st.info("Select at least one XAI method in the sidebar.")
        return

    model = get_xai_model(model_name)
    results = []
    with st.spinner("Generating visual explanations..."):
        for method in methods:
            results.append(
                generate_xai_visualization(
                    model=model,
                    image=image,
                    method=method,
                    class_idx=int(prediction["grade"]),
                    alpha=alpha,
                    model_name=model_name,
                )
            )

    st.subheader("Explainable AI visualizations")

    tabs = st.tabs([result.method for result in results])
    for tab, result in zip(tabs, results):
        with tab:
            xai_cols = st.columns(3)
            xai_cols[0].image(result.original, caption="Original X-ray", use_container_width=True)
            xai_cols[1].image(result.heatmap, caption=f"{result.method} heatmap", use_container_width=True)
            xai_cols[2].image(result.overlay, caption=f"{result.method} overlay", use_container_width=True)

    if len(results) > 1:
        st.subheader("Overlay comparison")
        comparison_cols = st.columns(len(results))
        for column, result in zip(comparison_cols, results):
            column.image(result.overlay, caption=result.method, use_container_width=True)

        pair_metrics, focus_metrics, similarity_matrix, average_score = build_xai_metrics(results)

        st.subheader("Quantitative XAI comparison")
        metric_cols = st.columns(2)
        metric_cols[0].dataframe(pair_metrics.style.format({
            "Pearson correlation": "{:.3f}",
            "Top 20% IoU": "{:.3f}",
            "Agreement score": "{:.3f}",
        }), use_container_width=True, hide_index=True)
        metric_cols[1].dataframe(focus_metrics.style.format({
            "Focus area ratio": "{:.3f}",
        }), use_container_width=True, hide_index=True)

        st.metric("Average XAI Agreement Score", f"{average_score:.3f}")
        st.write(interpret_agreement(average_score))

        with st.expander("Heatmap similarity matrix"):
            st.dataframe(
                similarity_matrix.style.format("{:.3f}"),
                use_container_width=True,
            )

    st.subheader("Deletion Check")
    deletion_metrics, average_drop = build_deletion_metrics(results, prediction, model_name=model_name)
    st.dataframe(
        deletion_metrics.style.format({
            "Original class confidence": "{:.3f}",
            "After deletion confidence": "{:.3f}",
            "Confidence drop": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Average confidence drop after deleting top 20% XAI region", f"{average_drop:.3f}")
    st.write(interpret_deletion(average_drop))

    st.subheader("Stability Check")
    stability_metrics, average_stability = build_stability_metrics(
        model=model,
        image=image,
        results=results,
        predicted_grade=int(prediction["grade"]),
        model_name=model_name,
    )
    st.dataframe(
        stability_metrics.style.format({
            "Pearson correlation": "{:.3f}",
            "Top 20% IoU": "{:.3f}",
            "Stability score": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Average Stability Score", f"{average_stability:.3f}")
    st.write(interpret_stability(average_stability))


def main() -> None:
    st.set_page_config(
        page_title="KneeXAI",
        page_icon="K",
        layout="wide",
    )

    st.title("KneeXAI: Explainable AI System for Osteoarthritis Severity Classification")
    st.warning(DISCLAIMER)

    with st.sidebar:
        model_options = {config["display_name"]: name for name, config in MODEL_CONFIGS.items()}
        st.header("Inference")
        selected_model_label = st.selectbox("Model", list(model_options.keys()))
        model_name = model_options[selected_model_label]
        use_api = st.toggle("Use FastAPI endpoint", value=False)
        api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
        st.header("Explainability")
        enable_xai = st.toggle("Show XAI visualization", value=True)
        methods = st.multiselect("XAI methods", XAI_METHODS, default=DEFAULT_XAI_METHODS)
        alpha = st.slider("Overlay strength", min_value=0.1, max_value=0.8, value=0.45, step=0.05)

    uploaded_file = st.file_uploader(
        "Upload a knee X-ray image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload an X-ray image to start prediction.")
        return

    image_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image_col, result_col = st.columns([1, 1])
    with image_col:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with result_col:
        try:
            if use_api:
                prediction = predict_with_api(image_bytes, uploaded_file.name, api_url, model_name=model_name)
            else:
                prediction = predict_locally(image_bytes, model_name=model_name)
            show_prediction(prediction)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    if enable_xai:
        try:
            show_xai(image, prediction, methods=methods, alpha=alpha, model_name=model_name)
        except Exception as exc:
            st.error(f"XAI visualization failed: {exc}")


if __name__ == "__main__":
    main()
