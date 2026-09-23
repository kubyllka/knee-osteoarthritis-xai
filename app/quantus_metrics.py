import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from app.xai import generate_xai_attributions
from app.config import MODEL_CONFIGS
from app.metric_registry import METRIC_REGISTRY


@dataclass(frozen=True)
class QuantusScore:
    metric_name: str
    metric_category: str
    score_direction: str
    value: float
    parameters: dict
    status: str = "ok"
    note: str | None = None


QUANTUS_METRIC_METADATA = METRIC_REGISTRY


def _load_quantus():
    """Import Quantus only when an on-demand evaluation is requested.

    This keeps Prediction and Research views usable on managed Windows machines
    where an application-control policy blocks a SciPy native DLL.
    """
    try:
        import quantus
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Quantus could not start because Windows blocked a required SciPy DLL. "
            "Prediction and Research Mode remain available; run Quantus in an approved "
            "Python/GPU environment or ask the system administrator to allow SciPy."
        ) from exc
    return quantus


def _first_score(scores) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != 1 or not np.isfinite(values[0]):
        raise RuntimeError("Quantus did not return one finite score for the evaluated sample.")
    return float(values[0])


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters(), torch.empty(0)).device


def perturb_spatial_pixels_all_channels(
    input_array: np.ndarray,
    spatial_indices: np.ndarray,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Replace selected ``(y, x)`` pixels in every input channel together.

    XAI maps in this project contain one importance value per spatial image
    location. A perturbation of such a feature must therefore affect the RGB
    values at that location together, rather than accidentally treating the
    first input channel as the complete image.
    """
    if input_array.ndim != 4:
        raise ValueError("Spatial perturbation expects shape (batch, channels, height, width).")

    batch_size, channels, height, width = input_array.shape
    indices = np.asarray(spatial_indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = np.broadcast_to(indices, (batch_size, len(indices)))
    if indices.ndim != 2 or indices.shape[0] != batch_size:
        raise ValueError("spatial_indices must have one flattened-position row per image.")
    if indices.shape[1] == 0 or np.any(indices < 0) or np.any(indices >= height * width):
        raise ValueError("spatial_indices must contain valid flattened image positions.")

    if baseline is None:
        # The model receives ImageNet-normalised tensors, so use a per-channel
        # mean in that same input space. This preserves the channel-specific
        # normalisation while removing local spatial evidence.
        baseline = input_array.mean(axis=(2, 3), keepdims=True)
    if baseline.shape != (batch_size, channels, 1, 1):
        raise ValueError("baseline must have shape (batch, channels, 1, 1).")

    perturbed = input_array.copy()
    for sample_index, sample_indices in enumerate(indices):
        rows = sample_indices // width
        columns = sample_indices % width
        # NumPy places the advanced spatial index before the channel axis, so
        # the indexed result is (selected_pixels, channels).
        perturbed[sample_index, :, rows, columns] = baseline[sample_index, :, 0, 0][np.newaxis, :]
    return perturbed


def spatial_faithfulness_correlation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    nr_runs: int,
    subset_size: int,
) -> float:
    """Faithfulness correlation with one feature per spatial RGB location.

    Quantus expands a one-channel attribution map to RGB before its standard
    implementation samples features. That makes an individual channel value,
    rather than a spatial image location, the selected feature. This explicit
    implementation retains Quantus' correlation principle but samples from the
    2-D map directly and replaces every selected location in all RGB channels.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("FaithfulnessCorrelation currently supports one image with shape (1, channels, height, width).")
    if attribution.ndim != 2 or tuple(attribution.shape) != tuple(input_tensor.shape[-2:]):
        raise ValueError("attribution spatial dimensions must match the model input.")

    flat_attribution = attribution.astype(np.float64, copy=False).reshape(-1)
    maximum = np.max(np.abs(flat_attribution))
    if not np.isfinite(maximum) or np.isclose(maximum, 0.0):
        raise ValueError("FaithfulnessCorrelation requires a finite, non-zero attribution map.")
    if not 0 < subset_size < len(flat_attribution):
        raise ValueError("subset_size must be between 1 and the number of spatial image pixels minus one.")

    normalised_attribution = flat_attribution / maximum
    input_array = input_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    baseline = input_array.mean(axis=(2, 3), keepdims=True)
    model.eval()
    with torch.no_grad():
        original_score = float(model(input_tensor)[0, target_class].item())

    attribution_sums: list[float] = []
    prediction_deltas: list[float] = []
    for _ in range(nr_runs):
        selected = np.random.choice(len(flat_attribution), subset_size, replace=False)
        perturbed_array = perturb_spatial_pixels_all_channels(input_array, selected, baseline)
        perturbed_tensor = torch.from_numpy(perturbed_array).to(device=input_tensor.device, dtype=input_tensor.dtype)
        with torch.no_grad():
            perturbed_score = float(model(perturbed_tensor)[0, target_class].item())
        attribution_sums.append(float(normalised_attribution[selected].sum()))
        prediction_deltas.append(original_score - perturbed_score)

    if np.isclose(np.std(attribution_sums), 0.0) or np.isclose(np.std(prediction_deltas), 0.0):
        return float("nan")
    return float(np.corrcoef(attribution_sums, prediction_deltas)[0, 1])


def evaluate_quantus_metrics(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    xai_method: str,
    model_name: str,
    faithfulness_runs: int = 50,
    max_sensitivity_samples: int = 8,
) -> list[QuantusScore]:
    """Evaluate one XAI map with the initial Quantus metric set."""
    if input_tensor.shape[0] != 1:
        raise ValueError("Quantus evaluation currently supports one sample at a time.")
    if attribution.ndim != 2:
        raise ValueError("attribution must have shape (height, width)")

    quantus = _load_quantus()
    model.eval()
    x_batch = input_tensor.detach().cpu().numpy().astype(np.float32)
    y_batch = np.asarray([target_class], dtype=np.int64)
    a_batch = attribution[np.newaxis, np.newaxis, :, :].astype(np.float32)
    feature_count = int(attribution.shape[0] * attribution.shape[1])
    subset_size = max(1, min(224, feature_count - 1))
    device = str(_model_device(model))
    is_ordinal = MODEL_CONFIGS.get(model_name, {}).get("output_type") == "ordinal_thresholds"
    model_output = "ordinal_class_log_score" if is_ordinal else "logit"

    def explain_func(model, inputs, targets, **_):
        return generate_xai_attributions(
            model=model,
            inputs=inputs,
            targets=targets,
            method=xai_method,
            model_name=model_name,
        )

    metric_scores: dict[str, object] = {}
    metric_statuses: dict[str, str] = {}
    metric_notes: dict[str, str] = {}
    metric_scores["FaithfulnessCorrelation"] = [
        spatial_faithfulness_correlation(
            model=model,
            input_tensor=input_tensor,
            attribution=attribution,
            target_class=target_class,
            nr_runs=faithfulness_runs,
            subset_size=subset_size,
        )
    ]
    try:
        metric_scores["MaxSensitivity"] = quantus.MaxSensitivity(
            nr_samples=max_sensitivity_samples,
            lower_bound=0.05,
            disable_warnings=True,
        )(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            explain_func=explain_func,
            device=device,
        )
    except AssertionError as exc:
        if "attribution vector are all equal to zero" not in str(exc):
            raise
        # A zero map after a MaxSensitivity perturbation is a legitimate
        # degenerate explanation, but Quantus deliberately refuses to score it.
        # Record it transparently rather than injecting an epsilon or losing the
        # entire sample × method result.
        metric_scores["MaxSensitivity"] = [float("nan")]
        metric_statuses["MaxSensitivity"] = "undefined_zero_perturbed_attribution"
        metric_notes["MaxSensitivity"] = (
            "Quantus could not evaluate MaxSensitivity because at least one "
            "perturbed input produced an all-zero attribution map."
        )

    metric_scores["Sparseness"] = quantus.Sparseness(disable_warnings=True)(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        )

    parameters = {
        "FaithfulnessCorrelation": {
            "attribution": "normalise_by_max",
            "abs": False,
            "nr_runs": faithfulness_runs,
            "subset_size": subset_size,
            "feature_unit": "spatial_pixel_rgb_group",
            "perturbation": "replace_selected_spatial_pixels_in_all_channels",
            "perturb_baseline": "per_channel_input_mean",
            "similarity": "pearson_correlation",
            "implementation": "spatial_rgb_grouped_faithfulness_correlation_v1",
            "model_output": model_output,
            "xai_target": "predicted_kl_class_score" if is_ordinal else "predicted_class_logit",
        },
        "MaxSensitivity": {
            "attribution": "raw_scale_preserving",
            "nr_samples": max_sensitivity_samples,
            "perturbation": "gaussian_noise",
            "lower_bound": 0.05,
            "upper_bound": None,
            "model_output": model_output,
            "xai_target": "predicted_kl_class_score" if is_ordinal else "predicted_class_logit",
        },
        "Sparseness": {"attribution": "raw_scale_preserving"},
    }

    return [
        QuantusScore(
            metric_name=name,
            metric_category=QUANTUS_METRIC_METADATA[name].category,
            score_direction=QUANTUS_METRIC_METADATA[name].score_direction,
            value=(float("nan") if name in metric_statuses else _first_score(score)),
            parameters=parameters[name],
            status=metric_statuses.get(name, "ok"),
            note=metric_notes.get(name),
        )
        for name, score in metric_scores.items()
    ]
