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
REGION_PERTURBATION_PATCH_SIZE = 16
REGION_PERTURBATION_REGIONS = 100


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


def replace_region_with_channel_means(
    arr: np.ndarray,
    indices: np.ndarray,
    channel_means: np.ndarray,
    spatial_plane_size: int,
    **_: object,
) -> np.ndarray:
    """Replace a flattened spatial patch with fixed per-channel means."""
    values = np.asarray(arr)
    selected = np.asarray(indices, dtype=np.int64)
    means = np.asarray(channel_means)
    if values.ndim != 2 or selected.ndim != 2 or selected.shape[0] != values.shape[0]:
        raise ValueError("Region perturbation expects batched flattened arrays and indices.")
    if means.ndim != 2 or means.shape[0] != values.shape[0]:
        raise ValueError("channel_means must have shape (batch, channels).")
    if spatial_plane_size <= 0:
        raise ValueError("spatial_plane_size must be positive.")

    perturbed = values.copy()
    for sample_index, sample_indices in enumerate(selected):
        channels = (sample_indices // spatial_plane_size) % means.shape[1]
        perturbed[sample_index, sample_indices] = means[sample_index, channels]
    return perturbed


def normalised_score_drop_auc(score_drops: np.ndarray) -> float:
    """Summarise a score-drop curve on a normalised 0-1 perturbation-step axis."""
    curve = np.asarray(score_drops, dtype=np.float64).reshape(-1)
    if curve.size == 0 or not np.isfinite(curve).all():
        raise ValueError("Region perturbation must return a finite, non-empty curve.")
    curve_with_origin = np.concatenate(([0.0], curve))
    fractions = np.linspace(0.0, 1.0, curve_with_origin.size)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(curve_with_origin, fractions))


def region_perturbation_score_drop_auc(
    quantus,
    model: torch.nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: np.ndarray,
    device: str,
    order: str = "morf",
    patch_size: int = REGION_PERTURBATION_PATCH_SIZE,
    regions_evaluation: int = REGION_PERTURBATION_REGIONS,
) -> float:
    """Run Quantus RegionPerturbation and calculate target-score-drop AUC."""
    height, width = x_batch.shape[-2:]
    padded_height = int(np.ceil(height / patch_size) * patch_size)
    padded_width = int(np.ceil(width / patch_size) * patch_size)
    curve = quantus.RegionPerturbation(
        patch_size=patch_size,
        order=order,
        regions_evaluation=regions_evaluation,
        abs=False,
        normalise=True,
        perturb_func=replace_region_with_channel_means,
        perturb_func_kwargs={
            "channel_means": x_batch.mean(axis=(2, 3)),
            "spatial_plane_size": padded_height * padded_width,
        },
        disable_warnings=True,
    )(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        channel_first=True,
        softmax=False,
        device=device,
    )
    return normalised_score_drop_auc(np.asarray(curve, dtype=float))


def random_order_region_control(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    repeats: int,
    seed: int,
) -> tuple[float, float, list[float]]:
    """Evaluate a method-independent random region-order sanity baseline."""
    if repeats <= 0:
        raise ValueError("Random-order control repeats must be positive.")
    quantus = _load_quantus()
    x_batch = input_tensor.detach().cpu().numpy().astype(np.float32)
    y_batch = np.asarray([target_class], dtype=np.int64)
    neutral_attribution = np.ones(
        (1, 1, input_tensor.shape[-2], input_tensor.shape[-1]),
        dtype=np.float32,
    )
    values = []
    for repeat_index in range(repeats):
        repeat_seed = int(seed + repeat_index)
        np.random.seed(repeat_seed)
        torch.manual_seed(repeat_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(repeat_seed)
        values.append(
            region_perturbation_score_drop_auc(
                quantus=quantus,
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=neutral_attribution,
                device=str(_model_device(model)),
                order="random",
            )
        )
    return float(np.mean(values)), float(np.std(values, ddof=1) if repeats > 1 else 0.0), values


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
    faithfulness_subset_size: int = 224,
    max_sensitivity_samples: int = 8,
    max_sensitivity_normalise: bool = False,
    requested_metrics: set[str] | frozenset[str] | None = None,
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
    if faithfulness_subset_size <= 0:
        raise ValueError("faithfulness_subset_size must be positive")
    if faithfulness_subset_size >= feature_count:
        raise ValueError(
            "faithfulness_subset_size must be smaller than the number of spatial attribution pixels "
            f"({feature_count})."
        )
    subset_size = faithfulness_subset_size
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

    requested = set(requested_metrics or {
        "FaithfulnessCorrelation",
        "Region perturbation score-drop AUC",
        "MaxSensitivity",
        "Sparseness",
    })
    supported = {
        "FaithfulnessCorrelation",
        "Region perturbation score-drop AUC",
        "MaxSensitivity",
        "Sparseness",
    }
    unknown = requested - supported
    if unknown:
        raise ValueError("Unsupported Quantus metrics: " + ", ".join(sorted(unknown)))

    metric_scores: dict[str, object] = {}
    metric_statuses: dict[str, str] = {}
    metric_notes: dict[str, str] = {}
    if "FaithfulnessCorrelation" in requested:
        correlation = spatial_faithfulness_correlation(
            model=model,
            input_tensor=input_tensor,
            attribution=attribution,
            target_class=target_class,
            nr_runs=faithfulness_runs,
            subset_size=subset_size,
        )
        metric_scores["FaithfulnessCorrelation"] = [correlation]
        if not np.isfinite(correlation):
            metric_statuses["FaithfulnessCorrelation"] = "undefined_degenerate_correlation"
            metric_notes["FaithfulnessCorrelation"] = (
                "FaithfulnessCorrelation is undefined because the attribution sums or "
                "target-score changes had zero variance across perturbation runs."
            )
    if "Region perturbation score-drop AUC" in requested:
        metric_scores["Region perturbation score-drop AUC"] = [
            region_perturbation_score_drop_auc(
                quantus=quantus,
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                device=device,
            )
        ]
    if "MaxSensitivity" in requested:
        try:
            max_sensitivity_kwargs = {
                "nr_samples": max_sensitivity_samples,
                "lower_bound": 0.05,
                "normalise": max_sensitivity_normalise,
                "disable_warnings": True,
            }
            if max_sensitivity_normalise:
                max_sensitivity_kwargs["normalise_func"] = quantus.normalise_by_max
            metric_scores["MaxSensitivity"] = quantus.MaxSensitivity(**max_sensitivity_kwargs)(
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
            metric_scores["MaxSensitivity"] = [float("nan")]
            metric_statuses["MaxSensitivity"] = "undefined_zero_perturbed_attribution"
            metric_notes["MaxSensitivity"] = (
                "Quantus could not evaluate MaxSensitivity because at least one "
                "perturbed input produced an all-zero attribution map."
            )

    if "Sparseness" in requested:
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
            "subset_fraction": subset_size / feature_count,
            "feature_unit": "spatial_pixel_rgb_group",
            "perturbation": "replace_selected_spatial_pixels_in_all_channels",
            "perturb_baseline": "per_channel_input_mean",
            "similarity": "pearson_correlation",
            "implementation": "spatial_rgb_grouped_faithfulness_correlation_v1",
            "model_output": model_output,
            "xai_target": "predicted_kl_class_score" if is_ordinal else "predicted_class_logit",
        },
        "Region perturbation score-drop AUC": {
            "attribution": "normalise_by_max",
            "abs": False,
            "patch_size": REGION_PERTURBATION_PATCH_SIZE,
            "regions_evaluation": REGION_PERTURBATION_REGIONS,
            "order": "morf",
            "perturbation": "replace_spatial_patch_in_all_channels",
            "perturb_baseline": "fixed_per_channel_input_mean",
            "summary": "trapezoidal_auc_of_target_score_drop_with_zero_origin",
            "auc_step_axis": "normalised_0_to_1_over_evaluated_regions",
            "softmax": False,
            "model_output": model_output,
            "xai_target": "predicted_kl_class_score" if is_ordinal else "predicted_class_logit",
        },
        "MaxSensitivity": {
            "attribution": (
                "quantus_normalise_by_max_separately"
                if max_sensitivity_normalise
                else "raw_scale_preserving"
            ),
            "normalise": max_sensitivity_normalise,
            "normalise_func": "normalise_by_max",
            "nr_samples": max_sensitivity_samples,
            "perturbation": "quantus_batch_uniform_noise",
            "lower_bound": 0.05,
            "upper_bound": None,
            "model_output": model_output,
            "xai_target": "predicted_kl_class_score" if is_ordinal else "predicted_class_logit",
        },
        "Sparseness": {
            "attribution": "quantus_normalise_by_max",
            "abs": True,
        },
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
