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

    metric_scores = {
        "FaithfulnessCorrelation": quantus.FaithfulnessCorrelation(
            nr_runs=faithfulness_runs,
            subset_size=subset_size,
            abs=False,
            normalise=True,
            perturb_baseline="mean",
            disable_warnings=True,
        )(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        ),
        "MaxSensitivity": quantus.MaxSensitivity(
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
        ),
        "Sparseness": quantus.Sparseness(disable_warnings=True)(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        ),
    }

    parameters = {
        "FaithfulnessCorrelation": {
            "attribution": "normalise_by_max",
            "abs": False,
            "nr_runs": faithfulness_runs,
            "subset_size": subset_size,
            "perturb_baseline": "mean",
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
            value=_first_score(score),
            parameters=parameters[name],
        )
        for name, score in metric_scores.items()
    ]
