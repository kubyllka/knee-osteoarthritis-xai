import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import quantus

from app.xai import generate_xai_attributions


@dataclass(frozen=True)
class QuantusScore:
    metric_name: str
    metric_category: str
    higher_is_better: bool
    value: float


QUANTUS_METRIC_METADATA = {
    "Infidelity": {"metric_category": "faithfulness", "higher_is_better": False},
    "MaxSensitivity": {"metric_category": "robustness", "higher_is_better": False},
    "Sparseness": {"metric_category": "complexity", "higher_is_better": True},
}


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
    infidelity_samples: int = 8,
    max_sensitivity_samples: int = 8,
) -> list[QuantusScore]:
    """Evaluate one XAI map with the initial Quantus metric set."""
    if input_tensor.shape[0] != 1:
        raise ValueError("Quantus evaluation currently supports one sample at a time.")
    if attribution.ndim != 2:
        raise ValueError("attribution must have shape (height, width)")

    model.eval()
    x_batch = input_tensor.detach().cpu().numpy().astype(np.float32)
    y_batch = np.asarray([target_class], dtype=np.int64)
    a_batch = attribution[np.newaxis, np.newaxis, :, :].astype(np.float32)
    patch_size = max(1, min(16, attribution.shape[0], attribution.shape[1]))
    device = str(_model_device(model))

    def explain_func(model, inputs, targets, **_):
        return generate_xai_attributions(
            model=model,
            inputs=inputs,
            targets=targets,
            method=xai_method,
            model_name=model_name,
        )

    metric_scores = {
        "Infidelity": quantus.Infidelity(
            perturb_patch_sizes=[patch_size],
            n_perturb_samples=infidelity_samples,
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

    return [
        QuantusScore(
            metric_name=name,
            metric_category=QUANTUS_METRIC_METADATA[name]["metric_category"],
            higher_is_better=QUANTUS_METRIC_METADATA[name]["higher_is_better"],
            value=_first_score(score),
        )
        for name, score in metric_scores.items()
    ]
