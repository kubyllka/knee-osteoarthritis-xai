"""Shared descriptions and display rules for XAI evaluation metrics."""

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    category: str
    score_direction: str
    description: str
    interpretation: str
    display_format: str
    source: str


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "Infidelity": MetricDefinition(
        category="faithfulness",
        score_direction="lower",
        description="Mean-squared mismatch between attribution-weighted input perturbation and model-output change.",
        interpretation="Lower values indicate better faithfulness. Compare only results produced with the same configuration.",
        display_format="{:.3e}",
        source="quantus",
    ),
    "MaxSensitivity": MetricDefinition(
        category="robustness",
        score_direction="lower",
        description="Largest explanation change observed under small input perturbations.",
        interpretation="Lower values indicate a more robust explanation.",
        display_format="{:.4f}",
        source="quantus",
    ),
    "Sparseness": MetricDefinition(
        category="complexity",
        score_direction="higher",
        description="How concentrated the attribution values are.",
        interpretation="Higher values mean a more concentrated map; this alone does not establish clinical usefulness.",
        display_format="{:.4f}",
        source="quantus",
    ),
    "Deletion confidence drop": MetricDefinition(
        category="custom",
        score_direction="higher",
        description="Decrease in predicted-class confidence after masking the top 20% XAI region.",
        interpretation="Higher values suggest that the selected region was influential for this prediction.",
        display_format="{:.4f}",
        source="custom",
    ),
    "Stability score": MetricDefinition(
        category="custom",
        score_direction="higher",
        description="Agreement of an explanation before and after +5% brightness and +5% contrast perturbation.",
        interpretation="Higher values indicate more stable spatial explanations under this specific perturbation.",
        display_format="{:.4f}",
        source="custom",
    ),
}


def metric_definition(metric_name: str) -> MetricDefinition | None:
    return METRIC_REGISTRY.get(metric_name)


def format_metric_value(metric_name: str, value: float) -> str:
    definition = metric_definition(metric_name)
    return (definition.display_format if definition else "{:.4f}").format(value)


def serialise_parameters(parameters: dict[str, Any]) -> str:
    """Store metric configuration in one portable CSV cell."""
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def parse_parameters(value: object) -> dict[str, Any] | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None
