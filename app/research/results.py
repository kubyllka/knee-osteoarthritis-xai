from pathlib import Path
import json

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "sample_id",
    "image_path",
    "true_kl",
    "predicted_kl",
    "is_correct",
    "error_distance",
    "confidence",
    "model_name",
    "xai_method",
    "metric_name",
    "metric_category",
    "metric_value",
)

OPTIONAL_COLUMNS = (
    "run_id",
    "split",
    "seed",
    "checkpoint_version",
    "target_class",
    "metric_source",
    "score_direction",
    "metric_parameters",
    "metric_status",
    "metric_note",
    "attribution_l1_norm",
    "attribution_l2_norm",
    "attribution_max_abs",
    "attribution_l2_after_max_normalisation",
    "attribution_nonzero_fraction",
)

VALID_METRIC_CATEGORIES = frozenset(
    {"faithfulness", "robustness", "complexity", "randomisation", "custom"}
)
VALID_METRIC_STATUSES = frozenset({
    "ok",
    "undefined_zero_perturbed_attribution",
    "undefined_degenerate_correlation",
})


class ResearchResultsValidationError(ValueError):
    """Raised when a research-results CSV does not follow the shared contract."""


def _require_non_empty_text(frame: pd.DataFrame, column: str) -> None:
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ResearchResultsValidationError(
            f"Column '{column}' must contain a non-empty value in every row."
        )
    frame[column] = values


def _integer_column(frame: pd.DataFrame, column: str, minimum: int, maximum: int) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna() & np.isfinite(values) & values.eq(np.floor(values))
    valid &= values.between(minimum, maximum)
    if not valid.all():
        raise ResearchResultsValidationError(
            f"Column '{column}' must contain integer values from {minimum} to {maximum}."
        )
    return values.astype("int64")


def _normalise_boolean(values: pd.Series, column: str) -> pd.Series:
    normalised = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = normalised.map(mapping)
    if result.isna().any():
        raise ResearchResultsValidationError(
            f"Column '{column}' must contain boolean values (true/false)."
        )
    return result.astype(bool)


def _finite_float_column(
    frame: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    if minimum is not None:
        valid &= values.ge(minimum)
    if maximum is not None:
        valid &= values.le(maximum)
    if not valid.all():
        bounds = "a finite numeric value"
        if minimum is not None and maximum is not None:
            bounds = f"a value from {minimum} to {maximum}"
        raise ResearchResultsValidationError(f"Column '{column}' must contain {bounds}.")
    return values.astype(float)


def validate_research_results(results: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise one-row-per-sample/method/metric research results."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in results.columns]
    if missing_columns:
        raise ResearchResultsValidationError(
            "Missing required columns: " + ", ".join(missing_columns)
        )
    if results.empty:
        raise ResearchResultsValidationError("Research results must contain at least one row.")

    cleaned = results.copy()
    for column in ("sample_id", "image_path", "model_name", "xai_method", "metric_name"):
        _require_non_empty_text(cleaned, column)

    cleaned["true_kl"] = _integer_column(cleaned, "true_kl", 0, 4)
    cleaned["predicted_kl"] = _integer_column(cleaned, "predicted_kl", 0, 4)
    cleaned["is_correct"] = _normalise_boolean(cleaned["is_correct"], "is_correct")
    cleaned["error_distance"] = _integer_column(cleaned, "error_distance", 0, 4)
    cleaned["confidence"] = _finite_float_column(cleaned, "confidence", 0.0, 1.0)

    # Legacy files had no status column; all their finite values remain valid.
    if "metric_status" not in cleaned.columns:
        cleaned["metric_status"] = "ok"
    cleaned["metric_status"] = cleaned["metric_status"].fillna("").astype(str).str.strip().str.lower()
    invalid_statuses = sorted(set(cleaned["metric_status"]) - VALID_METRIC_STATUSES)
    if invalid_statuses:
        raise ResearchResultsValidationError(
            "Unsupported metric_status values: " + ", ".join(invalid_statuses)
        )

    raw_metric_values = cleaned["metric_value"]
    metric_values = pd.to_numeric(raw_metric_values, errors="coerce")
    value_is_finite = metric_values.notna() & np.isfinite(metric_values)
    undefined = cleaned["metric_status"].ne("ok")
    value_is_empty = raw_metric_values.isna() | raw_metric_values.astype(str).str.strip().eq("")
    if (~undefined & ~value_is_finite).any() or (undefined & ~value_is_empty).any():
        raise ResearchResultsValidationError(
            "Column 'metric_value' must be finite for status 'ok' and empty for an undefined metric."
        )
    cleaned["metric_value"] = metric_values.astype(float)

    if "metric_note" not in cleaned.columns:
        cleaned["metric_note"] = pd.NA
    cleaned["metric_note"] = cleaned["metric_note"].astype("string").str.strip()
    if (undefined & (cleaned["metric_note"].isna() | cleaned["metric_note"].eq(""))).any():
        raise ResearchResultsValidationError(
            "An undefined metric row must include a non-empty 'metric_note'."
        )

    correlation_rows = cleaned["metric_name"].eq("FaithfulnessCorrelation") & ~undefined
    if not cleaned.loc[correlation_rows, "metric_value"].between(-1.0, 1.0).all():
        raise ResearchResultsValidationError(
            "FaithfulnessCorrelation metric_value must be between -1 and 1."
        )

    _require_non_empty_text(cleaned, "metric_category")
    cleaned["metric_category"] = cleaned["metric_category"].str.lower()
    unknown_categories = sorted(set(cleaned["metric_category"]) - VALID_METRIC_CATEGORIES)
    if unknown_categories:
        raise ResearchResultsValidationError(
            "Unsupported metric_category values: " + ", ".join(unknown_categories)
        )

    expected_correct = cleaned["true_kl"].eq(cleaned["predicted_kl"])
    if not cleaned["is_correct"].eq(expected_correct).all():
        raise ResearchResultsValidationError(
            "Column 'is_correct' must equal true_kl == predicted_kl."
        )

    expected_distance = (cleaned["true_kl"] - cleaned["predicted_kl"]).abs()
    if not cleaned["error_distance"].eq(expected_distance).all():
        raise ResearchResultsValidationError(
            "Column 'error_distance' must equal abs(true_kl - predicted_kl)."
        )

    if "target_class" in cleaned.columns:
        cleaned["target_class"] = _integer_column(cleaned, "target_class", 0, 4)
        if not cleaned["target_class"].eq(cleaned["predicted_kl"]).all():
            raise ResearchResultsValidationError(
                "Column 'target_class' must equal predicted_kl for prediction explanations."
            )

    if "seed" in cleaned.columns:
        cleaned["seed"] = _integer_column(cleaned, "seed", 0, 2**63 - 1)

    for column in ("run_id", "split", "checkpoint_version"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.strip()

    if "metric_source" in cleaned.columns:
        _require_non_empty_text(cleaned, "metric_source")
        invalid_sources = sorted(set(cleaned["metric_source"]) - {"quantus", "custom"})
        if invalid_sources:
            raise ResearchResultsValidationError(
                "Column 'metric_source' must contain quantus or custom."
            )

    if "score_direction" in cleaned.columns:
        _require_non_empty_text(cleaned, "score_direction")
        cleaned["score_direction"] = cleaned["score_direction"].str.lower()
        invalid_directions = sorted(set(cleaned["score_direction"]) - {"higher", "lower"})
        if invalid_directions:
            raise ResearchResultsValidationError(
                "Column 'score_direction' must contain higher or lower."
            )

    if "metric_parameters" in cleaned.columns:
        for value in cleaned["metric_parameters"].dropna():
            try:
                parsed = json.loads(str(value))
            except json.JSONDecodeError as exc:
                raise ResearchResultsValidationError(
                    "Column 'metric_parameters' must contain JSON objects."
                ) from exc
            if not isinstance(parsed, dict):
                raise ResearchResultsValidationError(
                    "Column 'metric_parameters' must contain JSON objects."
                )
        cleaned["metric_parameters"] = cleaned["metric_parameters"].astype("string").str.strip()

    return cleaned


def load_research_results(path: str | Path) -> pd.DataFrame:
    """Load, validate, and normalise a research-results CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Research results CSV not found: {csv_path}")

    return validate_research_results(pd.read_csv(csv_path))
