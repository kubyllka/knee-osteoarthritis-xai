"""Reusable descriptive and primary KL-stage statistics for Research Mode."""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

from app.research.results import validate_research_results


PREDICTION_METADATA_COLUMNS = (
    "image_path",
    "true_kl",
    "predicted_kl",
    "is_correct",
    "error_distance",
    "confidence",
    "target_class",
)
ROW_ID_COLUMNS = ("model_name", "sample_id", "xai_method", "metric_name")
TEST_ID_COLUMNS = ("model_name", "xai_method", "metric_name")
KL_GRADES = tuple(range(5))


class ResearchAnalysisError(ValueError):
    """Raised when loaded research rows are unsuitable for an inferential analysis."""


def _canonical_parameters(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return json.dumps(json.loads(str(value)), sort_keys=True, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResearchAnalysisError("metric_parameters must contain readable JSON for Results analysis.") from exc


def _require_prediction_metadata_consistency(results: pd.DataFrame) -> None:
    missing = [column for column in PREDICTION_METADATA_COLUMNS if column not in results.columns]
    if missing:
        raise ResearchAnalysisError(
            "Results analysis requires prediction metadata columns: " + ", ".join(missing)
        )
    conflicts = []
    for (model_name, sample_id), group in results.groupby(["model_name", "sample_id"], sort=False):
        inconsistent = [
            column for column in PREDICTION_METADATA_COLUMNS
            if group[column].nunique(dropna=False) != 1
        ]
        if inconsistent:
            conflicts.append(f"{model_name}/{sample_id}: {', '.join(inconsistent)}")
    if conflicts:
        preview = "; ".join(conflicts[:3])
        suffix = " …" if len(conflicts) > 3 else ""
        raise ResearchAnalysisError(
            "Prediction metadata conflict within model × sample_id: " + preview + suffix
        )


def _require_unique_rows(results: pd.DataFrame) -> None:
    duplicates = results.duplicated(list(ROW_ID_COLUMNS), keep=False)
    if duplicates.any():
        preview = results.loc[duplicates, list(ROW_ID_COLUMNS)].drop_duplicates().head(3)
        pairs = "; ".join(
            f"{row.model_name}/{row.sample_id}/{row.xai_method}/{row.metric_name}"
            for row in preview.itertuples(index=False)
        )
        raise ResearchAnalysisError("Duplicate experiment rows were selected: " + pairs)


def _require_metric_configuration_consistency(results: pd.DataFrame) -> None:
    if "metric_parameters" not in results.columns:
        raise ResearchAnalysisError("Results analysis requires metric_parameters for reproducible comparisons.")
    signatures = results["metric_parameters"].map(_canonical_parameters)
    checked = results.assign(_parameter_signature=signatures)
    mixed = (
        checked.groupby(list(TEST_ID_COLUMNS))["_parameter_signature"]
        .nunique(dropna=False)
    )
    if (mixed > 1).any():
        model_name, method, metric_name = mixed[mixed > 1].index[0]
        raise ResearchAnalysisError(
            "Incompatible metric configuration for "
            f"{model_name}/{method}/{metric_name}. Select one matching experiment run."
        )


def merge_research_results(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Validate and merge selected CSV frames without introducing pseudo-replication."""
    validated = [validate_research_results(frame) for frame in frames]
    if not validated:
        raise ResearchAnalysisError("Select at least one research-results CSV.")
    merged = pd.concat(validated, ignore_index=True)
    _require_unique_rows(merged)
    return merged


def prediction_records(results: pd.DataFrame) -> pd.DataFrame:
    """Return one consistent prediction record per model and image."""
    _require_prediction_metadata_consistency(results)
    return (
        results.sort_values(["model_name", "sample_id"])
        .drop_duplicates(["model_name", "sample_id"])
        [["model_name", "sample_id", *PREDICTION_METADATA_COLUMNS]]
        .reset_index(drop=True)
    )


def classifier_subset_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute classifier metrics for the images represented in the loaded XAI results."""
    records = prediction_records(results)
    overall_rows = []
    grade_rows = []
    confusion_matrices: dict[str, pd.DataFrame] = {}
    for model_name, group in records.groupby("model_name", sort=True):
        true = group["true_kl"].to_numpy()
        predicted = group["predicted_kl"].to_numpy()
        overall_rows.append({
            "Model": model_name,
            "Images": len(group),
            "Accuracy": accuracy_score(true, predicted),
            "Balanced accuracy": balanced_accuracy_score(true, predicted),
            "Macro F1": f1_score(true, predicted, average="macro", zero_division=0),
            "Quadratic weighted kappa": cohen_kappa_score(true, predicted, weights="quadratic"),
        })
        for grade in KL_GRADES:
            grade_group = group[group["true_kl"].eq(grade)]
            count = len(grade_group)
            correct = int(grade_group["is_correct"].sum())
            grade_rows.append({
                "Model": model_name,
                "True KL grade": f"KL{grade}",
                "Images": count,
                "Correct": correct,
                "Accuracy": correct / count if count else np.nan,
            })
        matrix = pd.crosstab(group["true_kl"], group["predicted_kl"], dropna=False)
        matrix = matrix.reindex(index=KL_GRADES, columns=KL_GRADES, fill_value=0)
        matrix.index = [f"True KL{grade}" for grade in KL_GRADES]
        matrix.columns = [f"Predicted KL{grade}" for grade in KL_GRADES]
        confusion_matrices[model_name] = matrix
    return pd.DataFrame(overall_rows), pd.DataFrame(grade_rows), confusion_matrices


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjustment, preserving the original index and missing values."""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.dropna().astype(float)
    ordered = finite.sort_values(kind="stable")
    maximum = 0.0
    total = len(ordered)
    for position, (index, value) in enumerate(ordered.items()):
        maximum = max(maximum, (total - position) * value)
        adjusted.loc[index] = min(maximum, 1.0)
    return adjusted


def epsilon_squared(h_statistic: float, total_n: int, groups: int) -> float:
    """Kruskal–Wallis epsilon-squared: max(0, (H - k + 1) / (N - k))."""
    if total_n <= groups:
        return np.nan
    return max(0.0, (float(h_statistic) - groups + 1) / (total_n - groups))


def _grade_summary(group: pd.DataFrame, grade: int) -> dict[str, float | int]:
    values = group.loc[group["true_kl"].eq(grade), "metric_value"].dropna()
    return {
        f"KL{grade} n": len(values),
        f"KL{grade} median": values.median() if len(values) else np.nan,
        f"KL{grade} Q1": values.quantile(0.25) if len(values) else np.nan,
        f"KL{grade} Q3": values.quantile(0.75) if len(values) else np.nan,
        f"KL{grade} IQR": values.quantile(0.75) - values.quantile(0.25) if len(values) else np.nan,
    }


def kl_stage_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise each model × XAI method × metric and apply Holm per model."""
    _require_prediction_metadata_consistency(results)
    _require_metric_configuration_consistency(results)
    status = results.get("metric_status", pd.Series("ok", index=results.index))
    numeric = results[status.eq("ok")].copy()
    numeric["metric_value"] = pd.to_numeric(numeric["metric_value"], errors="coerce")
    numeric = numeric[numeric["metric_value"].notna()]
    rows = []
    for identity, all_rows in results.groupby(list(TEST_ID_COLUMNS), sort=True):
        model_name, method, metric_name = identity
        group = numeric[
            numeric["model_name"].eq(model_name)
            & numeric["xai_method"].eq(method)
            & numeric["metric_name"].eq(metric_name)
        ]
        row: dict[str, object] = {
            "Model": model_name,
            "XAI method": method,
            "Metric": metric_name,
            "Valid N": len(group),
            "Excluded undefined": int((all_rows.get("metric_status", "ok") != "ok").sum()),
        }
        for grade in KL_GRADES:
            row.update(_grade_summary(group, grade))
        grade_values = [
            group.loc[group["true_kl"].eq(grade), "metric_value"].dropna().to_numpy()
            for grade in KL_GRADES
        ]
        non_empty = [values for values in grade_values if len(values)]
        row["KL groups"] = len(non_empty)
        if len(non_empty) < 2:
            row.update({"Kruskal–Wallis H": np.nan, "p raw": np.nan, "epsilon squared": np.nan, "Test status": "insufficient KL groups"})
        elif np.allclose(np.concatenate(non_empty), np.concatenate(non_empty)[0]):
            row.update({"Kruskal–Wallis H": 0.0, "p raw": 1.0, "epsilon squared": 0.0, "Test status": "all values identical"})
        else:
            statistic, p_value = kruskal(*non_empty)
            row.update({
                "Kruskal–Wallis H": statistic,
                "p raw": p_value,
                "epsilon squared": epsilon_squared(statistic, len(group), len(non_empty)),
                "Test status": "ok",
            })
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["p Holm"] = np.nan
    for model_name, indices in summary.groupby("Model").groups.items():
        summary.loc[list(indices), "p Holm"] = holm_adjust(summary.loc[list(indices), "p raw"])
    summary["Global significant"] = summary["p Holm"].lt(0.05)
    return summary


def dunn_posthoc(results: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    """Return Holm-corrected Dunn pairs only for Holm-significant global KL tests."""
    try:
        import scikit_posthocs as sp
    except ImportError as exc:
        raise ResearchAnalysisError("Dunn post-hoc requires the scikit-posthocs dependency.") from exc

    status = results.get("metric_status", pd.Series("ok", index=results.index))
    numeric = results[status.eq("ok")].copy()
    numeric["metric_value"] = pd.to_numeric(numeric["metric_value"], errors="coerce")
    numeric = numeric[numeric["metric_value"].notna()]
    comparisons = []
    significant = analysis[analysis["Global significant"]]
    for _, test in significant.iterrows():
        group = numeric[
            numeric["model_name"].eq(test["Model"])
            & numeric["xai_method"].eq(test["XAI method"])
            & numeric["metric_name"].eq(test["Metric"])
        ][["true_kl", "metric_value"]]
        if group["true_kl"].nunique() < 2:
            continue
        matrix = sp.posthoc_dunn(group, val_col="metric_value", group_col="true_kl", p_adjust="holm")
        for left in matrix.index:
            for right in matrix.columns:
                if int(left) >= int(right):
                    continue
                p_value = float(matrix.loc[left, right])
                comparisons.append({
                    "Model": test["Model"],
                    "XAI method": test["XAI method"],
                    "Metric": test["Metric"],
                    "KL pair": f"KL{int(left)} vs KL{int(right)}",
                    "p Dunn-Holm": p_value,
                    "Significant": p_value < 0.05,
                })
    return pd.DataFrame(comparisons)
