"""Reusable descriptive and primary KL-stage statistics for Research Mode."""

from __future__ import annotations

import json
from itertools import combinations
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal, rankdata, wilcoxon
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
XAI_METHOD_ORDER = (
    "Grad-CAM",
    "Grad-CAM++",
    "Eigen-CAM",
    "Integrated Gradients",
    "SmoothGrad",
)
METHOD_COMPARISON_METRICS = (
    "FaithfulnessCorrelation",
    "Deletion confidence drop",
    "MaxSensitivity",
    "Stability score",
    "Sparseness",
)
ARCHITECTURE_MODELS = ("densenet121", "resnet50")


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


def method_complete_case_matrices(results: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Return one five-method, complete-case matrix for every model and primary metric.

    Undefined or missing values are never imputed. A sample remains in a matrix only
    when all five methods have a finite ``metric_status == 'ok'`` value for that metric.
    The total number of candidate samples is retained in ``DataFrame.attrs``.
    """
    _require_unique_rows(results)
    _require_prediction_metadata_consistency(results)
    _require_metric_configuration_consistency(results)
    status = results.get("metric_status", pd.Series("ok", index=results.index))
    numeric = results[status.eq("ok")].copy()
    numeric["metric_value"] = pd.to_numeric(numeric["metric_value"], errors="coerce")
    numeric = numeric[numeric["metric_value"].notna() & np.isfinite(numeric["metric_value"])]

    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for model_name in sorted(results["model_name"].unique()):
        for metric_name in METHOD_COMPARISON_METRICS:
            all_rows = results[
                results["model_name"].eq(model_name)
                & results["metric_name"].eq(metric_name)
            ]
            candidate_samples = int(all_rows["sample_id"].nunique())
            valid_rows = numeric[
                numeric["model_name"].eq(model_name)
                & numeric["metric_name"].eq(metric_name)
            ]
            if valid_rows.empty:
                matrix = pd.DataFrame(columns=XAI_METHOD_ORDER, dtype=float)
                matrix.index.name = "sample_id"
            else:
                matrix = valid_rows.pivot(
                    index="sample_id", columns="xai_method", values="metric_value"
                ).reindex(columns=XAI_METHOD_ORDER)
                matrix = matrix.dropna(axis=0, how="any").sort_index()
            matrix.attrs["candidate_n"] = candidate_samples
            matrix.attrs["excluded_n"] = candidate_samples - len(matrix)
            available_methods = set(all_rows["xai_method"].unique())
            matrix.attrs["missing_methods"] = tuple(
                method for method in XAI_METHOD_ORDER if method not in available_methods
            )
            matrices[(str(model_name), metric_name)] = matrix
    return matrices


def method_comparison_values(matrices: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Convert complete-case matrices into long values for charts and summaries."""
    rows = []
    for (model_name, metric_name), matrix in matrices.items():
        if matrix.empty:
            continue
        long = (
            matrix.rename_axis(index="sample_id", columns="XAI method")
            .stack(future_stack=True)
            .rename("metric_value")
            .reset_index()
        )
        long.insert(0, "Metric", metric_name)
        long.insert(0, "Model", model_name)
        long["Valid N"] = len(matrix)
        long["Excluded N"] = int(matrix.attrs.get("excluded_n", 0))
        rows.append(long)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Model", "Metric", "sample_id", "XAI method", "metric_value", "Valid N", "Excluded N"]
    )


def method_comparison_descriptives(matrices: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Summarise each method using exactly the complete cases used for Friedman."""
    values = method_comparison_values(matrices)
    if values.empty:
        return pd.DataFrame(
            columns=["Model", "Metric", "XAI method", "N", "Median", "Q1", "Q3", "IQR"]
        )
    summary = (
        values.groupby(["Model", "Metric", "XAI method"], as_index=False)
        .agg(
            N=("metric_value", "size"),
            Median=("metric_value", "median"),
            Q1=("metric_value", lambda values: values.quantile(0.25)),
            Q3=("metric_value", lambda values: values.quantile(0.75)),
        )
    )
    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary["Metric"] = pd.Categorical(summary["Metric"], METHOD_COMPARISON_METRICS, ordered=True)
    summary["XAI method"] = pd.Categorical(summary["XAI method"], XAI_METHOD_ORDER, ordered=True)
    return summary.sort_values(["Model", "Metric", "XAI method"]).reset_index(drop=True)


def friedman_method_analysis(matrices: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Run one paired Friedman test per model and metric, with Holm per model."""
    rows = []
    for (model_name, metric_name), matrix in matrices.items():
        valid_n = len(matrix)
        missing_methods = tuple(matrix.attrs.get("missing_methods", ()))
        row: dict[str, object] = {
            "Model": model_name,
            "Metric": metric_name,
            "Valid N": valid_n,
            "Excluded N": int(matrix.attrs.get("excluded_n", 0)),
            "Friedman statistic": np.nan,
            "p raw": np.nan,
            "Kendall's W": np.nan,
            "Test status": "ok",
        }
        if missing_methods:
            row["Test status"] = "missing methods: " + ", ".join(missing_methods)
        elif valid_n < 2:
            row["Test status"] = "insufficient complete-case samples"
        elif np.allclose(matrix.to_numpy(), matrix.iloc[:, [0]].to_numpy()):
            row.update({"Friedman statistic": 0.0, "p raw": 1.0, "Kendall's W": 0.0, "Test status": "all paired method values identical"})
        else:
            statistic, p_value = friedmanchisquare(*(matrix[method].to_numpy() for method in XAI_METHOD_ORDER))
            if np.isfinite(statistic) and np.isfinite(p_value):
                row.update({
                    "Friedman statistic": float(statistic),
                    "p raw": float(p_value),
                    "Kendall's W": float(statistic) / (valid_n * (len(XAI_METHOD_ORDER) - 1)),
                })
            else:
                row["Test status"] = "undefined Friedman result"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["p Holm"] = np.nan
    for _, indices in summary.groupby("Model").groups.items():
        summary.loc[list(indices), "p Holm"] = holm_adjust(summary.loc[list(indices), "p raw"])
    summary["Friedman significant"] = summary["p Holm"].lt(0.05)
    summary["Metric"] = pd.Categorical(summary["Metric"], METHOD_COMPARISON_METRICS, ordered=True)
    return summary.sort_values(["Model", "Metric"]).reset_index(drop=True)


def wilcoxon_method_posthoc(
    matrices: dict[tuple[str, str], pd.DataFrame], friedman_summary: pd.DataFrame
) -> pd.DataFrame:
    """Run Holm-corrected paired Wilcoxon tests after a significant Friedman result."""
    comparisons = []
    significant = friedman_summary[friedman_summary["Friedman significant"]]
    for test in significant.itertuples(index=False):
        matrix = matrices[(test.Model, test.Metric)]
        pair_rows = []
        for left, right in combinations(XAI_METHOD_ORDER, 2):
            differences = matrix[left].to_numpy() - matrix[right].to_numpy()
            row: dict[str, object] = {
                "Model": test.Model,
                "Metric": test.Metric,
                "Method pair": f"{left} vs {right}",
                "Wilcoxon statistic": np.nan,
                "p Wilcoxon raw": np.nan,
                "Test status": "ok",
            }
            if np.allclose(differences, 0.0):
                row.update({"Wilcoxon statistic": 0.0, "p Wilcoxon raw": 1.0, "Test status": "all paired differences zero"})
            else:
                statistic, p_value = wilcoxon(
                    matrix[left], matrix[right], alternative="two-sided", zero_method="wilcox", method="auto"
                )
                row.update({"Wilcoxon statistic": float(statistic), "p Wilcoxon raw": float(p_value)})
            pair_rows.append(row)

        pair_frame = pd.DataFrame(pair_rows)
        pair_frame["p Wilcoxon Holm"] = holm_adjust(pair_frame["p Wilcoxon raw"])
        pair_frame["Significant"] = pair_frame["p Wilcoxon Holm"].lt(0.05)
        comparisons.append(pair_frame)
    return pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame(
        columns=[
            "Model", "Metric", "Method pair", "Wilcoxon statistic", "p Wilcoxon raw",
            "p Wilcoxon Holm", "Significant", "Test status",
        ]
    )


def _require_architecture_configuration_compatibility(results: pd.DataFrame) -> None:
    """Require each metric/method to use identical parameters for both architectures."""
    selected = results[results["model_name"].isin(ARCHITECTURE_MODELS)].copy()
    selected["_signature"] = selected["metric_parameters"].map(_canonical_parameters)
    signatures = selected.groupby(["xai_method", "metric_name"])["_signature"].nunique()
    mixed = signatures[signatures > 1]
    if not mixed.empty:
        method, metric = mixed.index[0]
        raise ResearchAnalysisError(
            "Architecture comparison requires matching metric parameters for DenseNet121 and ResNet50: "
            f"{method}/{metric} differs."
        )


def architecture_complete_case_pairs(
    results: pd.DataFrame, *, same_target_only: bool = False
) -> dict[tuple[str, str], pd.DataFrame]:
    """Pair DenseNet121 and ResNet50 values by sample for each method and metric.

    When ``same_target_only`` is set, only samples with the same predicted KL
    class in both architectures remain. This is a sensitivity analysis because
    XAI maps target the model's predicted class.
    """
    _require_unique_rows(results)
    _require_prediction_metadata_consistency(results)
    _require_metric_configuration_consistency(results)
    if not set(ARCHITECTURE_MODELS).issubset(set(results["model_name"].unique())):
        raise ResearchAnalysisError(
            "Architecture comparison requires both densenet121 and resnet50 result files."
        )
    _require_architecture_configuration_compatibility(results)

    predictions = prediction_records(results)
    predicted_wide = predictions[
        predictions["model_name"].isin(ARCHITECTURE_MODELS)
    ].pivot(index="sample_id", columns="model_name", values="predicted_kl")
    shared_prediction_samples = set(predicted_wide.dropna().index)
    if same_target_only:
        shared_prediction_samples = {
            sample_id
            for sample_id, row in predicted_wide.dropna().iterrows()
            if int(row["densenet121"]) == int(row["resnet50"])
        }

    selected = results[results["model_name"].isin(ARCHITECTURE_MODELS)].copy()
    status = selected.get("metric_status", pd.Series("ok", index=selected.index))
    numeric = selected[status.eq("ok")].copy()
    numeric["metric_value"] = pd.to_numeric(numeric["metric_value"], errors="coerce")
    numeric = numeric[numeric["metric_value"].notna() & np.isfinite(numeric["metric_value"])]

    pairs: dict[tuple[str, str], pd.DataFrame] = {}
    for method in XAI_METHOD_ORDER:
        for metric_name in METHOD_COMPARISON_METRICS:
            all_rows = selected[
                selected["xai_method"].eq(method) & selected["metric_name"].eq(metric_name)
            ]
            sample_sets = [
                set(all_rows.loc[all_rows["model_name"].eq(model), "sample_id"])
                for model in ARCHITECTURE_MODELS
            ]
            candidate_samples = set.intersection(*sample_sets) & shared_prediction_samples
            valid_rows = numeric[
                numeric["xai_method"].eq(method) & numeric["metric_name"].eq(metric_name)
            ]
            if valid_rows.empty:
                matrix = pd.DataFrame(columns=ARCHITECTURE_MODELS, dtype=float)
                matrix.index.name = "sample_id"
            else:
                matrix = valid_rows.pivot(
                    index="sample_id", columns="model_name", values="metric_value"
                ).reindex(index=sorted(candidate_samples), columns=ARCHITECTURE_MODELS)
                matrix = matrix.dropna(axis=0, how="any").sort_index()
            matrix.attrs["candidate_n"] = len(candidate_samples)
            matrix.attrs["excluded_n"] = len(candidate_samples) - len(matrix)
            matrix.attrs["same_target_only"] = same_target_only
            pairs[(method, metric_name)] = matrix
    return pairs


def architecture_comparison_values(pairs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Return paired architecture values in long form for metric-specific plots."""
    frames = []
    for (method, metric_name), matrix in pairs.items():
        if matrix.empty:
            continue
        long = (
            matrix.rename_axis(index="sample_id", columns="Model")
            .stack(future_stack=True)
            .rename("metric_value")
            .reset_index()
        )
        long.insert(0, "Metric", metric_name)
        long.insert(0, "XAI method", method)
        long["Paired N"] = len(matrix)
        long["Excluded N"] = int(matrix.attrs.get("excluded_n", 0))
        frames.append(long)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["XAI method", "Metric", "sample_id", "Model", "metric_value", "Paired N", "Excluded N"]
    )


def _architecture_distribution(prefix: str, values: pd.Series) -> dict[str, float]:
    return {
        f"{prefix} median": values.median(),
        f"{prefix} Q1": values.quantile(0.25),
        f"{prefix} Q3": values.quantile(0.75),
    }


def _rank_biserial_correlation(differences: np.ndarray) -> float:
    """Rank-biserial correlation where positive means DenseNet121 is higher."""
    nonzero = differences[~np.isclose(differences, 0.0)]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def architecture_comparison_analysis(pairs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Run paired DenseNet121-vs-ResNet50 Wilcoxon tests with Holm over 25 tests."""
    rows = []
    for (method, metric_name), matrix in pairs.items():
        paired_n = len(matrix)
        row: dict[str, object] = {
            "XAI method": method,
            "Metric": metric_name,
            "Paired N": paired_n,
            "Excluded N": int(matrix.attrs.get("excluded_n", 0)),
            "Median paired difference (DenseNet - ResNet)": np.nan,
            "Wilcoxon statistic": np.nan,
            "p raw": np.nan,
            "Rank-biserial correlation": np.nan,
            "Test status": "ok",
        }
        if paired_n:
            row.update(_architecture_distribution("DenseNet121", matrix["densenet121"]))
            row.update(_architecture_distribution("ResNet50", matrix["resnet50"]))
            differences = matrix["densenet121"].to_numpy() - matrix["resnet50"].to_numpy()
            row["Median paired difference (DenseNet - ResNet)"] = float(np.median(differences))
        else:
            row.update({
                "DenseNet121 median": np.nan, "DenseNet121 Q1": np.nan, "DenseNet121 Q3": np.nan,
                "ResNet50 median": np.nan, "ResNet50 Q1": np.nan, "ResNet50 Q3": np.nan,
            })
            row["Test status"] = "no complete paired samples"
        if paired_n < 2:
            if paired_n:
                row["Test status"] = "insufficient complete paired samples"
        elif np.allclose(differences, 0.0):
            row.update({
                "Wilcoxon statistic": 0.0,
                "p raw": 1.0,
                "Rank-biserial correlation": 0.0,
                "Test status": "all paired differences zero",
            })
        else:
            statistic, p_value = wilcoxon(
                matrix["densenet121"],
                matrix["resnet50"],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            row.update({
                "Wilcoxon statistic": float(statistic),
                "p raw": float(p_value),
                "Rank-biserial correlation": _rank_biserial_correlation(differences),
            })
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["p Holm"] = holm_adjust(summary["p raw"])
    summary["Architecture significant"] = summary["p Holm"].lt(0.05)
    summary["Metric"] = pd.Categorical(summary["Metric"], METHOD_COMPARISON_METRICS, ordered=True)
    summary["XAI method"] = pd.Categorical(summary["XAI method"], XAI_METHOD_ORDER, ordered=True)
    return summary.sort_values(["Metric", "XAI method"]).reset_index(drop=True)


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
