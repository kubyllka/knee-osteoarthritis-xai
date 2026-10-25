from io import BytesIO
import json
from pathlib import Path
import random
from zipfile import ZIP_DEFLATED, ZipFile

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image

from app.config import KL_LABELS, MODEL_CONFIGS
from app.evaluation import (
    build_deletion_metrics,
    build_stability_metrics,
    build_xai_metrics,
)
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.metric_registry import is_legacy_metric, metric_definition, parse_parameters
from app.quantus_metrics import evaluate_quantus_metrics
from app.research.analysis import (
    METHOD_COMPARISON_METRICS,
    ResearchAnalysisError,
    architecture_comparison_analysis,
    architecture_comparison_values,
    architecture_complete_case_pairs,
    classifier_subset_summary,
    dunn_posthoc,
    friedman_method_analysis,
    kl_stage_analysis,
    method_comparison_descriptives,
    method_comparison_values,
    method_complete_case_matrices,
    merge_research_results,
    prediction_correctness_analysis,
    prediction_correctness_values,
    prediction_records,
    wilcoxon_method_posthoc,
)
from app.research.results import ResearchResultsValidationError, load_research_results, validate_research_results
from app.research.toolkit import build_research_toolkit_zip
from app.xai import generate_xai_visualization, load_xai_model_by_name, predict_kl_with_xai_model, preprocess_pil_for_torch


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
RESEARCH_RESULTS_DIR = Path(__file__).resolve().parent / "results/research"
QUICK_FAITHFULNESS_RUNS = 8
QUICK_MAX_SENSITIVITY_SAMPLES = 2

# These orders make the two-model Results overview directly comparable. They
# deliberately do not imply that one metric or method is an overall winner.
PRIMARY_KL_METRIC_ORDER = [
    "FaithfulnessCorrelation",
    "Region perturbation score-drop AUC",
    "MaxSensitivity",
    "Stability score",
    "Sparseness",
]
KL_GRADE_LABELS = [f"KL{grade}" for grade in range(5)]


def apply_interface_styles() -> None:
    """Small, presentation-only styles; model and metric outputs are untouched."""
    st.markdown(
        """
        <style>
            .kneexai-brand {
                color: #17253E;
                font-size: 1.7rem;
                font-weight: 700;
                letter-spacing: -0.025em;
                line-height: 1.2;
                margin: 0 0 0.2rem;
            }
            .kneexai-subtitle {
                color: #475569;
                font-size: 0.98rem;
                margin: 0 0 0.55rem;
            }
            .kneexai-grade {
                color: #17253E;
                font-size: clamp(1.8rem, 3vw, 2.35rem);
                font-weight: 700;
                letter-spacing: -0.035em;
                line-height: 1.15;
                margin: 0 0 0.35rem;
            }
            .st-key-prediction-view img {
                max-height: 320px;
                object-fit: contain;
            }
            .st-key-prediction-view h2 {
                color: #17253E;
                font-size: 1.3rem;
                letter-spacing: -0.015em;
            }
            @media (max-width: 640px) {
                .kneexai-brand { font-size: 1.45rem; }
                .st-key-prediction-view img { max-height: 240px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_local_classifier(model_name: str) -> ONNXKneeOsteoarthritisClassifier:
    config = MODEL_CONFIGS[model_name]
    return ONNXKneeOsteoarthritisClassifier(config["onnx_path"], output_type=config.get("output_type", "multiclass"))


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


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def show_prediction(prediction: dict) -> None:
    grade = int(prediction["grade"])
    confidence = float(prediction["confidence"])

    st.caption("PREDICTED SEVERITY")
    st.markdown(f'<div class="kneexai-grade">KL Grade {grade}</div>', unsafe_allow_html=True)
    st.write(GRADE_MEANINGS.get(grade, KL_LABELS.get(grade, "Unknown grade")))

    metric_cols = st.columns(2)
    metric_cols[0].metric("Confidence", format_percent(confidence))
    metric_cols[1].metric("Inference time", f"{float(prediction.get('latency_ms', 0.0)):.1f} ms")

    probabilities = prediction["probabilities"]
    chart_data = pd.DataFrame(
        {
            "Grade": [label.replace("KL Grade ", "KL") for label in probabilities],
            "Probability": [float(probability) for probability in probabilities.values()],
        }
    )
    st.caption("Class scores · KL0–KL4")
    st.bar_chart(chart_data, x="Grade", y="Probability", color="#2563EB", height=220)


def all_option(values) -> list[str]:
    return ["All"] + sorted(str(value) for value in values if pd.notna(value))


def _metric_tooltip_format(display_format: str) -> str:
    return ".3f" if display_format == "{:.3f}" else ".4f"


def _metric_scale(metric_name: str):
    return alt.Scale(domain=[-1, 1]) if metric_name == "FaithfulnessCorrelation" else None


def _summary_by(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(columns, as_index=False)
        .agg(
            Samples=("sample_id", "nunique"),
            Median=("metric_value", "median"),
            Mean=("metric_value", "mean"),
            Q1=("metric_value", lambda values: values.quantile(0.25)),
            Q3=("metric_value", lambda values: values.quantile(0.75)),
        )
        .assign(IQR=lambda table: table["Q3"] - table["Q1"])
    )


def _format_summary(summary: pd.DataFrame, display_format: str):
    return summary.style.format(
        {
            "Median": display_format,
            "Mean": display_format,
            "Q1": display_format,
            "Q3": display_format,
            "IQR": display_format,
        }
    )


DIMENSIONS = (
    ("Faithfulness", "faithfulness"),
    ("Robustness", "robustness"),
    ("Complexity", "complexity"),
    ("Custom checks", "custom"),
)


def _metric_category(metric_name: str, csv_category: str) -> str:
    """Use current definitions for known metrics, including older CSVs."""
    definition = metric_definition(metric_name)
    return definition.category if definition else csv_category


def _is_custom_check(metric_name: str, csv_source: str | None) -> bool:
    """Keep project-specific checks visible even when their scientific dimension differs."""
    definition = metric_definition(metric_name)
    if definition:
        return metric_name in {"Deletion confidence drop", "Stability score"}
    return csv_source == "custom"


def _selected_metric_rows(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Keep every visual summary strictly within one metric scale."""
    selected = frame[frame["metric_name"] == metric_name].copy()
    if selected["metric_name"].nunique() != 1:
        raise ValueError("Dashboard summaries must contain exactly one metric.")
    return selected


def _median_by_method_and_grade(frame: pd.DataFrame, definition) -> pd.DataFrame:
    """One comparable median per method and true KL grade for the dashboard heatmap."""
    summary = _summary_by(frame, ["xai_method", "true_kl"])
    method_order = (
        summary.groupby("xai_method", as_index=False)["Median"].median()
        .sort_values("Median", ascending=definition.score_direction == "lower")["xai_method"]
        .tolist()
    )
    summary["KL grade"] = summary["true_kl"].map(lambda grade: f"KL{grade}")
    summary["xai_method"] = pd.Categorical(summary["xai_method"], categories=method_order, ordered=True)
    return summary


def _median_heatmap(frame: pd.DataFrame, definition, display_format: str):
    summary = _median_by_method_and_grade(frame, definition)
    summary["Median label"] = summary["Median"].map(lambda value: display_format.format(value))
    metric_name = frame["metric_name"].iloc[0]
    color_scale = alt.Scale(domain=[-1, 1], scheme="redblue") if metric_name == "FaithfulnessCorrelation" else alt.Scale(scheme="blues")
    rect = alt.Chart(summary).mark_rect(cornerRadius=3).encode(
        x=alt.X("KL grade:O", title="True KL grade", sort=["KL0", "KL1", "KL2", "KL3", "KL4"]),
        y=alt.Y("xai_method:N", title="XAI method", sort=summary["xai_method"].cat.categories.tolist()),
        color=alt.Color("Median:Q", title="Median score", scale=color_scale),
        tooltip=[
            "xai_method:N",
            "KL grade:O",
            "Samples:Q",
            alt.Tooltip("Median:Q", format=_metric_tooltip_format(display_format)),
            alt.Tooltip("IQR:Q", format=_metric_tooltip_format(display_format)),
        ],
    )
    labels = alt.Chart(summary).mark_text(fontSize=12).encode(
        x=alt.X("KL grade:O", sort=["KL0", "KL1", "KL2", "KL3", "KL4"]),
        y=alt.Y("xai_method:N", sort=summary["xai_method"].cat.categories.tolist()),
        text="Median label:N",
    )
    return (rect + labels).properties(height=max(150, 42 * len(summary["xai_method"].cat.categories)))


def _outcome_heatmap(frame: pd.DataFrame, definition, display_format: str):
    """Method-by-outcome medians without mixing metrics or drawing fragile facets."""
    summary = _summary_by(frame, ["xai_method", "is_correct"])
    summary["Prediction outcome"] = summary["is_correct"].map({False: "Incorrect", True: "Correct"})
    method_order = (
        summary.groupby("xai_method", as_index=False)["Median"].median()
        .sort_values("Median", ascending=definition.score_direction == "lower")["xai_method"]
        .tolist()
    )
    summary["xai_method"] = pd.Categorical(summary["xai_method"], categories=method_order, ordered=True)
    summary["Median label"] = summary["Median"].map(lambda value: display_format.format(value))
    metric_name = frame["metric_name"].iloc[0]
    color_scale = alt.Scale(domain=[-1, 1], scheme="redblue") if metric_name == "FaithfulnessCorrelation" else alt.Scale(scheme="blues")
    rect = alt.Chart(summary).mark_rect(cornerRadius=3).encode(
        x=alt.X("Prediction outcome:O", title="Prediction outcome", sort=["Incorrect", "Correct"]),
        y=alt.Y("xai_method:N", title="XAI method", sort=method_order),
        color=alt.Color("Median:Q", title="Median score", scale=color_scale),
        tooltip=[
            "xai_method:N",
            "Prediction outcome:O",
            "Samples:Q",
            alt.Tooltip("Median:Q", format=_metric_tooltip_format(display_format)),
            alt.Tooltip("IQR:Q", format=_metric_tooltip_format(display_format)),
        ],
    )
    labels = alt.Chart(summary).mark_text(fontSize=12).encode(
        x=alt.X("Prediction outcome:O", sort=["Incorrect", "Correct"]),
        y=alt.Y("xai_method:N", sort=method_order),
        text="Median label:N",
    )
    return (rect + labels).properties(height=max(150, 42 * len(method_order)))


def _median_by_kl_bar_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Wide table used by Streamlit's reliable grouped-bar renderer."""
    summary = _summary_by(frame, ["true_kl", "xai_method"])
    summary["KL grade"] = summary["true_kl"].map(lambda grade: f"KL{grade}")
    data = summary.pivot(index="KL grade", columns="xai_method", values="Median")
    data = data.reindex(["KL0", "KL1", "KL2", "KL3", "KL4"])
    data.index.name = "True KL grade"
    return data


def _summary_card_values(filtered: pd.DataFrame, definition) -> dict[str, object]:
    samples = filtered["sample_id"].nunique()
    sample_rows = filtered.drop_duplicates("sample_id")
    correct_rate = sample_rows["is_correct"].mean()
    by_method = _summary_by(filtered, ["xai_method"])
    by_grade = _summary_by(filtered, ["true_kl"])
    ascending = definition.score_direction == "lower"
    method = by_method.sort_values("Median", ascending=ascending).iloc[0]
    best_grade = by_grade.sort_values("Median", ascending=ascending).iloc[0]
    worst_grade = by_grade.sort_values("Median", ascending=not ascending).iloc[0]
    selected_label = "Lowest" if ascending else "Highest"
    opposite_label = "Highest" if ascending else "Lowest"

    return {
        "samples": samples,
        "correct_rate": correct_rate,
        "method_label": f"{selected_label} median for selected metric",
        "method": str(method["xai_method"]),
        "best_grade_label": f"{selected_label} median KL grade",
        "best_grade": int(best_grade["true_kl"]),
        "worst_grade_label": f"{opposite_label} median KL grade",
        "worst_grade": int(worst_grade["true_kl"]),
    }


def _summary_cards(filtered: pd.DataFrame, definition) -> None:
    values = _summary_card_values(filtered, definition)
    cards = st.columns(5)
    cards[0].metric("Unique images", str(values["samples"]))
    cards[1].metric("Correct predictions", f"{values['correct_rate']:.0%}")
    cards[2].metric(values["method_label"], values["method"])
    cards[3].metric(values["best_grade_label"], f"KL{values['best_grade']}")
    cards[4].metric(values["worst_grade_label"], f"KL{values['worst_grade']}")


def _show_metric_dashboard(filtered: pd.DataFrame, selected_metric: str) -> None:
    all_rows = filtered.copy()
    status = all_rows.get("metric_status", pd.Series("ok", index=all_rows.index))
    undefined_rows = all_rows[status.ne("ok")]
    filtered = all_rows[status.eq("ok")].copy()

    definition = metric_definition(selected_metric)
    if definition is None:
        st.warning("This legacy metric has no registered display metadata. Its raw rows are shown below.")
        with st.expander("Filtered research rows"):
            st.dataframe(all_rows, use_container_width=True, hide_index=True)
        return

    display_format = definition.display_format
    st.subheader(selected_metric)
    direction_text = (
        "Higher means more concentrated, not necessarily better."
        if definition.category == "complexity"
        else f"{definition.score_direction.title()} is better for this metric."
    )
    st.caption(f"**{direction_text}** {definition.description} {definition.interpretation}")
    if not undefined_rows.empty:
        st.warning(
            f"{len(undefined_rows)} metric row(s) from {undefined_rows['sample_id'].nunique()} image(s) "
            "were not evaluated because a perturbed input produced an all-zero attribution map. "
            "They are excluded from summaries and charts; inspect them in the detailed rows below."
        )
    if filtered.empty:
        st.info("No valid numeric values remain for this metric under the active filters.")
        with st.expander("Filtered research rows"):
            st.dataframe(all_rows, use_container_width=True, hide_index=True)
        return

    _summary_cards(filtered, definition)
    if definition.category == "complexity":
        st.info("Sparseness is descriptive: 'most concentrated' does not mean clinically best explanation.")
    if is_legacy_metric(selected_metric):
        st.warning("Legacy metric: Infidelity is retained only for historical CSV inspection and is excluded from new runs.")

    parameter_values = []
    if "metric_parameters" in filtered.columns:
        parameter_values = [parse_parameters(value) for value in filtered["metric_parameters"].dropna().unique()]
        parameter_values = [value for value in parameter_values if value is not None]
    if not parameter_values:
        st.warning("Legacy result file: exact metric configuration was not stored with these rows.")
    else:
        with st.expander("Metric configuration"):
            st.json(parameter_values[0] if len(parameter_values) == 1 else parameter_values)
        if selected_metric == "FaithfulnessCorrelation" and min(item.get("nr_runs", 0) for item in parameter_values) < 8:
            st.warning("This FaithfulnessCorrelation run uses fewer than 8 perturbation runs; it is a technical preview, not thesis evidence.")

    group_summary = _summary_by(filtered, ["true_kl", "xai_method"])
    if group_summary["Samples"].min() < 10:
        st.warning("Small-sample comparison: at least one KL × method group has fewer than 10 images. Inspect patterns, but do not draw statistical conclusions.")

    st.subheader(f"Median {selected_metric} by true KL grade")
    direction = (
        "Higher values mean a more concentrated map; this is descriptive, not automatically better."
        if definition.category == "complexity"
        else "Lower values are better." if definition.score_direction == "lower" else "Higher values are better."
    )
    st.caption(f"Each bar is the median across images in one KL-grade × XAI-method group. {direction} N and IQR are available in the table below.")
    st.bar_chart(
        _median_by_kl_bar_data(filtered),
        height=360,
        stack=False,
        use_container_width=True,
    )

    st.subheader("Median XAI score by true KL grade")
    st.caption("Each cell is the median for one XAI method and KL grade. Hover for sample count and IQR; do not compare different metrics with each other.")
    st.altair_chart(_median_heatmap(filtered, definition, display_format), use_container_width=True)

    st.subheader("Median XAI score: correct vs incorrect predictions")
    if filtered["is_correct"].nunique() < 2:
        st.info("No correct-vs-incorrect comparison is available under the active filters because all selected images have the same prediction outcome.")
    else:
        st.altair_chart(_outcome_heatmap(filtered, definition, display_format), use_container_width=True)

    st.subheader("KL grade × XAI method summary")
    summary = group_summary.rename(columns={"true_kl": "True KL grade"})
    st.dataframe(_format_summary(summary, display_format), use_container_width=True, hide_index=True)
    with st.expander("Filtered research rows"):
        st.dataframe(all_rows, use_container_width=True, hide_index=True)


def _format_results_analysis_table(table: pd.DataFrame):
    formats = {
        "Accuracy": "{:.1%}",
        "Balanced accuracy": "{:.1%}",
        "Macro F1": "{:.3f}",
        "Quadratic weighted kappa": "{:.3f}",
        "p raw": "{:.3g}",
        "p Holm": "{:.3g}",
        "Kruskal–Wallis H": "{:.3f}",
        "epsilon squared": "{:.3f}",
        "p Dunn-Holm": "{:.3g}",
        "Friedman statistic": "{:.3f}",
        "Kendall's W": "{:.3f}",
        "Wilcoxon statistic": "{:.3f}",
        "p Wilcoxon raw": "{:.3g}",
        "p Wilcoxon Holm": "{:.3g}",
        "Rank-biserial correlation": "{:.3f}",
        "Median paired difference (DenseNet - ResNet)": "{:.4g}",
        "Mann–Whitney U": "{:.3f}",
        "KL-adjusted permutation p raw": "{:.3g}",
        "KL-adjusted permutation p Holm": "{:.3g}",
    }
    for column in table.columns:
        if column in {"Median", "Q1", "Q3", "IQR"} or any(column.endswith(suffix) for suffix in ("median", "Q1", "Q3", "IQR")):
            formats[column] = "{:.4g}"
    return table.style.format(formats, na_rep="—")


def _classifier_accuracy_chart_data(classifier_by_grade: pd.DataFrame) -> pd.DataFrame:
    """Prepare one labelled accuracy bar for each model and true KL grade."""
    data = classifier_by_grade.copy()
    data["True KL grade"] = pd.Categorical(
        data["True KL grade"], categories=KL_GRADE_LABELS, ordered=True
    )
    data["Accuracy label"] = data["Accuracy"].map(lambda value: f"{value:.1%}")
    data["Correct / images"] = data.apply(
        lambda row: f"{int(row['Correct'])} / {int(row['Images'])}", axis=1
    )
    return data.sort_values(["True KL grade", "Model"]).reset_index(drop=True)


def _classifier_accuracy_chart(classifier_by_grade: pd.DataFrame):
    data = _classifier_accuracy_chart_data(classifier_by_grade)
    data["Model display"] = data["Model"].map(_model_label)
    model_order = [_model_label(name) for name in sorted(data["Model"].unique())]
    color_scale = alt.Scale(
        domain=model_order,
        range=["#2563eb", "#e76f51", "#7c3aed", "#0f766e"][: len(model_order)],
    )
    base = alt.Chart(data).encode(
        x=alt.X("True KL grade:N", title="True KL grade", sort=KL_GRADE_LABELS),
        xOffset=alt.XOffset("Model display:N", sort=model_order),
        color=alt.Color("Model display:N", title="Model", sort=model_order, scale=color_scale),
    )
    bars = base.mark_bar().encode(
        y=alt.Y(
            "Accuracy:Q",
            title="Classification accuracy",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("Model display:N", title="Model"),
            alt.Tooltip("True KL grade:N"),
            alt.Tooltip("Correct:Q"),
            alt.Tooltip("Images:Q", title="Images in XAI subset"),
            alt.Tooltip("Accuracy:Q", format=".1%"),
        ],
    )
    labels = base.mark_text(dy=-9, color="#1f2937", fontSize=11).encode(
        y=alt.Y("Accuracy:Q"),
        text="Accuracy label:N",
    )
    return (bars + labels).properties(height=330)


def _confusion_matrix_chart_data(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reshape existing classifier counts for display; do not recalculate predictions."""
    counts = matrix.copy()
    counts.index.name = "True KL"
    counts.columns.name = "Predicted KL"
    data = counts.stack().rename("Images").reset_index()
    data["True KL"] = data["True KL"].str.replace("True ", "", regex=False)
    data["Predicted KL"] = data["Predicted KL"].str.replace("Predicted ", "", regex=False)
    return data


def _confusion_matrix_chart(matrix: pd.DataFrame):
    data = _confusion_matrix_chart_data(matrix)
    max_count = max(int(data["Images"].max()), 1)
    base = alt.Chart(data).encode(
        x=alt.X("Predicted KL:N", sort=KL_GRADE_LABELS, title="Predicted KL grade"),
        y=alt.Y("True KL:N", sort=KL_GRADE_LABELS, title="True KL grade"),
    )
    tiles = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color("Images:Q", title="Images", scale=alt.Scale(domain=[0, max_count], scheme="blues")),
        tooltip=["True KL:N", "Predicted KL:N", alt.Tooltip("Images:Q", title="Images")],
    )
    labels = base.mark_text(fontSize=14, fontWeight="bold").encode(
        text="Images:Q",
        color=alt.condition(f"datum.Images > {max_count / 2}", alt.value("white"), alt.value("#17253E")),
    )
    return (tiles + labels).properties(height=330)


def _effect_map_data(analysis: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return a complete metric-by-method grid for one model's KL-stage effects."""
    selected = analysis[analysis["Model"].eq(model_name)].copy()
    keys = ["Metric", "XAI method"]
    if selected.duplicated(keys).any():
        raise ValueError(f"Effect map has duplicate model/metric/method rows for {model_name}.")
    grid = pd.MultiIndex.from_product(
        [PRIMARY_KL_METRIC_ORDER, XAI_METHODS], names=keys
    ).to_frame(index=False)
    columns = [
        "Metric",
        "XAI method",
        "epsilon squared",
        "Kruskal–Wallis H",
        "p raw",
        "p Holm",
        "Valid N",
        "Excluded undefined",
        "Global significant",
    ]
    data = grid.merge(selected[[column for column in columns if column in selected.columns]], on=keys, how="left")
    data["Effect label"] = data.apply(
        lambda row: (
            "—"
            if pd.isna(row["epsilon squared"])
            else f"{float(row['epsilon squared']):.3f}{'*' if bool(row.get('Global significant', False)) else ''}"
        ),
        axis=1,
    )
    data["Significance"] = data["Global significant"].map(
        {True: "Holm-significant", False: "Not Holm-significant"}
    ).fillna("No result")
    data["Metric"] = pd.Categorical(data["Metric"], categories=PRIMARY_KL_METRIC_ORDER, ordered=True)
    data["XAI method"] = pd.Categorical(data["XAI method"], categories=XAI_METHODS, ordered=True)
    return data.sort_values(["Metric", "XAI method"]).reset_index(drop=True)


def _effect_map_scale_max(analysis: pd.DataFrame) -> float:
    values = pd.to_numeric(analysis.get("epsilon squared"), errors="coerce").dropna()
    return max(float(values.max()), 0.01) if not values.empty else 1.0


def _effect_map_chart(effect_data: pd.DataFrame, model_name: str, scale_max: float):
    color = alt.condition(
        "isValid(datum['epsilon squared'])",
        alt.Color(
            "epsilon squared:Q",
            title="KL-stage effect (ε²)",
            scale=alt.Scale(domain=[0, scale_max], scheme="blues"),
        ),
        alt.value("#edf0f4"),
    )
    base = alt.Chart(effect_data).encode(
        x=alt.X("XAI method:N", title=None, sort=XAI_METHODS, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("Metric:N", title=None, sort=PRIMARY_KL_METRIC_ORDER),
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=color,
        tooltip=[
            alt.Tooltip("Metric:N"),
            alt.Tooltip("XAI method:N"),
            alt.Tooltip("epsilon squared:Q", title="ε²", format=".3f"),
            alt.Tooltip("Kruskal–Wallis H:Q", title="Kruskal–Wallis H", format=".3f"),
            alt.Tooltip("p raw:Q", format=".3g"),
            alt.Tooltip("p Holm:Q", format=".3g"),
            alt.Tooltip("Valid N:Q"),
            alt.Tooltip("Excluded undefined:Q"),
            alt.Tooltip("Significance:N"),
        ],
    )
    labels = base.mark_text(fontSize=11).encode(
        text="Effect label:N",
        color=alt.condition(
            "datum['epsilon squared'] >= 0.35", alt.value("white"), alt.value("#1f2937")
        ),
    )
    return (cells + labels).properties(title=model_name, height=260)


def _method_comparison_plot_data(values: pd.DataFrame, metric_name: str, model_name: str) -> pd.DataFrame:
    """Select one metric and model only; cross-metric plots are never valid."""
    data = values[
        values["Metric"].eq(metric_name) & values["Model"].eq(model_name)
    ].copy()
    data["XAI method"] = pd.Categorical(data["XAI method"], categories=XAI_METHODS, ordered=True)
    return data.sort_values(["XAI method", "sample_id"]).reset_index(drop=True)


def _method_comparison_y_scale(values: pd.DataFrame, metric_name: str) -> alt.Scale:
    if metric_name == "FaithfulnessCorrelation":
        return alt.Scale(domain=[-1, 1])
    metric_values = pd.to_numeric(
        values.loc[values["Metric"].eq(metric_name), "metric_value"], errors="coerce"
    ).dropna()
    if metric_values.empty:
        return alt.Scale()
    lower, upper = float(metric_values.min()), float(metric_values.max())
    if metric_name == "MaxSensitivity":
        # The current experiments contain positive values with a very long right
        # tail. A logarithmic scale retains outliers without flattening all boxes.
        return alt.Scale(type="log", domain=[lower * 0.8, upper * 1.2])
    padding = max((upper - lower) * 0.08, abs(upper) * 0.04, 0.01)
    return alt.Scale(domain=[lower - padding, upper + padding])


def _method_comparison_chart(
    values: pd.DataFrame,
    descriptives: pd.DataFrame,
    friedman_row: pd.Series,
    metric_name: str,
    model_name: str,
    y_scale: alt.Scale,
):
    data = _method_comparison_plot_data(values, metric_name, model_name)
    summary = descriptives[
        descriptives["Metric"].eq(metric_name) & descriptives["Model"].eq(model_name)
    ].copy()
    summary["Valid N"] = int(friedman_row["Valid N"])
    summary["Excluded N"] = int(friedman_row["Excluded N"])
    summary["XAI method"] = pd.Categorical(summary["XAI method"], categories=XAI_METHODS, ordered=True)
    summary["Median label"] = summary["Median"].map(lambda value: f"{value:.3g}")

    color_scale = alt.Scale(
        domain=XAI_METHODS,
        range=["#2563eb", "#e76f51", "#7c3aed", "#0f766e", "#d97706"],
    )
    base = alt.Chart(data).encode(
        x=alt.X("XAI method:N", title="XAI method", sort=XAI_METHODS, axis=alt.Axis(labelAngle=-30)),
        color=alt.Color("XAI method:N", legend=None, scale=color_scale),
    )
    boxes = base.mark_boxplot(size=42).encode(
        y=alt.Y(
            "metric_value:Q",
            title=f"{metric_name} (log scale)" if metric_name == "MaxSensitivity" else metric_name,
            scale=y_scale,
        ),
    )
    medians = alt.Chart(summary).mark_point(shape="diamond", filled=True, color="#111827", size=68).encode(
        x=alt.X("XAI method:N", sort=XAI_METHODS),
        y=alt.Y("Median:Q", scale=y_scale),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("XAI method:N"),
            alt.Tooltip("N:Q"),
            alt.Tooltip("Valid N:Q"),
            alt.Tooltip("Excluded N:Q"),
            alt.Tooltip("Median:Q", format=".4g"),
            alt.Tooltip("Q1:Q", format=".4g"),
            alt.Tooltip("Q3:Q", format=".4g"),
            alt.Tooltip("IQR:Q", format=".4g"),
        ],
    )
    labels = alt.Chart(summary).mark_text(dy=-13, color="#111827", fontSize=10).encode(
        x=alt.X("XAI method:N", sort=XAI_METHODS),
        y=alt.Y("Median:Q", scale=y_scale),
        text="Median label:N",
    )
    return (boxes + medians + labels).properties(title=_model_label(model_name), height=330)


def _method_metric_caption(metric_name: str) -> str:
    definition = metric_definition(metric_name)
    if definition is None:
        return "Metric metadata are unavailable."
    if metric_name == "Sparseness":
        return f"{definition.description} Concentration only: higher values are not automatically better."
    if metric_name == "Region perturbation score-drop AUC":
        return f"{definition.description} Higher is a stronger supporting faithfulness signal, but it is influenced partly by original model confidence."
    if metric_name == "MaxSensitivity":
        return f"{definition.description} Lower values are better. The Y-axis is logarithmic to retain the full positive range, including large outliers."
    return f"{definition.description} {definition.score_direction.title()} values are better."


def _architecture_comparison_plot_data(values: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Keep one metric scale in each architecture-comparison chart."""
    data = values[values["Metric"].eq(metric_name)].copy()
    data["XAI method"] = pd.Categorical(data["XAI method"], categories=XAI_METHODS, ordered=True)
    data["Model"] = pd.Categorical(data["Model"], categories=list(values["Model"].unique()), ordered=True)
    return data.sort_values(["XAI method", "Model", "sample_id"]).reset_index(drop=True)


def _architecture_median_summary(analysis: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    selected = analysis[analysis["Metric"].eq(metric_name)]
    rows = []
    for _, row_data in selected.iterrows():
        model_pair = tuple(row_data.get("Model pair", ("densenet121", "resnet50")))
        for model_name in model_pair:
            label = "DenseNet121" if model_name == "densenet121" else "ResNet50" if model_name == "resnet50" else model_name
            rows.append(
                {
                    "XAI method": row_data["XAI method"],
                    "Model": model_name,
                    "Model label": label,
                    "Median": row_data[f"{label} median"],
                    "Paired N": row_data["Paired N"],
                    "Excluded N": row_data["Excluded N"],
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["Median label"] = summary["Median"].map(lambda value: "—" if pd.isna(value) else f"{value:.3g}")
    summary["XAI method"] = pd.Categorical(summary["XAI method"], categories=XAI_METHODS, ordered=True)
    summary["Model"] = pd.Categorical(summary["Model"], categories=list(summary["Model"].unique()), ordered=True)
    return summary.sort_values(["XAI method", "Model"]).reset_index(drop=True)


def _architecture_comparison_chart(values: pd.DataFrame, analysis: pd.DataFrame, metric_name: str):
    data = _architecture_comparison_plot_data(values, metric_name)
    summary = _architecture_median_summary(analysis, metric_name)
    y_scale = _method_comparison_y_scale(values, metric_name)
    data["Model display"] = data["Model"].map(_model_label)
    summary["Model display"] = summary["Model"].map(_model_label)
    model_pair = [_model_label(name) for name in data["Model"].cat.categories]
    color_scale = alt.Scale(domain=model_pair, range=["#e76f51", "#2563eb"])
    base = alt.Chart(data).encode(
        x=alt.X("XAI method:N", title="XAI method", sort=XAI_METHODS, axis=alt.Axis(labelAngle=-30)),
        xOffset=alt.XOffset("Model display:N", sort=model_pair),
        color=alt.Color("Model display:N", title="Model", sort=model_pair, scale=color_scale),
    )
    boxes = base.mark_boxplot(size=30).encode(
        y=alt.Y(
            "metric_value:Q",
            title=f"{metric_name} (log scale)" if metric_name == "MaxSensitivity" else metric_name,
            scale=y_scale,
        ),
    )
    medians = alt.Chart(summary).mark_point(shape="diamond", filled=True, color="#111827", size=48).encode(
        x=alt.X("XAI method:N", sort=XAI_METHODS),
        xOffset=alt.XOffset("Model display:N", sort=model_pair),
        y=alt.Y("Median:Q", scale=y_scale),
        tooltip=[
            alt.Tooltip("Model label:N", title="Model"),
            alt.Tooltip("XAI method:N"),
            alt.Tooltip("Paired N:Q"),
            alt.Tooltip("Excluded N:Q"),
            alt.Tooltip("Median:Q", format=".4g"),
        ],
    )
    labels = alt.Chart(summary).mark_text(dy=-11, color="#111827", fontSize=9).encode(
        x=alt.X("XAI method:N", sort=XAI_METHODS),
        xOffset=alt.XOffset("Model display:N", sort=model_pair),
        y=alt.Y("Median:Q", scale=y_scale),
        text="Median label:N",
    )
    return (boxes + medians + labels).properties(height=350)


def _architecture_effect_heatmap_data(
    analysis: pd.DataFrame, models: tuple[str, str]
) -> pd.DataFrame:
    """Place existing paired-test effects on a fixed 5 × 5 display grid."""
    if "Model pair" in analysis.columns and any(
        tuple(pair) != models for pair in analysis["Model pair"]
    ):
        raise ValueError("Heatmap results do not match the selected model order.")
    keys = ["Metric", "XAI method"]
    if analysis.duplicated(keys).any():
        raise ValueError("Heatmap results contain duplicate method × metric tests.")
    grid = pd.MultiIndex.from_product(
        [METHOD_COMPARISON_METRICS, XAI_METHODS], names=keys
    ).to_frame(index=False)
    data = grid.merge(analysis, on=keys, how="left", validate="one_to_one")
    effect = pd.to_numeric(data["Rank-biserial correlation"], errors="coerce")
    adjusted_p = pd.to_numeric(data["p Holm"], errors="coerce")
    paired_n = pd.to_numeric(data["Paired N"], errors="coerce")
    valid = (
        data["Test status"].isin(("ok", "all paired differences zero"))
        & np.isfinite(effect) & effect.between(-1, 1)
        & np.isfinite(adjusted_p) & paired_n.ge(2)
    )
    data["Display effect"] = effect.where(valid)
    data["Display significant"] = valid & adjusted_p.lt(0.05)
    data["Cell label"] = [
        (f"{value:+.2f}".replace("-", "−") + ("*" if significant else ""))
        if pd.notna(value) else "N/A"
        for value, significant in zip(data["Display effect"], data["Display significant"])
    ]
    return data


def _architecture_figure_title(models: tuple[str, str]) -> str:
    names = {
        "densenet121": "DenseNet-121",
        "resnet50": "ResNet-50",
    }
    first, second = (
        names.get(MODEL_CONFIGS.get(model, {}).get("architecture"), _model_label(model))
        for model in models
    )
    return f"{first} variants" if first == second else f"{first} vs {second}"


def _architecture_effect_heatmap_figure(data: pd.DataFrame, models: tuple[str, str]):
    """Render the paired-model effect map without recalculating any tests."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.figure import Figure

    matrix = data["Display effect"].to_numpy(dtype=float).reshape(5, 5)
    fig = Figure(figsize=(10.6, 5.2), layout="constrained")
    ax = fig.subplots()
    cmap = LinearSegmentedColormap.from_list(
        "kneexai_soft_diverging",
        ["#B86960", "#E8CBC2", "#F7F8F7", "#CFDFEA", "#5B89B0"],
    ).with_extremes(bad="#edf0f4")  # Negative = warm red; positive = blue.
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=Normalize(-1, 1), aspect="auto")
    ax.set_xticks(range(5), ["Grad-CAM", "Grad-CAM++", "EigenCAM", "Integrated Gradients", "SmoothGrad"])
    ax.set_yticks(range(5), [
        "Faithfulness Correlation", "Region Perturbation", "MaxSensitivity",
        "Stability Score", "Sparseness",
    ])
    for tick in ax.get_xticklabels():
        tick.set_rotation(-35)
        tick.set_horizontalalignment("left")
        tick.set_rotation_mode("anchor")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="#F8FAFC", linewidth=1.5)
    ax.tick_params(axis="both", which="both", length=0, labelsize=10, colors="#475569")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, row in data.reset_index(drop=True).iterrows():
        value = row["Display effect"]
        text_color = "#17253E"
        if pd.notna(value):
            shade = np.asarray(cmap((float(value) + 1) / 2)[:3])
            linear = np.where(shade <= 0.04045, shade / 12.92, ((shade + 0.055) / 1.055) ** 2.4)
            luminance = float(np.dot(linear, (0.2126, 0.7152, 0.0722)))
            white_contrast = 1.05 / (luminance + 0.05)
            dark_contrast = (luminance + 0.05) / (0.018 + 0.05)
            text_color = "white" if white_contrast >= dark_contrast else "#17253E"
        ax.text(
            index % 5, index // 5, row["Cell label"], ha="center", va="center",
            color=text_color,
            weight="bold" if bool(row["Display significant"]) else "normal", fontsize=11,
        )
    ax.set_title(_architecture_figure_title(models), loc="left", color="#17253E", fontsize=12, pad=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Rank-biserial correlation", fontsize=9, color="#475569", labelpad=9)
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=9, colors="#475569", length=0)
    colorbar.outline.set_edgecolor("#E2E8F0")
    return fig


def _architecture_heatmap_bytes(figure, format_name: str) -> bytes:
    output = BytesIO()
    figure.savefig(output, format=format_name, dpi=300, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def _correctness_effect_heatmap_data(
    analysis: pd.DataFrame, model_name: str, *, kl_adjusted: bool
) -> pd.DataFrame:
    selected = analysis[analysis["Model"].eq(model_name)].copy()
    keys = ["Metric", "XAI method"]
    grid = pd.MultiIndex.from_product(
        [METHOD_COMPARISON_METRICS, XAI_METHODS], names=keys
    ).to_frame(index=False)
    data = grid.merge(selected, on=keys, how="left")
    p_raw = "KL-adjusted permutation p raw" if kl_adjusted else "p raw"
    p_holm = "KL-adjusted permutation p Holm" if kl_adjusted else "p Holm"
    significant = "KL-adjusted significant" if kl_adjusted else "Primary significant"
    data["Selected raw p"] = data[p_raw]
    data["Selected Holm p"] = data[p_holm]
    data["Selected significant"] = data[significant].eq(True)
    data["Effect label"] = data.apply(
        lambda row: "—" if pd.isna(row["Rank-biserial correlation"])
        else f"{float(row['Rank-biserial correlation']):.2f}{'*' if bool(row['Selected significant']) else ''}",
        axis=1,
    )
    data["Metric"] = pd.Categorical(data["Metric"], METHOD_COMPARISON_METRICS, ordered=True)
    data["XAI method"] = pd.Categorical(data["XAI method"], XAI_METHODS, ordered=True)
    return data.sort_values(["Metric", "XAI method"]).reset_index(drop=True)


def _correctness_effect_heatmap(data: pd.DataFrame, model_name: str):
    color = alt.condition(
        "isValid(datum['Rank-biserial correlation'])",
        alt.Color(
            "Rank-biserial correlation:Q",
            title="Correct − incorrect rank-biserial effect",
            scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
        ),
        alt.value("#edf0f4"),
    )
    base = alt.Chart(data).encode(
        x=alt.X("XAI method:N", title=None, sort=XAI_METHODS, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("Metric:N", title=None, sort=METHOD_COMPARISON_METRICS),
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=color,
        tooltip=[
            alt.Tooltip("Metric:N"),
            alt.Tooltip("XAI method:N"),
            alt.Tooltip("Correct N:Q"),
            alt.Tooltip("Incorrect N:Q"),
            alt.Tooltip("Excluded N:Q"),
            alt.Tooltip("Correct median:Q", format=".4g"),
            alt.Tooltip("Incorrect median:Q", format=".4g"),
            alt.Tooltip("Rank-biserial correlation:Q", format=".3f"),
            alt.Tooltip("Selected raw p:Q", format=".3g"),
            alt.Tooltip("Selected Holm p:Q", format=".3g"),
        ],
    )
    labels = base.mark_text(fontSize=11).encode(
        text="Effect label:N",
        color=alt.condition(
            "abs(datum['Rank-biserial correlation']) >= 0.45", alt.value("white"), alt.value("#1f2937")
        ),
    )
    return (cells + labels).properties(title=model_name, height=260)


def _correctness_distribution_chart(
    values: pd.DataFrame,
    analysis_row: pd.Series,
    metric_name: str,
):
    data = values[
        values["Model"].eq(analysis_row["Model"])
        & values["XAI method"].eq(analysis_row["XAI method"])
        & values["Metric"].eq(metric_name)
    ].copy()
    data["Prediction outcome"] = data["Correct"].map({True: "Correct", False: "Incorrect"})
    data["Prediction outcome"] = pd.Categorical(
        data["Prediction outcome"], categories=["Incorrect", "Correct"], ordered=True
    )
    y_scale = _method_comparison_y_scale(data, metric_name)
    return alt.Chart(data).mark_boxplot(size=55).encode(
        x=alt.X("Prediction outcome:N", title="Prediction outcome", sort=["Incorrect", "Correct"]),
        y=alt.Y(
            "metric_value:Q",
            title=f"{metric_name} (log scale)" if metric_name == "MaxSensitivity" else metric_name,
            scale=y_scale,
        ),
        color=alt.Color(
            "Prediction outcome:N",
            legend=None,
            scale=alt.Scale(domain=["Incorrect", "Correct"], range=["#e76f51", "#2563eb"]),
        ),
        tooltip=[
            alt.Tooltip("Prediction outcome:N"),
            alt.Tooltip("metric_value:Q", format=".4g"),
            alt.Tooltip("true_kl:Q", title="True KL grade"),
        ],
    ).properties(height=340)


@st.cache_data(show_spinner="Calculating KL-stratified correctness sensitivity tests…")
def _cached_prediction_correctness_analysis(results: pd.DataFrame) -> pd.DataFrame:
    return prediction_correctness_analysis(results)


@st.cache_data(show_spinner=False)
def _cached_classifier_summary(results: pd.DataFrame):
    return classifier_subset_summary(results)


@st.cache_data(show_spinner=False)
def _cached_kl_analysis(results: pd.DataFrame):
    return kl_stage_analysis(results)


@st.cache_data(show_spinner=False)
def _cached_method_analysis(results: pd.DataFrame):
    matrices = method_complete_case_matrices(results)
    return (
        matrices,
        method_comparison_values(matrices),
        method_comparison_descriptives(matrices),
        friedman_method_analysis(matrices),
    )


@st.cache_data(show_spinner=False)
def _cached_architecture_analysis(results: pd.DataFrame, models: tuple[str, str], same_target: bool):
    pairs = architecture_complete_case_pairs(results, models=models, same_target_only=same_target)
    return architecture_comparison_values(pairs), architecture_comparison_analysis(pairs)


@st.cache_data(show_spinner=False)
def _cached_dunn_analysis(results: pd.DataFrame, analysis: pd.DataFrame):
    return dunn_posthoc(results, analysis)


@st.cache_data(show_spinner=False)
def _cached_method_posthoc(matrices, summary):
    return wilcoxon_method_posthoc(matrices, summary)


@st.cache_data(show_spinner=False)
def _cached_toolkit() -> bytes:
    return build_research_toolkit_zip(Path(__file__).resolve().parent)


def _model_label(model_id: str) -> str:
    return MODEL_CONFIGS.get(model_id, {}).get("display_name", model_id)


def _validate_shared_cohort(results: pd.DataFrame) -> None:
    """Do not interpret colliding sample IDs as paired images when metadata differ."""
    records = prediction_records(results)
    shared = records[records.duplicated("sample_id", keep=False)]
    for field in ("image_path", "true_kl", "split"):
        field_rows = (
            shared if field in shared.columns
            else results.drop_duplicates(["model_name", "sample_id"])
        )
        conflicts = field_rows[field_rows.duplicated("sample_id", keep=False)].groupby("sample_id")[field].nunique(dropna=False)
        if (conflicts > 1).any():
            sample_id = str(conflicts[conflicts > 1].index[0])
            raise ResearchAnalysisError(
                f"Shared sample ID {sample_id} has conflicting {field} across models. "
                "Load CSVs from compatible image cohorts before comparing them."
            )
    if "selected_sample_content_sha256" in results.columns:
        hashes = results.groupby("model_name")["selected_sample_content_sha256"].first().dropna()
        model_sets = [frozenset(group["sample_id"]) for _, group in results.groupby("model_name")]
        if len(hashes) > 1 and hashes.nunique() > 1 and len(set(model_sets)) == 1:
            raise ResearchAnalysisError(
                "Loaded models record different selected-image content hashes. "
                "Do not combine these experiment cohorts."
            )


def _download_csv(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label, frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename, mime="text/csv", key=key,
    )


def _research_cohort(results: pd.DataFrame) -> pd.DataFrame:
    model_sets = {name: set(group["sample_id"]) for name, group in results.groupby("model_name")}
    options = ["All loaded images"]
    if len(model_sets) > 1:
        options.append("Images shared by all loaded models")
    base_manifest = Path(__file__).resolve().parent / "inputs/test/test_manifest.json"
    original_ids = set()
    if base_manifest.is_file():
        try:
            original_ids = {
                str(item["sample_id"])
                for item in json.loads(base_manifest.read_text(encoding="utf-8"))["samples"]
            }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if original_ids:
        options.append(f"Images in repository test manifest ({len(original_ids)})")
    different = len({frozenset(ids) for ids in model_sets.values()}) > 1
    if different:
        st.warning(
            "The loaded models use different image sets. Per-model summaries use available images; "
            "paired tests retain shared sample IDs only."
        )
        if "selected_sample_content_sha256" in results.columns:
            st.caption("Dataset content hashes differ across runs because their selected image cohorts differ; verify provenance before interpreting paired tests.")
    if len(model_sets) > 1 and not set.intersection(*model_sets.values()):
        st.warning("No sample IDs are shared by all loaded models; paired comparison is unavailable.")
    cohort = st.selectbox(
        "Analysis cohort", options, index=0,
        help="Applies to analysis sections. Explore & Export has additional local filters.",
        key="research-cohort",
    )
    if cohort == "Images shared by all loaded models":
        results = results[results["sample_id"].isin(set.intersection(*model_sets.values()))]
    elif cohort.startswith("Images in repository"):
        results = results[results["sample_id"].isin(original_ids)]
    return results.copy()


def _research_data_summary(results: pd.DataFrame) -> None:
    records = prediction_records(results)
    status = results.get("metric_status", pd.Series("ok", index=results.index))
    values = pd.to_numeric(results["metric_value"], errors="coerce")
    undefined = int((status.ne("ok") | ~np.isfinite(values)).sum())
    cards = st.columns(5)
    for card, label, value in zip(
        cards,
        ("Models", "Unique images", "XAI methods", "Metrics", "Undefined values"),
        (results["model_name"].nunique(), records["sample_id"].nunique(),
         results["xai_method"].nunique(), results["metric_name"].nunique(), undefined),
    ):
        card.metric(label, str(value))
    if undefined:
        st.warning(f"{undefined} metric result(s) are undefined and are excluded from tests without imputation.")
    absent_methods = [name for name in XAI_METHODS if name not in set(results["xai_method"])]
    absent_metrics = [name for name in METHOD_COMPARISON_METRICS if name not in set(results["metric_name"])]
    if absent_methods or absent_metrics:
        st.warning(
            "Incomplete XAI result set. Missing methods: "
            + (", ".join(absent_methods) or "none")
            + ". Missing metrics: " + (", ".join(absent_metrics) or "none") + "."
        )
    for model_name, model_rows in results.groupby("model_name", sort=True):
        missing_methods = set(XAI_METHODS) - set(model_rows["xai_method"])
        missing_metrics = set(METHOD_COMPARISON_METRICS) - set(model_rows["metric_name"])
        if missing_methods or missing_metrics:
            st.warning(
                f"{_model_label(model_name)} has incomplete results: "
                f"missing methods {', '.join(sorted(missing_methods)) or 'none'}; "
                f"missing metrics {', '.join(sorted(missing_metrics)) or 'none'}."
            )


def _show_research_overview(results: pd.DataFrame) -> None:
    st.subheader("Overview")
    st.caption("Classifier statistics describe the loaded XAI image subset, not automatically the full held-out test set.")
    try:
        summary, grades, matrices = _cached_classifier_summary(results)
    except ResearchAnalysisError as exc:
        st.error(f"Classifier-subset analysis unavailable: {exc}")
        return
    records = prediction_records(results)
    counts = records.groupby("true_kl")["sample_id"].nunique().reindex(range(5), fill_value=0)
    st.caption("Image distribution by true KL grade: " + " · ".join(f"KL{k}: {int(n)}" for k, n in counts.items()))
    with st.container(border=True):
        st.markdown("#### Classification accuracy by true KL grade")
        st.altair_chart(_classifier_accuracy_chart(grades), use_container_width=True)
    with st.container(border=True):
        st.markdown("#### Classification performance · loaded XAI subset")
        compact = summary.copy()
        compact["Model"] = compact["Model"].map(_model_label)
        st.dataframe(_format_results_analysis_table(compact), use_container_width=True, hide_index=True)
    selected = st.selectbox(
        "Confusion matrix model", list(matrices),
        format_func=_model_label, key="overview-confusion-model",
    )
    with st.container(border=True):
        st.caption(f"Confusion matrix · {_model_label(selected)}")
        st.altair_chart(_confusion_matrix_chart(matrices[selected]), use_container_width=True)
        with st.expander("View confusion matrix counts as a table"):
            st.dataframe(matrices[selected], use_container_width=True)
    outcomes = records.groupby("model_name")["is_correct"].agg(["sum", "count"]).reset_index()
    outcomes["Incorrect"] = outcomes["count"] - outcomes["sum"]
    outcomes["Model"] = outcomes["model_name"].map(_model_label)
    st.caption("Correct and incorrect predictions in the loaded XAI subset")
    st.dataframe(
        outcomes[["Model", "sum", "Incorrect"]].rename(columns={"sum": "Correct"}),
        use_container_width=True, hide_index=True,
    )
    with st.expander("Per-KL classifier details"):
        st.dataframe(_format_results_analysis_table(grades), use_container_width=True, hide_index=True)
        _download_csv("Download classifier subset summary", summary, "classifier_subset_summary.csv", "download-classifier-summary")
        _download_csv("Download per-KL classifier details", grades, "classifier_by_kl_grade.csv", "download-classifier-kl")


def _kl_distribution_chart(results: pd.DataFrame, model: str, metric: str):
    status = results.get("metric_status", pd.Series("ok", index=results.index))
    data = results[
        results["model_name"].eq(model) & results["metric_name"].eq(metric) & status.eq("ok")
    ].copy()
    data["metric_value"] = pd.to_numeric(data["metric_value"], errors="coerce")
    data = data[np.isfinite(data["metric_value"])]
    data["KL grade"] = data["true_kl"].map(lambda value: f"KL{value}")
    if data.empty:
        return None
    scale = _method_comparison_y_scale(
        data.rename(columns={"metric_name": "Metric"}), metric
    )
    return alt.Chart(data).mark_boxplot(size=24).encode(
        x=alt.X("KL grade:N", sort=KL_GRADE_LABELS, title="True KL grade"),
        xOffset=alt.XOffset("xai_method:N", sort=XAI_METHODS),
        y=alt.Y("metric_value:Q", title=metric, scale=scale),
        color=alt.Color("xai_method:N", title="XAI method", sort=XAI_METHODS),
        tooltip=[
            alt.Tooltip("KL grade:N"),
            alt.Tooltip("xai_method:N", title="XAI method"),
            alt.Tooltip("metric_value:Q", format=".4g"),
        ],
    ).properties(height=340)


def _show_kl_stage(results: pd.DataFrame) -> None:
    st.subheader("KL-stage Analysis")
    st.caption("Associations with true KL grade are metric-specific; they do not establish a causal KL effect or an overall XAI ranking.")
    try:
        analysis = _cached_kl_analysis(results)
    except ResearchAnalysisError as exc:
        st.error(f"KL-stage analysis unavailable: {exc}")
        return
    models = sorted(results["model_name"].unique())
    selectors = st.columns(2)
    model = selectors[0].selectbox("Model", models, format_func=_model_label, key="kl-model")
    metric = selectors[1].selectbox("XAI metric", list(METHOD_COMPARISON_METRICS), key="kl-metric")
    st.caption(_method_metric_caption(metric))
    with st.container(border=True):
        st.markdown("#### Distribution across KL0–KL4")
        chart = _kl_distribution_chart(results, model, metric)
        if chart is None:
            st.info("No finite values are available for this model and metric.")
        else:
            st.altair_chart(chart, use_container_width=True)
    st.markdown("#### KL-stage effect map")
    st.caption("Cells show ε² for one metric × method. * denotes a Holm-significant Kruskal–Wallis test. A common colour scale is used across models.")
    scale = _effect_map_scale_max(analysis)
    for offset in range(0, len(models), 2):
        columns = st.columns(min(2, len(models) - offset))
        for column, model_name in zip(columns, models[offset:offset + 2]):
            with column:
                with st.container(border=True):
                    st.altair_chart(
                        _effect_map_chart(_effect_map_data(analysis, model_name), _model_label(model_name), scale),
                        use_container_width=True,
                    )
    selected = analysis[analysis["Model"].eq(model) & analysis["Metric"].eq(metric)]
    st.markdown("#### Kruskal–Wallis results")
    st.caption(
        "Null hypothesis: this metric has the same distribution across true KL groups for a given model and XAI method. "
        "Holm correction is applied within each model; ε² = max(0, (H − k + 1) / (N − k)). "
        "The effect is not a measure of clinical explanation quality."
    )
    st.dataframe(_format_results_analysis_table(selected), use_container_width=True, hide_index=True)
    if not selected.empty:
        n_columns = [f"KL{k} n" for k in range(5)]
        if (selected[n_columns] < 10).any().any():
            st.warning("At least one selected KL group has fewer than ten valid images.")
        excluded = int(selected["Excluded undefined"].sum())
        if excluded:
            st.warning(f"{excluded} undefined observation(s) were excluded without imputation.")
    family_complete = (
        analysis.groupby("Model").size().eq(25).all()
        and analysis["KL groups"].eq(5).all()
    )
    if not family_complete:
        st.warning("The primary 25-test family or five KL groups are incomplete; interpret results as exploratory.")
    with st.expander("Complete KL-stage statistical table and CSV"):
        st.dataframe(_format_results_analysis_table(analysis), use_container_width=True, hide_index=True)
        _download_csv("Download KL-stage summary", analysis, "kl_stage_analysis.csv", "download-kl")
    with st.expander("Dunn post-hoc KL comparisons"):
        if not family_complete:
            st.info("Dunn results require the complete primary KL analysis for every selected model.")
        else:
            try:
                posthoc = _cached_dunn_analysis(results, analysis)
            except ResearchAnalysisError as exc:
                st.error(f"Dunn post-hoc unavailable: {exc}")
            else:
                if posthoc.empty:
                    st.info("No Holm-significant global KL test is available for Dunn pairwise comparisons.")
                else:
                    st.caption("Pairs are tested only after a Holm-significant global result; the ten pairwise p-values are Holm-adjusted within that test.")
                    st.dataframe(_format_results_analysis_table(posthoc), use_container_width=True, hide_index=True)
                    _download_csv("Download Dunn comparisons", posthoc, "dunn_posthoc_kl_pairs.csv", "download-dunn")


def _show_xai_methods(results: pd.DataFrame) -> None:
    st.subheader("XAI Methods")
    st.caption("Five methods are compared on paired complete-case images within each model and metric. There is no cross-metric score or universal winner.")
    metric = st.selectbox("XAI metric", list(METHOD_COMPARISON_METRICS), key="methods-metric")
    st.caption(_method_metric_caption(metric))
    try:
        matrices, values, descriptives, summary = _cached_method_analysis(results)
    except ResearchAnalysisError as exc:
        st.error(f"XAI-method analysis unavailable: {exc}")
        return
    st.caption(
        "Complete CSV exports cover every available model and metric in the active research cohort, "
        "regardless of the metric selected for the charts below. Wilcoxon pairs exist only after "
        "a Holm-significant Friedman test for that model and metric."
    )
    friedman_download, wilcoxon_download = st.columns(2)
    with friedman_download:
        _download_csv(
            "Download all Friedman paired tests", summary,
            "friedman_xai_method_analysis_all_metrics.csv", "download-friedman-all",
        )
    try:
        all_posthoc = _cached_method_posthoc(matrices, summary)
    except ResearchAnalysisError as exc:
        all_posthoc = None
        st.warning(f"Wilcoxon post-hoc unavailable: {exc}")
    with wilcoxon_download:
        if all_posthoc is not None and not all_posthoc.empty:
            _download_csv(
                "Download all Wilcoxon method pairs", all_posthoc,
                "wilcoxon_xai_method_pairs_all_metrics.csv", "download-method-posthoc-all",
            )
        elif all_posthoc is not None:
            st.caption("No Holm-significant Friedman tests; no Wilcoxon pairs were generated.")
    selected = summary[summary["Metric"].eq(metric)]
    if selected.empty or values[values["Metric"].eq(metric)].empty:
        st.info("No complete paired samples are available for this metric.")
        return
    scale = _method_comparison_y_scale(values, metric)
    rows = list(selected.iterrows())
    for offset in range(0, len(rows), 2):
        columns = st.columns(min(2, len(rows) - offset))
        for column, (_, row) in zip(columns, rows[offset:offset + 2]):
            with column:
                with st.container(border=True):
                    st.caption(f"{_model_label(str(row['Model']))}: valid N {int(row['Valid N'])}; excluded N {int(row['Excluded N'])}")
                    st.altair_chart(
                        _method_comparison_chart(values, descriptives, row, metric, str(row["Model"]), scale),
                        use_container_width=True,
                    )
    st.markdown("#### Median and IQR · complete-case images")
    st.dataframe(
        _format_results_analysis_table(descriptives[descriptives["Metric"].eq(metric)]),
        use_container_width=True, hide_index=True,
    )
    st.markdown("#### Friedman paired test")
    st.caption(
        "Null hypothesis: the five XAI methods have equal rank distributions on the same images. "
        "Holm corrects five metric tests per model; Kendall's W is the effect size."
    )
    st.dataframe(_format_results_analysis_table(selected), use_container_width=True, hide_index=True)
    with st.expander("Wilcoxon post-hoc method pairs"):
        if all_posthoc is None:
            st.error("Wilcoxon post-hoc unavailable for this cohort.")
        else:
            posthoc = all_posthoc[all_posthoc["Metric"].eq(metric)] if not all_posthoc.empty else all_posthoc
            if posthoc.empty:
                st.info("No Holm-significant Friedman result for this metric; no pairwise test is shown.")
            else:
                st.caption("Two-sided paired tests; Holm correction over ten method pairs within each significant model–metric test.")
                st.dataframe(_format_results_analysis_table(posthoc), use_container_width=True, hide_index=True)
                _download_csv("Download Wilcoxon pairs", posthoc, f"wilcoxon_{metric}.csv", "download-method-posthoc")
    with st.expander("Full XAI-method tables and CSV"):
        st.dataframe(_format_results_analysis_table(summary), use_container_width=True, hide_index=True)
        _download_csv("Download method descriptives", descriptives, "xai_method_descriptives.csv", "download-method-descriptive")


MODEL_COMPARISON_OPTIONS = (
    ("Architecture · ordinal BCE", ("densenet121", "resnet50")),
    ("Architecture · class-aware ordinal BCE", ("densenet121_class_aware", "resnet50_class_aware")),
    ("Training variant · DenseNet121", ("densenet121_class_aware", "densenet121")),
    ("Training variant · ResNet50", ("resnet50_class_aware", "resnet50")),
)


def _show_paired_models(results: pd.DataFrame) -> None:
    available = set(results["model_name"])
    choices = [(label, pair) for label, pair in MODEL_COMPARISON_OPTIONS if set(pair).issubset(available)]
    if not choices:
        st.info("Load a compatible model pair to enable paired model analysis.")
        return
    choice = st.selectbox("Comparison", [label for label, _ in choices], key="paired-model-choice")
    pair = next(models for label, models in choices if label == choice)
    if not set(results.loc[results["model_name"].eq(pair[0]), "sample_id"]) & set(
        results.loc[results["model_name"].eq(pair[1]), "sample_id"]
    ):
        st.warning("These models have no shared sample IDs, so a paired comparison is unavailable.")
        return
    mode = st.radio(
        "Analysis population",
        ["Primary · all paired images", "Sensitivity · same predicted KL target"],
        horizontal=True, key="paired-model-mode",
    )
    same_target = mode.startswith("Sensitivity")
    if same_target:
        st.warning("Sensitivity analysis retains only images with the same predicted KL target in both models.")
    else:
        st.caption("Different predicted KL targets are retained and are not hidden in this primary analysis.")
    try:
        values, summary = _cached_architecture_analysis(results, pair, same_target)
    except ResearchAnalysisError as exc:
        st.info(f"Paired model comparison unavailable: {exc}")
        return
    metric = st.selectbox("XAI metric", list(METHOD_COMPARISON_METRICS), key="paired-model-metric")
    st.caption(_method_metric_caption(metric))
    if values[values["Metric"].eq(metric)].empty:
        st.info("No finite paired values are available for this metric.")
    else:
        st.altair_chart(_architecture_comparison_chart(values, summary, metric), use_container_width=True)
    selected = summary[summary["Metric"].eq(metric)]
    st.caption(
        "Two-sided Wilcoxon signed-rank tests compare identical image × method × metric pairs. "
        "Holm correction covers 25 tests for this model pair. Paired N and excluded N are reported."
    )
    st.dataframe(_format_results_analysis_table(selected), use_container_width=True, hide_index=True)
    st.markdown("#### Paired-model effect sizes")
    heatmap_data = _architecture_effect_heatmap_data(summary, pair)
    mode_label = "Same target" if same_target else "All pairs"
    figure = _architecture_effect_heatmap_figure(heatmap_data, pair)
    png = _architecture_heatmap_bytes(figure, "png")
    svg = _architecture_heatmap_bytes(figure, "svg")
    figure.clear()
    st.image(png, use_container_width=True)
    st.caption(
        f"Model A: {_model_label(pair[0])}; Model B: {_model_label(pair[1])}; {mode_label}. "
        "Blue: higher values for A; warm red: higher values for B; * Holm-adjusted p < 0.05. "
        "Interpret each metric separately."
    )
    paired_images = int(values["sample_id"].nunique()) if not values.empty else 0
    reduced = heatmap_data[pd.to_numeric(heatmap_data["Paired N"], errors="coerce").lt(paired_images)]
    st.caption(
        f"{paired_images} unique images have at least one finite model pair in this mode. "
        f"{len(reduced)} of 25 method × metric tests use fewer pairs; see Paired N in the table."
    )
    suffix = "same_target" if same_target else "all_pairs"
    filename = f"model_comparison_{pair[0]}_vs_{pair[1]}_{suffix}_heatmap"
    png_col, svg_col = st.columns(2)
    with png_col:
        st.download_button(
            "Download heatmap (PNG)", png, file_name=f"{filename}.png", mime="image/png",
            key="download-paired-heatmap-png",
        )
    with svg_col:
        st.download_button(
            "Download heatmap (SVG)", svg, file_name=f"{filename}.svg", mime="image/svg+xml",
            key="download-paired-heatmap-svg",
        )
    with st.expander("All paired-model Wilcoxon results"):
        st.dataframe(_format_results_analysis_table(summary), use_container_width=True, hide_index=True)
        _download_csv(
            "Download paired-model results", summary,
            f"model_comparison_{pair[0]}_vs_{pair[1]}_{suffix}.csv", "download-paired-models",
        )


def _show_correctness(results: pd.DataFrame) -> None:
    st.subheader("Prediction Correctness")
    st.caption(
        "Compare correct and incorrect predictions within each model, not DenseNet-121 against ResNet-50. "
        "Correctness means predicted KL = true KL. Statistical significance does not prove clinical utility "
        "or explanation faithfulness. No observations are pooled across methods or metrics."
    )
    try:
        values = prediction_correctness_values(results)
        summary = _cached_prediction_correctness_analysis(results)
    except ResearchAnalysisError as exc:
        st.info(f"Prediction correctness analysis unavailable: {exc}")
        return
    mode = st.radio(
        "Inference mode",
        ["Primary · pooled correct vs incorrect", "KL-adjusted sensitivity · stratified permutation"],
        horizontal=True, key="correctness-mode",
    )
    adjusted = mode.startswith("KL-adjusted")
    st.caption(
        "Cells are pooled rank-biserial effects: positive means higher metric values for correct predictions. "
        "A * denotes Holm significance in this mode. The direction of preference depends on the metric."
    )
    models = sorted(summary["Model"].unique())
    for offset in range(0, len(models), 2):
        columns = st.columns(min(2, len(models) - offset))
        for column, model in zip(columns, models[offset:offset + 2]):
            with column:
                st.altair_chart(
                    _correctness_effect_heatmap(
                        _correctness_effect_heatmap_data(summary, model, kl_adjusted=adjusted),
                        _model_label(model),
                    ),
                    use_container_width=True,
                )
    st.caption(
        "Primary: two-sided Mann–Whitney U with Holm correction across 25 tests per model. "
        "Sensitivity: 10,000 fixed-seed permutations within true-KL strata, with a separate Holm correction."
    )
    selectors = st.columns(3)
    model = selectors[0].selectbox("Model", models, format_func=_model_label, key="correctness-model")
    metric = selectors[1].selectbox("Metric", list(METHOD_COMPARISON_METRICS), key="correctness-metric")
    method = selectors[2].selectbox("XAI method", XAI_METHODS, key="correctness-method")
    chosen = summary[
        summary["Model"].eq(model) & summary["Metric"].eq(metric) & summary["XAI method"].eq(method)
    ]
    if chosen.empty:
        st.info("No comparison exists for this selection.")
        return
    row = chosen.iloc[0]
    st.altair_chart(_correctness_distribution_chart(values, row, metric), use_container_width=True)
    p_column = "KL-adjusted permutation p Holm" if adjusted else "p Holm"
    st.caption(
        f"Correct N: {int(row['Correct N'])}; incorrect N: {int(row['Incorrect N'])}; "
        f"excluded N: {int(row['Excluded N'])}; adjusted p: {row[p_column]:.3g}. "
        "KL adjustment is a sensitivity analysis, not a replacement for the primary comparison."
    )
    with st.expander("All prediction-correctness statistics"):
        st.dataframe(_format_results_analysis_table(summary), use_container_width=True, hide_index=True)
        _download_csv(
            "Download prediction-correctness results", summary,
            "prediction_correctness_analysis.csv", "download-correctness",
        )


def _show_model_comparison(results: pd.DataFrame) -> None:
    st.subheader("Model Comparison")
    st.caption(
        "Paired model configurations are compared on the same images, XAI methods and metrics. "
        "All tests are metric-specific; no overall architecture winner is inferred."
    )
    _show_paired_models(results)


def _show_explore_export(results: pd.DataFrame) -> None:
    st.subheader("Explore & Export")
    st.caption("Filters below affect only this exploratory view. Other research sections use the selected analysis cohort without these filters.")
    columns = st.columns(4)
    model_ids = sorted(results["model_name"].unique())
    model = columns[0].selectbox(
        "Model", [*model_ids, "All"],
        format_func=lambda item: "All models" if item == "All" else _model_label(item),
        key="explore-model",
    )
    method = columns[1].selectbox("XAI method", ["All", *[m for m in XAI_METHODS if m in set(results["xai_method"])]], key="explore-method")
    grades = columns[2].multiselect("True KL grade", list(range(5)), default=list(range(5)), key="explore-kl")
    correctness = columns[3].selectbox("Prediction correctness", ["All", "Correct", "Incorrect"], key="explore-correctness")
    filtered = results[results["true_kl"].isin(grades)].copy()
    if model != "All":
        filtered = filtered[filtered["model_name"].eq(model)]
    if method != "All":
        filtered = filtered[filtered["xai_method"].eq(method)]
    if correctness != "All":
        filtered = filtered[filtered["is_correct"].eq(correctness == "Correct")]
    st.caption(
        f"Exploratory scope: {filtered['sample_id'].nunique()} image(s), {len(filtered)} metric row(s) "
        f"from the {len(results)}-row analysis cohort."
    )
    if not filtered.empty:
        filtered["dashboard_category"] = [
            _metric_category(metric_name, category)
            for metric_name, category in zip(filtered["metric_name"], filtered["metric_category"])
        ]
        sources = filtered.get("metric_source", pd.Series(None, index=filtered.index))
        filtered["is_custom_check"] = [
            _is_custom_check(metric_name, source)
            for metric_name, source in zip(filtered["metric_name"], sources)
        ]
        dimension = st.radio(
            "Evaluation dimension", [label for label, _ in DIMENSIONS],
            horizontal=True, key="explore-dimension",
        )
        category = dict(DIMENSIONS)[dimension]
        category_rows = (
            filtered[filtered["is_custom_check"]]
            if category == "custom" else filtered[filtered["dashboard_category"].eq(category)]
        )
        if category_rows.empty:
            st.info(f"No {dimension.lower()} results match the exploratory filters.")
        else:
            metrics = sorted(category_rows["metric_name"].unique())
            metric = st.selectbox("Metric", metrics, key=f"explore-metric-{category}")
            metric_rows = _selected_metric_rows(category_rows, metric)
            if model == "All" and metric_rows["model_name"].nunique() > 1:
                displayed_model = st.selectbox(
                    "Model to display", sorted(metric_rows["model_name"].unique()),
                    format_func=_model_label, key="explore-displayed-model",
                )
                st.caption("Switch models here to inspect each separately. The filtered CSV download still includes all selected models.")
                _show_metric_dashboard(metric_rows[metric_rows["model_name"].eq(displayed_model)], metric)
            else:
                _show_metric_dashboard(metric_rows, metric)
        with st.expander("Filtered metric rows and CSV"):
            st.dataframe(filtered.drop(columns=["dashboard_category", "is_custom_check"]), use_container_width=True, hide_index=True)
            _download_csv(
                "Download filtered results", filtered.drop(columns=["dashboard_category", "is_custom_check"]),
                "filtered_research_results.csv", "download-filtered",
            )
    else:
        st.info("No rows match the exploratory filters.")


def _show_research_toolkit() -> None:
    """Offer the data-free runner before the user needs to have any results."""
    with st.container(border=True):
        try:
            toolkit = _cached_toolkit()
        except OSError as exc:
            st.error(f"Research toolkit cannot be packaged: {exc}")
            return
        heading, action = st.columns([3, 1], vertical_alignment="center")
        with heading:
            st.markdown("**Reproduce an XAI experiment**")
            st.caption("Runner, statistical analysis and setup guide. No CSV required to download.")
        with action:
            st.download_button(
                "Research Toolkit (.zip)", toolkit,
                file_name="kneexai_research_toolkit.zip", mime="application/zip",
                key="download-research-toolkit-top", use_container_width=True,
            )
        with st.expander("How to run the experiment · detailed instructions"):
            st.markdown(
                "1. Install dependencies and provide matching `.pth` and `.onnx` models.\n"
                "2. Prepare a labelled JSON manifest and run `scripts/run_research.py`.\n"
                "3. Load the resulting CSV here, or run `scripts/analyze_research_results.py` offline."
            )
            try:
                guide = (Path(__file__).resolve().parent / "research_toolkit/README.md").read_text(encoding="utf-8")
            except OSError as exc:
                st.error(f"Research toolkit guide is unavailable: {exc}")
            else:
                st.markdown(guide)
            st.download_button(
                "Download Research Toolkit (.zip)", toolkit,
                file_name="kneexai_research_toolkit.zip", mime="application/zip",
                key="download-research-toolkit-guide",
            )


def show_research_mode() -> None:
    st.header("Research Mode")
    st.caption("Run or reproduce an XAI experiment, then analyze its precomputed results here.")

    _show_research_toolkit()

    with st.container(border=True):
        source_mode = st.radio(
            "Research results source",
            ["Repository CSV files", "Upload CSV files"],
            horizontal=True, key="research-source",
        )
        try:
            if source_mode == "Upload CSV files":
                uploads = st.file_uploader(
                    "Upload one or more research CSV files", type=["csv"],
                    accept_multiple_files=True, key="research-upload",
                )
                if not uploads:
                    st.info("Upload at least one research CSV to begin.")
                    return
                frames = [validate_research_results(pd.read_csv(upload)) for upload in uploads]
                source_names = [upload.name for upload in uploads]
            else:
                available = sorted(RESEARCH_RESULTS_DIR.glob("*.csv"))
                selected = st.multiselect(
                    "Repository research CSV files", available,
                    format_func=lambda path: path.name, key="research-repository-files",
                )
                if not selected:
                    st.info("Select one or more repository CSV files to begin.")
                    return
                frames = [load_research_results(path) for path in selected]
                source_names = [path.name for path in selected]
                if any(path.name.startswith("demo_") for path in selected):
                    st.warning("Synthetic demo data must not be used for research conclusions.")
            results = merge_research_results(frames)
            _validate_shared_cohort(results)
        except (OSError, ValueError, pd.errors.ParserError,
                ResearchResultsValidationError, ResearchAnalysisError) as exc:
            st.error(f"Research CSV validation failed: {exc}")
            return
        results = results[results["metric_name"].ne("Focus area ratio")].copy()
        if results.empty:
            st.warning("No supported metric rows are available in these files.")
            return
        st.caption("Loaded: " + ", ".join(source_names))
        results = _research_cohort(results)
        if results.empty:
            st.info("No images are available in this analysis cohort.")
            return
        _research_data_summary(results)

    section = st.radio(
        "Research section",
        [
            "Overview", "KL-stage Analysis", "XAI Methods", "Model Comparison",
            "Prediction Correctness", "Explore & Export",
        ],
        horizontal=True, key="research-navigation",
    )
    try:
        if section == "Overview":
            _show_research_overview(results)
        elif section == "KL-stage Analysis":
            _show_kl_stage(results)
        elif section == "XAI Methods":
            _show_xai_methods(results)
        elif section == "Model Comparison":
            _show_model_comparison(results)
        elif section == "Prediction Correctness":
            _show_correctness(results)
        else:
            _show_explore_export(results)
    except ResearchAnalysisError as exc:
        st.error(f"{section} is unavailable for this cohort: {exc}")


def _xai_gallery_images_zip(results: list) -> bytes:
    """Export the original and overlays as separate, unscaled PNG files."""
    archive = BytesIO()
    images = [("01_original_xray.png", results[0].original)]
    for index, result in enumerate(results, start=2):
        method_slug = result.method.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")
        images.append((f"{index:02d}_{method_slug}_overlay.png", result.overlay))
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for filename, image in images:
            png = BytesIO()
            image.save(png, format="PNG")
            bundle.writestr(filename, png.getvalue())
    return archive.getvalue()


@st.fragment
def _show_xai_gallery_download(results: list) -> None:
    st.download_button(
        "Download images for a report (.zip)",
        data=_xai_gallery_images_zip(results),
        file_name="kneexai_explanation_images.zip",
        mime="application/zip",
        key="download-xai-gallery-images",
    )


def _show_overlay_gallery(results: list) -> None:
    """Show original plus selected overlays in fixed three-column rows."""
    st.markdown("#### Overlay comparison")
    st.caption("Original radiograph and the overlays from the selected methods, shown at the same size.")
    gallery_items = [("Original X-ray", results[0].original)] + [
        (result.method, result.overlay) for result in results
    ]
    with st.container(key="overlay-gallery"):
        for offset in range(0, len(gallery_items), 3):
            comparison_cols = st.columns(3, gap="small")
            for column, (label, visual) in zip(comparison_cols, gallery_items[offset:offset + 3]):
                with column:
                    # Display-only enlargement avoids tiny native-size images in wide columns.
                    preview = visual.resize((672, 672), Image.Resampling.LANCZOS)
                    st.image(preview, use_container_width=True)
                    st.caption(label)
    _show_xai_gallery_download(results)


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
    _, parity_tensor = preprocess_pil_for_torch(image)
    with torch.no_grad():
        xai_grade = predict_kl_with_xai_model(model, parity_tensor, model_name)
    if xai_grade != int(prediction["grade"]):
        st.error(
            f"XAI stopped: ONNX prediction is KL{prediction['grade']}, but the PyTorch XAI checkpoint predicts KL{xai_grade}."
        )
        return
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

    st.subheader("Model Explainability")
    st.caption("Compare the original radiograph with each method's heatmap and overlay.")

    tabs = st.tabs([result.method for result in results])
    for tab, result in zip(tabs, results):
        with tab:
            xai_cols = st.columns(3, gap="small")
            for column, label, visual in zip(
                xai_cols,
                ("Original X-ray", "XAI Heatmap", "Overlay"),
                (result.original, result.heatmap, result.overlay),
            ):
                with column:
                    with st.container(border=True):
                        st.caption(label)
                        st.image(visual, use_container_width=True)

    if len(results) > 1:
        _show_overlay_gallery(results)

        pair_metrics, similarity_matrix, average_score = build_xai_metrics(results)

    st.subheader("Explanation Analysis")
    st.caption("Single-image checks describe different properties of the explanations; no one score establishes clinical validity.")
    if len(results) > 1:
        with st.container(border=True):
            st.markdown("#### XAI Agreement")
            st.metric("Average agreement score", f"{average_score:.3f}")
            st.caption("Pairwise similarity of the maps, combining Pearson correlation and overlap of their top 20% regions.")
            with st.expander("Pairwise results and similarity matrix"):
                st.dataframe(pair_metrics.style.format({
                    "Pearson correlation": "{:.3f}",
                    "Top 20% IoU": "{:.3f}",
                    "Agreement score": "{:.3f}",
                }), use_container_width=True, hide_index=True)
                st.dataframe(similarity_matrix.style.format("{:.3f}"), use_container_width=True)

    deletion_metrics, average_drop = build_deletion_metrics(
        results,
        prediction,
        model_name=model_name,
        predict_fn=predict_locally,
    )
    with st.container(border=True):
        st.markdown("#### Deletion Check")
        st.metric("Average confidence drop", f"{average_drop:.3f}")
        st.caption("Change in predicted-class confidence after masking the top 20% highlighted region. It also depends on the original confidence.")
        with st.expander("Per-method deletion results"):
            st.dataframe(
                deletion_metrics.style.format({
                    "Original class confidence": "{:.3f}",
                    "After deletion confidence": "{:.3f}",
                    "Confidence drop": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    stability_metrics, average_stability = build_stability_metrics(
        model=model,
        image=image,
        results=results,
        predicted_grade=int(prediction["grade"]),
        model_name=model_name,
    )
    with st.container(border=True):
        st.markdown("#### Stability Check")
        st.metric("Average stability score", f"{average_stability:.3f}")
        st.caption("Similarity of explanations before and after the existing mild brightness and contrast perturbation.")
        with st.expander("Per-method stability results"):
            st.dataframe(
                stability_metrics.style.format({
                    "Pearson correlation": "{:.3f}",
                    "Top 20% IoU": "{:.3f}",
                    "Stability score": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Optional Quantus Evaluation")
    st.caption("Optional exploratory evaluation for the selected image. Not a dataset-level research result. The quick profile uses 8 faithfulness runs and 2 sensitivity samples.")
    if st.button("Evaluate selected XAI methods", key=f"quantus-{model_name}-{','.join(methods)}"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        _, input_tensor = preprocess_pil_for_torch(image)
        score_rows = []
        try:
            with st.spinner("Evaluating Quantus metrics for the selected explanations..."):
                for result in results:
                    for score in evaluate_quantus_metrics(
                        model=model,
                        input_tensor=input_tensor,
                        attribution=result.raw_attribution,
                        target_class=result.class_idx,
                        xai_method=result.method,
                        model_name=model_name,
                        faithfulness_runs=QUICK_FAITHFULNESS_RUNS,
                        max_sensitivity_samples=QUICK_MAX_SENSITIVITY_SAMPLES,
                    ):
                        score_rows.append({
                            "XAI method": result.method,
                            "Metric": score.metric_name,
                            "Value": score.value,
                            "Parameters": score.parameters,
                        })
        except Exception as exc:
            st.error(f"Quantus evaluation failed: {exc}")
            return

        quantus_results = pd.DataFrame(score_rows)
        for metric_name, metric_rows in quantus_results.groupby("Metric", sort=False):
            definition = metric_definition(metric_name)
            direction = (
                f"{definition.score_direction.title()} is better"
                if definition is not None
                else "See experiment documentation for interpretation"
            )
            st.markdown(f"**{metric_name} - {direction}**")
            if definition is not None:
                st.caption(definition.description)
            st.dataframe(
                metric_rows[["XAI method", "Value"]].style.format(
                    {"Value": definition.display_format if definition else "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander(f"{metric_name} configuration"):
                st.json(metric_rows.iloc[0]["Parameters"])


def main() -> None:
    st.set_page_config(
        page_title="KneeXAI",
        page_icon="K",
        layout="wide",
    )

    apply_interface_styles()
    st.markdown('<div class="kneexai-brand">KneeXAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kneexai-subtitle">Knee Osteoarthritis Classification &amp; Explainability</div>',
        unsafe_allow_html=True,
    )
    st.caption(DISCLAIMER)

    mode = st.segmented_control(
        "Application mode", ["Prediction", "Research"], default="Prediction", label_visibility="collapsed"
    )
    if mode == "Research":
        show_research_mode()
        return

    with st.sidebar:
        model_options = {config["display_name"]: name for name, config in MODEL_CONFIGS.items()}
        st.subheader("Main settings")
        selected_model_label = st.selectbox("Model", list(model_options.keys()))
        model_name = model_options[selected_model_label]
        enable_xai = st.toggle("Show XAI visualization", value=True)
        methods = st.multiselect("XAI methods", XAI_METHODS, default=DEFAULT_XAI_METHODS)
        alpha = st.slider("Overlay strength", min_value=0.1, max_value=0.8, value=0.45, step=0.05)
        with st.expander("Advanced settings"):
            use_api = st.toggle("Use FastAPI endpoint", value=False)
            api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")

    with st.container(key="prediction-view"):
        st.subheader("Prediction Results")
        uploaded_file = st.file_uploader(
            "Upload a knee X-ray image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )

        if uploaded_file is None:
            st.info("Upload an X-ray image to start prediction.")
            return

        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        try:
            if use_api:
                prediction = predict_with_api(image_bytes, uploaded_file.name, api_url, model_name=model_name)
            else:
                prediction = predict_locally(image_bytes, model_name=model_name)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        image_col, result_col = st.columns([0.9, 1.1], gap="large")
        with image_col:
            with st.container(key="prediction-image", border=True):
                st.caption("UPLOADED X-RAY")
                st.image(image, use_container_width=True)
                st.caption(uploaded_file.name)

        with result_col:
            with st.container(border=True):
                show_prediction(prediction)

        if enable_xai:
            try:
                show_xai(image, prediction, methods=methods, alpha=alpha, model_name=model_name)
            except Exception as exc:
                st.error(f"XAI visualization failed: {exc}")


if __name__ == "__main__":
    main()
