from io import BytesIO
import json
from pathlib import Path
import random

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
    wilcoxon_method_posthoc,
)
from app.research.results import ResearchResultsValidationError, load_research_results, validate_research_results
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
    concentration = definition.category == "complexity"

    return {
        "samples": samples,
        "correct_rate": correct_rate,
        "method_label": "Most concentrated method" if concentration else "Best method by median",
        "method": str(method["xai_method"]),
        "best_grade_label": "Most concentrated KL grade" if concentration else "Best KL grade",
        "best_grade": int(best_grade["true_kl"]),
        "worst_grade_label": "Least concentrated KL grade" if concentration else "Worst KL grade",
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
    st.caption(f"**{definition.score_direction.title()} is better.** {definition.description} {definition.interpretation}")
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
    model_order = sorted(data["Model"].unique())
    color_scale = alt.Scale(
        domain=model_order,
        range=["#2563eb", "#e76f51", "#7c3aed", "#0f766e"][: len(model_order)],
    )
    base = alt.Chart(data).encode(
        x=alt.X("True KL grade:N", title="True KL grade", sort=KL_GRADE_LABELS),
        xOffset=alt.XOffset("Model:N", sort=model_order),
        color=alt.Color("Model:N", title="Model", sort=model_order, scale=color_scale),
    )
    bars = base.mark_bar().encode(
        y=alt.Y(
            "Accuracy:Q",
            title="Classification accuracy",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("Model:N"),
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
    return (boxes + medians + labels).properties(title=model_name, height=330)


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
    model_pair = list(data["Model"].cat.categories)
    color_scale = alt.Scale(domain=model_pair, range=["#e76f51", "#2563eb"])
    base = alt.Chart(data).encode(
        x=alt.X("XAI method:N", title="XAI method", sort=XAI_METHODS, axis=alt.Axis(labelAngle=-30)),
        xOffset=alt.XOffset("Model:N", sort=model_pair),
        color=alt.Color("Model:N", title="Model", sort=model_pair, scale=color_scale),
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
        xOffset=alt.XOffset("Model:N", sort=model_pair),
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
        xOffset=alt.XOffset("Model:N", sort=model_pair),
        y=alt.Y("Median:Q", scale=y_scale),
        text="Median label:N",
    )
    return (boxes + medians + labels).properties(height=350)


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


def _show_results_analysis(results: pd.DataFrame) -> None:
    st.subheader("Research overview")
    st.caption(
        "This overview uses the selected image cohort from the loaded XAI result files. Top filters apply only to "
        "Explore metrics, so they cannot distort classifier-subset accuracy or primary KL-stage analysis."
    )

    model_sets = {model_id: set(group["sample_id"]) for model_id, group in results.groupby("model_name")}
    different_cohorts = len({frozenset(ids) for ids in model_sets.values()}) > 1
    cohort_options = ["All loaded images"]
    if len(model_sets) > 1:
        cohort_options.append("Images shared by all loaded models")
    original_ids = set()
    base_manifest = Path(__file__).resolve().parent / "inputs/test/test_manifest.json"
    if base_manifest.is_file():
        try:
            original_ids = {
                str(sample["sample_id"])
                for sample in json.loads(base_manifest.read_text(encoding="utf-8"))["samples"]
            }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if original_ids:
        cohort_options.append("Original 150-image subset")
    if different_cohorts:
        st.warning(
            "Loaded models do not have identical image sets. Per-model summaries use their own available images; "
            "paired tests use only shared sample IDs. Select a common cohort for direct comparisons."
        )
    cohort = st.selectbox(
        "Results-analysis cohort", cohort_options,
        index=1 if different_cohorts and len(model_sets) > 1 else 0,
    )
    if cohort == "Images shared by all loaded models":
        results = results[results["sample_id"].isin(set.intersection(*model_sets.values()))].copy()
    elif cohort == "Original 150-image subset":
        results = results[results["sample_id"].isin(original_ids)].copy()
    if results.empty:
        st.info("No images are available in the selected cohort.")
        return

    try:
        classifier_summary, classifier_by_grade, confusion_matrices = classifier_subset_summary(results)
        analysis = kl_stage_analysis(results)
        method_matrices = method_complete_case_matrices(results)
        method_values = method_comparison_values(method_matrices)
        method_descriptives = method_comparison_descriptives(method_matrices)
        friedman_summary = friedman_method_analysis(method_matrices)
    except ResearchAnalysisError as exc:
        st.error(f"Results analysis cannot be calculated: {exc}")
        return

    st.markdown("#### Classification accuracy by true KL grade")
    st.caption(
        "Each bar shows correct predictions divided by all images of that true KL grade. "
        "These are results for the loaded XAI subset, not the full held-out classifier test set."
    )
    st.altair_chart(_classifier_accuracy_chart(classifier_by_grade), use_container_width=True)

    st.markdown("#### KL-stage effect map")
    st.caption(
        "Each cell is the Kruskal–Wallis epsilon-squared effect size (ε²) for one XAI method and metric. "
        "A * marks a Holm-significant global KL-stage test. Both models use the same colour scale."
    )
    model_names = sorted(analysis["Model"].unique())
    effect_scale_max = _effect_map_scale_max(analysis)
    map_columns = st.columns(min(len(model_names), 2)) if model_names else []
    for index, model_name in enumerate(model_names):
        # Streamlit columns are reused gracefully if an uploaded set has more than two models.
        with map_columns[index % len(map_columns)]:
            st.altair_chart(
                _effect_map_chart(_effect_map_data(analysis, model_name), model_name, effect_scale_max),
                use_container_width=True,
            )
    st.caption(
        "ε² describes the strength of association between true KL stage and this specific metric; it is not an "
        "XAI-method quality score or leaderboard. A result that is significant for one architecture and not the other "
        "does not by itself establish an architecture difference. Sparseness describes concentration, while region "
        "perturbation is a configuration-dependent supporting faithfulness check."
    )

    st.markdown("#### XAI-method comparison")
    st.caption(
        "Each metric is analysed separately with paired complete-case samples. The five methods are evaluated on the "
        "same images; no overall XAI score, method leaderboard, or KL-stratified method test is created here."
    )
    metric_tabs = st.tabs(list(METHOD_COMPARISON_METRICS))
    for tab, metric_name in zip(metric_tabs, METHOD_COMPARISON_METRICS):
        with tab:
            st.caption(_method_metric_caption(metric_name))
            metric_tests = friedman_summary[friedman_summary["Metric"].eq(metric_name)]
            if metric_tests.empty or method_values[method_values["Metric"].eq(metric_name)].empty:
                st.info("No complete paired values are available for this metric in the loaded files.")
                continue
            model_columns = st.columns(min(len(metric_tests), 2))
            y_scale = _method_comparison_y_scale(method_values, metric_name)
            for index, (_, test_row) in enumerate(metric_tests.iterrows()):
                with model_columns[index % len(model_columns)]:
                    st.caption(
                        f"Complete paired samples: {int(test_row['Valid N'])}; "
                        f"excluded: {int(test_row['Excluded N'])}."
                    )
                    st.altair_chart(
                        _method_comparison_chart(
                            method_values,
                            method_descriptives,
                            test_row,
                            metric_name,
                            str(test_row["Model"]),
                            y_scale,
                        ),
                        use_container_width=True,
                    )
            st.markdown("**Complete-case descriptive summary**")
            metric_descriptives = method_descriptives[method_descriptives["Metric"].eq(metric_name)]
            st.dataframe(_format_results_analysis_table(metric_descriptives), use_container_width=True, hide_index=True)

    with st.expander("Friedman XAI-method statistics"):
        st.caption(
            "Each test compares the five paired XAI methods within one model and metric. "
            "Holm correction is applied across the five metric-level Friedman tests separately for each model. "
            "Kendall's W = χ² / (N × 4)."
        )
        st.dataframe(_format_results_analysis_table(friedman_summary), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Friedman XAI-method summary CSV",
            data=friedman_summary.to_csv(index=False).encode("utf-8"),
            file_name="friedman_xai_method_analysis.csv",
            mime="text/csv",
        )

    with st.expander("Wilcoxon post-hoc method pairs"):
        posthoc_methods = wilcoxon_method_posthoc(method_matrices, friedman_summary)
        if posthoc_methods.empty:
            st.info("No Holm-significant Friedman result is available for pairwise XAI-method comparison.")
        else:
            st.caption(
                "Two-sided paired Wilcoxon signed-rank tests are shown only after a Holm-significant Friedman test. "
                "Holm correction is applied across the ten method pairs within that metric."
            )
            st.dataframe(_format_results_analysis_table(posthoc_methods), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Wilcoxon XAI-method pairs CSV",
                data=posthoc_methods.to_csv(index=False).encode("utf-8"),
                file_name="wilcoxon_xai_method_pairs.csv",
                mime="text/csv",
            )

    st.markdown("#### Paired model comparison")
    st.caption(
        "Compare architectures within one training variant, or training variants within one architecture. "
        "Only the same image, XAI method and metric are paired; no overall model winner is inferred."
    )
    possible_pairs = [
        ("Architecture: ordinal BCE", ("densenet121", "resnet50")),
        ("Architecture: class-aware ordinal BCE", ("densenet121_class_aware", "resnet50_class_aware")),
        ("Training variant: DenseNet121", ("densenet121_class_aware", "densenet121")),
        ("Training variant: ResNet50", ("resnet50_class_aware", "resnet50")),
    ]
    available_pairs = [(label, models) for label, models in possible_pairs if set(models).issubset(model_sets)]
    if not available_pairs:
        st.info("Load a compatible pair of model CSV files to enable paired model comparisons.")
    else:
        comparison_name = st.selectbox("Model comparison", [label for label, _ in available_pairs])
        selected_pair = next(models for label, models in available_pairs if label == comparison_name)
        try:
            architecture_primary_pairs = architecture_complete_case_pairs(results, models=selected_pair)
            architecture_same_target_pairs = architecture_complete_case_pairs(
                results, same_target_only=True, models=selected_pair
            )
        except ResearchAnalysisError as exc:
            st.info(f"Paired model comparison is unavailable: {exc}")
            architecture_primary_pairs = None
        if architecture_primary_pairs is not None:
            architecture_mode = st.radio(
                "Paired model comparison mode",
                ["Primary: all paired samples", "Sensitivity: same predicted KL target"],
                horizontal=True,
            )
            sensitivity_mode = architecture_mode.startswith("Sensitivity")
            architecture_pairs = (
                architecture_same_target_pairs if sensitivity_mode else architecture_primary_pairs
            )
            architecture_values = architecture_comparison_values(architecture_pairs)
            architecture_summary = architecture_comparison_analysis(architecture_pairs)
            if sensitivity_mode:
                st.warning(
                    "Sensitivity analysis: only images for which both models predicted the same KL class are retained. "
                    "This reduces confounding from different XAI target classes, but analyses a smaller selected cohort."
                )
            else:
                st.caption(
                    "Primary analysis includes every complete finite pair, even when the two models predicted different KL classes."
                )

            architecture_tabs = st.tabs(list(METHOD_COMPARISON_METRICS))
            for tab, metric_name in zip(architecture_tabs, METHOD_COMPARISON_METRICS):
                with tab:
                    st.caption(_method_metric_caption(metric_name))
                    metric_values = architecture_values[architecture_values["Metric"].eq(metric_name)]
                    if metric_values.empty:
                        st.info("No complete paired values are available for this metric.")
                        continue
                    st.altair_chart(
                        _architecture_comparison_chart(architecture_values, architecture_summary, metric_name),
                        use_container_width=True,
                    )
                    metric_summary = architecture_summary[architecture_summary["Metric"].eq(metric_name)]
                    st.dataframe(_format_results_analysis_table(metric_summary), use_container_width=True, hide_index=True)

            with st.expander("Paired-model Wilcoxon statistics"):
                st.caption(
                    "One two-sided paired Wilcoxon signed-rank test is calculated per XAI method × metric. "
                    "Holm correction is applied across 25 tests for the selected model pair. "
                    f"Positive paired effects mean higher values for {selected_pair[0]} than {selected_pair[1]}."
                )
                st.dataframe(_format_results_analysis_table(architecture_summary), use_container_width=True, hide_index=True)
                suffix = "same_target" if sensitivity_mode else "all_pairs"
                st.download_button(
                    "Download paired-model comparison CSV",
                    data=architecture_summary.to_csv(index=False).encode("utf-8"),
                    file_name=f"model_comparison_{selected_pair[0]}_vs_{selected_pair[1]}_{suffix}.csv",
                    mime="text/csv",
                )

    st.markdown("#### Prediction correctness analysis")
    st.caption(
        "Correctness is derived as predicted KL = true KL. Each comparison remains within one model, XAI method, "
        "and metric; a significant result is not evidence that an explanation is faithful or clinically valid."
    )
    try:
        correctness_values = prediction_correctness_values(results)
        correctness_summary = _cached_prediction_correctness_analysis(results)
    except ResearchAnalysisError as exc:
        st.info(f"Prediction correctness analysis is unavailable: {exc}")
    else:
        correctness_mode = st.radio(
            "Prediction correctness analysis mode",
            ["Primary: pooled correct vs incorrect", "KL-adjusted sensitivity: stratified permutation"],
            horizontal=True,
        )
        kl_adjusted_mode = correctness_mode.startswith("KL-adjusted")
        if kl_adjusted_mode:
            st.warning(
                "KL-adjusted sensitivity analysis shuffles correctness labels only within each true-KL stratum, preserving "
                "stratum-specific group sizes. It tests whether the pooled effect remains after accounting for KL composition."
            )
        else:
            st.caption(
                "Primary analysis compares all valid correct and incorrect predictions for the selected model × method × metric."
            )

        st.caption(
            "Heatmap values are pooled rank-biserial effects (correct − incorrect): positive means correct predictions have "
            "higher metric values. A * marks Holm significance in the selected mode; whether higher is preferable depends on the metric."
        )
        correctness_models = sorted(correctness_summary["Model"].unique())
        heatmap_columns = st.columns(min(len(correctness_models), 2))
        for index, model_name in enumerate(correctness_models):
            with heatmap_columns[index % len(heatmap_columns)]:
                heatmap_data = _correctness_effect_heatmap_data(
                    correctness_summary, model_name, kl_adjusted=kl_adjusted_mode
                )
                st.altair_chart(_correctness_effect_heatmap(heatmap_data, model_name), use_container_width=True)

        st.markdown("**Individual distribution**")
        selectors = st.columns(3)
        selected_model = selectors[0].selectbox(
            "Model", correctness_models, key="correctness-distribution-model"
        )
        selected_metric = selectors[1].selectbox(
            "Metric", list(METHOD_COMPARISON_METRICS), key="correctness-distribution-metric"
        )
        selected_method = selectors[2].selectbox(
            "XAI method", XAI_METHODS, key="correctness-distribution-method"
        )
        selected_test = correctness_summary[
            correctness_summary["Model"].eq(selected_model)
            & correctness_summary["Metric"].eq(selected_metric)
            & correctness_summary["XAI method"].eq(selected_method)
        ].iloc[0]
        st.altair_chart(
            _correctness_distribution_chart(correctness_values, selected_test, selected_metric),
            use_container_width=True,
        )
        selected_p = (
            selected_test["KL-adjusted permutation p Holm"]
            if kl_adjusted_mode else selected_test["p Holm"]
        )
        st.caption(
            f"Correct N: {int(selected_test['Correct N'])}; incorrect N: {int(selected_test['Incorrect N'])}; "
            f"excluded: {int(selected_test['Excluded N'])}; Holm-adjusted p: {selected_p:.3g}."
        )

        with st.expander("Prediction-correctness statistics"):
            st.caption(
                "The table includes primary Mann–Whitney and KL-adjusted permutation results. "
                "Holm correction is applied separately across 25 tests for each model in each analysis mode."
            )
            st.dataframe(_format_results_analysis_table(correctness_summary), use_container_width=True, hide_index=True)
            st.download_button(
                "Download prediction-correctness analysis CSV",
                data=correctness_summary.to_csv(index=False).encode("utf-8"),
                file_name="prediction_correctness_analysis.csv",
                mime="text/csv",
            )

    grade_n_columns = [f"KL{grade} n" for grade in range(5)]
    tests_per_model = analysis.groupby("Model").size()
    complete_primary_family = tests_per_model.eq(25).all() and analysis["KL groups"].eq(5).all()
    if not complete_primary_family:
        st.warning(
            "Incomplete primary test family: thesis-level inference expects five metrics, five XAI methods, "
            "and all five KL groups for each selected model. Displayed statistics are exploratory."
        )
    if (analysis[grade_n_columns] < 10).any().any():
        st.warning("Small-sample warning: at least one KL group has fewer than 10 valid values.")
    excluded = int(analysis["Excluded undefined"].sum())
    if excluded:
        st.warning(
            f"{excluded} undefined metric row(s) were excluded without imputation; their valid companion metrics remain included."
        )

    with st.expander("Classifier subset details"):
        st.caption("These metrics describe only images included in the loaded XAI CSV files, not the full held-out classifier test set.")
        st.dataframe(_format_results_analysis_table(classifier_summary), use_container_width=True, hide_index=True)
        st.dataframe(_format_results_analysis_table(classifier_by_grade), use_container_width=True, hide_index=True)
        for model_name, matrix in confusion_matrices.items():
            st.markdown(f"**{model_name}: confusion matrix**")
            st.dataframe(matrix, use_container_width=True)

    with st.expander("KL-stage statistical table"):
        st.caption(
            "Kruskal–Wallis tests compare true KL groups within one model × XAI method × metric. "
            "Holm correction is applied separately within each model. "
            "ε² = max(0, (H − k + 1) / (N − k))."
        )
        st.dataframe(_format_results_analysis_table(analysis), use_container_width=True, hide_index=True)
        st.download_button(
            "Download KL-stage summary CSV",
            data=analysis.to_csv(index=False).encode("utf-8"),
            file_name="kl_stage_analysis.csv",
            mime="text/csv",
        )

    with st.expander("Dunn post-hoc KL comparisons"):
        if not complete_primary_family:
            st.info(
                "Dunn post-hoc comparisons require the complete primary KL analysis: five metrics, "
                "five XAI methods, and all five KL groups for every selected model."
            )
            return
        try:
            posthoc = dunn_posthoc(results, analysis)
        except ResearchAnalysisError as exc:
            st.error(f"Post-hoc comparisons are unavailable: {exc}")
            return
        if posthoc.empty:
            st.info("No Holm-significant global KL tests were available for post-hoc comparison.")
        else:
            st.caption("Dunn pairwise tests are calculated only after a Holm-significant global test; each set of ten KL pairs uses Holm correction.")
            st.dataframe(_format_results_analysis_table(posthoc), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Dunn post-hoc CSV",
                data=posthoc.to_csv(index=False).encode("utf-8"),
                file_name="dunn_posthoc_kl_pairs.csv",
                mime="text/csv",
            )


def show_research_mode() -> None:
    st.header("Research Mode")
    st.caption(
        "Explore precomputed XAI evaluation results. This mode does not run models or metrics."
    )

    available_files = sorted(RESEARCH_RESULTS_DIR.glob("*.csv"))
    source_mode = st.radio(
        "Research results source",
        ["Repository CSV files", "Upload CSV files"],
        horizontal=True,
    )

    try:
        if source_mode == "Upload CSV files":
            uploaded_files = st.file_uploader(
                "Upload one or more research-results CSV files",
                type=["csv"],
                accept_multiple_files=True,
            )
            if not uploaded_files:
                st.info("Upload one or more CSV files to begin.")
                return
            frames = [validate_research_results(pd.read_csv(uploaded_file)) for uploaded_file in uploaded_files]
            source_names = [uploaded_file.name for uploaded_file in uploaded_files]
        else:
            selected_paths = st.multiselect(
                "Research result files",
                available_files,
                format_func=lambda path: path.name,
            )
            if not selected_paths:
                st.info("Select one or more repository CSV files to begin.")
                return
            frames = [load_research_results(path) for path in selected_paths]
            source_names = [path.name for path in selected_paths]
            if any(path.name.startswith("demo_") for path in selected_paths):
                st.warning("Demo data only: synthetic values must not be used in research conclusions.")
        results = merge_research_results(frames)
        st.caption("Loaded: " + ", ".join(source_names))
    except (OSError, pd.errors.ParserError, ResearchResultsValidationError, ResearchAnalysisError) as exc:
        st.error(f"Could not load research results: {exc}")
        return

    results = results[results["metric_name"] != "Focus area ratio"].copy()
    if results.empty:
        st.warning("This file contains no supported, non-legacy metrics.")
        return

    filters = st.columns(4)
    selected_model = filters[0].selectbox("Model", all_option(results["model_name"].unique()))
    selected_method = filters[1].selectbox("XAI method", all_option(results["xai_method"].unique()))
    selected_grades = filters[2].multiselect(
        "True KL grade",
        options=[0, 1, 2, 3, 4],
        default=[0, 1, 2, 3, 4],
    )
    correctness = filters[3].selectbox("Prediction correctness", ["All", "Correct", "Incorrect"])

    filtered = results[results["true_kl"].isin(selected_grades)].copy()
    if selected_model != "All":
        filtered = filtered[filtered["model_name"] == selected_model]
    if selected_method != "All":
        filtered = filtered[filtered["xai_method"] == selected_method]
    if correctness == "Correct":
        filtered = filtered[filtered["is_correct"]]
    elif correctness == "Incorrect":
        filtered = filtered[~filtered["is_correct"]]

    filtered["dashboard_category"] = [
        _metric_category(metric, category)
        for metric, category in zip(filtered["metric_name"], filtered["metric_category"])
    ]
    metric_sources = filtered.get("metric_source", pd.Series(None, index=filtered.index))
    filtered["is_custom_check"] = [
        _is_custom_check(metric, source)
        for metric, source in zip(filtered["metric_name"], metric_sources)
    ]
    results_tab, dashboard_tab = st.tabs(["Results analysis", "Explore metrics"])
    with results_tab:
        _show_results_analysis(results)
    with dashboard_tab:
        if filtered.empty:
            st.warning("No rows match the selected exploratory filters.")
            return
        tabs = st.tabs([label for label, _ in DIMENSIONS])
        for tab, (label, category) in zip(tabs, DIMENSIONS):
            with tab:
                category_rows = (
                    filtered[filtered["is_custom_check"]]
                    if category == "custom"
                    else filtered[filtered["dashboard_category"] == category]
                )
                if category_rows.empty:
                    if category == "custom":
                        st.info("No custom checks match the active filters.")
                    else:
                        st.info(f"No {label.lower()} metrics match the active filters.")
                    continue
                if category == "custom":
                    st.caption(
                        "Supporting project-specific checks. Legacy deletion rows and stability appear here; "
                        "new research runs use Quantus Region Perturbation under Faithfulness."
                    )
                metrics = sorted(category_rows["metric_name"].unique())
                selected_metric = metrics[0] if len(metrics) == 1 else st.selectbox("Metric", metrics, key=f"research-metric-{category}")
                _show_metric_dashboard(_selected_metric_rows(category_rows, selected_metric), selected_metric)


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
        st.markdown("#### Overlay comparison")
        with st.container(key="overlay-gallery"):
            for offset in range(0, len(results), 3):
                comparison_cols = st.columns(min(3, len(results) - offset), gap="small")
                for column, result in zip(comparison_cols, results[offset:offset + 3]):
                    with column:
                        with st.container(border=True):
                            st.caption(result.method)
                            st.image(result.overlay, use_container_width=True)

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
