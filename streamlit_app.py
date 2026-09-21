from io import BytesIO
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
    interpret_agreement,
    interpret_deletion,
    interpret_stability,
)
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.metric_registry import is_legacy_metric, metric_definition, parse_parameters
from app.quantus_metrics import evaluate_quantus_metrics
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
RESEARCH_RESULTS_DIR = Path("results/research")
QUICK_FAITHFULNESS_RUNS = 8
QUICK_MAX_SENSITIVITY_SAMPLES = 2


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

    st.subheader(f"Predicted class: Grade {grade}")
    st.write(f"**Meaning:** {GRADE_MEANINGS.get(grade, KL_LABELS.get(grade, 'Unknown grade'))}")

    metric_cols = st.columns(2)
    metric_cols[0].metric("Confidence", format_percent(confidence))
    metric_cols[1].metric("Latency", f"{float(prediction.get('latency_ms', 0.0)):.1f} ms")

    probabilities = prediction["probabilities"]
    chart_data = pd.DataFrame(
        {
            "Grade": [label.replace("KL Grade ", "Grade ") for label in probabilities],
            "Probability": [float(probability) for probability in probabilities.values()],
        }
    )
    st.bar_chart(chart_data, x="Grade", y="Probability", height=280)


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
    definition = metric_definition(selected_metric)
    if definition is None:
        st.warning("This legacy metric has no registered display metadata. Its raw rows are shown below.")
        with st.expander("Filtered research rows"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)
        return

    display_format = definition.display_format
    st.subheader(selected_metric)
    st.caption(f"**{definition.score_direction.title()} is better.** {definition.description} {definition.interpretation}")
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
        st.dataframe(filtered, use_container_width=True, hide_index=True)


def show_research_mode() -> None:
    st.header("Research Mode")
    st.caption(
        "Explore precomputed XAI evaluation results. This mode does not run models or metrics."
    )

    available_files = sorted(RESEARCH_RESULTS_DIR.glob("*.csv"))
    source_options = [f"Repository file: {path.name}" for path in available_files]
    source_options.append("Upload CSV for this session")
    selected_source = st.selectbox("Research results source", source_options)

    try:
        if selected_source == "Upload CSV for this session":
            uploaded_file = st.file_uploader("Upload a research-results CSV", type=["csv"])
            if uploaded_file is None:
                st.info("Choose a CSV file or select a repository file above.")
                return
            results = validate_research_results(pd.read_csv(uploaded_file))
            st.caption(f"Loaded uploaded file: {uploaded_file.name}")
        else:
            selected_path = available_files[source_options.index(selected_source)]
            results = load_research_results(selected_path)
            st.caption(f"Loaded: {selected_path}")
            if selected_path.name.startswith("demo_"):
                st.warning("Demo data only: these values are synthetic and must not be used in research conclusions.")
    except (OSError, pd.errors.ParserError, ResearchResultsValidationError) as exc:
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

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    filtered["dashboard_category"] = [
        _metric_category(metric, category)
        for metric, category in zip(filtered["metric_name"], filtered["metric_category"])
    ]
    tabs = st.tabs([label for label, _ in DIMENSIONS])
    for tab, (label, category) in zip(tabs, DIMENSIONS):
        with tab:
            category_rows = filtered[filtered["dashboard_category"] == category]
            if category_rows.empty:
                if category == "custom":
                    st.info("No dataset-wide custom agreement metrics are available yet. This tab is reserved for future agreement checks.")
                else:
                    st.info(f"No {label.lower()} metrics match the active filters.")
                continue
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

    st.subheader("Explainable AI visualizations")

    tabs = st.tabs([result.method for result in results])
    for tab, result in zip(tabs, results):
        with tab:
            xai_cols = st.columns(3)
            xai_cols[0].image(result.original, caption="Original X-ray", use_container_width=True)
            xai_cols[1].image(result.heatmap, caption=f"{result.method} heatmap", use_container_width=True)
            xai_cols[2].image(result.overlay, caption=f"{result.method} overlay", use_container_width=True)

    if len(results) > 1:
        st.subheader("Overlay comparison")
        comparison_cols = st.columns(len(results))
        for column, result in zip(comparison_cols, results):
            column.image(result.overlay, caption=result.method, use_container_width=True)

        pair_metrics, similarity_matrix, average_score = build_xai_metrics(results)

        st.subheader("Quantitative XAI comparison")
        st.dataframe(pair_metrics.style.format({
            "Pearson correlation": "{:.3f}",
            "Top 20% IoU": "{:.3f}",
            "Agreement score": "{:.3f}",
        }), use_container_width=True, hide_index=True)

        st.metric("Average XAI Agreement Score", f"{average_score:.3f}")
        st.write(interpret_agreement(average_score))

        with st.expander("Heatmap similarity matrix"):
            st.dataframe(
                similarity_matrix.style.format("{:.3f}"),
                use_container_width=True,
            )

    st.subheader("Deletion Check")
    deletion_metrics, average_drop = build_deletion_metrics(
        results,
        prediction,
        model_name=model_name,
        predict_fn=predict_locally,
    )
    st.dataframe(
        deletion_metrics.style.format({
            "Original class confidence": "{:.3f}",
            "After deletion confidence": "{:.3f}",
            "Confidence drop": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Average confidence drop after deleting top 20% XAI region", f"{average_drop:.3f}")
    st.write(interpret_deletion(average_drop))

    st.subheader("Stability Check")
    stability_metrics, average_stability = build_stability_metrics(
        model=model,
        image=image,
        results=results,
        predicted_grade=int(prediction["grade"]),
        model_name=model_name,
    )
    st.dataframe(
        stability_metrics.style.format({
            "Pearson correlation": "{:.3f}",
            "Top 20% IoU": "{:.3f}",
            "Stability score": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Average Stability Score", f"{average_stability:.3f}")
    st.write(interpret_stability(average_stability))

    st.subheader("Quantus evaluation (exploratory)")
    st.caption("Runs only when requested. This quick profile uses 8 faithfulness runs and 2 sensitivity samples; it is not a dataset-level research result.")
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

    st.title("KneeXAI: Explainable AI System for Osteoarthritis Severity Classification")
    st.warning(DISCLAIMER)

    mode = st.radio("Application mode", ["Prediction", "Research"], horizontal=True)
    if mode == "Research":
        show_research_mode()
        return

    with st.sidebar:
        model_options = {config["display_name"]: name for name, config in MODEL_CONFIGS.items()}
        st.header("Inference")
        selected_model_label = st.selectbox("Model", list(model_options.keys()))
        model_name = model_options[selected_model_label]
        use_api = st.toggle("Use FastAPI endpoint", value=False)
        api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
        st.header("Explainability")
        enable_xai = st.toggle("Show XAI visualization", value=True)
        methods = st.multiselect("XAI methods", XAI_METHODS, default=DEFAULT_XAI_METHODS)
        alpha = st.slider("Overlay strength", min_value=0.1, max_value=0.8, value=0.45, step=0.05)

    uploaded_file = st.file_uploader(
        "Upload a knee X-ray image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload an X-ray image to start prediction.")
        return

    image_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image_col, result_col = st.columns([0.65, 1.35])
    with image_col:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with result_col:
        try:
            if use_api:
                prediction = predict_with_api(image_bytes, uploaded_file.name, api_url, model_name=model_name)
            else:
                prediction = predict_locally(image_bytes, model_name=model_name)
            show_prediction(prediction)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    if enable_xai:
        try:
            show_xai(image, prediction, methods=methods, alpha=alpha, model_name=model_name)
        except Exception as exc:
            st.error(f"XAI visualization failed: {exc}")


if __name__ == "__main__":
    main()
