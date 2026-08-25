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
from app.metric_registry import format_metric_value, metric_definition, parse_parameters
from app.quantus_metrics import evaluate_quantus_metrics
from app.research.results import ResearchResultsValidationError, load_research_results, validate_research_results
from app.xai import generate_xai_visualization, load_xai_model_by_name, preprocess_pil_for_torch


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


@st.cache_resource
def get_local_classifier(model_name: str) -> ONNXKneeOsteoarthritisClassifier:
    return ONNXKneeOsteoarthritisClassifier(MODEL_CONFIGS[model_name]["onnx_path"])


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

    visible_metrics = sorted(
        metric for metric in results["metric_name"].unique() if metric != "Focus area ratio"
    )
    if not visible_metrics:
        st.warning("This file contains no supported, non-legacy metrics.")
        return

    filters = st.columns(4)
    selected_model = filters[0].selectbox("Model", all_option(results["model_name"].unique()))
    selected_method = filters[1].selectbox("XAI method", all_option(results["xai_method"].unique()))
    selected_metric = filters[2].selectbox("Metric", visible_metrics)
    correctness = filters[3].selectbox("Prediction correctness", ["All", "Correct", "Incorrect"])
    selected_grades = st.multiselect(
        "True KL grades",
        options=[0, 1, 2, 3, 4],
        default=[0, 1, 2, 3, 4],
    )

    filtered = results[results["true_kl"].isin(selected_grades)].copy()
    if selected_model != "All":
        filtered = filtered[filtered["model_name"] == selected_model]
    if selected_method != "All":
        filtered = filtered[filtered["xai_method"] == selected_method]
    filtered = filtered[filtered["metric_name"] == selected_metric]
    if correctness == "Correct":
        filtered = filtered[filtered["is_correct"]]
    elif correctness == "Incorrect":
        filtered = filtered[~filtered["is_correct"]]

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    definition = metric_definition(selected_metric)
    if definition is None:
        st.warning("Legacy metric: no registered description or display rule is available.")
        display_format = "{:.4f}"
        direction = filtered.get("score_direction", pd.Series(["unknown"])).iloc[0]
    else:
        display_format = definition.display_format
        direction = definition.score_direction

    st.subheader(selected_metric)
    if definition is not None:
        info = st.columns(3)
        info[0].metric("Category", definition.category.title())
        info[1].metric("Score direction", f"{definition.score_direction.title()} is better")
        info[2].metric("Metric rows", str(len(filtered)))
        st.caption(f"{definition.description} {definition.interpretation}")
    else:
        st.caption(f"Score direction: {direction}. Interpret this legacy metric only with its original experiment documentation.")

    if "metric_parameters" not in filtered.columns or filtered["metric_parameters"].isna().all():
        st.info("Legacy result file: metric configuration was not saved with these rows.")
    else:
        parameter_values = [parse_parameters(value) for value in filtered["metric_parameters"].dropna().unique()]
        with st.expander("Metric configuration"):
            st.json(parameter_values[0] if len(parameter_values) == 1 else parameter_values)

    summary = (
        filtered.groupby(["true_kl", "xai_method"], as_index=False)
        .agg(
            Samples=("sample_id", "nunique"),
            Mean=("metric_value", "mean"),
            Median=("metric_value", "median"),
            Standard_deviation=("metric_value", "std"),
        )
        .rename(columns={"true_kl": "True KL grade"})
    )

    if summary["Samples"].min() < 10:
        st.warning("Small-sample view: fewer than 10 images are available in at least one group. Use this for inspection, not statistical conclusions.")

    st.subheader("Summary by true KL grade and XAI method")
    st.dataframe(
        summary.style.format({"Mean": display_format, "Median": display_format, "Standard_deviation": display_format}),
        use_container_width=True,
        hide_index=True,
    )

    st.altair_chart(
        alt.Chart(summary).mark_bar().encode(
            x=alt.X("True KL grade:O", title="True KL grade"),
            y=alt.Y("Mean:Q", title=f"Mean {selected_metric}"),
            color=alt.Color("xai_method:N", title="XAI method"),
            xOffset="xai_method:N",
            tooltip=["True KL grade:O", "xai_method:N", "Samples:Q", alt.Tooltip("Mean:Q", format=".3e" if display_format == "{:.3e}" else ".4f"), alt.Tooltip("Median:Q", format=".3e" if display_format == "{:.3e}" else ".4f")],
        ),
        use_container_width=True,
    )

    st.subheader("Individual metric values by true KL grade")
    st.altair_chart(
        alt.Chart(filtered).mark_circle(size=75, opacity=0.75).encode(
            x=alt.X("true_kl:O", title="True KL grade"),
            y=alt.Y("metric_value:Q", title=selected_metric),
            color=alt.Color("xai_method:N", title="XAI method"),
            tooltip=["sample_id:N", "xai_method:N", "true_kl:O", "predicted_kl:O", alt.Tooltip("metric_value:Q", format=".3e" if display_format == "{:.3e}" else ".4f")],
        ),
        use_container_width=True,
    )

    correctness_summary = (
        filtered.groupby(["is_correct", "xai_method"], as_index=False)
        .agg(Samples=("sample_id", "nunique"), Mean=("metric_value", "mean"), Median=("metric_value", "median"))
        .replace({"is_correct": {True: "Correct", False: "Incorrect"}})
        .rename(columns={"is_correct": "Prediction"})
    )
    st.subheader("Correct vs incorrect predictions")
    st.dataframe(
        correctness_summary.style.format({"Mean": display_format, "Median": display_format}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Filtered research rows"):
        st.dataframe(filtered, use_container_width=True, hide_index=True)


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
        xai_grade = int(model(parity_tensor).argmax(dim=1).item())
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
    st.caption("Runs only when requested. This quick profile uses 2 perturbation samples and is not a dataset-level research result.")
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
                        infidelity_samples=2,
                        max_sensitivity_samples=2,
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
