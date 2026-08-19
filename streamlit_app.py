from io import BytesIO

import pandas as pd
import requests
import streamlit as st
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
from app.xai import generate_xai_visualization, load_xai_model_by_name


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

        pair_metrics, focus_metrics, similarity_matrix, average_score = build_xai_metrics(results)

        st.subheader("Quantitative XAI comparison")
        metric_cols = st.columns(2)
        metric_cols[0].dataframe(pair_metrics.style.format({
            "Pearson correlation": "{:.3f}",
            "Top 20% IoU": "{:.3f}",
            "Agreement score": "{:.3f}",
        }), use_container_width=True, hide_index=True)
        metric_cols[1].dataframe(focus_metrics.style.format({
            "Focus area ratio": "{:.3f}",
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


def main() -> None:
    st.set_page_config(
        page_title="KneeXAI",
        page_icon="K",
        layout="wide",
    )

    st.title("KneeXAI: Explainable AI System for Osteoarthritis Severity Classification")
    st.warning(DISCLAIMER)

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

    image_col, result_col = st.columns([1, 1])
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
