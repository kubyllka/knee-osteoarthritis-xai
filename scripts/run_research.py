import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODEL_CONFIGS
from app.evaluation import build_deletion_metrics, build_stability_metrics, focus_area_ratio
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.quantus_metrics import QuantusScore, evaluate_quantus_metrics
from app.research.manifest import DatasetSample, load_dataset_manifest
from app.research.results import validate_research_results
from app.xai import generate_xai_visualization, load_xai_model_by_name, preprocess_pil_for_torch


DEFAULT_METHODS = ["Grad-CAM", "SmoothGrad"]


def prediction_to_dict(prediction) -> dict:
    return {
        "grade": prediction.grade,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "latency_ms": prediction.latency_ms,
    }


def relative_image_path(sample: DatasetSample) -> str:
    try:
        return str(sample.image_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(sample.image_path)


def metric_row(
    sample: DatasetSample,
    prediction: dict,
    model_name: str,
    method: str,
    metric_name: str,
    metric_category: str,
    metric_value: float,
    run_id: str,
    seed: int,
    checkpoint_version: str,
) -> dict:
    predicted_kl = int(prediction["grade"])
    return {
        "sample_id": sample.sample_id,
        "image_path": relative_image_path(sample),
        "true_kl": sample.true_kl,
        "predicted_kl": predicted_kl,
        "is_correct": sample.true_kl == predicted_kl,
        "error_distance": abs(sample.true_kl - predicted_kl),
        "confidence": float(prediction["confidence"]),
        "model_name": model_name,
        "xai_method": method,
        "metric_name": metric_name,
        "metric_category": metric_category,
        "metric_value": float(metric_value),
        "run_id": run_id,
        "split": sample.split,
        "seed": seed,
        "checkpoint_version": checkpoint_version,
        "target_class": predicted_kl,
    }


def custom_metric_rows(
    sample: DatasetSample,
    prediction: dict,
    model_name: str,
    xai_result,
    xai_model,
    classifier: ONNXKneeOsteoarthritisClassifier,
    run_id: str,
    seed: int,
    checkpoint_version: str,
) -> list[dict]:
    def predict_fn(image_bytes: bytes, _: str) -> dict:
        return prediction_to_dict(classifier.predict(image_bytes))

    deletion_table, _ = build_deletion_metrics(
        [xai_result],
        prediction,
        model_name=model_name,
        predict_fn=predict_fn,
    )
    stability_table, _ = build_stability_metrics(
        model=xai_model,
        image=xai_result.original,
        results=[xai_result],
        predicted_grade=int(prediction["grade"]),
        model_name=model_name,
    )

    return [
        metric_row(
            sample,
            prediction,
            model_name,
            xai_result.method,
            "Deletion confidence drop",
            "custom",
            float(deletion_table.iloc[0]["Confidence drop"]),
            run_id,
            seed,
            checkpoint_version,
        ),
        metric_row(
            sample,
            prediction,
            model_name,
            xai_result.method,
            "Stability score",
            "custom",
            float(stability_table.iloc[0]["Stability score"]),
            run_id,
            seed,
            checkpoint_version,
        ),
        metric_row(
            sample,
            prediction,
            model_name,
            xai_result.method,
            "Focus area ratio",
            "custom",
            focus_area_ratio(xai_result.cam),
            run_id,
            seed,
            checkpoint_version,
        ),
    ]


def quantus_metric_rows(
    sample: DatasetSample,
    prediction: dict,
    model_name: str,
    method: str,
    scores: list[QuantusScore],
    run_id: str,
    seed: int,
    checkpoint_version: str,
) -> list[dict]:
    return [
        metric_row(
            sample,
            prediction,
            model_name,
            method,
            score.metric_name,
            score.metric_category,
            score.value,
            run_id,
            seed,
            checkpoint_version,
        )
        for score in scores
    ]


def run_research(
    manifest_path: Path,
    model_name: str,
    methods: list[str],
    output_path: Path,
    run_id: str,
    seed: int,
    limit: int | None,
    infidelity_samples: int,
    max_sensitivity_samples: int,
) -> pd.DataFrame:
    samples = load_dataset_manifest(manifest_path)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No samples selected for the research run.")

    model_config = MODEL_CONFIGS[model_name]
    classifier = ONNXKneeOsteoarthritisClassifier(model_config["onnx_path"])
    xai_model = load_xai_model_by_name(model_name)
    checkpoint_version = model_config["checkpoint_path"].name
    rows = []

    for sample_index, sample in enumerate(samples, start=1):
        image_bytes = sample.image_path.read_bytes()
        prediction = prediction_to_dict(classifier.predict(image_bytes))
        image = Image.open(sample.image_path).convert("RGB")
        _, input_tensor = preprocess_pil_for_torch(image)
        print(
            f"[{sample_index}/{len(samples)}] {sample.sample_id}: "
            f"true KL{sample.true_kl}, predicted KL{prediction['grade']}"
        )

        for method in methods:
            xai_result = generate_xai_visualization(
                model=xai_model,
                image=image,
                method=method,
                class_idx=int(prediction["grade"]),
                model_name=model_name,
            )
            quantus_scores = evaluate_quantus_metrics(
                model=xai_model,
                input_tensor=input_tensor,
                attribution=xai_result.cam,
                target_class=xai_result.class_idx,
                xai_method=method,
                model_name=model_name,
                infidelity_samples=infidelity_samples,
                max_sensitivity_samples=max_sensitivity_samples,
            )
            rows.extend(
                quantus_metric_rows(
                    sample,
                    prediction,
                    model_name,
                    method,
                    quantus_scores,
                    run_id,
                    seed,
                    checkpoint_version,
                )
            )
            rows.extend(
                custom_metric_rows(
                    sample,
                    prediction,
                    model_name,
                    xai_result,
                    xai_model,
                    classifier,
                    run_id,
                    seed,
                    checkpoint_version,
                )
            )

    results = validate_research_results(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline XAI evaluation and write Research Mode results.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "inputs" / "manifest.json")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="resnet50")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--infidelity-samples", type=int, default=8)
    parser.add_argument("--max-sensitivity-samples", type=int, default=8)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.infidelity_samples <= 0 or args.max_sensitivity_samples <= 0:
        parser.error("Quantus sample counts must be positive")

    if args.run_id is None:
        args.run_id = datetime.now().strftime("research-%Y%m%d-%H%M%S")
    if args.output is None:
        args.output = PROJECT_ROOT / "results" / "research" / f"{args.run_id}.csv"
    return args


if __name__ == "__main__":
    arguments = parse_args()
    dataframe = run_research(
        manifest_path=arguments.manifest,
        model_name=arguments.model,
        methods=arguments.methods,
        output_path=arguments.output,
        run_id=arguments.run_id,
        seed=arguments.seed,
        limit=arguments.limit,
        infidelity_samples=arguments.infidelity_samples,
        max_sensitivity_samples=arguments.max_sensitivity_samples,
    )
    print(f"Saved {len(dataframe)} metric rows to: {arguments.output}")
