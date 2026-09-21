import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODEL_CONFIGS
from app.evaluation import build_deletion_metrics, build_stability_metrics
from app.metric_registry import metric_definition, parse_parameters, serialise_parameters
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.quantus_metrics import QuantusScore, evaluate_quantus_metrics
from app.research.manifest import DatasetSample, load_dataset_manifest
from app.research.results import load_research_results, validate_research_results
from app.xai import generate_xai_visualization, load_xai_model_by_name, predict_kl_with_xai_model, preprocess_pil_for_torch


DEFAULT_METHODS = ["Grad-CAM", "SmoothGrad"]
ORDINAL_LOGIT_PARITY_ATOL = 1e-2
ORDINAL_THRESHOLD_MARGIN = 1e-3
RUNNER_METRICS = frozenset(
    {
        "FaithfulnessCorrelation",
        "MaxSensitivity",
        "Sparseness",
        "Deletion confidence drop",
        "Stability score",
    }
)


def ordinal_parity_warning(
    onnx_logits: np.ndarray,
    pytorch_logits: np.ndarray,
    onnx_grade: int,
    pytorch_grade: int,
) -> str | None:
    """Accept only numerically borderline ordinal grade differences.

    ONNX Runtime and PyTorch can differ in the last few decimal places. For an
    ordinal threshold logit extremely close to zero, that may flip one binary
    threshold while still representing the same checkpoint. A material logit
    difference remains a hard error, preventing accidental checkpoint/ONNX
    mismatches from entering a research run.
    """
    onnx_values = np.asarray(onnx_logits, dtype=float).reshape(-1)
    pytorch_values = np.asarray(pytorch_logits, dtype=float).reshape(-1)
    if onnx_values.shape != pytorch_values.shape:
        raise RuntimeError(
            "ONNX/PyTorch ordinal output shapes differ: "
            f"{onnx_values.shape} versus {pytorch_values.shape}."
        )
    max_difference = float(np.max(np.abs(onnx_values - pytorch_values)))
    if max_difference > ORDINAL_LOGIT_PARITY_ATOL:
        raise RuntimeError(
            "ONNX/PyTorch ordinal logits differ materially "
            f"(max absolute difference {max_difference:.6f} > {ORDINAL_LOGIT_PARITY_ATOL}). "
            "Export ONNX again from the selected checkpoint and verify the model files."
        )
    if onnx_grade == pytorch_grade:
        return None

    different_thresholds = (onnx_values > 0.0) != (pytorch_values > 0.0)
    threshold_distance = np.maximum(np.abs(onnx_values), np.abs(pytorch_values))
    if not different_thresholds.any() or np.any(threshold_distance[different_thresholds] > ORDINAL_THRESHOLD_MARGIN):
        raise RuntimeError(
            f"Prediction mismatch: ONNX predicted KL{onnx_grade}, but the PyTorch XAI checkpoint "
            f"predicted KL{pytorch_grade}. The disagreement is not within the "
            f"{ORDINAL_THRESHOLD_MARGIN} ordinal-threshold margin."
        )
    return (
        f"near-threshold ordinal parity: ONNX KL{onnx_grade}, PyTorch KL{pytorch_grade}, "
        f"max logit difference {max_difference:.6f}. XAI targets the ONNX predicted class."
    )


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested XAI device without silently falling back from CUDA."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch installation cannot access a CUDA GPU. "
                "Install a CUDA-compatible PyTorch build or use --device cpu."
            )
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")


def set_research_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch generators used by a research run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    metric_source: str,
    score_direction: str,
    metric_parameters: dict,
    metric_status: str = "ok",
    metric_note: str | None = None,
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
        "metric_source": metric_source,
        "score_direction": score_direction,
        "metric_parameters": serialise_parameters(metric_parameters),
        "metric_status": metric_status,
        "metric_note": metric_note,
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
    device_name: str,
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

    deletion_definition = metric_definition("Deletion confidence drop")
    stability_definition = metric_definition("Stability score")
    ordinal_parameters = (
        {"model_output": "ordinal_class_probability", "xai_target": "predicted_kl_class_score"}
        if MODEL_CONFIGS[model_name].get("output_type") == "ordinal_thresholds"
        else {"model_output": "class_probability", "xai_target": "predicted_class_logit"}
    )
    return [
        metric_row(
            sample,
            prediction,
            model_name,
            xai_result.method,
            "Deletion confidence drop",
            deletion_definition.category,
            float(deletion_table.iloc[0]["Confidence drop"]),
            run_id,
            seed,
            checkpoint_version,
            deletion_definition.source,
            deletion_definition.score_direction,
            {"masking": "top_20_percent", "replacement": "image_mean", "device": device_name, **ordinal_parameters},
        ),
        metric_row(
            sample,
            prediction,
            model_name,
            xai_result.method,
            "Stability score",
            stability_definition.category,
            float(stability_table.iloc[0]["Stability score"]),
            run_id,
            seed,
            checkpoint_version,
            stability_definition.source,
            stability_definition.score_direction,
            {"brightness_multiplier": 1.05, "contrast_multiplier": 1.05, "device": device_name, **ordinal_parameters},
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
    device_name: str,
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
            "quantus",
            score.score_direction,
            {**score.parameters, "device": device_name},
            score.status,
            score.note,
        )
        for score in scores
    ]


def _validate_resume_configuration(
    existing: pd.DataFrame,
    *,
    model_name: str,
    run_id: str,
    seed: int,
    checkpoint_version: str,
    faithfulness_runs: int,
    max_sensitivity_samples: int,
    device_name: str,
) -> None:
    """Reject a resume attempt that would mix incompatible experimental settings."""
    expected = {
        "model_name": model_name,
        "run_id": run_id,
        "seed": seed,
        "checkpoint_version": checkpoint_version,
    }
    for column, value in expected.items():
        if column not in existing.columns or set(existing[column].dropna().astype(str)) != {str(value)}:
            raise ValueError(f"Cannot resume: existing output has a different {column}.")

    expected_parameters = {
        "FaithfulnessCorrelation": {"nr_runs": faithfulness_runs},
        "MaxSensitivity": {"nr_samples": max_sensitivity_samples},
    }
    for metric_name, expected_values in expected_parameters.items():
        metric_rows = existing[existing["metric_name"] == metric_name]
        if metric_rows.empty:
            continue
        for value in metric_rows.get("metric_parameters", pd.Series(dtype="object")).dropna():
            parameters = parse_parameters(value)
            if parameters is None:
                raise ValueError("Cannot resume: existing output has no readable metric configuration.")
            if parameters.get("device") != device_name:
                raise ValueError("Cannot resume: existing output was created on a different device.")
            for key, expected_value in expected_values.items():
                if parameters.get(key) != expected_value:
                    raise ValueError(f"Cannot resume: existing {metric_name} uses a different {key}.")

    # All other fields are fixed by this pipeline version. Multiple parameter
    # objects for a metric would mean an interrupted file mixed experiments.
    for metric_name, metric_rows in existing.groupby("metric_name"):
        parameter_texts = set(metric_rows.get("metric_parameters", pd.Series(dtype="object")).dropna().astype(str))
        if not parameter_texts:
            raise ValueError(f"Cannot resume: {metric_name} has no stored metric configuration.")
        if len(parameter_texts) != 1:
            raise ValueError(f"Cannot resume: {metric_name} contains mixed metric configurations.")
        parameters = parse_parameters(next(iter(parameter_texts)))
        if parameters is None or parameters.get("device") != device_name:
            raise ValueError(f"Cannot resume: {metric_name} was created with a different or unreadable device configuration.")


def completed_combinations(rows: pd.DataFrame) -> set[tuple[str, str]]:
    """Return sample/method pairs which already contain every runner metric."""
    completed = set()
    for (sample_id, method), group in rows.groupby(["sample_id", "xai_method"]):
        if RUNNER_METRICS.issubset(set(group["metric_name"])):
            completed.add((str(sample_id), str(method)))
    return completed


def _write_validated_rows(rows: list[dict], output_path: Path) -> pd.DataFrame:
    results = validate_research_results(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def run_research(
    manifest_path: Path,
    model_name: str,
    methods: list[str],
    output_path: Path,
    run_id: str,
    seed: int,
    limit: int | None,
    faithfulness_runs: int,
    max_sensitivity_samples: int,
    device: str = "auto",
    resume: bool = False,
) -> pd.DataFrame:
    set_research_seed(seed)
    selected_device = resolve_device(device)
    device_name = str(selected_device)
    samples = load_dataset_manifest(manifest_path)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No samples selected for the research run.")

    model_config = MODEL_CONFIGS[model_name]
    classifier = ONNXKneeOsteoarthritisClassifier(
        model_config["onnx_path"], output_type=model_config.get("output_type", "multiclass")
    )
    xai_model = load_xai_model_by_name(model_name, device=selected_device)
    checkpoint_version = model_config["checkpoint_path"].name
    rows: list[dict] = []
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --resume to continue this run or choose a new --output path."
            )
        existing = load_research_results(output_path)
        _validate_resume_configuration(
            existing,
            model_name=model_name,
            run_id=run_id,
            seed=seed,
            checkpoint_version=checkpoint_version,
            faithfulness_runs=faithfulness_runs,
            max_sensitivity_samples=max_sensitivity_samples,
            device_name=device_name,
        )
        rows = existing.to_dict("records")
    complete_pairs = completed_combinations(pd.DataFrame(rows)) if rows else set()
    print(f"XAI device: {device_name}")

    for sample_index, sample in enumerate(samples, start=1):
        image_bytes = sample.image_path.read_bytes()
        prediction = prediction_to_dict(classifier.predict(image_bytes))
        image = Image.open(sample.image_path).convert("RGB")
        _, input_tensor = preprocess_pil_for_torch(image)
        input_tensor = input_tensor.to(selected_device)
        with torch.no_grad():
            xai_prediction = predict_kl_with_xai_model(xai_model, input_tensor, model_name)
        if model_config.get("output_type") == "ordinal_thresholds":
            base_model = xai_model.base_model
            with torch.no_grad():
                pytorch_logits = base_model(input_tensor).detach().cpu().numpy()
            onnx_logits = classifier.session.run(
                None,
                {classifier.input_name: input_tensor.detach().cpu().numpy()},
            )[0]
            warning = ordinal_parity_warning(
                onnx_logits,
                pytorch_logits,
                int(prediction["grade"]),
                xai_prediction,
            )
            if warning:
                print(f"  Warning for {sample.sample_id}: {warning}")
        elif xai_prediction != int(prediction["grade"]):
            raise RuntimeError(
                f"Prediction mismatch for {sample.sample_id}: ONNX predicted KL{prediction['grade']}, "
                f"but the PyTorch XAI checkpoint predicted KL{xai_prediction}."
            )
        print(
            f"[{sample_index}/{len(samples)}] {sample.sample_id}: "
            f"true KL{sample.true_kl}, predicted KL{prediction['grade']}"
        )

        for method in methods:
            pair = (sample.sample_id, method)
            if pair in complete_pairs:
                print(f"  {method}: already complete, skipped")
                continue
            # An interrupted write may contain only some of a pair's metrics.
            # Replace that incomplete set rather than introducing duplicates.
            rows = [
                row for row in rows
                if not (str(row["sample_id"]) == sample.sample_id and str(row["xai_method"]) == method)
            ]
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
                attribution=xai_result.raw_attribution,
                target_class=xai_result.class_idx,
                xai_method=method,
                model_name=model_name,
                faithfulness_runs=faithfulness_runs,
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
                    device_name,
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
                    device_name,
                )
            )
            _write_validated_rows(rows, output_path)
            complete_pairs.add(pair)

    return _write_validated_rows(rows, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline XAI evaluation and write Research Mode results.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "inputs" / "manifest.json")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="resnet50")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--faithfulness-runs", type=int, default=50)
    parser.add_argument("--max-sensitivity-samples", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue a matching partially completed output CSV.")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.faithfulness_runs <= 0 or args.max_sensitivity_samples <= 0:
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
        faithfulness_runs=arguments.faithfulness_runs,
        max_sensitivity_samples=arguments.max_sensitivity_samples,
        device=arguments.device,
        resume=arguments.resume,
    )
    print(f"Saved {len(dataframe)} metric rows to: {arguments.output}")
