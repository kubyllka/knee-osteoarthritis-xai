import argparse
import hashlib
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

from app.config import MODEL_CONFIGS, RESEARCH_MODEL_IDS
from app.evaluation import build_stability_metrics
from app.metric_registry import metric_definition, parse_parameters, serialise_parameters
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.quantus_metrics import (
    REGION_PERTURBATION_PATCH_SIZE,
    REGION_PERTURBATION_REGIONS,
    QuantusScore,
    evaluate_quantus_metrics,
    random_order_region_control,
)
from app.research.experiment import FINAL_RESEARCH_PROFILE
from app.research.manifest import DatasetSample, load_dataset_manifest, validate_final_test_manifest
from app.research.run_metadata import (
    ensure_matching_metadata,
    metadata_path,
    package_versions,
    sample_content_digest,
    sha256_file,
    write_dataframe_atomic,
)
from app.research.results import load_research_results, validate_research_results
from app.xai import generate_xai_visualization, load_xai_model_by_name, predict_kl_with_xai_model, preprocess_pil_for_torch


DEFAULT_METHODS = ["Grad-CAM", "SmoothGrad"]
ORDINAL_LOGIT_PARITY_ATOL = 1e-2
ORDINAL_THRESHOLD_MARGIN = 1e-3
RUNNER_METRICS = frozenset(
    {
        "FaithfulnessCorrelation",
        "Region perturbation score-drop AUC",
        "MaxSensitivity",
        "Sparseness",
        "Stability score",
    }
)
METRIC_KEYS = {
    "faithfulness-correlation": "FaithfulnessCorrelation",
    "region-perturbation": "Region perturbation score-drop AUC",
    "max-sensitivity": "MaxSensitivity",
    "stability": "Stability score",
    "sparseness": "Sparseness",
}
QUANTUS_METRICS = frozenset(RUNNER_METRICS - {"Stability score"})


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


def combination_seed(base_seed: int, *parts: object) -> int:
    """Create a stable seed independent of run order and resume position."""
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


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


def select_manifest_samples(
    samples: list[DatasetSample],
    limit: int | None,
    samples_per_kl: int | None,
) -> list[DatasetSample]:
    """Select either a leading smoke-test subset or a KL-balanced subset."""
    if limit is not None and samples_per_kl is not None:
        raise ValueError("Use either limit or samples_per_kl, not both.")
    if samples_per_kl is not None:
        selected = []
        for grade in range(5):
            grade_samples = [sample for sample in samples if sample.true_kl == grade]
            if len(grade_samples) < samples_per_kl:
                raise ValueError(
                    f"Manifest has only {len(grade_samples)} KL{grade} samples; "
                    f"cannot select {samples_per_kl} per KL grade."
                )
            selected.extend(grade_samples[:samples_per_kl])
        return selected
    return samples[:limit] if limit is not None else samples


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
    attribution_diagnostics: dict[str, float] | None = None,
) -> dict:
    predicted_kl = int(prediction["grade"])
    row = {
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
    if attribution_diagnostics is not None:
        row.update(attribution_diagnostics)
    return row


def attribution_norm_diagnostics(attribution: np.ndarray) -> dict[str, float]:
    """Return scale diagnostics relevant to MaxSensitivity's denominator."""
    values = np.asarray(attribution, dtype=np.float64)
    absolute = np.abs(values)
    maximum = float(absolute.max()) if absolute.size else 0.0
    return {
        "attribution_l1_norm": float(absolute.sum()),
        "attribution_l2_norm": float(np.linalg.norm(values.reshape(-1), ord=2)),
        "attribution_max_abs": maximum,
        "attribution_l2_after_max_normalisation": (
            float(np.linalg.norm((values / maximum).reshape(-1), ord=2))
            if np.isfinite(maximum) and maximum > 0.0
            else float("nan")
        ),
        "attribution_nonzero_fraction": float(np.count_nonzero(absolute) / absolute.size)
        if absolute.size
        else 0.0,
    }


def custom_metric_rows(
    sample: DatasetSample,
    prediction: dict,
    model_name: str,
    xai_result,
    xai_model,
    run_id: str,
    seed: int,
    checkpoint_version: str,
    device_name: str,
) -> list[dict]:
    diagnostics = attribution_norm_diagnostics(xai_result.raw_attribution)
    stability_table, _ = build_stability_metrics(
        model=xai_model,
        image=xai_result.original,
        results=[xai_result],
        predicted_grade=int(prediction["grade"]),
        model_name=model_name,
    )

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
            "Stability score",
            stability_definition.category,
            float(stability_table.iloc[0]["Stability score"]),
            run_id,
            seed,
            checkpoint_version,
            stability_definition.source,
            stability_definition.score_direction,
            {"brightness_multiplier": 1.05, "contrast_multiplier": 1.05,
             "device": device_name, **ordinal_parameters},
            attribution_diagnostics=diagnostics,
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
    attribution: np.ndarray,
) -> list[dict]:
    diagnostics = attribution_norm_diagnostics(attribution)
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
            metric_definition(score.metric_name).source,
            score.score_direction,
            {**score.parameters, "device": device_name},
            score.status,
            score.note,
            diagnostics,
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
    faithfulness_subset_size: int = 224,
    max_sensitivity_normalise: bool = False,
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
        "FaithfulnessCorrelation": {
            "nr_runs": faithfulness_runs,
            "subset_size": faithfulness_subset_size,
        },
        "MaxSensitivity": {
            "nr_samples": max_sensitivity_samples,
            "normalise": max_sensitivity_normalise,
        },
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
    for (metric_name, method), metric_rows in existing.groupby(["metric_name", "xai_method"]):
        parameter_texts = set(metric_rows.get("metric_parameters", pd.Series(dtype="object")).dropna().astype(str))
        if not parameter_texts:
            raise ValueError(f"Cannot resume: {metric_name}/{method} has no stored metric configuration.")
        if len(parameter_texts) != 1:
            raise ValueError(f"Cannot resume: {metric_name}/{method} contains mixed metric configurations.")
        parameters = parse_parameters(next(iter(parameter_texts)))
        if parameters is None or parameters.get("device") != device_name:
            raise ValueError(f"Cannot resume: {metric_name} was created with a different or unreadable device configuration.")


def completed_combinations(rows: pd.DataFrame) -> set[tuple[str, str]]:
    """Return sample/method pairs which already contain every requested metric."""
    return completed_combinations_for_metrics(rows, RUNNER_METRICS)


def completed_combinations_for_metrics(
    rows: pd.DataFrame,
    required_metrics: set[str] | frozenset[str],
) -> set[tuple[str, str]]:
    completed = set()
    for (sample_id, method), group in rows.groupby(["sample_id", "xai_method"]):
        if set(required_metrics).issubset(set(group["metric_name"])):
            completed.add((str(sample_id), str(method)))
    return completed


def _write_validated_rows(rows: list[dict], output_path: Path, metadata_fields: dict | None = None) -> pd.DataFrame:
    results = validate_research_results(pd.DataFrame(rows))
    results["xai_map_target"] = np.where(
        results["xai_method"].eq("Eigen-CAM"), "class_agnostic", "predicted_kl"
    )
    if metadata_fields:
        for key, value in metadata_fields.items():
            results[key] = value
    write_dataframe_atomic(results, output_path)
    return results


def _write_random_control_rows(rows: list[dict], output_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    write_dataframe_atomic(frame, output_path)
    return frame


def _validate_random_control_resume(
    frame: pd.DataFrame,
    *,
    model_name: str,
    checkpoint_version: str,
    device_name: str,
    repeats: int,
    base_seed: int,
) -> None:
    required = {
        "sample_id",
        "model_name",
        "checkpoint_version",
        "device",
        "repeats",
        "seed",
        "patch_size",
        "regions_evaluation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Cannot resume random control; missing columns: " + ", ".join(missing))
    expected_constants = {
        "model_name": str(model_name),
        "checkpoint_version": str(checkpoint_version),
        "device": str(device_name),
        "repeats": str(repeats),
        "patch_size": str(REGION_PERTURBATION_PATCH_SIZE),
        "regions_evaluation": str(REGION_PERTURBATION_REGIONS),
    }
    for column, expected in expected_constants.items():
        actual = set(frame[column].dropna().astype(str))
        if actual != {expected}:
            raise ValueError(f"Cannot resume random control: existing output has a different {column}.")
    for row in frame.itertuples(index=False):
        expected_seed = combination_seed(base_seed, model_name, row.sample_id, "random-region-control")
        if int(row.seed) != expected_seed:
            raise ValueError("Cannot resume random control: existing output has incompatible seeds.")


def build_run_metadata(
    *, manifest_path: Path, samples: list[DatasetSample], model_name: str,
    model_config: dict, run_id: str, seed: int, methods: list[str],
    metrics: frozenset[str], device_name: str, faithfulness_runs: int,
    faithfulness_subset_size: int, max_sensitivity_samples: int,
    max_sensitivity_normalise: bool, profile_name: str | None,
    random_control_enabled: bool, random_control_repeats: int,
) -> dict:
    """Freeze the exact model, image contents, and method/metric configuration."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": model_name,
        "architecture": model_config.get("architecture", model_name),
        "training_variant": model_config.get("training_variant", "legacy"),
        "kl1_loss_multiplier": model_config.get("kl1_loss_multiplier"),
        "checkpoint_filename": model_config["checkpoint_path"].name,
        "checkpoint_sha256": sha256_file(model_config["checkpoint_path"]),
        "onnx_filename": model_config["onnx_path"].name,
        "onnx_sha256": sha256_file(model_config["onnx_path"]),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_sample_content_sha256": sample_content_digest(samples),
        "selected_sample_count": len(samples),
        "seed": seed,
        "methods": list(methods),
        "metrics": sorted(metrics),
        "device": device_name,
        "profile": profile_name or "custom",
        "faithfulness_runs": faithfulness_runs,
        "faithfulness_subset_size": faithfulness_subset_size,
        "max_sensitivity_samples": max_sensitivity_samples,
        "max_sensitivity_normalise": max_sensitivity_normalise,
        "random_control_enabled": random_control_enabled,
        "random_control_repeats": random_control_repeats if random_control_enabled else None,
        "xai_and_metric_settings": FINAL_RESEARCH_PROFILE.metadata() if profile_name == "final_research" else None,
        "implementation_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in (
                "app/preprocessing.py", "app/inference.py", "app/ordinal.py",
                "app/xai.py", "app/quantus_metrics.py", "app/evaluation.py",
                "app/research/experiment.py", "scripts/run_research.py",
            )
        },
        "package_versions": package_versions(),
    }


def run_research(
    manifest_path: Path,
    model_name: str,
    methods: list[str],
    output_path: Path,
    run_id: str,
    seed: int,
    limit: int | None,
    samples_per_kl: int | None,
    faithfulness_runs: int,
    max_sensitivity_samples: int,
    faithfulness_subset_size: int = 224,
    max_sensitivity_normalise: bool = False,
    device: str = "auto",
    resume: bool = False,
    metrics: set[str] | frozenset[str] | None = None,
    random_control_output: Path | None = None,
    random_control_repeats: int = 1,
    profile_name: str | None = None,
) -> pd.DataFrame:
    set_research_seed(seed)
    selected_device = resolve_device(device)
    device_name = str(selected_device)
    selected_metrics = frozenset(metrics or RUNNER_METRICS)
    unknown_metrics = selected_metrics - RUNNER_METRICS
    if unknown_metrics:
        raise ValueError("Unsupported research metrics: " + ", ".join(sorted(unknown_metrics)))
    if profile_name is not None:
        if profile_name != "final_research":
            raise ValueError(f"Unsupported research profile: {profile_name}")
        profile = FINAL_RESEARCH_PROFILE
        if (
            tuple(methods) != profile.methods
            or selected_metrics != frozenset(profile.metrics)
            or faithfulness_runs != profile.faithfulness_runs
            or faithfulness_subset_size != profile.faithfulness_subset_size
            or max_sensitivity_samples != profile.max_sensitivity_samples
            or max_sensitivity_normalise != profile.max_sensitivity_normalise
        ):
            raise ValueError("final_research settings differ from the shared fixed profile.")
    if random_control_output is not None and random_control_output.resolve() == output_path.resolve():
        raise ValueError("The random-control output must be different from the main research output.")
    samples = select_manifest_samples(load_dataset_manifest(manifest_path), limit, samples_per_kl)
    if not samples:
        raise ValueError("No samples selected for the research run.")
    if profile_name == "final_research":
        full_cohort = limit is None and (samples_per_kl is None or samples_per_kl == 50)
        if device_name != "cuda" and full_cohort:
            raise RuntimeError("The full final_research run requires CUDA; use --device cuda on a GPU host.")
        if full_cohort:
            originals = load_dataset_manifest(PROJECT_ROOT / "inputs/test/test_manifest.json")
            validate_final_test_manifest(samples, originals)
        if (REGION_PERTURBATION_PATCH_SIZE, REGION_PERTURBATION_REGIONS) != (
            FINAL_RESEARCH_PROFILE.region_patch_size, FINAL_RESEARCH_PROFILE.region_count
        ):
            raise RuntimeError("Region Perturbation implementation no longer matches final_research profile.")

    model_config = MODEL_CONFIGS[model_name]
    for name in ("checkpoint_path", "onnx_path"):
        if not model_config[name].is_file():
            raise FileNotFoundError(
                f"{model_name}: required {name} is missing: {model_config[name]}. "
                "Place the trusted model file locally or export ONNX from its matching checkpoint."
            )
    run_metadata = build_run_metadata(
        manifest_path=manifest_path, samples=samples, model_name=model_name,
        model_config=model_config, run_id=run_id, seed=seed, methods=methods,
        metrics=selected_metrics, device_name=device_name,
        faithfulness_runs=faithfulness_runs,
        faithfulness_subset_size=faithfulness_subset_size,
        max_sensitivity_samples=max_sensitivity_samples,
        max_sensitivity_normalise=max_sensitivity_normalise,
        profile_name=profile_name,
        random_control_enabled=random_control_output is not None,
        random_control_repeats=random_control_repeats,
    )
    if output_path.exists() and resume and not metadata_path(output_path).exists() and profile_name is None:
        print("Warning: resuming a legacy CSV without a run-metadata sidecar; only row-level checks are available.")
    else:
        ensure_matching_metadata(output_path, run_metadata, resume=resume)
    metadata_fields = {
        "model_id": model_name,
        "experiment_profile": run_metadata["profile"],
        "model_architecture": run_metadata["architecture"],
        "training_variant": run_metadata["training_variant"],
        "kl1_loss_multiplier": run_metadata["kl1_loss_multiplier"],
        "manifest_sha256": run_metadata["manifest_sha256"],
        "selected_sample_content_sha256": run_metadata["selected_sample_content_sha256"],
        "checkpoint_sha256": run_metadata["checkpoint_sha256"],
        "onnx_sha256": run_metadata["onnx_sha256"],
    }
    classifier = ONNXKneeOsteoarthritisClassifier(
        model_config["onnx_path"], output_type=model_config.get("output_type", "multiclass")
    )
    xai_model = load_xai_model_by_name(model_name, device=selected_device)
    checkpoint_version = model_config["checkpoint_path"].name
    rows: list[dict] = []
    random_control_rows: list[dict] = []
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
            faithfulness_subset_size=faithfulness_subset_size,
            max_sensitivity_normalise=max_sensitivity_normalise,
        )
        rows = existing.to_dict("records")
    complete_pairs = (
        completed_combinations_for_metrics(pd.DataFrame(rows), selected_metrics)
        if rows else set()
    )
    print(f"XAI device: {device_name}")

    if random_control_output is not None and random_control_output.exists():
        if not resume:
            raise FileExistsError(
                f"Random-control output already exists: {random_control_output}. "
                "Use --resume or choose a new path."
            )
        random_control_frame = pd.read_csv(random_control_output)
        _validate_random_control_resume(
            random_control_frame,
            model_name=model_name,
            checkpoint_version=checkpoint_version,
            device_name=device_name,
            repeats=random_control_repeats,
            base_seed=seed,
        )
        random_control_rows = random_control_frame.to_dict("records")
    completed_control_samples = {
        str(row["sample_id"]) for row in random_control_rows
        if str(row.get("model_name")) == model_name
    }

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

        if random_control_output is not None and sample.sample_id not in completed_control_samples:
            control_seed = combination_seed(seed, model_name, sample.sample_id, "random-region-control")
            control_mean, control_std, control_values = random_order_region_control(
                model=xai_model,
                input_tensor=input_tensor,
                target_class=int(prediction["grade"]),
                repeats=random_control_repeats,
                seed=control_seed,
            )
            random_control_rows.append({
                "sample_id": sample.sample_id,
                "image_path": relative_image_path(sample),
                "true_kl": sample.true_kl,
                "predicted_kl": int(prediction["grade"]),
                "model_name": model_name,
                "target_class": int(prediction["grade"]),
                "random_order_auc_mean": control_mean,
                "random_order_auc_std": control_std,
                "random_order_auc_values": serialise_parameters({"values": control_values}),
                "repeats": random_control_repeats,
                "seed": control_seed,
                "patch_size": REGION_PERTURBATION_PATCH_SIZE,
                "regions_evaluation": REGION_PERTURBATION_REGIONS,
                "checkpoint_version": checkpoint_version,
                "device": device_name,
            })
            _write_random_control_rows(random_control_rows, random_control_output)
            completed_control_samples.add(sample.sample_id)

        for method in methods:
            pair = (sample.sample_id, method)
            if pair in complete_pairs:
                print(f"  {method}: already complete, skipped")
                continue
            # An interrupted write may contain only some of a pair's metrics.
            # Replace that incomplete set rather than introducing duplicates.
            rows = [
                row for row in rows
                if not (
                    str(row["sample_id"]) == sample.sample_id
                    and str(row["xai_method"]) == method
                    and str(row["metric_name"]) in selected_metrics
                )
            ]
            set_research_seed(combination_seed(seed, model_name, sample.sample_id, method, "attribution"))
            xai_result = generate_xai_visualization(
                model=xai_model,
                image=image,
                method=method,
                class_idx=int(prediction["grade"]),
                model_name=model_name,
            )
            selected_quantus_metrics = set(selected_metrics & QUANTUS_METRICS)
            if selected_quantus_metrics:
                quantus_scores = []
                for metric_name in sorted(selected_quantus_metrics):
                    set_research_seed(
                        combination_seed(seed, model_name, sample.sample_id, method, metric_name)
                    )
                    quantus_scores.extend(
                        evaluate_quantus_metrics(
                            model=xai_model,
                            input_tensor=input_tensor,
                            attribution=xai_result.raw_attribution,
                            target_class=xai_result.class_idx,
                            xai_method=method,
                            model_name=model_name,
                            faithfulness_runs=faithfulness_runs,
                            faithfulness_subset_size=faithfulness_subset_size,
                            max_sensitivity_samples=max_sensitivity_samples,
                            max_sensitivity_normalise=max_sensitivity_normalise,
                            requested_metrics={metric_name},
                        )
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
                        xai_result.raw_attribution,
                    )
                )
            if "Stability score" in selected_metrics:
                set_research_seed(combination_seed(seed, model_name, sample.sample_id, method, "stability"))
                rows.extend(
                    custom_metric_rows(
                        sample,
                        prediction,
                        model_name,
                        xai_result,
                        xai_model,
                        run_id,
                        seed,
                        checkpoint_version,
                        device_name,
                    )
                )
            _write_validated_rows(rows, output_path, metadata_fields)
            complete_pairs.add(pair)

    return _write_validated_rows(rows, output_path, metadata_fields)


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline XAI evaluation and write Research Mode results.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "inputs" / "manifest.json")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="resnet50")
    parser.add_argument("--all-models", action="store_true", help="Run all four ordinal model IDs sequentially.")
    parser.add_argument("--profile", choices=["final_research"], default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "research")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-kl", type=int, default=None)
    parser.add_argument("--faithfulness-runs", type=int, default=None)
    parser.add_argument(
        "--faithfulness-subset-size",
        type=int,
        default=None,
        help="Number of spatial RGB pixel groups perturbed per FaithfulnessCorrelation run.",
    )
    parser.add_argument("--max-sensitivity-samples", type=int, default=None)
    parser.add_argument(
        "--max-sensitivity-normalise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Normalise the original and every perturbed attribution independently by their "
            "maximum absolute value inside Quantus MaxSensitivity."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_KEYS),
        default=None,
        help="Metrics to calculate. Use metric keys such as faithfulness-correlation or region-perturbation.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue a matching partially completed output CSV.")
    parser.add_argument(
        "--random-control-output",
        type=Path,
        default=None,
        help="Optional separate CSV for the method-independent random region-order sanity control.",
    )
    parser.add_argument("--random-control-repeats", type=int, default=1)
    args = parser.parse_args()

    if args.profile == "final_research":
        profile = FINAL_RESEARCH_PROFILE
        profile_values = {
            "methods": list(profile.methods),
            "faithfulness_runs": profile.faithfulness_runs,
            "faithfulness_subset_size": profile.faithfulness_subset_size,
            "max_sensitivity_samples": profile.max_sensitivity_samples,
            "max_sensitivity_normalise": profile.max_sensitivity_normalise,
        }
        for name, expected in profile_values.items():
            actual = getattr(args, name)
            if actual is not None and actual != expected:
                parser.error(f"--profile final_research fixes {name} to {expected}; do not override it")
            setattr(args, name, expected)
        final_metric_keys = [key for key, name in METRIC_KEYS.items() if name in profile.metrics]
        if args.metrics is not None and set(args.metrics) != set(final_metric_keys):
            parser.error("--profile final_research requires all five fixed metrics")
        args.metrics = final_metric_keys
    else:
        args.methods = args.methods or DEFAULT_METHODS
        args.metrics = args.metrics or list(METRIC_KEYS)
        args.faithfulness_runs = args.faithfulness_runs if args.faithfulness_runs is not None else 50
        args.faithfulness_subset_size = args.faithfulness_subset_size if args.faithfulness_subset_size is not None else 224
        args.max_sensitivity_samples = args.max_sensitivity_samples if args.max_sensitivity_samples is not None else 8
        args.max_sensitivity_normalise = (
            args.max_sensitivity_normalise if args.max_sensitivity_normalise is not None else False
        )

    if args.all_models and args.output is not None:
        parser.error("--all-models creates one CSV per model; use --output-dir, not --output")
    if args.all_models and args.random_control_output is not None:
        parser.error("--all-models requires separate random-control outputs; run models individually for that control")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.samples_per_kl is not None and args.samples_per_kl <= 0:
        parser.error("--samples-per-kl must be positive")
    if args.limit is not None and args.samples_per_kl is not None:
        parser.error("Use either --limit or --samples-per-kl, not both")
    if (
        args.faithfulness_runs <= 0
        or args.faithfulness_subset_size <= 0
        or args.max_sensitivity_samples <= 0
    ):
        parser.error("Quantus sample counts must be positive")
    if args.random_control_repeats <= 0:
        parser.error("--random-control-repeats must be positive")

    if args.run_id is None:
        args.run_id = datetime.now().strftime("research-%Y%m%d-%H%M%S")
    if args.output is None and not args.all_models:
        args.output = args.output_dir / f"{args.run_id}.csv"
    return args


if __name__ == "__main__":
    arguments = parse_args()
    model_ids = RESEARCH_MODEL_IDS if arguments.all_models else (arguments.model,)
    for model_id in model_ids:
        config = MODEL_CONFIGS[model_id]
        for key in ("checkpoint_path", "onnx_path"):
            if not config[key].is_file():
                raise FileNotFoundError(f"{model_id}: missing {key}: {config[key]}")
    for model_id in model_ids:
        run_id = f"{arguments.run_id}-{model_id}" if arguments.all_models else arguments.run_id
        output_path = arguments.output_dir / f"{run_id}.csv" if arguments.all_models else arguments.output
        dataframe = run_research(
            manifest_path=arguments.manifest,
            model_name=model_id,
            methods=arguments.methods,
            output_path=output_path,
            run_id=run_id,
            seed=arguments.seed,
            limit=arguments.limit,
            samples_per_kl=arguments.samples_per_kl,
            faithfulness_runs=arguments.faithfulness_runs,
            faithfulness_subset_size=arguments.faithfulness_subset_size,
            max_sensitivity_samples=arguments.max_sensitivity_samples,
            max_sensitivity_normalise=arguments.max_sensitivity_normalise,
            device=arguments.device,
            resume=arguments.resume,
            metrics={METRIC_KEYS[key] for key in arguments.metrics},
            random_control_output=arguments.random_control_output,
            random_control_repeats=arguments.random_control_repeats,
            profile_name=arguments.profile,
        )
        print(f"Saved {len(dataframe)} metric rows to: {output_path}")
