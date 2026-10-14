import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from app.config import MODEL_CONFIGS, RESEARCH_MODEL_IDS
from app.metric_registry import metric_definition
from app.research.analysis import architecture_complete_case_pairs, merge_research_results
from app.research.experiment import FINAL_RESEARCH_PROFILE
from app.research.manifest import DatasetManifestValidationError, DatasetSample, validate_final_test_manifest
from app.research.run_metadata import (
    ensure_matching_metadata,
    metadata_path,
    sample_content_digest,
    write_dataframe_atomic,
)
from app.research.results import load_research_results
from scripts.build_final_test_manifest import build_final_manifest
from scripts.run_research import parse_args
from scripts.run_research import quantus_metric_rows
from app.quantus_metrics import QuantusScore
from streamlit_app import _architecture_comparison_plot_data, _architecture_median_summary


class FinalResearchTests(unittest.TestCase):
    def test_four_distinct_model_ids_share_only_two_architectures(self) -> None:
        self.assertEqual(len(RESEARCH_MODEL_IDS), 4)
        self.assertEqual(len(set(RESEARCH_MODEL_IDS)), 4)
        self.assertEqual(
            {MODEL_CONFIGS[name]["architecture"] for name in RESEARCH_MODEL_IDS},
            {"resnet50", "densenet121"},
        )
        self.assertEqual(
            {MODEL_CONFIGS[name]["training_variant"] for name in RESEARCH_MODEL_IDS},
            {"ordinal_bce", "class_aware_ordinal_bce"},
        )
        self.assertTrue(all(MODEL_CONFIGS[name]["num_outputs"] == 4 for name in RESEARCH_MODEL_IDS))

    def test_final_profile_and_custom_faithfulness_source(self) -> None:
        profile = FINAL_RESEARCH_PROFILE
        self.assertEqual(len(profile.methods), 5)
        self.assertEqual(len(profile.metrics), 5)
        self.assertEqual((profile.faithfulness_runs, profile.faithfulness_subset_size), (50, 1024))
        self.assertEqual((profile.max_sensitivity_samples, profile.max_sensitivity_normalise), (8, True))
        self.assertEqual(metric_definition("FaithfulnessCorrelation").source, "custom")

    def test_new_faithfulness_row_records_custom_source(self) -> None:
        sample = DatasetSample("one", Path(__file__).resolve(), 0, "test")
        score = QuantusScore("FaithfulnessCorrelation", "faithfulness", "higher", 0.3, {"nr_runs": 50})
        rows = quantus_metric_rows(
            sample, {"grade": 0, "confidence": 0.6}, "resnet50", "Grad-CAM", [score],
            "run", 42, "checkpoint.pth", "cpu", np.ones((2, 2), dtype=np.float32),
        )
        self.assertEqual(rows[0]["metric_source"], "custom")

    def test_cli_final_profile_cannot_be_overridden(self) -> None:
        with patch.object(sys, "argv", ["run_research.py", "--profile", "final_research", "--all-models"]):
            args = parse_args()
        self.assertEqual(args.faithfulness_subset_size, 1024)
        self.assertEqual(args.faithfulness_runs, 50)
        self.assertTrue(args.max_sensitivity_normalise)
        self.assertEqual(args.methods, list(FINAL_RESEARCH_PROFILE.methods))
        with patch.object(sys, "argv", [
            "run_research.py", "--profile", "final_research", "--faithfulness-subset-size", "224"
        ]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_final_manifest_preserves_original_ids_and_rejects_duplicate_contents(self) -> None:
        root = Path(__file__).resolve().parent
        token = uuid.uuid4().hex
        paths = [root / f"_final_test_{token}_{grade}.png" for grade in range(5)]
        try:
            samples = []
            for grade, path in enumerate(paths):
                path.write_bytes(f"image-{grade}".encode())
                samples.append(DatasetSample(f"old-{grade}", path, grade, "test"))
            validate_final_test_manifest(samples, samples, samples_per_kl=1)
            changed = list(samples)
            changed[0] = DatasetSample("new-id", changed[0].image_path, 0, "test")
            with self.assertRaisesRegex(DatasetManifestValidationError, "Original sample"):
                validate_final_test_manifest(changed, samples, samples_per_kl=1)
            samples[-1].image_path.write_bytes(samples[0].image_path.read_bytes())
            with self.assertRaisesRegex(DatasetManifestValidationError, "duplicate image file contents"):
                validate_final_test_manifest(samples, samples, samples_per_kl=1)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

    def test_builder_reports_missing_20_per_grade_without_writing_a_manifest(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(DatasetManifestValidationError, "20 additional"):
            build_final_manifest(
                project_root / "inputs/test/test_manifest.json",
                project_root / "inputs/test",
            )
        self.assertFalse((project_root / "inputs/test/final_manifest_250.json").exists())

    def test_atomic_csv_and_resume_identity_reject_changed_manifest(self) -> None:
        output = Path(__file__).resolve().parent / f"_final_test_{uuid.uuid4().hex}.csv"
        try:
            initial = {"model_id": "resnet50", "manifest_sha256": "one", "seed": 42}
            ensure_matching_metadata(output, initial, resume=False)
            self.assertTrue(metadata_path(output).exists())
            write_dataframe_atomic(pd.DataFrame([{"sample_id": "one"}]), output)
            self.assertEqual(pd.read_csv(output).iloc[0]["sample_id"], "one")
            ensure_matching_metadata(output, initial, resume=True)
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                ensure_matching_metadata(output, {**initial, "manifest_sha256": "two"}, resume=True)
        finally:
            output.unlink(missing_ok=True)
            metadata_path(output).unlink(missing_ok=True)

    def test_new_model_csv_round_trips_through_research_loader(self) -> None:
        output = Path(__file__).resolve().parent / f"_final_test_{uuid.uuid4().hex}.csv"
        try:
            frame = pd.DataFrame([{
                "sample_id": "one", "image_path": "inputs/test/0/example.png",
                "true_kl": 0, "predicted_kl": 0, "is_correct": True, "error_distance": 0,
                "confidence": 0.6, "model_name": "resnet50_class_aware",
                "model_id": "resnet50_class_aware", "model_architecture": "resnet50",
                "training_variant": "class_aware_ordinal_bce", "experiment_profile": "final_research",
                "xai_method": "Grad-CAM", "metric_name": "FaithfulnessCorrelation",
                "metric_category": "faithfulness", "metric_value": 0.2,
                "metric_source": "custom", "metric_status": "ok",
                "metric_parameters": json.dumps({"subset_size": 1024}),
                "target_class": 0, "seed": 42,
            }])
            write_dataframe_atomic(frame, output)
            loaded = load_research_results(output)
            self.assertEqual(loaded.iloc[0]["model_id"], "resnet50_class_aware")
            self.assertEqual(loaded.iloc[0]["metric_source"], "custom")
        finally:
            output.unlink(missing_ok=True)

    def test_selected_sample_digest_changes_with_image_bytes(self) -> None:
        path = Path(__file__).resolve().parent / f"_final_test_{uuid.uuid4().hex}.png"
        try:
            path.write_bytes(b"first")
            sample = DatasetSample("one", path, 0, "test")
            first = sample_content_digest([sample])
            path.write_bytes(b"second")
            self.assertNotEqual(first, sample_content_digest([sample]))
        finally:
            path.unlink(missing_ok=True)

    def test_class_aware_model_pair_uses_only_matching_sample_id_and_image(self) -> None:
        rows = []
        for model, score in (("resnet50_class_aware", 0.2), ("densenet121_class_aware", 0.4)):
            rows.append({
                "sample_id": "sample-one", "image_path": "inputs/test/0/example.png",
                "true_kl": 0, "predicted_kl": 0, "is_correct": True, "error_distance": 0,
                "confidence": 0.6, "model_name": model, "xai_method": "Grad-CAM",
                "metric_name": "Sparseness", "metric_category": "complexity", "metric_value": score,
                "target_class": 0, "metric_parameters": json.dumps({"abs": True}),
            })
        results = merge_research_results([pd.DataFrame(rows)])
        pair = ("densenet121_class_aware", "resnet50_class_aware")
        matrices = architecture_complete_case_pairs(results, models=pair)
        self.assertEqual(len(matrices[("Grad-CAM", "Sparseness")]), 1)
        altered = results.copy()
        altered.loc[altered["model_name"].eq("resnet50_class_aware"), "image_path"] = "different.png"
        with self.assertRaisesRegex(ValueError, "image_path differs"):
            architecture_complete_case_pairs(altered, models=pair)

    def test_class_aware_pair_chart_keeps_dynamic_model_ids(self) -> None:
        metric = "Sparseness"
        values = pd.DataFrame([
            {"Metric": metric, "XAI method": "Grad-CAM", "Model": model,
             "sample_id": "one", "metric_value": value}
            for model, value in (("densenet121_class_aware", 0.4), ("resnet50_class_aware", 0.2))
        ])
        plotted = _architecture_comparison_plot_data(values, metric)
        self.assertEqual(set(plotted["Model"].astype(str)),
                         {"densenet121_class_aware", "resnet50_class_aware"})
        summary = pd.DataFrame([{
            "Metric": metric, "XAI method": "Grad-CAM",
            "Model pair": ("densenet121_class_aware", "resnet50_class_aware"),
            "densenet121_class_aware median": 0.4,
            "resnet50_class_aware median": 0.2,
            "Paired N": 1, "Excluded N": 0,
        }])
        medians = _architecture_median_summary(summary, metric)
        self.assertEqual(set(medians["Model"].astype(str)),
                         {"densenet121_class_aware", "resnet50_class_aware"})


if __name__ == "__main__":
    unittest.main()
