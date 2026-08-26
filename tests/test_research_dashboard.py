import unittest
from unittest.mock import patch

import pandas as pd

from app.metric_registry import metric_definition
from streamlit_app import _median_by_method_and_grade, _selected_metric_rows, _summary_card_values
from scripts.run_research import (
    RUNNER_METRICS,
    _validate_resume_configuration,
    completed_combinations,
    resolve_device,
)


class ResearchDashboardAndRunnerTests(unittest.TestCase):
    def test_dashboard_category_routing_matches_evaluation_dimensions(self) -> None:
        self.assertEqual(metric_definition("FaithfulnessCorrelation").category, "faithfulness")
        self.assertEqual(metric_definition("Deletion confidence drop").category, "faithfulness")
        self.assertEqual(metric_definition("MaxSensitivity").category, "robustness")
        self.assertEqual(metric_definition("Stability score").category, "robustness")
        self.assertEqual(metric_definition("Sparseness").category, "complexity")

    def test_completed_combinations_requires_every_metric(self) -> None:
        full = [{"sample_id": "one", "xai_method": "Grad-CAM", "metric_name": metric} for metric in RUNNER_METRICS]
        partial = [{"sample_id": "two", "xai_method": "Grad-CAM", "metric_name": "Sparseness"}]
        completed = completed_combinations(pd.DataFrame(full + partial))
        self.assertEqual(completed, {("one", "Grad-CAM")})

    def test_cpu_device_is_always_available(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")

    @patch("scripts.run_research.torch.cuda.is_available", return_value=True)
    def test_auto_device_selects_cuda_when_available(self, _available) -> None:
        self.assertEqual(resolve_device("auto").type, "cuda")

    def test_resume_rejects_changed_quantus_configuration(self) -> None:
        rows = pd.DataFrame(
            [{
                "model_name": "resnet50",
                "run_id": "run-1",
                "seed": 42,
                "checkpoint_version": "checkpoint.pth",
                "metric_name": "FaithfulnessCorrelation",
                "metric_parameters": '{"device":"cpu","nr_runs":50}',
            }]
        )
        with self.assertRaisesRegex(ValueError, "nr_runs"):
            _validate_resume_configuration(
                rows,
                model_name="resnet50",
                run_id="run-1",
                seed=42,
                checkpoint_version="checkpoint.pth",
                faithfulness_runs=8,
                max_sensitivity_samples=8,
                device_name="cpu",
            )

    def test_summary_cards_use_one_metric_and_the_correct_direction(self) -> None:
        rows = pd.DataFrame(
            [
                {"sample_id": "one", "xai_method": "A", "true_kl": 0, "is_correct": True, "metric_name": "FaithfulnessCorrelation", "metric_value": 0.8},
                {"sample_id": "two", "xai_method": "B", "true_kl": 1, "is_correct": False, "metric_name": "FaithfulnessCorrelation", "metric_value": 0.2},
                {"sample_id": "one", "xai_method": "A", "true_kl": 0, "is_correct": True, "metric_name": "MaxSensitivity", "metric_value": 0.1},
            ]
        )
        selected = _selected_metric_rows(rows, "FaithfulnessCorrelation")
        values = _summary_card_values(selected, metric_definition("FaithfulnessCorrelation"))
        self.assertEqual(selected["metric_name"].nunique(), 1)
        self.assertEqual(values["method"], "A")
        self.assertEqual(values["best_grade"], 0)
        self.assertEqual(values["worst_grade"], 1)

    def test_heatmap_uses_one_median_per_method_and_kl_grade(self) -> None:
        rows = pd.DataFrame(
            [
                {"sample_id": "one", "xai_method": "A", "true_kl": 0, "metric_value": 0.2},
                {"sample_id": "two", "xai_method": "A", "true_kl": 0, "metric_value": 0.6},
                {"sample_id": "three", "xai_method": "B", "true_kl": 0, "metric_value": 0.1},
            ]
        )
        summary = _median_by_method_and_grade(rows, metric_definition("FaithfulnessCorrelation"))
        self.assertEqual(len(summary), 2)
        self.assertEqual(float(summary.loc[summary["xai_method"] == "A", "Median"].iloc[0]), 0.4)


if __name__ == "__main__":
    unittest.main()
