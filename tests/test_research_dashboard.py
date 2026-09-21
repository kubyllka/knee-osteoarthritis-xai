import unittest
from unittest.mock import patch

import pandas as pd

from app.metric_registry import metric_definition
from streamlit_app import (
    _classifier_accuracy_chart_data,
    _effect_map_data,
    _effect_map_scale_max,
    _is_custom_check,
    _median_by_method_and_grade,
    _selected_metric_rows,
    _summary_card_values,
)
from scripts.run_research import (
    RUNNER_METRICS,
    _validate_resume_configuration,
    completed_combinations,
    ordinal_parity_warning,
    resolve_device,
)


class ResearchDashboardAndRunnerTests(unittest.TestCase):
    def test_dashboard_category_routing_matches_evaluation_dimensions(self) -> None:
        self.assertEqual(metric_definition("FaithfulnessCorrelation").category, "faithfulness")
        self.assertEqual(metric_definition("Deletion confidence drop").category, "faithfulness")
        self.assertEqual(metric_definition("MaxSensitivity").category, "robustness")
        self.assertEqual(metric_definition("Stability score").category, "robustness")
        self.assertEqual(metric_definition("Sparseness").category, "complexity")

    def test_custom_checks_are_visible_in_their_own_tab(self) -> None:
        self.assertTrue(_is_custom_check("Deletion confidence drop", "custom"))
        self.assertTrue(_is_custom_check("Stability score", "custom"))
        self.assertFalse(_is_custom_check("MaxSensitivity", "quantus"))

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

    def test_near_threshold_ordinal_parity_is_reported_but_accepted(self) -> None:
        warning = ordinal_parity_warning(
            onnx_logits=[[-0.000024, -1.9, -6.45, -10.46]],
            pytorch_logits=[[0.000083, -1.901, -6.454, -10.466]],
            onnx_grade=0,
            pytorch_grade=1,
        )
        self.assertIn("near-threshold", warning)

    def test_material_ordinal_parity_difference_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "materially"):
            ordinal_parity_warning(
                onnx_logits=[[0.0, -1.0, -2.0, -3.0]],
                pytorch_logits=[[0.1, -1.0, -2.0, -3.0]],
                onnx_grade=0,
                pytorch_grade=1,
            )

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

    def test_accuracy_overview_keeps_per_grade_counts_without_a_method_ranking(self) -> None:
        summary = pd.DataFrame(
            [
                {"Model": "resnet50", "True KL grade": "KL0", "Images": 30, "Correct": 25, "Accuracy": 25 / 30},
                {"Model": "densenet121", "True KL grade": "KL0", "Images": 30, "Correct": 20, "Accuracy": 20 / 30},
            ]
        )

        chart_data = _classifier_accuracy_chart_data(summary)

        self.assertEqual(chart_data["Correct / images"].tolist(), ["20 / 30", "25 / 30"])
        self.assertTrue(chart_data["Accuracy label"].str.endswith("%").all())
        self.assertFalse(any("method" in column.lower() or "rank" in column.lower() for column in chart_data.columns))

    def test_effect_map_is_a_complete_grid_with_effects_not_rankings(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "Model": "resnet50",
                    "Metric": "Deletion confidence drop",
                    "XAI method": "Grad-CAM",
                    "epsilon squared": 0.4,
                    "Kruskal–Wallis H": 10.0,
                    "p raw": 0.01,
                    "p Holm": 0.02,
                    "Valid N": 150,
                    "Excluded undefined": 1,
                    "Global significant": True,
                }
            ]
        )

        effect_data = _effect_map_data(analysis, "resnet50")

        self.assertEqual(len(effect_data), 25)
        selected = effect_data[
            effect_data["Metric"].eq("Deletion confidence drop")
            & effect_data["XAI method"].eq("Grad-CAM")
        ].iloc[0]
        self.assertEqual(selected["Effect label"], "0.400*")
        self.assertEqual(selected["Significance"], "Holm-significant")
        self.assertEqual(effect_data["Effect label"].eq("—").sum(), 24)
        self.assertFalse(any("rank" in column.lower() for column in effect_data.columns))

    def test_effect_maps_use_one_shared_scale_for_both_models(self) -> None:
        analysis = pd.DataFrame(
            {"Model": ["resnet50", "densenet121"], "epsilon squared": [0.18, 0.52]}
        )

        self.assertEqual(_effect_map_scale_max(analysis), 0.52)


if __name__ == "__main__":
    unittest.main()
