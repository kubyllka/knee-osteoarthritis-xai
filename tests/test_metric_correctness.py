import unittest

import numpy as np
import torch

from app.metric_registry import format_metric_value, metric_definition
from app.research.results import ResearchResultsValidationError, validate_research_results
from app.xai import normalize_cam, resize_attribution
from scripts.run_research import set_research_seed
from test_research_results import valid_results


class MetricCorrectnessTests(unittest.TestCase):
    def test_display_normalisation_does_not_change_raw_attribution(self) -> None:
        raw = np.array([[2.0, 4.0], [8.0, 16.0]], dtype=np.float32)
        metric_map = resize_attribution(raw)
        display_map = normalize_cam(raw)

        self.assertAlmostEqual(float(metric_map.max()), 16.0)
        self.assertAlmostEqual(float(display_map.min()), 0.0)
        self.assertAlmostEqual(float(display_map.max()), 1.0)
        self.assertFalse(np.allclose(metric_map, display_map))

    def test_seed_reproduces_numpy_and_torch_sequences(self) -> None:
        set_research_seed(42)
        first = (np.random.random(3), torch.rand(3))
        set_research_seed(42)
        second = (np.random.random(3), torch.rand(3))

        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_metric_metadata_accepts_json_and_legacy_rows(self) -> None:
        modern = valid_results()
        modern["metric_source"] = "quantus"
        modern["score_direction"] = "lower"
        modern["metric_parameters"] = '{"n_perturb_samples":8}'
        loaded = validate_research_results(modern)
        self.assertEqual(loaded.loc[0, "score_direction"], "lower")

        legacy = valid_results()
        self.assertNotIn("metric_parameters", validate_research_results(legacy).columns)

    def test_invalid_metric_parameter_json_is_rejected(self) -> None:
        invalid = valid_results()
        invalid["metric_parameters"] = "not-json"
        with self.assertRaisesRegex(ResearchResultsValidationError, "metric_parameters"):
            validate_research_results(invalid)

    def test_infidelity_has_scientific_display_rule(self) -> None:
        self.assertEqual(metric_definition("Infidelity").score_direction, "lower")
        self.assertEqual(format_metric_value("Infidelity", 750866100.0), "7.509e+08")


if __name__ == "__main__":
    unittest.main()
