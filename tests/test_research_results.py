from pathlib import Path
import unittest

import pandas as pd

from app.research.results import (
    ResearchResultsValidationError,
    load_research_results,
    validate_research_results,
)


def valid_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "image_path": "data/test/sample_001.png",
                "true_kl": 2,
                "predicted_kl": 2,
                "is_correct": True,
                "error_distance": 0,
                "confidence": 0.84,
                "model_name": "example-model",
                "xai_method": "Grad-CAM",
                "metric_name": "Infidelity",
                "metric_category": "faithfulness",
                "metric_value": 0.12,
                "run_id": "baseline-01",
                "split": "test",
                "seed": 42,
                "checkpoint_version": "best",
                "target_class": 2,
            }
        ]
    )


class ResearchResultsTests(unittest.TestCase):
    def test_repository_demo_file_follows_the_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        demo_path = project_root / "results" / "research" / "demo_xai_metrics.csv"

        loaded = load_research_results(demo_path)

        self.assertEqual(len(loaded), 20)

    def test_valid_in_memory_results_are_normalised(self) -> None:
        loaded = validate_research_results(valid_results())

        self.assertEqual(loaded.loc[0, "true_kl"], 2)
        self.assertTrue(loaded.loc[0, "is_correct"])

    def test_missing_required_column_is_rejected(self) -> None:
        invalid = valid_results().drop(columns="metric_value")

        with self.assertRaisesRegex(ResearchResultsValidationError, "Missing required columns"):
            validate_research_results(invalid)

    def test_invalid_kl_is_rejected(self) -> None:
        invalid = valid_results()
        invalid.loc[0, "true_kl"] = 5

        with self.assertRaisesRegex(ResearchResultsValidationError, "true_kl"):
            validate_research_results(invalid)

    def test_non_numeric_metric_value_is_rejected(self) -> None:
        invalid = valid_results()
        invalid["metric_value"] = invalid["metric_value"].astype(object)
        invalid.loc[0, "metric_value"] = "not-a-number"

        with self.assertRaisesRegex(ResearchResultsValidationError, "metric_value"):
            validate_research_results(invalid)


if __name__ == "__main__":
    unittest.main()
