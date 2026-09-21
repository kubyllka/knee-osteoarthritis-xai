import unittest

import numpy as np
import pandas as pd

from app.research.analysis import (
    ResearchAnalysisError,
    classifier_subset_summary,
    dunn_posthoc,
    epsilon_squared,
    holm_adjust,
    kl_stage_analysis,
    merge_research_results,
)


def make_rows(method: str = "Grad-CAM", metric: str = "Deletion confidence drop") -> pd.DataFrame:
    rows = []
    for grade in range(5):
        for index in range(2):
            predicted = grade if index == 0 else max(0, grade - 1)
            is_correct = predicted == grade
            rows.append({
                "sample_id": f"kl{grade}-{index}",
                "image_path": f"inputs/kl{grade}-{index}.png",
                "true_kl": grade,
                "predicted_kl": predicted,
                "is_correct": is_correct,
                "error_distance": abs(grade - predicted),
                "confidence": 0.9 if is_correct else 0.6,
                "model_name": "model-a",
                "xai_method": method,
                "metric_name": metric,
                "metric_category": "faithfulness",
                "metric_value": float(grade + index / 10),
                "run_id": "run-a",
                "split": "test",
                "seed": 42,
                "checkpoint_version": "checkpoint.pth",
                "target_class": predicted,
                "metric_source": "custom",
                "score_direction": "higher",
                "metric_parameters": '{"masking":"top_20_percent"}',
                "metric_status": "ok",
                "metric_note": None,
            })
    return pd.DataFrame(rows)


class ResearchAnalysisTests(unittest.TestCase):
    def test_classifier_summary_uses_one_prediction_per_sample(self) -> None:
        results = merge_research_results([make_rows(), make_rows(method="SmoothGrad")])
        overall, by_grade, confusion = classifier_subset_summary(results)

        self.assertEqual(int(overall.loc[0, "Images"]), 10)
        self.assertEqual(len(by_grade), 5)
        self.assertEqual(confusion["model-a"].shape, (5, 5))

    def test_duplicate_experiment_rows_are_rejected(self) -> None:
        rows = make_rows()
        with self.assertRaisesRegex(ResearchAnalysisError, "Duplicate experiment rows"):
            merge_research_results([rows, rows])

    def test_conflicting_prediction_metadata_is_rejected(self) -> None:
        rows = make_rows()
        conflicting = make_rows(method="SmoothGrad")
        conflicting.loc[0, "confidence"] = 0.2
        with self.assertRaisesRegex(ResearchAnalysisError, "Prediction metadata conflict"):
            classifier_subset_summary(merge_research_results([rows, conflicting]))

    def test_kl_summary_reports_per_grade_n_and_undefined_count(self) -> None:
        rows = make_rows()
        rows.loc[5, "metric_value"] = None
        rows.loc[5, "metric_status"] = "undefined_zero_perturbed_attribution"
        rows.loc[5, "metric_note"] = "A perturbed input produced an all-zero attribution map."
        results = merge_research_results([rows])

        summary = kl_stage_analysis(results)

        self.assertEqual(len(summary), 1)
        self.assertEqual(int(summary.loc[0, "KL2 n"]), 1)
        self.assertEqual(int(summary.loc[0, "Excluded undefined"]), 1)

    def test_holm_adjustment_and_fixed_epsilon_squared_formula(self) -> None:
        adjusted = holm_adjust(pd.Series([0.01, 0.02, 0.04], index=["a", "b", "c"]))

        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.04)
        self.assertAlmostEqual(epsilon_squared(10.0, total_n=30, groups=5), 0.24)

    def test_dunn_posthoc_runs_only_after_significant_global_test(self) -> None:
        results = merge_research_results([make_rows()])
        summary = kl_stage_analysis(results)
        summary.loc[0, "Global significant"] = True

        posthoc = dunn_posthoc(results, summary)

        self.assertEqual(len(posthoc), 10)
        self.assertTrue(posthoc["p Dunn-Holm"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
