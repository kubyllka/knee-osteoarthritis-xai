import unittest

import numpy as np
import pandas as pd

from app.research.analysis import (
    ARCHITECTURE_MODELS,
    METHOD_COMPARISON_METRICS,
    ResearchAnalysisError,
    architecture_comparison_analysis,
    architecture_complete_case_pairs,
    classifier_subset_summary,
    dunn_posthoc,
    epsilon_squared,
    friedman_method_analysis,
    holm_adjust,
    kl_stage_analysis,
    method_comparison_descriptives,
    method_complete_case_matrices,
    merge_research_results,
    prediction_correctness_analysis,
    prediction_correctness_values,
    stratified_correctness_permutation_p_value,
    _stratified_permuted_correctness_labels,
    wilcoxon_method_posthoc,
)


def make_rows(method: str = "Grad-CAM", metric: str = "Region perturbation score-drop AUC") -> pd.DataFrame:
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


def make_paired_rows(
    model_name: str = "model-a", metric: str = "Region perturbation score-drop AUC", identical_first_pair: bool = False
) -> pd.DataFrame:
    methods = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Integrated Gradients", "SmoothGrad"]
    frames = []
    for position, method in enumerate(methods):
        frame = make_rows(method=method, metric=metric).copy()
        frame["model_name"] = model_name
        # Each method receives a fixed offset, producing paired method differences.
        # The optional first identical pair exercises Wilcoxon's all-zero safeguard.
        offset = 0 if identical_first_pair and position == 1 else position
        frame["metric_value"] = (frame["true_kl"] + 1) * 0.01 + offset * 0.1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_architecture_rows(metric: str = "Region perturbation score-drop AUC") -> pd.DataFrame:
    dense = make_paired_rows(model_name="densenet121", metric=metric)
    resnet = make_paired_rows(model_name="resnet50", metric=metric)
    resnet["metric_value"] = resnet["metric_value"] - 0.02
    return pd.concat([dense, resnet], ignore_index=True)


def make_correctness_rows(model_name: str = "model-a") -> pd.DataFrame:
    frames = []
    methods = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Integrated Gradients", "SmoothGrad"]
    for metric in METHOD_COMPARISON_METRICS:
        for position, method in enumerate(methods):
            frame = make_rows(method=method, metric=metric).copy()
            frame["model_name"] = model_name
            correct = frame["predicted_kl"].eq(frame["true_kl"])
            frame["metric_value"] = 0.05 + position * 0.02 + frame["true_kl"] * 0.01 + correct * 0.2
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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

    def test_method_comparison_uses_complete_cases_for_every_method(self) -> None:
        rows = make_paired_rows()
        undefined = rows.index[(rows["xai_method"] == "Grad-CAM") & (rows["sample_id"] == "kl2-1")][0]
        rows.loc[undefined, ["metric_value", "metric_status", "metric_note"]] = [
            None,
            "undefined_zero_perturbed_attribution",
            "A perturbed input produced an all-zero attribution map.",
        ]

        matrices = method_complete_case_matrices(merge_research_results([rows]))
        matrix = matrices[("model-a", "Region perturbation score-drop AUC")]
        descriptives = method_comparison_descriptives(matrices)

        self.assertEqual(len(matrix), 9)
        self.assertEqual(matrix.attrs["candidate_n"], 10)
        self.assertEqual(matrix.attrs["excluded_n"], 1)
        self.assertTrue(descriptives["N"].eq(9).all())

    def test_friedman_reports_kendalls_w_and_holm_per_model(self) -> None:
        rows = pd.concat(
            [make_paired_rows(metric=metric) for metric in METHOD_COMPARISON_METRICS],
            ignore_index=True,
        )
        matrices = method_complete_case_matrices(merge_research_results([rows]))
        summary = friedman_method_analysis(matrices)
        deletion = summary[summary["Metric"].eq("Region perturbation score-drop AUC")].iloc[0]

        expected_w = deletion["Friedman statistic"] / (deletion["Valid N"] * 4)
        self.assertAlmostEqual(deletion["Kendall's W"], expected_w)
        self.assertEqual(len(summary), 5)
        self.assertTrue((summary["p Holm"] >= summary["p raw"]).all())
        self.assertFalse(any("rank" in column.lower() for column in summary.columns))

    def test_wilcoxon_outputs_ten_pairs_only_after_significant_friedman(self) -> None:
        significant_rows = make_paired_rows(identical_first_pair=True)
        nonsignificant_rows = make_paired_rows(metric="Sparseness")
        for method in nonsignificant_rows["xai_method"].unique():
            mask = nonsignificant_rows["xai_method"].eq(method)
            nonsignificant_rows.loc[mask, "metric_value"] = make_rows()["metric_value"].to_numpy()
        rows = pd.concat([significant_rows, nonsignificant_rows], ignore_index=True)
        matrices = method_complete_case_matrices(merge_research_results([rows]))
        summary = friedman_method_analysis(matrices)
        summary.loc[summary["Metric"].eq("Region perturbation score-drop AUC"), "Friedman significant"] = True
        summary.loc[summary["Metric"].eq("Sparseness"), "Friedman significant"] = False

        posthoc = wilcoxon_method_posthoc(matrices, summary)

        self.assertEqual(len(posthoc), 10)
        self.assertTrue(posthoc["p Wilcoxon Holm"].between(0, 1).all())
        zero_pair = posthoc[posthoc["Method pair"].eq("Grad-CAM vs Grad-CAM++")].iloc[0]
        self.assertEqual(zero_pair["Test status"], "all paired differences zero")

    def test_architecture_comparison_uses_only_complete_finite_pairs(self) -> None:
        rows = make_architecture_rows()
        undefined = rows.index[
            (rows["model_name"] == "resnet50")
            & (rows["xai_method"] == "Grad-CAM")
            & (rows["sample_id"] == "kl2-1")
        ][0]
        rows.loc[undefined, ["metric_value", "metric_status", "metric_note"]] = [
            None,
            "undefined_zero_perturbed_attribution",
            "A perturbed input produced an all-zero attribution map.",
        ]

        pairs = architecture_complete_case_pairs(merge_research_results([rows]))
        matrix = pairs[("Grad-CAM", "Region perturbation score-drop AUC")]
        summary = architecture_comparison_analysis(pairs)
        result = summary[
            summary["XAI method"].eq("Grad-CAM")
            & summary["Metric"].eq("Region perturbation score-drop AUC")
        ].iloc[0]

        self.assertEqual(tuple(matrix.columns), ARCHITECTURE_MODELS)
        self.assertEqual(len(matrix), 9)
        self.assertEqual(matrix.attrs["candidate_n"], 10)
        self.assertEqual(matrix.attrs["excluded_n"], 1)
        self.assertEqual(int(result["Paired N"]), 9)
        self.assertEqual(int(result["Excluded N"]), 1)
        self.assertGreater(result["Median paired difference (DenseNet - ResNet)"], 0)
        self.assertGreater(result["Rank-biserial correlation"], 0)

    def test_architecture_same_target_sensitivity_uses_matching_predictions_only(self) -> None:
        rows = make_architecture_rows()
        mismatch = (rows["model_name"] == "resnet50") & (rows["sample_id"] == "kl1-0")
        rows.loc[mismatch, ["predicted_kl", "target_class", "is_correct", "error_distance"]] = [2, 2, False, 1]

        pairs = architecture_complete_case_pairs(merge_research_results([rows]), same_target_only=True)
        matrix = pairs[("Grad-CAM", "Region perturbation score-drop AUC")]

        self.assertEqual(matrix.attrs["candidate_n"], 9)
        self.assertEqual(len(matrix), 9)
        self.assertTrue(matrix.attrs["same_target_only"])

    def test_architecture_holm_correction_covers_25_method_metric_tests(self) -> None:
        rows = pd.concat(
            [make_architecture_rows(metric=metric) for metric in METHOD_COMPARISON_METRICS],
            ignore_index=True,
        )
        summary = architecture_comparison_analysis(
            architecture_complete_case_pairs(merge_research_results([rows]))
        )

        self.assertEqual(len(summary), 25)
        self.assertTrue((summary["p Holm"] >= summary["p raw"]).all())
        self.assertFalse(any("winner" in column.lower() or "leaderboard" in column.lower() for column in summary.columns))

    def test_correctness_analysis_reports_groups_exclusions_and_positive_effect(self) -> None:
        rows = make_correctness_rows()
        undefined = rows.index[
            (rows["xai_method"] == "Grad-CAM")
            & (rows["metric_name"] == "Region perturbation score-drop AUC")
            & (rows["sample_id"] == "kl2-1")
        ][0]
        rows.loc[undefined, ["metric_value", "metric_status", "metric_note"]] = [
            None,
            "undefined_zero_perturbed_attribution",
            "A perturbed input produced an all-zero attribution map.",
        ]
        results = merge_research_results([rows])
        values = prediction_correctness_values(results)
        summary = prediction_correctness_analysis(results, permutations=50)
        result = summary[
            summary["XAI method"].eq("Grad-CAM")
            & summary["Metric"].eq("Region perturbation score-drop AUC")
        ].iloc[0]

        self.assertEqual(len(values[values["Metric"].eq("Region perturbation score-drop AUC")]), 49)
        self.assertEqual(int(result["Correct N"]) + int(result["Incorrect N"]), 9)
        self.assertEqual(int(result["Excluded N"]), 1)
        self.assertGreater(result["Correct median"], result["Incorrect median"])
        self.assertGreater(result["Rank-biserial correlation"], 0)

    def test_stratified_permutations_preserve_kl_group_counts_and_are_reproducible(self) -> None:
        true_kl = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        labels = np.array([True, True, False, True, False, False, True, False])
        first = _stratified_permuted_correctness_labels(
            true_kl, labels, permutations=25, rng=np.random.default_rng(42)
        )
        second = _stratified_permuted_correctness_labels(
            true_kl, labels, permutations=25, rng=np.random.default_rng(42)
        )

        self.assertTrue(np.array_equal(first, second))
        for grade in np.unique(true_kl):
            self.assertTrue((first[:, true_kl == grade].sum(axis=1) == labels[true_kl == grade].sum()).all())

    def test_stratified_permutation_uses_two_sided_plus_one_correction(self) -> None:
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        true_kl = np.array([0, 0, 0, 1, 1, 1])
        labels = np.array([True, True, False, True, False, False])
        p_first = stratified_correctness_permutation_p_value(values, true_kl, labels, seed=42, permutations=100)
        p_second = stratified_correctness_permutation_p_value(values, true_kl, labels, seed=42, permutations=100)

        self.assertEqual(p_first, p_second)
        self.assertGreaterEqual(p_first, 1 / 101)
        self.assertAlmostEqual(p_first * 101, round(p_first * 101))

    def test_correctness_holm_is_separate_per_model_and_never_mixes_metrics(self) -> None:
        rows = pd.concat([make_correctness_rows("model-a"), make_correctness_rows("model-b")], ignore_index=True)
        summary = prediction_correctness_analysis(merge_research_results([rows]), permutations=20)

        self.assertEqual(len(summary), 50)
        self.assertTrue(summary.groupby("Model").size().eq(25).all())
        self.assertTrue((summary["p Holm"] >= summary["p raw"]).all())
        self.assertTrue((summary["KL-adjusted permutation p Holm"] >= summary["KL-adjusted permutation p raw"]).all())


if __name__ == "__main__":
    unittest.main()
