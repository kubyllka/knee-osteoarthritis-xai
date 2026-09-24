import unittest
from unittest.mock import patch

import numpy as np
import torch

from app.quantus_metrics import (
    evaluate_quantus_metrics,
    normalised_score_drop_auc,
    perturb_spatial_pixels_all_channels,
    random_order_region_control,
    replace_region_with_channel_means,
)


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(1, 2, 3))
        return torch.stack((mean, -mean, mean * 0.5, mean * 0.25, mean * 0.1), dim=1)


class QuantusMetricsTests(unittest.TestCase):
    def test_spatial_faithfulness_perturbation_replaces_all_rgb_channels(self) -> None:
        """A selected spatial feature must never perturb just one RGB channel."""
        image = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
        baseline = np.array([[[[100.0]], [[200.0]], [[300.0]]]], dtype=np.float32)

        # Flattened spatial positions 1 and 2 are (0, 1) and (1, 0).
        perturbed = perturb_spatial_pixels_all_channels(
            image,
            spatial_indices=np.array([1, 2]),
            baseline=baseline,
        )

        expected_rgb = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        self.assertTrue(np.array_equal(perturbed[0, :, 0, 1], expected_rgb))
        self.assertTrue(np.array_equal(perturbed[0, :, 1, 0], expected_rgb))
        # Unselected pixels remain exactly unchanged.
        self.assertTrue(np.array_equal(perturbed[0, :, 0, 0], image[0, :, 0, 0]))
        self.assertTrue(np.array_equal(perturbed[0, :, 1, 1], image[0, :, 1, 1]))

    def test_region_perturbation_replaces_same_patch_in_all_channels(self) -> None:
        image = np.arange(12, dtype=np.float32).reshape(1, -1)
        # Spatial position 1 in each 2x2 channel plane.
        indices = np.array([[1, 5, 9]])
        perturbed = replace_region_with_channel_means(
            image,
            indices,
            channel_means=np.array([[100.0, 200.0, 300.0]], dtype=np.float32),
            spatial_plane_size=4,
        )

        self.assertEqual(perturbed[0, 1], 100.0)
        self.assertEqual(perturbed[0, 5], 200.0)
        self.assertEqual(perturbed[0, 9], 300.0)
        self.assertTrue(np.array_equal(perturbed[0, [0, 2, 4, 6, 8, 10]], image[0, [0, 2, 4, 6, 8, 10]]))

    def test_region_perturbation_auc_has_zero_origin_and_normalised_axis(self) -> None:
        self.assertAlmostEqual(normalised_score_drop_auc(np.array([1.0, 3.0])), 1.25)

    @patch("app.quantus_metrics.region_perturbation_score_drop_auc")
    @patch("app.quantus_metrics._load_quantus", return_value=object())
    def test_random_order_control_is_fixed_seed_reproducible(self, _load, mock_region) -> None:
        mock_region.side_effect = lambda **_: float(np.random.random())
        inputs = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

        first = random_order_region_control(TinyClassifier(), inputs, target_class=0, repeats=3, seed=42)
        second = random_order_region_control(TinyClassifier(), inputs, target_class=0, repeats=3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first[2]), 3)

    @patch("app.quantus_metrics.generate_xai_attributions")
    def test_initial_quantus_metric_set_returns_metadata_and_scores(self, mock_attributions) -> None:
        attribution = np.arange(16, dtype=np.float32).reshape(4, 4)
        mock_attributions.return_value = attribution[np.newaxis, np.newaxis, :, :]
        model = TinyClassifier()
        input_tensor = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 47

        try:
            scores = evaluate_quantus_metrics(
                model=model,
                input_tensor=input_tensor,
                attribution=attribution,
                target_class=0,
                xai_method="Grad-CAM",
                model_name="resnet50",
                faithfulness_runs=2,
                faithfulness_subset_size=4,
                max_sensitivity_samples=2,
            )
        except RuntimeError as exc:
            if "Windows blocked" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(
            [score.metric_name for score in scores],
            ["FaithfulnessCorrelation", "Region perturbation score-drop AUC", "MaxSensitivity", "Sparseness"],
        )
        self.assertTrue(all(np.isfinite(score.value) for score in scores))
        self.assertGreaterEqual(scores[0].value, -1.0)
        self.assertLessEqual(scores[0].value, 1.0)
        self.assertEqual(scores[0].score_direction, "higher")
        self.assertEqual(scores[0].parameters["attribution"], "normalise_by_max")
        self.assertEqual(scores[0].parameters["feature_unit"], "spatial_pixel_rgb_group")
        self.assertEqual(scores[0].parameters["subset_size"], 4)
        self.assertAlmostEqual(scores[0].parameters["subset_fraction"], 0.25)
        self.assertEqual(
            scores[0].parameters["perturbation"],
            "replace_selected_spatial_pixels_in_all_channels",
        )
        self.assertEqual(mock_attributions.call_count, 2)
        region = scores[1]
        self.assertEqual(region.parameters["patch_size"], 16)
        self.assertEqual(region.parameters["order"], "morf")
        self.assertFalse(region.parameters["softmax"])

    @patch("app.quantus_metrics._load_quantus")
    def test_zero_perturbed_map_marks_only_max_sensitivity_undefined(self, mock_quantus) -> None:
        class ReturningMetric:
            def __init__(self, value, **_):
                self.value = value

            def __call__(self, **_):
                return [self.value]

        class ZeroMapMaxSensitivity:
            def __init__(self, **_):
                pass

            def __call__(self, **_):
                raise AssertionError("The elements in the attribution vector are all equal to zero")

        class FakeQuantus:
            FaithfulnessCorrelation = lambda *_, **__: ReturningMetric(0.2)
            MaxSensitivity = ZeroMapMaxSensitivity
            Sparseness = lambda *_, **__: ReturningMetric(0.4)

        mock_quantus.return_value = FakeQuantus
        scores = evaluate_quantus_metrics(
            model=TinyClassifier(),
            input_tensor=torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 47,
            attribution=np.arange(16, dtype=np.float32).reshape(4, 4),
            target_class=0,
            xai_method="Grad-CAM",
            model_name="resnet50",
            faithfulness_runs=2,
            faithfulness_subset_size=4,
            max_sensitivity_samples=2,
            requested_metrics={"FaithfulnessCorrelation", "MaxSensitivity", "Sparseness"},
        )

        self.assertEqual([score.status for score in scores], ["ok", "undefined_zero_perturbed_attribution", "ok"])
        self.assertTrue(np.isnan(scores[1].value))
        self.assertIn("all-zero attribution", scores[1].note)

    @patch("app.quantus_metrics._load_quantus")
    def test_sensitivity_parameters_are_explicit_and_recorded(self, mock_quantus) -> None:
        constructor_kwargs = {}

        class CapturingMaxSensitivity:
            def __init__(self, **kwargs):
                constructor_kwargs.update(kwargs)

            def __call__(self, **_):
                return [0.25]

        class FakeQuantus:
            MaxSensitivity = CapturingMaxSensitivity
            normalise_by_max = staticmethod(lambda values: values / np.max(np.abs(values)))

        mock_quantus.return_value = FakeQuantus
        scores = evaluate_quantus_metrics(
            model=TinyClassifier(),
            input_tensor=torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 47,
            attribution=np.arange(16, dtype=np.float32).reshape(4, 4),
            target_class=0,
            xai_method="Grad-CAM++",
            model_name="resnet50",
            faithfulness_subset_size=8,
            max_sensitivity_samples=3,
            max_sensitivity_normalise=True,
            requested_metrics={"MaxSensitivity"},
        )

        self.assertTrue(constructor_kwargs["normalise"])
        self.assertIs(constructor_kwargs["normalise_func"], FakeQuantus.normalise_by_max)
        self.assertEqual(constructor_kwargs["nr_samples"], 3)
        self.assertEqual(scores[0].parameters["normalise"], True)
        self.assertEqual(scores[0].parameters["attribution"], "quantus_normalise_by_max_separately")

    @patch("app.quantus_metrics._load_quantus", return_value=object())
    def test_faithfulness_subset_size_must_fit_spatial_map(self, _load) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than"):
            evaluate_quantus_metrics(
                model=TinyClassifier(),
                input_tensor=torch.zeros((1, 3, 4, 4), dtype=torch.float32),
                attribution=np.ones((4, 4), dtype=np.float32),
                target_class=0,
                xai_method="Grad-CAM",
                model_name="resnet50",
                faithfulness_subset_size=16,
                requested_metrics={"FaithfulnessCorrelation"},
            )


if __name__ == "__main__":
    unittest.main()
