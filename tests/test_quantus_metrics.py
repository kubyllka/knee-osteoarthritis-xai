import unittest
from unittest.mock import patch

import numpy as np
import torch

from app.quantus_metrics import evaluate_quantus_metrics


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(1, 2, 3))
        return torch.stack((mean, -mean, mean * 0.5, mean * 0.25, mean * 0.1), dim=1)


class QuantusMetricsTests(unittest.TestCase):
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
                max_sensitivity_samples=2,
            )
        except RuntimeError as exc:
            if "Windows blocked" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual([score.metric_name for score in scores], ["FaithfulnessCorrelation", "MaxSensitivity", "Sparseness"])
        self.assertTrue(all(np.isfinite(score.value) for score in scores))
        self.assertGreaterEqual(scores[0].value, -1.0)
        self.assertLessEqual(scores[0].value, 1.0)
        self.assertEqual(scores[0].score_direction, "higher")
        self.assertEqual(scores[0].parameters["attribution"], "normalise_by_max")
        self.assertEqual(mock_attributions.call_count, 2)


if __name__ == "__main__":
    unittest.main()
