import unittest

import torch

from app.config import MODEL_CONFIGS
from app.model_definition import create_densenet121_ordinal_model
from app.ordinal import ordinal_class_log_scores, ordinal_logits_to_class
from app.xai import create_xai_model, get_target_layer_and_transform


class DenseNetConfigurationTests(unittest.TestCase):
    def test_densenet_is_a_selectable_model(self) -> None:
        config = MODEL_CONFIGS["densenet121"]
        self.assertEqual(config["display_name"], "DenseNet-121 (ordinal BCE)")
        self.assertEqual(config["onnx_path"].name, "densenet121_ordinal_kl.onnx")
        self.assertEqual(
            config["checkpoint_path"].name,
            "densenet121_ordinal_finetune_block3_4_best_21092026.pth",
        )
        self.assertEqual(config["output_type"], "ordinal_thresholds")

    def test_densenet_ordinal_classifier_outputs_four_thresholds(self) -> None:
        model = create_densenet121_ordinal_model(num_classes=5)
        self.assertEqual(model.classifier[1].out_features, 512)
        self.assertEqual(model.classifier[4].out_features, 4)

    def test_ordinal_decode_counts_positive_thresholds(self) -> None:
        logits = torch.tensor([[-2.0, 3.0, 1.0, -1.0]])
        self.assertEqual(int(ordinal_logits_to_class(logits).item()), 2)

    def test_ordinal_class_scores_produce_five_differentiable_targets(self) -> None:
        logits = torch.tensor([[0.2, -0.3, 0.4, -0.5]], requires_grad=True)
        scores = ordinal_class_log_scores(logits)
        self.assertEqual(tuple(scores.shape), (1, 5))
        scores[0, 3].backward()
        self.assertIsNotNone(logits.grad)

    def test_densenet_has_a_valid_xai_target_layer(self) -> None:
        model = create_xai_model("densenet121")
        target_layer, _ = get_target_layer_and_transform(model, "densenet121")
        self.assertIs(target_layer, model.features.denseblock4)


if __name__ == "__main__":
    unittest.main()
