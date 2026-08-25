import unittest

import torch

from app.config import MODEL_CONFIGS
from app.model_definition import create_densenet121_model
from app.xai import create_xai_model, get_target_layer_and_transform


class DenseNetConfigurationTests(unittest.TestCase):
    def test_densenet_is_a_selectable_model(self) -> None:
        config = MODEL_CONFIGS["densenet121"]
        self.assertEqual(config["display_name"], "DenseNet-121")
        self.assertEqual(config["onnx_path"].name, "densenet121_kl.onnx")
        self.assertEqual(
            config["checkpoint_path"].name,
            "densenet121_finetune_block3_4_best.pth",
        )

    def test_densenet_classifier_outputs_five_kl_classes(self) -> None:
        model = create_densenet121_model(num_classes=5)
        self.assertEqual(model.classifier[1].out_features, 512)
        self.assertEqual(model.classifier[4].out_features, 5)

    def test_densenet_has_a_valid_xai_target_layer(self) -> None:
        model = create_xai_model("densenet121")
        target_layer, _ = get_target_layer_and_transform(model, "densenet121")
        self.assertIs(target_layer, model.features.denseblock4)


if __name__ == "__main__":
    unittest.main()
