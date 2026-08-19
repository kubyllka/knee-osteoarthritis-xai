import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import IMAGE_SIZE, KL_LABELS, NUM_CLASSES, PYTORCH_CHECKPOINT_PATH
from app.model_definition import create_vit_model, vit_reshape_transform


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    cleaned_checkpoint = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(cleaned_checkpoint, strict=True)
    return model


def preprocess_for_torch(image_path: Path) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(array).unsqueeze(0)
    return resized, tensor


class ViTGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        output[:, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        activations = vit_reshape_transform(self.activations)
        gradients = vit_reshape_transform(self.gradients)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx, probabilities.detach().cpu().numpy()[0]


def save_gradcam_images(original: Image.Image, cam: np.ndarray, output_dir: Path, alpha: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    original_rgb = np.asarray(original)
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap, alpha, 0)

    Image.fromarray(original_rgb).save(output_dir / "original_resized.jpg")
    Image.fromarray(heatmap).save(output_dir / "gradcam_heatmap.jpg")
    Image.fromarray(overlay).save(output_dir / "gradcam_overlay.jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for ViT KL Grade classifier.")
    parser.add_argument("--checkpoint", type=Path, default=PYTORCH_CHECKPOINT_PATH)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"))
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--class-idx", type=int, choices=range(NUM_CLASSES), default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = create_vit_model(num_classes=args.num_classes, freeze_backbone=False)
    model = load_checkpoint(model, args.checkpoint)
    model.eval()

    target_layer = model.encoder.layers[-1].ln_1
    original, input_tensor = preprocess_for_torch(args.image)

    gradcam = ViTGradCAM(model, target_layer)
    try:
        cam, predicted_class, probabilities = gradcam.generate(input_tensor, args.class_idx)
    finally:
        gradcam.remove_hooks()

    save_gradcam_images(original, cam, args.output_dir, args.alpha)

    print(f"Predicted class: {predicted_class}")
    print(f"Label: {KL_LABELS[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class]:.4f}")
    print(f"Saved original image: {args.output_dir / 'original_resized.jpg'}")
    print(f"Saved Grad-CAM heatmap: {args.output_dir / 'gradcam_heatmap.jpg'}")
    print(f"Saved Grad-CAM overlay: {args.output_dir / 'gradcam_overlay.jpg'}")


if __name__ == "__main__":
    main()
