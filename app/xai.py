from dataclasses import dataclass
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import (
    IMAGE_SIZE,
    MODEL_CONFIGS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    PYTORCH_CHECKPOINT_PATH,
)
from app.model_definition import (
    create_densenet121_model,
    create_resnet50_model,
    create_vit_model,
    identity_reshape_transform,
    vit_reshape_transform,
)


@dataclass
class XAIResult:
    original: Image.Image
    heatmap: Image.Image
    overlay: Image.Image
    cam: np.ndarray
    raw_attribution: np.ndarray
    method: str
    class_idx: int


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    """Load a local, trusted training checkpoint into an XAI model.

    Newer PyTorch releases default to ``weights_only=True``. Older training
    checkpoints may contain harmless NumPy scalar metadata alongside weights,
    which requires the trusted-file fallback below.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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


def create_xai_model(model_name: str) -> torch.nn.Module:
    if model_name == "resnet50":
        return create_resnet50_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    if model_name == "densenet121":
        return create_densenet121_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    if model_name == "vit":
        return create_vit_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    raise ValueError(f"Unsupported model name: {model_name}")


def get_target_layer_and_transform(model: torch.nn.Module, model_name: str):
    if model_name == "resnet50":
        return model.layer4[-1], identity_reshape_transform
    if model_name == "densenet121":
        # DenseNet applies an in-place ReLU immediately after norm5. A full
        # backward hook on norm5 conflicts with that operation in PyTorch, so
        # use the preceding final dense block as the Grad-CAM target instead.
        return model.features.denseblock4, identity_reshape_transform
    if model_name == "vit":
        return model.encoder.layers[-1].ln_1, vit_reshape_transform
    raise ValueError(f"Unsupported model name: {model_name}")


def load_xai_model(
    checkpoint_path: Path = PYTORCH_CHECKPOINT_PATH,
    model_name: str = "vit",
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")

    model = create_xai_model(model_name)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    return model


def load_xai_model_by_name(model_name: str) -> torch.nn.Module:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model name: {model_name}")
    checkpoint_path = MODEL_CONFIGS[model_name]["checkpoint_path"]
    return load_xai_model(checkpoint_path=checkpoint_path, model_name=model_name)


def preprocess_pil_for_torch(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.asarray(NORMALIZE_MEAN, dtype=np.float32)
    std = np.asarray(NORMALIZE_STD, dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(array).unsqueeze(0)
    return resized, tensor


class ActivationXAI:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, reshape_transform):
        self.model = model.eval()
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
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

    def generate(
        self,
        input_tensor: torch.Tensor,
        method: str = "Grad-CAM",
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        method_key = method.lower().replace("+", "plus").replace("-", "").replace(" ", "")

        if method_key == "integratedgradients":
            return self._integrated_gradients(input_tensor, class_idx=class_idx), self._resolve_class_idx(input_tensor, class_idx)

        if method_key == "smoothgrad":
            return self._smooth_grad(input_tensor, class_idx=class_idx), self._resolve_class_idx(input_tensor, class_idx)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        if method_key == "eigencam":
            if self.activations is None:
                raise RuntimeError("Eigen-CAM hooks did not capture activations.")
            return self._eigen_cam(self.activations), class_idx

        self.model.zero_grad(set_to_none=True)
        output[:, class_idx].backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("XAI hooks did not capture activations or gradients.")

        if method_key == "gradcamplusplus":
            cam = self._grad_cam_plus_plus(self.activations, self.gradients)
        else:
            cam = self._grad_cam(self.activations, self.gradients)

        return cam, class_idx

    def _resolve_class_idx(self, input_tensor: torch.Tensor, class_idx: int | None) -> int:
        if class_idx is not None:
            return class_idx

        with torch.no_grad():
            output = self.model(input_tensor)
        return int(output.argmax(dim=1).item())

    def _integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        steps: int = 16,
    ) -> np.ndarray:
        target_class = self._resolve_class_idx(input_tensor, class_idx)
        baseline = torch.zeros_like(input_tensor)
        total_gradients = torch.zeros_like(input_tensor)

        # Integrated Gradients accumulates input gradients along a path from
        # a neutral baseline to the real image, reducing single-gradient noise.
        for alpha in torch.linspace(0.0, 1.0, steps + 1, device=input_tensor.device)[1:]:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            output = self.model(interpolated)
            self.model.zero_grad(set_to_none=True)
            output[:, target_class].sum().backward()
            total_gradients += interpolated.grad.detach()

        average_gradients = total_gradients / steps
        attribution = (input_tensor - baseline) * average_gradients
        saliency = attribution.abs().sum(dim=1).squeeze(0)
        return saliency.cpu().numpy()

    def _smooth_grad(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        samples: int = 12,
        noise_std: float = 0.10,
    ) -> np.ndarray:
        target_class = self._resolve_class_idx(input_tensor, class_idx)
        total_saliency = torch.zeros_like(input_tensor[:, :1])

        # SmoothGrad averages saliency over noisy copies of the same image,
        # which suppresses isolated noisy pixels and keeps stable evidence.
        for _ in range(samples):
            noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_std
            noisy_input = noisy_input.detach().requires_grad_(True)

            output = self.model(noisy_input)
            self.model.zero_grad(set_to_none=True)
            output[:, target_class].sum().backward()
            total_saliency += noisy_input.grad.detach().abs().mean(dim=1, keepdim=True)

        saliency = (total_saliency / samples).squeeze(0).squeeze(0)
        return saliency.cpu().numpy()

    def _grad_cam(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        activations = self.reshape_transform(activations)
        gradients = self.reshape_transform(gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        return torch.relu(cam).cpu().numpy()

    def _grad_cam_plus_plus(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        activations = self.reshape_transform(activations)
        gradients = self.reshape_transform(gradients)

        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        denominator = 2.0 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = grad_2 / (denominator + 1e-8)
        weights = (alpha * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        return torch.relu(cam).cpu().numpy()

    def _eigen_cam(self, activations: torch.Tensor) -> np.ndarray:
        activations = self.reshape_transform(activations).squeeze(0)
        feature_map = activations.cpu().numpy()
        channels, height, width = feature_map.shape
        flattened = feature_map.reshape(channels, height * width)
        flattened = flattened - flattened.mean(axis=1, keepdims=True)
        _, _, vh = np.linalg.svd(flattened, full_matrices=False)
        cam = vh[0].reshape(height, width)
        if abs(cam.min()) > abs(cam.max()):
            cam = -cam
        return np.maximum(cam, 0.0)


def resize_attribution(attribution: np.ndarray) -> np.ndarray:
    """Resize an attribution without changing its numerical scale."""
    return cv2.resize(attribution, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Create a 0-1 display map; never use this visual normalisation for Quantus."""
    resized = resize_attribution(cam)
    resized = resized - resized.min()
    return resized / (resized.max() + 1e-8)


def colorize_cam(original: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> tuple[Image.Image, Image.Image]:
    original_rgb = np.asarray(original.convert("RGB"))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap, alpha, 0)
    return Image.fromarray(heatmap), Image.fromarray(overlay)


def generate_xai_visualization(
    model: torch.nn.Module,
    image: Image.Image,
    method: str = "Grad-CAM",
    class_idx: int | None = None,
    alpha: float = 0.45,
    model_name: str = "vit",
) -> XAIResult:
    original, input_tensor = preprocess_pil_for_torch(image)
    target_layer, reshape_transform = get_target_layer_and_transform(model, model_name)
    explainer = ActivationXAI(model, target_layer, reshape_transform)
    try:
        raw_attribution, explained_class = explainer.generate(input_tensor, method=method, class_idx=class_idx)
    finally:
        explainer.remove_hooks()

    cam = normalize_cam(raw_attribution)
    heatmap, overlay = colorize_cam(original, cam, alpha=alpha)
    return XAIResult(
        original=original,
        heatmap=heatmap,
        overlay=overlay,
        cam=cam,
        raw_attribution=resize_attribution(raw_attribution),
        method=method,
        class_idx=explained_class,
    )


def generate_xai_attributions(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    method: str,
    model_name: str,
) -> np.ndarray:
    """Generate raw, scale-preserving attribution maps for evaluation libraries."""
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (batch, channels, height, width)")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same batch size")

    device = next(model.parameters()).device
    target_layer, reshape_transform = get_target_layer_and_transform(model, model_name)
    explainer = ActivationXAI(model, target_layer, reshape_transform)
    cams = []
    try:
        for input_array, target in zip(inputs, targets):
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device=device, dtype=torch.float32)
            raw_attribution, _ = explainer.generate(
                input_tensor,
                method=method,
                class_idx=int(target),
            )
            cams.append(resize_attribution(raw_attribution))
    finally:
        explainer.remove_hooks()

    return np.asarray(cams, dtype=np.float32)[:, np.newaxis, :, :]
