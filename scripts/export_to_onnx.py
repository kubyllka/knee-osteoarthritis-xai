import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    ONNX_MODEL_PATH,
    PYTORCH_CHECKPOINT_PATH,
    RESNET_ONNX_MODEL_PATH,
    RESNET_PYTORCH_CHECKPOINT_PATH,
)
from app.model_definition import create_resnet50_model, create_vit_model


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


def create_model(architecture: str, num_classes: int) -> torch.nn.Module:
    if architecture == "resnet50":
        return create_resnet50_model(num_classes=num_classes, freeze_backbone=True)
    return create_vit_model(num_classes=num_classes, freeze_backbone=True)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    num_classes: int,
    architecture: str,
) -> None:
    output_path = Path(str(output_path).strip())

    model = create_model(architecture=architecture, num_classes=num_classes)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Exported {architecture} ONNX model to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained KL Grade classifier to ONNX.")
    parser.add_argument("--architecture", choices=["vit", "resnet50"], default="vit")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = (
            RESNET_PYTORCH_CHECKPOINT_PATH
            if args.architecture == "resnet50"
            else PYTORCH_CHECKPOINT_PATH
        )
    if args.output is None:
        args.output = RESNET_ONNX_MODEL_PATH if args.architecture == "resnet50" else ONNX_MODEL_PATH

    return args


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(args.checkpoint, args.output, args.num_classes, args.architecture)
