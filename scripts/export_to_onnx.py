import argparse
import pickle
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    DENSENET_ONNX_MODEL_PATH,
    DENSENET_PYTORCH_CHECKPOINT_PATH,
    MODEL_CONFIGS,
    ONNX_MODEL_PATH,
    PYTORCH_CHECKPOINT_PATH,
    RESNET_ONNX_MODEL_PATH,
    RESNET_PYTORCH_CHECKPOINT_PATH,
)
from app.model_definition import (
    create_densenet121_model,
    create_densenet121_ordinal_model,
    create_resnet50_model,
    create_resnet50_ordinal_model,
    create_vit_model,
)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    """Load a local, trusted training checkpoint for export.

    ``weights_only=False`` is used only as a fallback for legacy checkpoints
    that embed NumPy scalar training metadata. Do not use untrusted files.
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


def create_model(architecture: str, num_classes: int, output_type: str = "multiclass") -> torch.nn.Module:
    if architecture == "resnet50":
        factory = create_resnet50_ordinal_model if output_type == "ordinal_thresholds" else create_resnet50_model
        return factory(num_classes=num_classes, freeze_backbone=True)
    if architecture == "densenet121":
        factory = create_densenet121_ordinal_model if output_type == "ordinal_thresholds" else create_densenet121_model
        return factory(num_classes=num_classes, freeze_backbone=True)
    if architecture == "vit":
        return create_vit_model(num_classes=num_classes, freeze_backbone=True)
    raise ValueError(f"Unsupported architecture: {architecture}")


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    num_classes: int,
    architecture: str,
    output_type: str = "multiclass",
) -> None:
    output_path = Path(str(output_path).strip())

    model = create_model(architecture=architecture, num_classes=num_classes, output_type=output_type)
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
    parser.add_argument("--model-id", choices=sorted(MODEL_CONFIGS), default=None,
                        help="Use the configured architecture, checkpoint, output path, and output type.")
    parser.add_argument("--architecture", choices=["vit", "resnet50", "densenet121"], default="vit")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = parser.parse_args()

    if args.model_id is not None:
        config = MODEL_CONFIGS[args.model_id]
        args.architecture = config.get("architecture", args.model_id)
        args.checkpoint = args.checkpoint or config["checkpoint_path"]
        args.output = args.output or config["onnx_path"]
        args.output_type = config.get("output_type", "multiclass")
        return args

    if args.checkpoint is None:
        args.checkpoint = {
            "vit": PYTORCH_CHECKPOINT_PATH,
            "resnet50": RESNET_PYTORCH_CHECKPOINT_PATH,
            "densenet121": DENSENET_PYTORCH_CHECKPOINT_PATH,
        }[args.architecture]
    if args.output is None:
        args.output = {
            "vit": ONNX_MODEL_PATH,
            "resnet50": RESNET_ONNX_MODEL_PATH,
            "densenet121": DENSENET_ONNX_MODEL_PATH,
        }[args.architecture]

    args.output_type = MODEL_CONFIGS.get(args.architecture, {}).get("output_type", "multiclass")

    return args


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(args.checkpoint, args.output, args.num_classes, args.architecture, args.output_type)
