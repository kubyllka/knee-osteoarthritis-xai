from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

ONNX_MODEL_PATH = MODEL_DIR / "vit_kl.onnx"
PYTORCH_CHECKPOINT_PATH = MODEL_DIR / "vit_phase2_best.pth"
RESNET_ONNX_MODEL_PATH = MODEL_DIR / "resnet50_ordinal_kl.onnx"
RESNET_PYTORCH_CHECKPOINT_PATH = MODEL_DIR / "resnet50_ordinal_finetune_layer3_4_best_21092026.pth"
DENSENET_ONNX_MODEL_PATH = MODEL_DIR / "densenet121_ordinal_kl.onnx"
DENSENET_PYTORCH_CHECKPOINT_PATH = MODEL_DIR / "densenet121_ordinal_finetune_block3_4_best_21092026.pth"

MODEL_CONFIGS = {
    "vit": {
        "display_name": "ViT-B/16",
        "onnx_path": ONNX_MODEL_PATH,
        "checkpoint_path": PYTORCH_CHECKPOINT_PATH,
    },
    "resnet50": {
        "display_name": "ResNet-50 (ordinal BCE)",
        "architecture": "resnet50",
        "training_variant": "ordinal_bce",
        "kl1_loss_multiplier": 1.0,
        "onnx_path": RESNET_ONNX_MODEL_PATH,
        "checkpoint_path": RESNET_PYTORCH_CHECKPOINT_PATH,
        "output_type": "ordinal_thresholds",
        "num_outputs": 4,
    },
    "densenet121": {
        "display_name": "DenseNet-121 (ordinal BCE)",
        "architecture": "densenet121",
        "training_variant": "ordinal_bce",
        "kl1_loss_multiplier": 1.0,
        "onnx_path": DENSENET_ONNX_MODEL_PATH,
        "checkpoint_path": DENSENET_PYTORCH_CHECKPOINT_PATH,
        "output_type": "ordinal_thresholds",
        "num_outputs": 4,
    },
    "resnet50_class_aware": {
        "display_name": "ResNet-50 (class-aware ordinal BCE)",
        "architecture": "resnet50",
        "training_variant": "class_aware_ordinal_bce",
        "kl1_loss_multiplier": 1.5,
        "onnx_path": MODEL_DIR / "resnet50_class_aware_ordinal.onnx",
        "checkpoint_path": MODEL_DIR / "resnet50_class_aware_ordinal.pth",
        "output_type": "ordinal_thresholds",
        "num_outputs": 4,
    },
    "densenet121_class_aware": {
        "display_name": "DenseNet-121 (class-aware ordinal BCE)",
        "architecture": "densenet121",
        "training_variant": "class_aware_ordinal_bce",
        "kl1_loss_multiplier": 1.5,
        "onnx_path": MODEL_DIR / "densenet121_class_aware_ordinal.onnx",
        "checkpoint_path": MODEL_DIR / "densenet121_class_aware_ordinal.pth",
        "output_type": "ordinal_thresholds",
        "num_outputs": 4,
    },
}

RESEARCH_MODEL_IDS = (
    "resnet50",
    "resnet50_class_aware",
    "densenet121",
    "densenet121_class_aware",
)

NUM_CLASSES = 5
IMAGE_SIZE = 224

KL_LABELS = {
    0: "KL Grade 0 - no radiographic osteoarthritis",
    1: "KL Grade 1 - doubtful joint space narrowing / possible osteophytes",
    2: "KL Grade 2 - mild osteoarthritis",
    3: "KL Grade 3 - moderate osteoarthritis",
    4: "KL Grade 4 - severe osteoarthritis",
}

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
