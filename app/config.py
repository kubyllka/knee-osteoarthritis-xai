from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

ONNX_MODEL_PATH = MODEL_DIR / "vit_kl.onnx"
PYTORCH_CHECKPOINT_PATH = MODEL_DIR / "vit_phase2_best.pth"
RESNET_ONNX_MODEL_PATH = MODEL_DIR / "resnet50_kl.onnx"
RESNET_PYTORCH_CHECKPOINT_PATH = MODEL_DIR / "resnet50_phase2_best.pth"
DENSENET_ONNX_MODEL_PATH = MODEL_DIR / "densenet121_kl.onnx"
DENSENET_PYTORCH_CHECKPOINT_PATH = MODEL_DIR / "densenet121_finetune_block3_4_best.pth"

MODEL_CONFIGS = {
    "vit": {
        "display_name": "ViT-B/16",
        "onnx_path": ONNX_MODEL_PATH,
        "checkpoint_path": PYTORCH_CHECKPOINT_PATH,
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "onnx_path": RESNET_ONNX_MODEL_PATH,
        "checkpoint_path": RESNET_PYTORCH_CHECKPOINT_PATH,
    },
    "densenet121": {
        "display_name": "DenseNet-121",
        "onnx_path": DENSENET_ONNX_MODEL_PATH,
        "checkpoint_path": DENSENET_PYTORCH_CHECKPOINT_PATH,
    },
}

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
