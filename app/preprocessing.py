from io import BytesIO

import numpy as np
from PIL import Image

from app.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    return image.convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image).astype(np.float32) / 255.0

    mean = np.asarray(NORMALIZE_MEAN, dtype=np.float32)
    std = np.asarray(NORMALIZE_STD, dtype=np.float32)
    array = (array - mean) / std

    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)
