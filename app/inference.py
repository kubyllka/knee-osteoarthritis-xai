from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from app.config import KL_LABELS, ONNX_MODEL_PATH
from app.preprocessing import load_image, preprocess_image, softmax


@dataclass
class Prediction:
    grade: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class ONNXKneeOsteoarthritisClassifier:
    def __init__(self, model_path: Path = ONNX_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. Run scripts/export_to_onnx.py first."
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()

    def predict(self, image_bytes: bytes) -> Prediction:
        image = load_image(image_bytes)
        input_array = preprocess_image(image)

        start = perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        latency_ms = (perf_counter() - start) * 1000

        logits = outputs[0]
        probabilities = softmax(logits)[0]
        grade = int(np.argmax(probabilities))

        return Prediction(
            grade=grade,
            label=KL_LABELS[grade],
            confidence=float(probabilities[grade]),
            probabilities={
                f"KL Grade {idx}": float(probability)
                for idx, probability in enumerate(probabilities)
            },
            latency_ms=latency_ms,
        )
