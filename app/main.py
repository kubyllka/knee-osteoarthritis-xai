from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import MODEL_CONFIGS
from app.inference import ONNXKneeOsteoarthritisClassifier
from app.schemas import HealthResponse, ImageUploadRequest, PredictionResponse

classifiers: dict[str, ONNXKneeOsteoarthritisClassifier] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifiers
    classifiers = {
        model_name: ONNXKneeOsteoarthritisClassifier(
            config["onnx_path"], output_type=config.get("output_type", "multiclass")
        )
        for model_name, config in MODEL_CONFIGS.items()
        if config["onnx_path"].exists()
    }
    yield


app = FastAPI(
    title="Knee Osteoarthritis KL Grade Classifier",
    description="ONNX Runtime inference API for X-ray knee osteoarthritis KL Grade 0-4 classification.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=bool(classifiers),
        runtime_providers=next(iter(classifiers.values())).providers if classifiers else [],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_name: str = Form("resnet50_class_aware")):
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=422, detail=f"Unsupported model_name: {model_name}")

    classifier = classifiers.get(model_name)
    if classifier is None:
        expected_path = MODEL_CONFIGS[model_name]["onnx_path"]
        raise HTTPException(status_code=503, detail=f"Model is not loaded: {expected_path}")

    image_bytes = await file.read()
    try:
        ImageUploadRequest(
            filename=file.filename or "",
            content_type=file.content_type or "",
            size_bytes=len(image_bytes),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        prediction = classifier.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    return PredictionResponse(
        grade=prediction.grade,
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        latency_ms=prediction.latency_ms,
    )
