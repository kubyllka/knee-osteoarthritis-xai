from pydantic import BaseModel, Field, field_validator


class ImageUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str
    size_bytes: int = Field(..., gt=0)

    @field_validator("content_type")
    @classmethod
    def validate_image_content_type(cls, value: str) -> str:
        if not value.startswith("image/"):
            raise ValueError("Uploaded file must be an image")
        return value


class PredictionResponse(BaseModel):
    grade: int = Field(..., ge=0, le=4, description="Predicted Kellgren-Lawrence grade")
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    runtime_providers: list[str]
