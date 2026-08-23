import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetManifestValidationError(ValueError):
    """Raised when a dataset manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class DatasetSample:
    """One labelled X-ray sample available to an offline research experiment."""

    sample_id: str
    image_path: Path
    true_kl: int
    split: str


def parse_dataset_manifest(
    payload: Any,
    project_root: Path = PROJECT_ROOT,
) -> list[DatasetSample]:
    """Validate manifest content and resolve sample paths relative to the project root."""
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise DatasetManifestValidationError("Manifest must be a JSON object with a 'samples' list.")
    if not payload["samples"]:
        raise DatasetManifestValidationError("Manifest must contain at least one sample.")

    samples = []
    sample_ids = set()
    for index, item in enumerate(payload["samples"]):
        if not isinstance(item, dict):
            raise DatasetManifestValidationError(f"Sample at index {index} must be a JSON object.")

        sample_id = str(item.get("sample_id", "")).strip()
        image_path_value = str(item.get("image_path", "")).strip()
        split = str(item.get("split", "unspecified")).strip() or "unspecified"
        true_kl = item.get("true_kl")

        if not sample_id:
            raise DatasetManifestValidationError(f"Sample at index {index} has an empty sample_id.")
        if sample_id in sample_ids:
            raise DatasetManifestValidationError(f"Duplicate sample_id: {sample_id}")
        if not image_path_value:
            raise DatasetManifestValidationError(f"Sample '{sample_id}' has an empty image_path.")
        if isinstance(true_kl, bool) or not isinstance(true_kl, int) or not 0 <= true_kl <= 4:
            raise DatasetManifestValidationError(
                f"Sample '{sample_id}' must have an integer true_kl from 0 to 4."
            )

        image_path = Path(image_path_value)
        resolved_path = image_path if image_path.is_absolute() else project_root / image_path
        if not resolved_path.is_file():
            raise DatasetManifestValidationError(
                f"Image for sample '{sample_id}' was not found: {resolved_path}"
            )

        sample_ids.add(sample_id)
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                image_path=resolved_path,
                true_kl=true_kl,
                split=split,
            )
        )

    return samples


def load_dataset_manifest(path: str | Path) -> list[DatasetSample]:
    """Load and validate a JSON dataset manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetManifestValidationError(
            f"Invalid JSON in dataset manifest: {manifest_path}"
        ) from exc

    return parse_dataset_manifest(payload)
