"""Freeze the original 150 images plus exactly 20 new test images per KL grade."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.research.manifest import (
    DatasetManifestValidationError,
    load_dataset_manifest,
    parse_dataset_manifest,
    validate_final_test_manifest,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_final_manifest(base_manifest: Path, test_root: Path, project_root: Path = PROJECT_ROOT) -> dict:
    project_root = project_root.resolve()
    base_manifest = base_manifest.resolve()
    test_root = test_root.resolve()
    originals = load_dataset_manifest(base_manifest)
    original_paths = {sample.image_path.resolve() for sample in originals}
    rows = [
        {
            "sample_id": sample.sample_id,
            "image_path": sample.image_path.relative_to(project_root).as_posix(),
            "true_kl": sample.true_kl,
            "split": sample.split,
        }
        for sample in originals
    ]
    for grade in range(5):
        class_directory = test_root / str(grade)
        if not class_directory.is_dir():
            raise DatasetManifestValidationError(f"Missing KL{grade} test directory: {class_directory}")
        candidates = sorted(
            path for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.resolve() not in original_paths
        )
        if len(candidates) != 20:
            raise DatasetManifestValidationError(
                f"KL{grade} needs exactly 20 additional test images; found {len(candidates)} in {class_directory}."
            )
        for path in candidates:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({
                "sample_id": f"kl-{grade}-extra-{digest[:16]}",
                "image_path": path.relative_to(project_root).as_posix(),
                "true_kl": grade,
                "split": "test",
            })
    payload = {
        "description": "Frozen 250-image KL-balanced test cohort; original 150 sample IDs are unchanged.",
        "selection_rule": "All original samples plus exactly 20 new files per KL folder, sorted by path.",
        "samples": rows,
    }
    final_samples = parse_dataset_manifest(payload, project_root=project_root)
    validate_final_test_manifest(final_samples, originals)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=PROJECT_ROOT / "inputs/test/test_manifest.json")
    parser.add_argument("--test-root", type=Path, default=PROJECT_ROOT / "inputs/test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "inputs/test/final_manifest_250.json")
    args = parser.parse_args()
    payload = build_final_manifest(args.base_manifest, args.test_root)
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}. Preserve frozen manifests; choose another path.")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(payload['samples'])} test images; saved {output}")


if __name__ == "__main__":
    main()
