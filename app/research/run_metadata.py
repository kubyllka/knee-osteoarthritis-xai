"""Reproducible run identity and recoverable, atomic result writes."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_content_digest(samples) -> str:
    """Identify the selected images, labels, split, and file contents."""
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item.sample_id):
        item = [sample.sample_id, sample.true_kl, sample.split, sha256_file(sample.image_path)]
        digest.update(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = ("torch", "torchvision", "onnxruntime", "quantus", "numpy", "scipy")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".run.json")


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write(path, lambda file: frame.to_csv(file, index=False))


def write_json_atomic(payload: dict, path: Path) -> None:
    _atomic_write(path, lambda file: json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False))


def ensure_matching_metadata(output_path: Path, expected: dict, *, resume: bool) -> None:
    """Never resume an output with a different model, data, or evaluation protocol."""
    sidecar = metadata_path(output_path)
    if sidecar.exists():
        if not resume:
            raise FileExistsError(f"Run metadata already exists: {sidecar}. Use --resume or a new output path.")
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if existing != expected:
            differences = sorted(key for key in set(existing) | set(expected) if existing.get(key) != expected.get(key))
            raise ValueError("Cannot resume: run metadata differs in " + ", ".join(differences) + ".")
        return
    if output_path.exists():
        raise ValueError(f"Cannot safely resume {output_path}: matching run metadata is missing.")
    write_json_atomic(expected, sidecar)
