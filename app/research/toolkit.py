"""Explicit, data-free source bundle for reproducing research analyses."""

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


TOOLKIT_FILES = (
    "README.md",
    "requirements.txt",
    "scripts/run_research.py",
    "scripts/analyze_research_results.py",
    "scripts/export_to_onnx.py",
    "notebooks/kaggle_research_runner.ipynb",
    "app/__init__.py",
    "app/config.py",
    "app/evaluation.py",
    "app/inference.py",
    "app/metric_registry.py",
    "app/model_definition.py",
    "app/ordinal.py",
    "app/preprocessing.py",
    "app/quantus_metrics.py",
    "app/xai.py",
    "app/research/__init__.py",
    "app/research/analysis.py",
    "app/research/experiment.py",
    "app/research/manifest.py",
    "app/research/results.py",
    "app/research/run_metadata.py",
)


def build_research_toolkit_zip(project_root: Path) -> bytes:
    """Package only reviewed source files; never walk models, inputs, or results."""
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for relative_path in TOOLKIT_FILES:
            source = project_root / ("research_toolkit/README.md" if relative_path == "README.md" else relative_path)
            if not source.is_file():
                raise FileNotFoundError(f"Research toolkit source is missing: {source}")
            bundle.write(source, relative_path)
    return archive.getvalue()
