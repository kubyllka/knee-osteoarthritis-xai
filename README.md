# Knee Osteoarthritis XAI

An educational and research project for classifying knee osteoarthritis from X-ray images and visualizing model predictions.

ResNet-50 and DenseNet-121 predict the Kellgren-Lawrence grade (KL Grade 0-4) using four ordinal threshold outputs. The application supports separate Ordinal BCE and Class-aware Ordinal BCE checkpoints for each architecture. The project includes ONNX Runtime inference, a FastAPI service, and a Streamlit interface.

The project implements Grad-CAM, Grad-CAM++, Eigen-CAM, Integrated Gradients, and SmoothGrad to interpret predictions. The interface also supports comparing explanations across methods and evaluating their stability.

Research Mode reads one or more precomputed experiment CSV files. It reports classifier-subset performance and metric-specific statistical comparisons; its exploration view presents one XAI metric at a time, so metrics with different numerical scales are never averaged together. Prediction Mode can optionally run a small on-demand XAI evaluation for the selected methods; use the offline runner for reproducible dataset-level experiments.

## Features

- KL grade prediction with per-grade scores and a reported confidence value;
- interactive XAI heatmaps and overlays for five explanation methods;
- deletion and stability checks for a single prediction;
- optional exploratory XAI evaluation: custom spatial RGB-grouped FaithfulnessCorrelation, Quantus-based Region Perturbation score-drop AUC, MaxSensitivity, and Sparseness;
- Research Mode for filtering and comparing precomputed experiment results by model, XAI method, true KL grade, and prediction correctness.

## Application modes

**Prediction Mode** is intended for analysing one uploaded X-ray. It generates a model prediction and selected XAI maps. Quantitative XAI evaluation is optional and uses a small exploratory profile, so it does not run automatically on every upload.

**Research Mode** is intended for already computed results. Select compatible CSV files to combine models without manual merging. Results analysis derives classifier-subset statistics and performs metric-specific KL0–KL4 Kruskal–Wallis, Holm, epsilon-squared, and conditional Dunn post-hoc analysis. The exploration view remains organised by Faithfulness, Robustness, Complexity, and Custom checks. A CSV row represents one `sample x XAI method x metric` value; details of the results format are in [`results/research/README.md`](results/research/README.md).

## Project structure

- `app/` - API, inference, preprocessing, and XAI modules;
- `models/` - local PyTorch checkpoints and ONNX models (not included in Git);
- `inputs/` - sample X-ray images;
- `scripts/` - model export, Grad-CAM, and benchmark utilities;
- `streamlit_app.py` - web interface for image analysis.

## Getting started

Python 3.12 is recommended. Model weights are **not included** in this repository; the application can open Research Mode with compatible CSV files, but Prediction Mode needs model files before it can make a prediction. The expected filenames for each model ID are defined in [`app/config.py`](app/config.py). There is currently no public model-download URL.

Create an environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For Prediction Mode, place a trusted `.pth` checkpoint and its matching `.onnx` file under `models/` using the names in `app/config.py`. For example, with a supplied class-aware ResNet-50 checkpoint, export its ONNX counterpart:

```powershell
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --model-id resnet50_class_aware
```

Then launch the interface and **select the model for which you supplied files** in the sidebar:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

To run the optional FastAPI service, use `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload`. On macOS/Linux, use `python3 -m venv .venv`, `.venv/bin/python` and `.venv/bin/streamlit` instead. Research Mode can inspect the bundled example CSVs without model weights; these example and older experiment results are not substitutes for a new final study.

## Research experiments

The offline runner, model-file names, CUDA setup, manifest requirements, resume procedure, Kaggle notebook, and exploratory parameter checks are documented in [`docs/research_runs.md`](docs/research_runs.md). The CSV format is described in [`results/research/README.md`](results/research/README.md). Current `inputs/test/test_manifest.json` contains **250**, not 150, images; read the manifest warning in the research guide before attempting to reproduce an original-150-preserving final run.

## Notes on XAI evaluation

New runs use a custom spatial RGB-grouped implementation of the Faithfulness Correlation principle (`metric_source=custom`), a Quantus-based Region Perturbation score-drop AUC, Quantus MaxSensitivity and Sparseness, and a custom Stability score. Eigen-CAM maps are class-agnostic even though their evaluation measures the output for the predicted KL class. Compare values only within a compatible image cohort and metric configuration. Old Infidelity and custom Deletion confidence drop rows are legacy data; no individual metric establishes clinical validity or overall explanation quality.

> This project is intended for educational and research purposes only. It is not a clinical diagnostic tool.
