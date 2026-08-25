# Knee Osteoarthritis XAI

An educational and research project for classifying knee osteoarthritis from X-ray images and visualizing model predictions.

The model predicts the Kellgren-Lawrence grade (KL Grade 0-4). The project includes ResNet-50 architectures, ONNX Runtime inference, a FastAPI service, and a Streamlit interface.

The project implements Grad-CAM, Grad-CAM++, Eigen-CAM, Integrated Gradients, and SmoothGrad to interpret predictions. The interface also supports comparing explanations across methods and evaluating their stability.

Research Mode reads precomputed experiment CSV files and presents one XAI metric at a time, so metrics with different numerical scales are never averaged together. Prediction Mode can optionally run a small on-demand Quantus comparison for the selected XAI methods; use the offline runner for reproducible dataset-level experiments.

## Features

- KL grade prediction with probability distribution and confidence;
- interactive XAI heatmaps and overlays for five explanation methods;
- deletion and stability checks for a single prediction;
- optional Quantus metrics: FaithfulnessCorrelation, MaxSensitivity, and Sparseness;
- Research Mode for filtering and comparing precomputed experiment results by model, XAI method, true KL grade, and prediction correctness.

## Application modes

**Prediction Mode** is intended for analysing one uploaded X-ray. It generates a model prediction and selected XAI maps. Quantus evaluation is optional and uses a small exploratory profile so it does not run automatically on every upload.

**Research Mode** is intended for already computed results. It is organised by evaluation dimension: Faithfulness, Robustness, Complexity, and future Custom checks. Within a dimension, select one metric to view its definition, configuration, XAI quality by true KL grade, and correct-versus-incorrect prediction comparison. A CSV row represents one `sample x XAI method x metric` value; details of the results format are in [`results/research/README.md`](results/research/README.md).

## Project structure

- `app/` - API, inference, preprocessing, and XAI modules;
- `models/` - trained PyTorch weights and ONNX models;
- `inputs/` - sample X-ray images;
- `scripts/` - model export, Grad-CAM, and benchmark utilities;
- `streamlit_app.py` - web interface for image analysis.

## Getting started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows, the project can also be started from its virtual environment:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

To run the API:

```bash
uvicorn app.main:app --reload
```

## Running an offline research experiment

Place images listed in `inputs/manifest.json` locally, ensure the corresponding ONNX and PyTorch checkpoint files are available under `models/`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad
```

The runner writes a validated partial CSV to `results/research/` after each completed image/method pair. Use `--device auto` to select CUDA when available, or `--device cuda` to require it; continue a matching interrupted run with `--resume`. CUDA setup is documented separately in [`requirements-gpu.md`](requirements-gpu.md). The small one-image / one-perturbation command is a pipeline check only; it is not sufficient for scientific conclusions.

To add DenseNet-121, place its checkpoint at `models/densenet121_finetune_block3_4_best.pth` and export the matching ONNX file:

```powershell
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --architecture densenet121
```

The expected ONNX output path is `models/densenet121_kl.onnx`; DenseNet-121 will then appear in the application model selector.
Only export checkpoints you trust: older PyTorch checkpoints can contain training metadata and require trusted deserialisation during export.

## Notes on XAI evaluation

Metric values should be compared only within the same model, dataset split, XAI configuration, and Quantus configuration. New runs use the bounded, higher-is-better `FaithfulnessCorrelation` metric for faithfulness; old Infidelity rows are legacy data and are not comparable to new runs. No individual metric alone establishes explanation quality.

> This project is intended for educational and research purposes only. It is not a clinical diagnostic tool.
