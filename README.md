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
- `models/` - PyTorch checkpoints and ONNX models; the two final class-aware pairs use Git LFS;
- `inputs/` - sample X-ray images;
- `scripts/` - model export, Grad-CAM, and benchmark utilities;
- `streamlit_app.py` - web interface for image analysis.

## Run without Docker

Python 3.12 is recommended. Install Git LFS before cloning, or run `git lfs pull` after cloning, to obtain the two published class-aware checkpoint/ONNX pairs. A source archive that does not include LFS objects is not sufficient for Prediction Mode. Research Mode can open compatible CSV files without models. The expected filenames for each model ID are defined in [`app/config.py`](app/config.py). Historical ordinal-BCE checkpoints are optional local files and are not published via LFS.

On Windows PowerShell, create an environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, the equivalent setup is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For Prediction Mode, the class-aware `.pth` checkpoints and matching `.onnx` files should be present under `models/` after `git lfs pull`. ONNX is used for inference; PyTorch checkpoints are needed for XAI. If an ONNX file is missing but its trusted checkpoint is available, regenerate it with:

```powershell
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --model-id resnet50_class_aware
```

Use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe` on macOS/Linux. Then launch the interface and **select the model for which you supplied files** in the sidebar:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

On macOS/Linux, run `.venv/bin/streamlit run streamlit_app.py` instead. To use the optional FastAPI endpoint, start it in a **second terminal** with `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload` (or `.venv/bin/python -m uvicorn app.main:app --reload`). Then enable **Use FastAPI endpoint** under Advanced settings; its local URL is `http://127.0.0.1:8000`. Without that toggle, Prediction Mode runs inference locally. Research Mode can inspect compatible CSVs without model weights; example and older experiment results are not substitutes for a new final study.

## Run with Docker Compose

Install Docker with the Compose plugin. The same `compose.yaml` starts **two CPU services**: Streamlit (`web`) and FastAPI (`api`). This is an application setup, not the CUDA dataset-wide research runner.

1. Run `git lfs pull` on the host to obtain the two class-aware model pairs under `models/`. For historical ordinal-BCE models, supply matching `.onnx` and `.pth` files yourself using the names in [`app/config.py`](app/config.py). Compose mounts these files but does not download them or bake them into the image.
2. From the repository root, start both services:

   ```bash
   docker compose up --build -d
   ```

3. Open **http://127.0.0.1:8501** for Streamlit. The API health endpoint is **http://127.0.0.1:8000/health**. A running API can still report `model_loaded: false` if no ONNX model is present. Select a model whose files you provided.

4. Prediction Mode uses the model mounted in `web` by default. To route its inference through the `api` service, enable **Use FastAPI endpoint** in Advanced settings; inside Compose, the URL field defaults to `http://api:8000`. Keep that internal address in the Streamlit field, not `127.0.0.1`.

Useful commands:

```bash
docker compose ps
docker compose logs -f web api
docker compose down
```

Compose mounts `models/`, `inputs/` and `results/research/` read-only rather than baking model weights, radiographs or research CSVs into the image. Add your own CSVs to `results/research/` on the host or upload them in Research Mode. `docker compose down` stops the services without deleting those host files. The **Download Research Toolkit** feature is included in the web container. For full GPU XAI experiments, use the separate [research instructions](docs/research_runs.md) and [CUDA requirements](requirements-gpu.md).

## Research experiments

The offline runner, model-file names, CUDA setup, manifest requirements, resume procedure, Kaggle notebook, and exploratory parameter checks are documented in [`docs/research_runs.md`](docs/research_runs.md). The CSV format is described in [`results/research/README.md`](results/research/README.md). Current `inputs/test/test_manifest.json` contains **250**, not 150, images; read the manifest warning in the research guide before attempting to reproduce an original-150-preserving final run.

## Notes on XAI evaluation

New runs use a custom spatial RGB-grouped implementation of the Faithfulness Correlation principle (`metric_source=custom`), a Quantus-based Region Perturbation score-drop AUC, Quantus MaxSensitivity and Sparseness, and a custom Stability score. Eigen-CAM maps are class-agnostic even though their evaluation measures the output for the predicted KL class. Compare values only within a compatible image cohort and metric configuration. Old Infidelity and custom Deletion confidence drop rows are legacy data; no individual metric establishes clinical validity or overall explanation quality.

> This project is intended for educational and research purposes only. It is not a clinical diagnostic tool.
