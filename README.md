# Knee Osteoarthritis XAI

An educational and research project for classifying knee osteoarthritis from X-ray images and visualizing model predictions.

ResNet-50 and DenseNet-121 predict the Kellgren-Lawrence grade (KL Grade 0-4) using four ordinal threshold outputs. The application supports separate Ordinal BCE and Class-aware Ordinal BCE checkpoints for each architecture. The project includes ONNX Runtime inference, a FastAPI service, and a Streamlit interface.

The project implements Grad-CAM, Grad-CAM++, Eigen-CAM, Integrated Gradients, and SmoothGrad to interpret predictions. The interface also supports comparing explanations across methods and evaluating their stability.

Research Mode reads one or more precomputed experiment CSV files. Its Results analysis view reports classifier-subset performance and metric-specific KL-stage statistics; its exploration view presents one XAI metric at a time, so metrics with different numerical scales are never averaged together. Prediction Mode can optionally run a small on-demand Quantus comparison for the selected XAI methods; use the offline runner for reproducible dataset-level experiments.

## Features

- KL grade prediction with probability distribution and confidence;
- interactive XAI heatmaps and overlays for five explanation methods;
- deletion and stability checks for a single prediction;
- optional Quantus metrics: FaithfulnessCorrelation, Region Perturbation score-drop AUC, MaxSensitivity, and Sparseness;
- Research Mode for filtering and comparing precomputed experiment results by model, XAI method, true KL grade, and prediction correctness.

## Application modes

**Prediction Mode** is intended for analysing one uploaded X-ray. It generates a model prediction and selected XAI maps. Quantus evaluation is optional and uses a small exploratory profile so it does not run automatically on every upload.

**Research Mode** is intended for already computed results. Select compatible CSV files to combine models without manual merging. Results analysis derives classifier-subset statistics and performs metric-specific KL0–KL4 Kruskal–Wallis, Holm, epsilon-squared, and conditional Dunn post-hoc analysis. The exploration view remains organised by Faithfulness, Robustness, Complexity, and Custom checks. A CSV row represents one `sample x XAI method x metric` value; details of the results format are in [`results/research/README.md`](results/research/README.md).

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

### Final four-model study

The model IDs are `resnet50`, `resnet50_class_aware`, `densenet121`, and
`densenet121_class_aware`. The class-aware file names below are expected local
names, **not files included with this repository**:

Class-aware training used the same four-output architecture and a KL1 training
loss multiplier of 1.5; inference and XAI do not apply a loss multiplier.

```text
models/resnet50_class_aware_ordinal.pth
models/resnet50_class_aware_ordinal.onnx
models/densenet121_class_aware_ordinal.pth
models/densenet121_class_aware_ordinal.onnx
```

Place trusted checkpoints there, or update their paths in `app/config.py`.
Export each matching ONNX model after the `.pth` files have been supplied:

```powershell
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --model-id resnet50_class_aware
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --model-id densenet121_class_aware
```

The original 150-image test subset is frozen in `inputs/test/test_manifest.json`.
Add exactly 20 new, independently labelled test images to each corresponding
`inputs/test/0` through `inputs/test/4` directory. Create the 250-image manifest
while retaining all original sample IDs:

```powershell
.\.venv\Scripts\python.exe scripts\build_final_test_manifest.py
```

No final manifest is generated until all additional images pass duplicate,
path, KL and test-split checks. The full GPU experiment uses one fixed
`final_research` profile: five XAI methods, five metrics, Faithfulness 50 runs /
1024 spatial RGB pixels, and MaxSensitivity 8 samples with independent
max-normalisation. On a CUDA host:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --manifest inputs/test/final_manifest_250.json --profile final_research --all-models --device cuda --run-id final-xai-250-v1
```

Use `--model resnet50_class_aware` instead of `--all-models` for one model. Use
the identical command plus `--resume` after an interruption. Each model gets a
separate CSV and `.csv.run.json` sidecar with model/checkpoint hashes, selected
image-content digest, parameters and library versions. The CSV is atomically
replaced after each completed image-method pair. A full `final_research` run
cannot silently start on CPU; a `--limit 1` smoke check needs a different run
ID/output from the full study.

For Kaggle, attach the project source, labelled 250-image manifest/images and
matching checkpoint/ONNX files, then use
`notebooks/kaggle_research_runner.ipynb`. Set `MODEL_SELECTION` to one model ID
or `all`, adjust the Kaggle Dataset paths and `RUN_ID_PREFIX`, and run the cells.
Old 150-image CSVs remain separate; use the Research Mode cohort selector to
inspect the original 150 versus the loaded full cohort.

### Other exploratory runs

Place images listed in `inputs/manifest.json` locally, ensure the corresponding ONNX and PyTorch checkpoint files are available under `models/`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad
```

The runner writes a validated partial CSV to `results/research/` after each completed image/method pair. Use `--device auto` to select CUDA when available, or `--device cuda` to require it; continue a matching interrupted run with `--resume`. CUDA setup is documented separately in [`requirements-gpu.md`](requirements-gpu.md). The small one-image / one-perturbation command is a pipeline check only; it is not sufficient for scientific conclusions.

New runs calculate five metric-specific results: FaithfulnessCorrelation, Region Perturbation score-drop AUC, MaxSensitivity, Stability score, and Sparseness. To recalculate only the two revised faithfulness metrics, use:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics faithfulness-correlation region-perturbation
```

An optional method-independent random-order sanity baseline is written to a separate file and is never included in the main metric statistics:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics region-perturbation --samples-per-kl 2 --random-control-output results/research/resnet50-random-control.csv --random-control-repeats 10
```

The final ordinal checkpoints expected by the application are:

```text
models/resnet50_ordinal_finetune_layer3_4_best_21092026.pth
models/densenet121_ordinal_finetune_block3_4_best_21092026.pth
```

They produce four threshold logits. The KL prediction uses the validation-time rule `sum(sigmoid(logits) > 0.5)`. Class-dependent XAI methods and Quantus use a differentiable five-class ordinal score derived from those thresholds; Eigen-CAM remains class-independent.

For a balanced 50-image parameter sensitivity check, run each configuration into a separate CSV. `224`, `1024`, and `2048` spatial pixels correspond to approximately `0.45%`, `2.04%`, and `4.08%` of a 224 x 224 attribution map:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics faithfulness-correlation --samples-per-kl 10 --faithfulness-runs 30 --faithfulness-subset-size 224 --output results/research/resnet50-faithfulness-224.csv
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics faithfulness-correlation --samples-per-kl 10 --faithfulness-runs 30 --faithfulness-subset-size 1024 --output results/research/resnet50-faithfulness-1024.csv
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics faithfulness-correlation --samples-per-kl 10 --faithfulness-runs 30 --faithfulness-subset-size 2048 --output results/research/resnet50-faithfulness-2048.csv
```

Compare MaxSensitivity with and without independent max-normalisation of the original and perturbed maps:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics max-sensitivity --samples-per-kl 10 --max-sensitivity-samples 8 --no-max-sensitivity-normalise --output results/research/resnet50-maxsensitivity-raw.csv
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "SmoothGrad" "Integrated Gradients" --metrics max-sensitivity --samples-per-kl 10 --max-sensitivity-samples 8 --max-sensitivity-normalise --output results/research/resnet50-maxsensitivity-normalised.csv
```

New result rows also store the initial map's L1, L2, maximum-absolute, max-normalised L2, and non-zero-fraction diagnostics. These columns are descriptive checks for scale-related MaxSensitivity extremes; they are not additional XAI quality metrics.

Export the matching ONNX files with:

```powershell
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --architecture resnet50
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --architecture densenet121
```

The expected output paths are `models/resnet50_ordinal_kl.onnx` and `models/densenet121_ordinal_kl.onnx`.
Only export checkpoints you trust: older PyTorch checkpoints can contain training metadata and require trusted deserialisation during export.

## Notes on XAI evaluation

Metric values should be compared only within a compatible image cohort, XAI configuration, and metric configuration. New runs use a custom spatial RGB-grouped implementation of the Faithfulness Correlation principle (`metric_source=custom`) and a Quantus Region Perturbation curve summarised by score-drop AUC. Eigen-CAM maps are class-agnostic even though their evaluation measures the output for the predicted KL class. Old Infidelity and custom Deletion confidence drop rows are legacy data and are not comparable to the revised runs. No individual metric alone establishes explanation quality.

> This project is intended for educational and research purposes only. It is not a clinical diagnostic tool.
