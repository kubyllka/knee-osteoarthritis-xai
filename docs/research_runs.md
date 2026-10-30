# Offline XAI research runs

This guide describes the command-line experiment pipeline. It is separate from the one-image Prediction Mode preview. Run commands from the repository root after installing `requirements.txt`; see the [main README](../README.md) for environment setup and [`requirements-gpu.md`](../requirements-gpu.md) for CUDA. The CLI does not include model weights or provide a model-download URL.

## Model files

The four ordinal research model IDs are `resnet50`, `resnet50_class_aware`, `densenet121`, and `densenet121_class_aware`. Put **matching, trusted** PyTorch checkpoints and ONNX models under `models/`, using the exact paths in [`app/config.py`](../app/config.py):

| Model ID | Checkpoint (`models/`) | ONNX (`models/`) |
| --- | --- | --- |
| `resnet50` | `resnet50_ordinal_finetune_layer3_4_best_21092026.pth` | `resnet50_ordinal_kl.onnx` |
| `densenet121` | `densenet121_ordinal_finetune_block3_4_best_21092026.pth` | `densenet121_ordinal_kl.onnx` |
| `resnet50_class_aware` | `resnet50_class_aware_ordinal.pth` | `resnet50_class_aware_ordinal.onnx` |
| `densenet121_class_aware` | `densenet121_class_aware_ordinal.pth` | `densenet121_class_aware_ordinal.onnx` |

If only a checkpoint is available, export its matching ONNX file, for example:

```powershell
.\.venv\Scripts\python.exe scripts\export_to_onnx.py --model-id resnet50_class_aware
```

Repeat with another `--model-id` as needed. Do not substitute an unrelated checkpoint: the runner checks ONNX/PyTorch prediction agreement. Older PyTorch checkpoint formats can require deserialisation of training metadata, so export only files from a trusted source. Class-aware training used a KL1 loss multiplier of 1.5; inference and XAI do not apply this multiplier. All four ordinal models produce four threshold logits. The predicted KL grade is `sum(sigmoid(logits) > 0.5)`. Class-dependent XAI methods evaluate a differentiable predicted-class score; Eigen-CAM maps are class-independent.

## Manifest status in this checkout

`inputs/manifest.json` lists **40 demo images**. `inputs/test/test_manifest.json` currently lists **250 test images (50 per KL grade)**, not the original 150-image subset described in older instructions. The separate `inputs/test/final_manifest_250.json` is not present.

`scripts/build_final_test_manifest.py` is designed to start from a *150-image* base manifest and find exactly 20 additional files in each KL folder. It therefore **cannot be run with the current 250-image `inputs/test/test_manifest.json` as its base**: it finds zero additional files. The runner's `final_research` full-cohort check also reads `inputs/test/test_manifest.json` as its original subset. With the current 250-image file as both base and selected manifest, it can validate balance and duplicates but cannot independently establish that the earlier 150 IDs were preserved.

Before claiming an original-150-preserving final reproduction, supply the frozen 150-image base manifest at `inputs/test/test_manifest.json` and then build a separate 250-image manifest from the same image folders:

```powershell
.\.venv\Scripts\python.exe scripts\build_final_test_manifest.py
```

Do not replace the 250-image manifest with a guessed 150-image list: its labels and sample IDs must come from the actual earlier cohort. Until the two manifests are restored separately, the 250-image file can be used for a technical run, but the original-subset preservation claim is not independently checked by this checkout.

## Final GPU profile

`--profile final_research` fixes five methods and five metrics: FaithfulnessCorrelation, Region Perturbation score-drop AUC, MaxSensitivity, Stability score, and Sparseness. It uses 50 faithfulness runs with 1,024 spatial RGB pixel groups, 8 independently max-normalised MaxSensitivity samples, 16 × 16 region patches, and 100 region perturbations. See [`app/research/experiment.py`](../app/research/experiment.py) for the full frozen profile. FaithfulnessCorrelation is the project's **custom spatial RGB-grouped implementation** (`metric_source=custom`), not a direct Quantus `FaithfulnessCorrelation` call. Region Perturbation, MaxSensitivity, and Sparseness use Quantus components; Stability score is custom.

Once the separate original and final manifests are in place, a full CUDA run is:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --manifest inputs/test/final_manifest_250.json --profile final_research --all-models --device cuda --run-id final-xai-250-v1
```

Use `--model resnet50_class_aware` instead of `--all-models` for one model. Use the **same** command with `--resume` after an interruption. The runner writes one validated CSV and `.csv.run.json` metadata sidecar per model, updating the CSV after each completed image-method pair. Resume checks run, model, checkpoint, manifest, device, seed, and metric settings. A full `final_research` run requires CUDA. Give a `--limit 1` smoke test a different run ID and output path; it is not dataset-level evidence.

On Kaggle, use [`notebooks/kaggle_research_runner.ipynb`](../notebooks/kaggle_research_runner.ipynb) after attaching the project source, labelled images/manifests, and matching checkpoint/ONNX files. Set its dataset paths, `MODEL_SELECTION`, and `RUN_ID_PREFIX` before running. The notebook calls the same runner; it does not implement a second metric pipeline.

## Exploratory runs

For a small non-final run, provide the appropriate local model files and use a labelled manifest. The default `inputs/manifest.json` is demonstration-only:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad --device auto
```

`--device auto` chooses CUDA when available, otherwise CPU; `--device cuda` requires CUDA. The non-final defaults are 50 FaithfulnessCorrelation runs with 224 spatial RGB pixel groups, 8 MaxSensitivity samples, and `normalise=False` for MaxSensitivity. These differ from the frozen final profile and should not be mixed in a statistical comparison.

To calculate only the two faithfulness metrics for all five methods:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "Integrated Gradients" "SmoothGrad" --metrics faithfulness-correlation region-perturbation
```

The optional random region-order sanity baseline is saved separately and is not an XAI-method metric:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" "Grad-CAM++" "Eigen-CAM" "Integrated Gradients" "SmoothGrad" --metrics region-perturbation --samples-per-kl 2 --random-control-output results/research/resnet50-random-control.csv --random-control-repeats 10
```

For a 50-image parameter-sensitivity study, use `--samples-per-kl 10` with one output CSV per configuration. For FaithfulnessCorrelation, compare `--faithfulness-subset-size 224`, `1024`, and `2048` at the same `--faithfulness-runs 30`. These sizes are approximately 0.45%, 2.04%, and 4.08% of a 224 × 224 map. For MaxSensitivity, compare `--no-max-sensitivity-normalise` and `--max-sensitivity-normalise` with the same `--max-sensitivity-samples 8`. Use distinct `--output` filenames for each run; do not merge configurations into one metric comparison. Initial attribution-map norms stored in the CSV are diagnostics, not additional XAI quality metrics.

Metric values are comparable only within a compatible image cohort, XAI configuration, and metric configuration. Old Infidelity and custom Deletion confidence drop rows are legacy data, not replacements for the current five-metric profile. The CSV contract is documented in [`results/research/README.md`](../results/research/README.md).
