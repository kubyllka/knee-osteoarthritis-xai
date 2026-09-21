# Research results contract

This directory stores outputs created by the offline XAI evaluation pipeline. Each CSV row represents one metric value for one `sample_id`, XAI method, and model.

## Required columns

```text
sample_id, image_path, true_kl, predicted_kl, is_correct,
error_distance, confidence, model_name, xai_method,
metric_name, metric_category, metric_value
```

- `true_kl` and `predicted_kl` are integer Kellgren-Lawrence grades from 0 to 4.
- `is_correct` must equal `true_kl == predicted_kl`.
- `error_distance` must equal `abs(true_kl - predicted_kl)`.
- `confidence` is the predicted-class probability from 0 to 1.
- `metric_category` must be one of `faithfulness`, `robustness`, `complexity`, `randomisation`, or `custom`.
- `metric_value` must be finite and numeric. Its direction is determined later by metric metadata, not by this contract.

## Optional reproducibility columns

`run_id`, `split`, `seed`, `checkpoint_version`, and `target_class` may be included. When present, `target_class` must equal `predicted_kl`, because the current analysis explains the model prediction.

New pipeline outputs also include `metric_source` (`quantus` or `custom`), `score_direction` (`lower` or `higher`), and `metric_parameters` (a JSON object). These make values reproducible and interpretable. Older CSV files remain valid but are shown as legacy data because they do not retain the metric configuration.

New runs use `FaithfulnessCorrelation` as the primary faithfulness metric. It ranges from -1 to 1, where higher is better, and uses a recorded, normalised attribution protocol. `Infidelity` may appear in older files only and is treated as legacy data. `MaxSensitivity` is lower-is-better; `Sparseness` is higher for more concentrated maps, not necessarily clinically better. Deletion confidence drop is displayed under Faithfulness and stability score under Robustness; both remain custom implementations. The `Custom checks` dashboard tab is intentionally empty until a future dataset-wide agreement metric is generated.

For the final ordinal-BCE ResNet-50 and DenseNet-121 models, `target_class` is decoded by counting threshold probabilities above 0.5. Class-dependent XAI and Quantus evaluate the differentiable score of that original predicted KL class, recorded as `xai_target: predicted_kl_class_score`; image masking never changes the target class being evaluated.

Research comparisons by disease severity use `true_kl`. Comparisons of correct and incorrect predictions use `is_correct`.

`demo_xai_metrics.csv` is synthetic data included only to demonstrate the Research Mode interface. It is not an experiment result and must not be used in analysis or reported conclusions.

## Creating real results

Run the offline pipeline from the project root. It reads `inputs/manifest.json` by default and writes a valid Research Mode CSV. A full experiment requires a labelled JSON manifest with real image paths, known `true_kl` labels, and split values. The six-image repository manifest is demonstration-only; labels are never guessed from directory names.

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad
```

The runner writes a validated partial CSV after every completed `sample × XAI method`. Continue an interrupted, matching run with `--resume`:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --device auto --resume --run-id research-20260825
```

`--device auto` selects CUDA only when the installed PyTorch build can access it; otherwise it uses CPU. `--device cuda` fails clearly rather than silently using CPU. The base `requirements.txt` is CPU-oriented; see [`requirements-gpu.md`](../../requirements-gpu.md) for the separate CUDA setup. ONNX prediction remains compatible with the base CPU runtime.

For a quick preview run on one sample, use `--limit 1 --faithfulness-runs 8 --max-sensitivity-samples 2`. This is only an exploratory check. Dataset-level thesis runs should use the default 50 FaithfulnessCorrelation perturbations and 8 MaxSensitivity samples with a fixed seed.
