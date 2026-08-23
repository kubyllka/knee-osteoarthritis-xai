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

Research comparisons by disease severity use `true_kl`. Comparisons of correct and incorrect predictions use `is_correct`.

`demo_xai_metrics.csv` is synthetic data included only to demonstrate the Research Mode interface. It is not an experiment result and must not be used in analysis or reported conclusions.

## Creating real results

Run the offline pipeline from the project root. It reads `inputs/manifest.json` by default and writes a valid Research Mode CSV.

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad
```

For a quick smoke run on one sample, use `--limit 1 --infidelity-samples 1 --max-sensitivity-samples 1`. Use GPU-enabled Kaggle or another CUDA runtime for larger research runs.
