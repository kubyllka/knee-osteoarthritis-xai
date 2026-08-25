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

`Infidelity` is an MSE-based, lower-is-better faithfulness score and may be numerically large. It must only be compared between rows produced with the same parameters. `MaxSensitivity` is lower-is-better; `Sparseness` is higher for more concentrated maps, not necessarily clinically better. The custom deletion and stability checks are reported separately.

Research comparisons by disease severity use `true_kl`. Comparisons of correct and incorrect predictions use `is_correct`.

`demo_xai_metrics.csv` is synthetic data included only to demonstrate the Research Mode interface. It is not an experiment result and must not be used in analysis or reported conclusions.

## Creating real results

Run the offline pipeline from the project root. It reads `inputs/manifest.json` by default and writes a valid Research Mode CSV.

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad
```

For a quick smoke run on one sample, use `--limit 1 --infidelity-samples 1 --max-sensitivity-samples 1`. This is only a technical check. Use a fixed seed and a larger perturbation count on GPU for research runs.
