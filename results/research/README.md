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
- `metric_value` must be finite and numeric when `metric_status` is `ok` (or absent in a legacy CSV). For an allowed undefined status, the column is present but its value must be empty. Its direction is determined by metric metadata.

## Optional reproducibility columns

`run_id`, `split`, `seed`, `checkpoint_version`, and `target_class` may be included. When present, `target_class` must equal `predicted_kl`, because the current analysis explains the model prediction.

Final-study rows also store `model_id` (equal to `model_name`),
`model_architecture`, `training_variant`, `kl1_loss_multiplier`,
`experiment_profile`, and SHA-256
digests for the manifest, selected image contents, checkpoint and ONNX model.
A matching `.csv.run.json` sidecar records the full run identity and package
versions. Resume rejects a different identity; partial CSV writes are atomic.

New pipeline outputs also include `metric_source` (`quantus` or `custom`), `score_direction` (`lower` or `higher`), and `metric_parameters` (a JSON object). These make values reproducible and interpretable. Older CSV files remain valid but are shown as legacy data because they do not retain the metric configuration.

`metric_status` is `ok` for a usable numeric value. It can be `undefined_zero_perturbed_attribution` only for `MaxSensitivity`: Quantus encountered an all-zero explanation after its prescribed input perturbation and therefore did not define a score. Such a row has an empty `metric_value` and a `metric_note`; Research Mode excludes it from numerical summaries while showing the exclusion count. It is not replaced with an artificial epsilon value.

New runs use `FaithfulnessCorrelation` as the primary faithfulness metric. It ranges from -1 to 1, where higher is better, and uses the project's custom spatial RGB-grouped implementation of the Quantus correlation principle (`metric_source=custom`). Old CSVs retain their original source labels. `Infidelity` is legacy only. `MaxSensitivity` is lower-is-better; `Sparseness` describes concentration, not clinical quality. Legacy Deletion confidence drop and current Stability score remain supporting custom checks.

For ordinal ResNet-50 and DenseNet-121 model variants, `target_class` is decoded by counting threshold probabilities above 0.5. Class-dependent XAI and evaluation use a differentiable score of the original predicted KL class. Eigen-CAM itself is class-agnostic; its metric evaluation still uses the predicted-class score. New rows record this distinction as `xai_map_target`.

The offline runner verifies ONNX and PyTorch ordinal logits before evaluation. It stops for a material logit disagreement, which can indicate mismatched model files. A sign change is accepted only when it is within a documented near-zero numerical margin; in that case the runner warns and XAI still targets the ONNX KL prediction saved in the CSV.

Research comparisons by disease severity use `true_kl`. Comparisons of correct and incorrect predictions use `is_correct`.

Research Mode can combine compatible CSV files and calculate a primary KL-stage analysis. It requires consistent prediction metadata within each `model × sample_id` and rejects duplicate `model × sample_id × XAI method × metric` rows. It reports per-KL `n`, median, Q1, Q3, and IQR, then applies Kruskal–Wallis with `ε² = max(0, (H - k + 1) / (N - k))` and Holm correction across the 25 method × metric tests within each model. Dunn pairwise KL comparisons are available only after a Holm-significant global test. These results remain metric-specific and must not be converted into an overall XAI-method ranking.

`demo_xai_metrics.csv` is synthetic data included only to demonstrate the Research Mode interface. It is not an experiment result and must not be used in analysis or reported conclusions.

## Creating real results

Run the offline pipeline from the project root. It reads `inputs/manifest.json` by default and writes a valid Research Mode CSV. A full experiment requires a labelled JSON manifest with real image paths, known `true_kl` labels, and split values. The **40-image** repository demo manifest is demonstration-only; labels are never guessed from directory names. See [`docs/research_runs.md`](../../docs/research_runs.md) for the current 250-image manifest caveat and final-run instructions.

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --methods "Grad-CAM" SmoothGrad
```

The runner writes a validated partial CSV after every completed `sample × XAI method`. Continue an interrupted run by repeating its original command, output path, run ID, model, manifest and settings with `--resume`; for example, if the initial run used the following ID:

```powershell
.\.venv\Scripts\python.exe scripts\run_research.py --model resnet50 --device auto --resume --run-id research-20260825
```

`--device auto` selects CUDA only when the installed PyTorch build can access it; otherwise it uses CPU. `--device cuda` fails clearly rather than silently using CPU. The base `requirements.txt` is CPU-oriented; see [`requirements-gpu.md`](../../requirements-gpu.md) for the separate CUDA setup. ONNX prediction remains compatible with the base CPU runtime.

For a quick preview run on one sample, use `--limit 1 --faithfulness-runs 8 --max-sensitivity-samples 2` and a separate run ID/output. This is only an exploratory check. The dataset-level `final_research` profile fixes 50 FaithfulnessCorrelation runs over 1,024 spatial RGB groups, 8 normalised MaxSensitivity samples, and the other protocol settings; it must not be conflated with the non-final defaults.
