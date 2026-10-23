# KneeXAI Research Toolkit

This source-only bundle reproduces the repository's XAI evaluation and statistical analysis. It does not contain model weights, radiographs, manifests, research CSVs, or credentials. The two scripts share the application's implementation; the statistical script does not reimplement test formulas.

For Kaggle GPU runs, open `notebooks/kaggle_research_runner.ipynb` and set its Toolkit, checkpoint, and manifest paths to your own attached Kaggle datasets. The notebook stages the source, exports ONNX from the selected trusted checkpoint, and calls the same `scripts/run_research.py`; it does not contain a second metrics implementation. Its default `LIMIT = 1` and smoke-test run ID are **not** a final experiment. Before a full run, set `LIMIT = None`, use a new run ID, and provide both the final 250-image manifest and the original 150-image manifest. Keep the image paths valid after staging: either use absolute Kaggle paths in both manifests or place the images under the staged project root at their recorded relative paths. A resumed run must reuse the same working copy, model files, manifests, run ID, and output CSV/metadata sidecar.

## Environment and inputs

Use a Python environment compatible with the pinned versions in `requirements.txt` (Python 3.12 is recommended). Install with `python -m pip install -r requirements.txt`. This requirements file pins a PyTorch version, but does **not** guarantee CUDA support: for GPU runs, replace its `torch`/`torchvision` installation with matching CUDA wheels for your machine using the official PyTorch installer, then verify `torch.cuda.is_available()`. CPU runs use `--device cpu`. The runner may be slow on CPU.

Place your own matching `.pth` and `.onnx` files in `models/` using the exact names in `app/config.py`. `scripts/export_to_onnx.py` can export a compatible local checkpoint; run `python scripts/export_to_onnx.py --help` for its current options. Do not substitute unrelated checkpoints. The runner checks ONNX/PyTorch prediction parity and run metadata. Image files should be placed under the project root, or use paths accepted by `app/research/manifest.py`.

The JSON manifest must contain known KL labels and split values. A minimal example (provide real files at these paths):

```json
{
  "samples": [
    {"sample_id": "image-001", "image_path": "inputs/test/image-001.png", "true_kl": 2, "split": "test"},
    {"sample_id": "image-002", "image_path": "inputs/test/image-002.png", "true_kl": 3, "split": "test"}
  ]
}
```

The `final_research` profile requires the final labelled cohort specified in the runner and fixes the five methods and metric parameters. For a full run, also provide the original 150-image manifest at `inputs/test/test_manifest.json`: the runner verifies that the final cohort preserves those original IDs and labels. Neither manifest is bundled because it contains user data paths. A two-image example is for inspecting the JSON structure only; it cannot pass final-profile cohort validation. For a technical smoke test use a non-final run with `--limit`, then use a new run ID for final results. Resume requires the same input and configuration; use `--resume` with the original output path and run ID.

## Run XAI evaluation

```bash
python scripts/run_research.py --manifest inputs/test/final_manifest.json --model resnet50_class_aware --profile final_research --run-id final-resnet-v1 --output-dir results/research --seed 42 --device cuda
```

Replace the manifest and model with your own validated files. Other model IDs are in `app/config.py`; `--all-models` runs all four ordinal configurations sequentially. On CPU use `--device cpu`. The runner saves validated metric rows to a CSV and run metadata to a sidecar file; it writes partial results for resumption. Required result-CSV columns are `sample_id`, `image_path`, `true_kl`, `predicted_kl`, `is_correct`, `error_distance`, `confidence`, `model_name`, `xai_method`, `metric_name`, `metric_category`, and `metric_value`. The runner additionally records status, run ID, configuration, and provenance fields; the exact contract is in `app/research/results.py`. Undefined metric values are retained as undefined, never replaced with zero. An optional random-order sanity control can be written separately with `--random-control-output`.

## Analyze existing research CSVs

```bash
python scripts/analyze_research_results.py results/research/resnet.csv results/research/densenet.csv --output-dir results/statistics --analysis all
```

For a faster subset, use `--analysis classifier`, `kl`, `methods`, or `models`; `--analysis correctness` runs the existing 10,000-permutation KL-stratified sensitivity analysis and can take longer. `--analysis all` includes it. The CLI rejects incompatible duplicate rows or conflicting prediction metadata. It writes separate classifier-subset, per-KL, Friedman/Wilcoxon, Kruskal–Wallis/Dunn, eligible paired-model, and correctness CSVs when the relevant data are present. Missing compatible model pairs or unavailable post-hoc tests are reported, not fabricated. Model comparisons require shared `sample_id` values and matching method/metric configuration; they never pool unlike metrics. These classifier figures describe only the loaded XAI subset, not necessarily the full test set.

To see all supported flags, run either script with `--help`. The package intentionally excludes the Streamlit interface and datasets; it is for command-line reproduction, not a standalone clinical product.
