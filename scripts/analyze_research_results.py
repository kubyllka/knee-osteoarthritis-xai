"""Run the same research statistics used by Streamlit on saved result CSVs."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.research.analysis import (  # noqa: E402
    ResearchAnalysisError,
    architecture_comparison_analysis,
    architecture_complete_case_pairs,
    classifier_subset_summary,
    dunn_posthoc,
    friedman_method_analysis,
    kl_stage_analysis,
    merge_research_results,
    method_comparison_descriptives,
    method_complete_case_matrices,
    prediction_correctness_analysis,
    prediction_records,
    wilcoxon_method_posthoc,
)
from app.research.results import load_research_results  # noqa: E402


MODEL_PAIRS = (
    ("densenet121", "resnet50"),
    ("densenet121_class_aware", "resnet50_class_aware"),
    ("densenet121_class_aware", "densenet121"),
    ("resnet50_class_aware", "resnet50"),
)
ANALYSES = ("classifier", "kl", "methods", "models", "correctness")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze precomputed KneeXAI research CSVs without Streamlit.")
    parser.add_argument("csv_files", nargs="+", type=Path, help="One compatible research CSV per model.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for statistical CSV outputs.")
    parser.add_argument("--analysis", choices=("all", *ANALYSES), default="all")
    return parser.parse_args(argv)


def analyze(csv_files: list[Path], output_dir: Path, analysis: str = "all") -> list[Path]:
    results = merge_research_results([load_research_results(path) for path in csv_files])
    prediction_records(results)  # Reject conflicting model/sample prediction metadata before any export.
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(name, frame):
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)

    selected = set(ANALYSES) if analysis == "all" else {analysis}
    if "classifier" in selected:
        summary, grades, matrices = classifier_subset_summary(results)
        save("classifier_subset_summary", summary)
        save("classifier_by_kl_grade", grades)
        for model, matrix in matrices.items():
            save(f"confusion_matrix_{model}", matrix.rename_axis("True KL grade").reset_index())
    if "kl" in selected:
        primary = kl_stage_analysis(results)
        save("kl_stage_analysis", primary)
        complete_family = (
            not primary.empty
            and primary.groupby("Model").size().eq(25).all()
            and primary["KL groups"].eq(5).all()
        )
        if not complete_family:
            print("Dunn post-hoc unavailable: the 25-test family or KL groups are incomplete.", file=sys.stderr)
        else:
            try:
                posthoc = dunn_posthoc(results, primary)
            except ResearchAnalysisError as exc:
                print(f"Dunn post-hoc unavailable: {exc}", file=sys.stderr)
            else:
                save("dunn_posthoc_kl_pairs", posthoc)
    if "methods" in selected:
        matrices = method_complete_case_matrices(results)
        friedman = friedman_method_analysis(matrices)
        save("xai_method_descriptives", method_comparison_descriptives(matrices))
        save("friedman_xai_method_analysis", friedman)
        save("wilcoxon_xai_method_pairs", wilcoxon_method_posthoc(matrices, friedman))
    if "models" in selected:
        available = set(results["model_name"])
        pairs_found = False
        for first, second in MODEL_PAIRS:
            if {first, second}.issubset(available):
                pairs_found = True
                for same_target, suffix in ((False, "all_pairs"), (True, "same_target")):
                    try:
                        pairs = architecture_complete_case_pairs(
                            results, same_target_only=same_target, models=(first, second)
                        )
                        summary = architecture_comparison_analysis(pairs)
                    except ResearchAnalysisError as exc:
                        print(f"Model comparison {first} vs {second} ({suffix}) unavailable: {exc}", file=sys.stderr)
                        continue
                    save(f"model_comparison_{first}_vs_{second}_{suffix}", summary)
        if not pairs_found:
            print("Model comparison unavailable: load at least one compatible model pair.", file=sys.stderr)
    if "correctness" in selected:
        save("prediction_correctness_analysis", prediction_correctness_analysis(results))
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        written = analyze(args.csv_files, args.output_dir, args.analysis)
    except (OSError, ValueError, ResearchAnalysisError) as exc:
        print(f"Research analysis failed: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
