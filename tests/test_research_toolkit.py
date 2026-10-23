import unittest
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pandas as pd

from app.research.toolkit import TOOLKIT_FILES, build_research_toolkit_zip
from scripts.analyze_research_results import analyze, parse_args
from app.metric_registry import metric_definition
from streamlit_app import (
    _confusion_matrix_chart,
    _confusion_matrix_chart_data,
    _kl_distribution_chart,
    _model_label,
    _show_research_toolkit,
    _summary_card_values,
    _validate_shared_cohort,
    show_research_mode,
)
from app.research.analysis import ResearchAnalysisError


ROOT = Path(__file__).resolve().parent.parent


class ResearchToolkitTests(unittest.TestCase):
    def test_bundle_has_only_allowlisted_source_and_documentation(self):
        with ZipFile(BytesIO(build_research_toolkit_zip(ROOT))) as bundle:
            names = set(bundle.namelist())
            self.assertEqual(names, set(TOOLKIT_FILES))
            self.assertIn("scripts/run_research.py", names)
            self.assertIn("scripts/analyze_research_results.py", names)
            self.assertIn("notebooks/kaggle_research_runner.ipynb", names)
            self.assertIn("app/research/analysis.py", names)
            self.assertFalse(any(name.startswith(("models/", "inputs/", "results/")) for name in names))
            self.assertFalse(any(name.endswith((".pth", ".onnx", ".csv", ".env")) for name in names))

    def test_cli_contract(self):
        args = parse_args(["one.csv", "--output-dir", "out", "--analysis", "classifier"])
        self.assertEqual(args.analysis, "classifier")
        self.assertEqual(args.csv_files, [Path("one.csv")])

    def test_kaggle_notebook_in_bundle_is_parseable_and_has_no_saved_outputs(self):
        with ZipFile(BytesIO(build_research_toolkit_zip(ROOT))) as bundle:
            notebook = json.loads(bundle.read("notebooks/kaggle_research_runner.ipynb"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(not cell["outputs"] for cell in code_cells))
        for cell in code_cells:
            compile("".join(cell["source"]), "kaggle_research_runner.ipynb", "exec")
        source = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("scripts/run_research.py", source)
        self.assertNotIn("shutil.rmtree", source)

    def test_cli_exports_classifier_subset_without_streamlit(self):
        source = ROOT / "results/research/densenet121-demo-20260921-133343.csv"
        self.assertTrue(source.is_file())
        with patch.object(Path, "mkdir"), patch.object(pd.DataFrame, "to_csv"):
            outputs = analyze([source], ROOT / "unused-output", "classifier")
            self.assertEqual(len(outputs), 3)
            self.assertTrue(any(path.name == "classifier_subset_summary.csv" for path in outputs))

    def test_kl_plot_is_single_metric_and_model(self):
        source = ROOT / "results/research/densenet121-xai-full-20260924-full (6).csv"
        if not source.is_file():
            self.skipTest("Full research CSV is not available")
        from app.research.results import load_research_results

        chart = _kl_distribution_chart(load_research_results(source), "densenet121", "Sparseness")
        self.assertIsNotNone(chart)
        data = chart.data
        self.assertEqual(set(data["metric_name"]), {"Sparseness"})
        self.assertEqual(set(data["model_name"]), {"densenet121"})

    def test_display_name_does_not_change_model_id(self):
        self.assertIn("class-aware", _model_label("resnet50_class_aware"))
        self.assertEqual(_model_label("unknown"), "unknown")

    def test_toolkit_is_present_before_any_csv_is_selected(self):
        with patch("streamlit_app._show_research_toolkit") as toolkit, patch("streamlit_app.st") as streamlit:
            streamlit.radio.return_value = "Repository CSV files"
            streamlit.multiselect.return_value = []
            show_research_mode()
        toolkit.assert_called_once_with()
        streamlit.info.assert_called_once()

    def test_toolkit_card_has_guide_and_download(self):
        with patch("streamlit_app.st") as streamlit, patch("streamlit_app._cached_toolkit", return_value=b"ZIP"):
            streamlit.columns.return_value = [MagicMock(), MagicMock()]
            _show_research_toolkit()
        streamlit.expander.assert_called_once_with("How to run the experiment · detailed instructions")
        self.assertTrue(any("KneeXAI Research Toolkit" in str(call) for call in streamlit.markdown.call_args_list))
        self.assertEqual(streamlit.download_button.call_count, 2)
        keys = {call.kwargs["key"] for call in streamlit.download_button.call_args_list}
        self.assertEqual(keys, {"download-research-toolkit-top", "download-research-toolkit-guide"})

    def test_confusion_heatmap_preserves_table_counts(self):
        matrix = pd.DataFrame(
            [[3, 1], [2, 4]],
            index=["True KL0", "True KL1"],
            columns=["Predicted KL0", "Predicted KL1"],
        )
        data = _confusion_matrix_chart_data(matrix)
        self.assertEqual(int(data["Images"].sum()), int(matrix.to_numpy().sum()))
        self.assertEqual(set(data["True KL"]), {"KL0", "KL1"})
        self.assertEqual(set(data["Predicted KL"]), {"KL0", "KL1"})
        self.assertIn("Images", _confusion_matrix_chart(matrix).to_json())

    def test_metric_summary_labels_name_the_direction_not_a_winner(self):
        frame = pd.DataFrame({
            "sample_id": ["a", "b", "c", "d"],
            "xai_method": ["Grad-CAM", "Grad-CAM", "SmoothGrad", "SmoothGrad"],
            "true_kl": [0, 1, 0, 1],
            "metric_value": [0.2, 0.4, 0.7, 0.6],
            "is_correct": [True, False, True, False],
        })
        self.assertEqual(
            _summary_card_values(frame, metric_definition("MaxSensitivity"))["method_label"],
            "Lowest median for selected metric",
        )
        self.assertEqual(
            _summary_card_values(frame, metric_definition("Sparseness"))["method_label"],
            "Highest median for selected metric",
        )

    def test_shared_id_with_different_image_is_rejected_before_pairing(self):
        source = ROOT / "results/research/densenet121-demo-20260921-133343.csv"
        from app.research.results import load_research_results

        dense = load_research_results(source)
        resnet = dense.copy()
        resnet["model_name"] = "resnet50"
        resnet.loc[resnet["sample_id"].eq(resnet.iloc[0]["sample_id"]), "image_path"] = "another/image.png"
        with self.assertRaisesRegex(ResearchAnalysisError, "conflicting image_path"):
            _validate_shared_cohort(pd.concat([dense, resnet], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
