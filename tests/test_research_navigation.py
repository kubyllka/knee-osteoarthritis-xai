import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from streamlit_app import _show_model_comparison, show_research_mode


class ResearchNavigationTests(unittest.TestCase):
    def test_model_comparison_has_no_comparison_type_selector(self):
        with patch("streamlit_app.st") as streamlit, patch("streamlit_app._show_paired_models") as paired:
            results = pd.DataFrame()
            _show_model_comparison(results)

        paired.assert_called_once_with(results)
        streamlit.radio.assert_not_called()

    def test_top_level_sections_render_independently(self):
        results = pd.DataFrame({"metric_name": ["MaxSensitivity"]})
        sections = ("Model Comparison", "Prediction Correctness")
        for section in sections:
            with self.subTest(section=section):
                with (
                    patch("streamlit_app.st") as streamlit,
                    patch("streamlit_app._show_research_toolkit"),
                    patch("streamlit_app.load_research_results", return_value=results),
                    patch("streamlit_app.merge_research_results", return_value=results),
                    patch("streamlit_app._validate_shared_cohort"),
                    patch("streamlit_app._research_cohort", return_value=results),
                    patch("streamlit_app._research_data_summary"),
                    patch("streamlit_app._show_model_comparison") as paired,
                    patch("streamlit_app._show_correctness") as correctness,
                ):
                    streamlit.container.return_value = MagicMock()
                    streamlit.radio.side_effect = ["Repository CSV files", section]
                    streamlit.multiselect.return_value = [Path("example.csv")]
                    show_research_mode()

                navigation = streamlit.radio.call_args_list[-1]
                self.assertEqual(
                    navigation.args[1][-3:],
                    ["Model Comparison", "Prediction Correctness", "Explore & Export"],
                )
                if section == "Model Comparison":
                    paired.assert_called_once_with(results)
                    correctness.assert_not_called()
                else:
                    correctness.assert_called_once_with(results)
                    paired.assert_not_called()


if __name__ == "__main__":
    unittest.main()
