import unittest

import pandas as pd

from app.research.analysis import (
    METHOD_COMPARISON_METRICS,
    XAI_METHOD_ORDER,
    architecture_comparison_analysis,
)
from streamlit_app import (
    _architecture_effect_heatmap_data,
    _architecture_effect_heatmap_figure,
    _architecture_heatmap_bytes,
)


MODELS = ("densenet121_class_aware", "resnet50_class_aware")


def example_summary() -> pd.DataFrame:
    rows = []
    for metric in METHOD_COMPARISON_METRICS:
        for method in XAI_METHOD_ORDER:
            rows.append({
                "Metric": metric,
                "XAI method": method,
                "Model pair": MODELS,
                "Paired N": 150,
                "Rank-biserial correlation": 0.44,
                "p Holm": 0.2,
                "Architecture significant": False,
                "Test status": "ok",
            })
    return pd.DataFrame(rows)


class ArchitectureHeatmapTests(unittest.TestCase):
    def test_all_25_cells_match_existing_results_and_holm_stars(self):
        summary = example_summary()
        summary.loc[0, ["Rank-biserial correlation", "p Holm", "Architecture significant"]] = [
            -0.92, 0.01, True,
        ]
        data = _architecture_effect_heatmap_data(summary, MODELS)

        self.assertEqual(len(data), 25)
        self.assertEqual(data.iloc[0]["Cell label"], "−0.92*")
        self.assertEqual(data.iloc[1]["Cell label"], "+0.44")
        for _, row in data.iterrows():
            original = summary[
                summary["Metric"].eq(row["Metric"])
                & summary["XAI method"].eq(row["XAI method"])
            ].iloc[0]
            self.assertEqual(row["Display effect"], original["Rank-biserial correlation"])
            self.assertEqual(row["Display significant"], original["p Holm"] < 0.05)

    def test_undefined_and_missing_tests_are_na_not_zero(self):
        summary = example_summary().iloc[:-1].copy()
        summary.loc[0, ["Rank-biserial correlation", "Test status"]] = [
            0.0, "insufficient complete paired samples",
        ]
        data = _architecture_effect_heatmap_data(summary, MODELS)

        self.assertEqual(data.iloc[0]["Cell label"], "N/A")
        self.assertEqual(data.iloc[-1]["Cell label"], "N/A")
        self.assertTrue(pd.isna(data.iloc[0]["Display effect"]))
        self.assertTrue(pd.isna(data.iloc[-1]["Display effect"]))

    def test_model_order_reverses_signed_rank_effect(self):
        first = "densenet121"
        second = "resnet50"
        matrix = pd.DataFrame({first: [2.0, 3.0, 4.0, 5.0], second: [1.0, 2.0, 3.0, 4.0]})
        matrix.attrs["models"] = (first, second)
        forward = architecture_comparison_analysis({(XAI_METHOD_ORDER[0], METHOD_COMPARISON_METRICS[0]): matrix})
        reversed_matrix = matrix[[second, first]].copy()
        reversed_matrix.attrs["models"] = (second, first)
        reverse = architecture_comparison_analysis({
            (XAI_METHOD_ORDER[0], METHOD_COMPARISON_METRICS[0]): reversed_matrix,
        })

        self.assertEqual(float(forward.iloc[0]["Rank-biserial correlation"]), 1.0)
        self.assertEqual(float(reverse.iloc[0]["Rank-biserial correlation"]), -1.0)
        self.assertEqual(
            _architecture_effect_heatmap_data(reverse, (second, first)).iloc[0]["Cell label"],
            "−1.00",
        )

    def test_png_and_svg_export_use_same_analysis_grid(self):
        data = _architecture_effect_heatmap_data(example_summary(), MODELS)
        figure = _architecture_effect_heatmap_figure(data, MODELS, "Same target")
        self.assertIn("Same target", figure.axes[0].get_title(loc="left"))
        self.assertEqual(len(figure.axes[0].texts), 25)
        self.assertTrue(_architecture_heatmap_bytes(figure, "png").startswith(b"\x89PNG"))
        self.assertIn(b"<svg", _architecture_heatmap_bytes(figure, "svg"))


if __name__ == "__main__":
    unittest.main()
