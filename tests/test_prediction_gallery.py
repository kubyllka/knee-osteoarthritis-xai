import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from PIL import Image

from streamlit_app import _show_overlay_gallery, _xai_gallery_images_zip


class PredictionGalleryTests(unittest.TestCase):
    def test_five_methods_use_two_equal_three_column_rows(self):
        original = Image.new("RGB", (224, 224), "white")
        methods = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Integrated Gradients", "SmoothGrad"]
        results = [
            SimpleNamespace(method=method, original=original, overlay=Image.new("RGB", (224, 224)))
            for method in methods
        ]

        with patch("streamlit_app.st") as streamlit:
            streamlit.columns.side_effect = [
                [MagicMock(), MagicMock(), MagicMock()],
                [MagicMock(), MagicMock(), MagicMock()],
            ]
            _show_overlay_gallery(results)

        self.assertEqual(streamlit.columns.call_count, 2)
        self.assertTrue(all(call.args == (3,) for call in streamlit.columns.call_args_list))
        self.assertEqual(streamlit.image.call_count, 6)
        self.assertEqual(
            [call.args[0] for call in streamlit.caption.call_args_list[1:]],
            ["Original X-ray", *methods],
        )
        self.assertTrue(all(call.args[0].size == (672, 672) for call in streamlit.image.call_args_list))

        with ZipFile(BytesIO(_xai_gallery_images_zip(results))) as bundle:
            self.assertEqual(len(bundle.namelist()), 6)
            self.assertEqual(bundle.namelist()[0], "01_original_xray.png")
            self.assertTrue(all(Image.open(BytesIO(bundle.read(name))).size == (224, 224)
                                for name in bundle.namelist()))


if __name__ == "__main__":
    unittest.main()
