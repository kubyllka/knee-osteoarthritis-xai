import unittest
from pathlib import Path

from app.research.manifest import (
    DatasetManifestValidationError,
    load_dataset_manifest,
    parse_dataset_manifest,
)


class DatasetManifestTests(unittest.TestCase):
    def test_demo_manifest_loads_all_local_images(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        manifest = load_dataset_manifest(project_root / "inputs" / "manifest.json")

        self.assertEqual(len(manifest), 6)
        self.assertEqual([sample.true_kl for sample in manifest], [0, 1, 1, 2, 3, 4])
        self.assertTrue(all(sample.image_path.is_file() for sample in manifest))

    def test_invalid_kl_is_rejected(self) -> None:
        payload = {
            "samples": [
                {"sample_id": "example", "image_path": "inputs/kl-0-1.png", "true_kl": 5}
            ]
        }

        with self.assertRaisesRegex(DatasetManifestValidationError, "true_kl"):
            parse_dataset_manifest(payload)

    def test_duplicate_sample_id_is_rejected(self) -> None:
        payload = {
            "samples": [
                {"sample_id": "example", "image_path": "inputs/kl-0-1.png", "true_kl": 0},
                {"sample_id": "example", "image_path": "inputs/kl-1-1.png", "true_kl": 1},
            ]
        }

        with self.assertRaisesRegex(DatasetManifestValidationError, "Duplicate sample_id"):
            parse_dataset_manifest(payload)


if __name__ == "__main__":
    unittest.main()
