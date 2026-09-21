
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "inputs" / "test"
OUTPUT_FILE = PROJECT_ROOT / "test_manifest.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

samples = []

for kl in range(5):
    class_dir = TEST_DIR / str(kl)

    if not class_dir.is_dir():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    images = sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for index, image_path in enumerate(images, start=1):
        samples.append({
            "sample_id": f"kl-{kl}-{index}",
            "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "true_kl": kl,
            "split": "test",
        })

manifest = {
    "description": "Knee X-ray test samples. Paths are relative to the project root.",
    "samples": samples,
}

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2, ensure_ascii=False)

print(f"Manifest saved: {OUTPUT_FILE}")
print(f"Total test images: {len(samples)}")

for kl in range(5):
    count = sum(sample["true_kl"] == kl for sample in samples)
    print(f"KL{kl}: {count} images")