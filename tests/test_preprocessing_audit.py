"""Synthetic, dataset-free tests for the preprocessing fidelity audit."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from reproduction.cadvit import preprocessing_audit as audit


ROOT = Path(__file__).resolve().parents[1]


def synthetic_v1(image):
    """Injected stub for audit orchestration tests; not a CAD-ViT replacement."""
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(oriented.resize((512, 512), Image.Resampling.BILINEAR)).copy()


def synthetic_v2(image):
    return np.clip(synthetic_v1(image).astype(np.int16) + 12, 0, 255).astype(np.uint8)


class PreprocessingAuditTests(unittest.TestCase):
    def test_documented_metrics_and_rgb_histograms(self):
        original = np.full((2, 3, 3), 100, dtype=np.uint8)
        changed = np.full((2, 3, 3), 110, dtype=np.uint8)
        images = {
            "original": original,
            "paper_described": original.copy(),
            "historical_v1": original.copy(),
            "historical_v2_bg10": changed,
        }
        rows, histograms = audit.image_metrics(images)
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]["brightness_luma_mean"], 100)
        self.assertAlmostEqual(rows[0]["contrast_luma_std_population"], 0)
        self.assertIsNone(rows[0]["pixel_mae_from_v1_aligned"])
        self.assertAlmostEqual(rows[2]["pixel_mae_from_v1_aligned"], 0)
        self.assertAlmostEqual(rows[3]["pixel_mae_from_v1_aligned"], 10)
        self.assertAlmostEqual(rows[3]["mean_rgb_distance_from_original"], np.sqrt(300))
        self.assertEqual(int(histograms["original"]["R"][100]), 6)
        self.assertEqual(int(histograms["historical_v2_bg10"]["B"][110]), 6)

    def test_train_manifest_guard_and_test_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            image = root / "fundus.png"
            Image.new("RGB", (31, 25), (30, 70, 130)).save(image)
            manifest = root / "manifest.csv"
            manifest.write_text("image_path,split\nfundus.png,train\n", encoding="utf-8")
            self.assertEqual(audit.validate_source(image, "train", manifest, root), image.resolve())
            with self.assertRaises(ValueError):
                audit.validate_source(image, "test", manifest, root)
            manifest.write_text("image_path,split\nfundus.png,test\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.validate_source(image, "train", manifest, root)
            with self.assertRaises(ValueError):
                audit.validate_source(image, "audit", manifest, root)
            manifest.write_text("image_path,split\n../fundus.png,train\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.validate_source(image, "train", manifest, root)

    def test_exif_comparison_and_saved_artifacts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary, \
                tempfile.TemporaryDirectory(dir=ROOT) as derived:
            root = Path(temporary)
            pixels = np.zeros((25, 41, 3), dtype=np.uint8)
            pixels[3:-3, 4:-4] = (40, 100, 190)
            pixels[7:15, 26:35] = (230, 60, 40)
            exif = Image.Exif()
            exif[274] = 6
            image_path = root / "oriented.jpg"
            Image.fromarray(pixels).save(image_path, format="JPEG", exif=exif)
            result = audit.audit_image(
                image_path, "audit", v1_fn=synthetic_v1, v2_fn=synthetic_v2
            )
            self.assertEqual(result["exif_orientation"], 6)
            self.assertEqual(result["images"]["original"].shape, (41, 25, 3))
            self.assertTrue(result["paper_v1_identical"])
            self.assertTrue(np.array_equal(
                result["images"]["paper_described"], result["images"]["historical_v1"]
            ))
            with self.assertRaises(ValueError):
                audit.save_audit(result, root / "derived")
            paths = audit.save_audit(result, derived)
            self.assertEqual(len(paths), 8)  # Four views, two figures, two CSVs.
            self.assertTrue(all(path.is_file() for path in paths.values()))
            with paths["metrics_csv"].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            with paths["histograms_csv"].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4 * 3 * 256)

    def test_corrupt_and_all_dark_images_fail_explicitly(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            corrupt = root / "corrupt.png"
            corrupt.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Unreadable or damaged image"):
                audit.audit_image(corrupt, "audit", v1_fn=synthetic_v1, v2_fn=synthetic_v2)
            dark = root / "dark.png"
            Image.new("RGB", (32, 24), (0, 0, 0)).save(dark)

            def historical_dark_error(_image):
                raise ValueError("Gambar seluruhnya gelap; crop tidak dapat ditentukan")

            with self.assertRaisesRegex(ValueError, "Gambar seluruhnya gelap"):
                audit.audit_image(dark, "audit", v1_fn=historical_dark_error, v2_fn=synthetic_v2)


if __name__ == "__main__":
    unittest.main()
