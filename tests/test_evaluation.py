"""Synthetic-only checks for V2 final-evaluation guards and metric definitions."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.cadvit import evaluation
from reproduction.cadvit.model import ARCH_CONFIG, CADViTCandidate


class EvaluationTests(unittest.TestCase):
    def test_five_class_metrics_and_orientation(self):
        result = evaluation.classification_metrics([0, 1, 2, 3, 4], [0, 1, 1, 3, 4])
        self.assertEqual(result["confusion_matrix"][2][1], 1)
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["accuracy"], 0.8)
        self.assertAlmostEqual(result["f1_macro"], 11 / 15)
        self.assertAlmostEqual(result["specificity_macro"], 0.95)
        self.assertEqual(result["per_class"]["2"]["recall"], 0.0)
        with self.assertRaises(ValueError):
            evaluation.classification_metrics([0], [5])

    def test_preprocessing_source_constants(self):
        config = evaluation.preprocessing_config_from_source()
        self.assertEqual(config["PREPROCESS_CONFIG"]["image_size"], 512)
        self.assertEqual(config["BG_V2_CONFIG"]["background_sigma"], 10.0)

    def test_manifest_is_group_safe_without_image_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            image_root = root / "APTOS2019"
            image_root.mkdir()
            manifest = root / "APTOS2019_seed42.csv"
            n = 3662
            targets = (round(0.8 * n), round(0.1 * n))
            splits = (["train"] * targets[0] + ["val"] * targets[1]
                      + ["test"] * (n - sum(targets)))
            counts = {part: [0] * 5 for part in ("train", "val", "test")}
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_id", "label", "image_path",
                                                          "group_id", "sha256", "split"])
                writer.writeheader()
                for i, split in enumerate(splits):
                    label = i % 5
                    counts[split][label] += 1
                    writer.writerow({"image_id": f"id{i}", "label": label,
                                     "image_path": f"unopened{i}.png", "group_id": f"group{i}",
                                     "sha256": f"{i:064x}", "split": split})
            digest = evaluation.file_sha256(manifest)
            metadata = {"dataset": "APTOS2019", "seed": 42,
                        "manifest_file_sha256": digest, "manifest_sha256": digest,
                        "root": str(image_root.resolve()),
                        "counts": {f"{part}:{label}": value for part, row in counts.items()
                                   for label, value in enumerate(row)}}
            manifest.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            result = evaluation.validate_manifest(manifest, "APTOS2019", image_root, digest)
            self.assertEqual(result["split_sizes"], {"train": 2930, "val": 366, "test": 366})
            # Deliberately duplicate one group across train and validation; no image is opened.
            text = manifest.read_text(encoding="utf-8").replace("group2930", "group0", 1)
            manifest.write_text(text, encoding="utf-8")
            changed = evaluation.file_sha256(manifest)
            metadata["manifest_file_sha256"] = changed
            metadata["manifest_sha256"] = changed
            manifest.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bocor antar split"):
                evaluation.validate_manifest(manifest, "APTOS2019", image_root, changed)

    def test_checkpoint_metadata_uses_validation_best(self):
        model = CADViTCandidate()
        config = evaluation.preprocessing_config_from_source()
        record = {"pipeline_version": "V2_BG10_STABLE",
                  "experiment": {"dataset": "APTOS2019", "seed": 42, "epochs": 80,
                                 "physical_batch_size": 16, "accumulation": 1,
                                 "balance": False, "augment": True,
                                 "ablation": "MFB+Attention", "learning_rate": 0.000186},
                  "preprocessing": {"base": config["PREPROCESS_CONFIG"],
                                    "ben_graham": config["BG_V2_CONFIG"]},
                  "manifest_sha256": "a" * 64, "model": ARCH_CONFIG,
                  "parameter_count": 5_282_101}
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best = {"record": record, "model": state, "epoch": 68,
                "best_f1": evaluation.RUNS["APTOS2019"]["best_f1"]}
        last = {**best, "epoch": 79}
        checked = evaluation.validate_checkpoint_pair(best, last, "APTOS2019", "a" * 64)
        self.assertEqual(checked["best_epoch"], 69)
        with self.assertRaisesRegex(ValueError, "Epoch best/last"):
            evaluation.validate_checkpoint_pair({**best, "epoch": 64}, last, "APTOS2019", "a" * 64)


if __name__ == "__main__":
    unittest.main()
