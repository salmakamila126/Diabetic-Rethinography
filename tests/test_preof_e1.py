"""Synthetic validation-cache checks for read-only PReOF E1 analysis."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np

from novelty import preof_e1 as e1
from novelty import preof_phase1 as phase


TRUTH = [0, 1, 2, 3, 4, 1, 3, 2]
PRED = {"p11": [0, 1, 2, 2, 4, 0, 3, 1],
        "p12": [0, 2, 2, 3, 4, 1, 2, 1],
        "p21": [1, 1, 2, 3, 4, 1, 3, 2],
        "p22": [0, 0, 2, 3, 4, 1, 4, 2]}


def synthetic_cache():
    rows = []
    for i, grade in enumerate(TRUTH):
        item = {"image_id": f"v{i}", "true_grade": grade, "group_id": f"g{i // 2}",
                "split": "val", "image_sha256": f"{i + 1:x}" * 64}
        for key in phase.KEYS:
            probability = np.full(5, 0.02)
            probability[PRED[key][i]] = 0.92
            item[key] = {"probabilities": probability.tolist(),
                         "logits": np.log(probability).tolist()}
        rows.append(item)
    return rows


class E1Tests(unittest.TestCase):
    def test_metrics_transitions_oracle_and_bootstrap(self):
        rows = synthetic_cache()
        report = e1.analyze(rows, "APTOS2019", {"synthetic": True}, bootstrap=40)
        self.assertEqual(report["sample_count"], 8)
        self.assertEqual(report["conditions"]["p11"]["correct"], 5)
        self.assertEqual(report["conditions"]["p22"]["correct"], 6)
        self.assertEqual(report["conditions"]["p11"]["grade_mae"], 3 / 8)
        self.assertEqual(report["conditions"]["p11"]["high_confidence_wrong"], 3)
        self.assertEqual(sum(map(sum, report["conditions"]["p11"]["confusion_matrix"])), 8)
        self.assertEqual(report["within_model"]["V1_within_model"]["correct_to_wrong"], 2)
        self.assertEqual(report["within_model"]["V1_within_model"]["wrong_to_correct"], 2)
        self.assertEqual(report["within_model"]["V2_within_model"]["wrong_to_correct"], 2)
        self.assertEqual(report["expert_complementarity"]["native_V1_only_correct"], 2)
        self.assertEqual(report["expert_complementarity"]["native_V2_only_correct"], 3)
        self.assertEqual(report["expert_complementarity"]["native_oracle_accuracy"], 1)
        for name, data in report["within_model"].items():
            self.assertEqual(sum(map(sum, data["prediction_transition_matrix_native_rows_cross_columns"])), 8,
                             name)
            self.assertEqual(data["both_correct"] + data["correct_to_wrong"] +
                             data["wrong_to_correct"] + data["both_wrong"], 8)
        intervals = report["paired_group_bootstrap"]["intervals"]
        self.assertAlmostEqual(intervals["V2_within_model_accuracy_delta"]["point"], 1 / 8)
        self.assertEqual(intervals, e1.analyze(rows, "APTOS2019", {}, bootstrap=40)["paired_group_bootstrap"]["intervals"])

    def test_cache_provenance_full_validation_and_report_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "datasets/APTOS2019"
            dataset_root.mkdir(parents=True)
            manifest = root / "cad_vit_runs/manifests/APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            rows = synthetic_cache()
            manifest_rows = [{"image_id": row["image_id"], "label": str(row["true_grade"]),
                              "image_path": f'missing_{row["image_id"]}.png',
                              "group_id": row["group_id"], "sha256": row["image_sha256"],
                              "split": "val"} for row in rows]
            manifest_rows.append({**manifest_rows[0], "image_id": "test_image", "split": "test",
                                  "image_path": "missing_test.png"})
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            spec = {**phase.SPECS["APTOS2019"], "val_count": 8,
                    "manifest_sha256": phase.sha256(manifest)}
            manifest.with_suffix(".json").write_text(json.dumps({
                "dataset": "APTOS2019", "seed": 42, "root": str(dataset_root),
                "manifest_file_sha256": spec["manifest_sha256"]}), encoding="utf-8")
            source = phase.REPO / "reproduction/archive/CAD_ViT_original_105_cells.ipynb"
            provenance = {"checkpoint_sha256": {v: spec[f"{v}_sha256"] for v in ("v1", "v2")},
                          "checkpoint_epoch": {v: spec[f"{v}_epoch"] for v in ("v1", "v2")},
                          "checkpoint_record_sha256": {v: "a" * 64 for v in ("v1", "v2")},
                          "v2_historical_source": {"path": str(source), "sha256": phase.sha256(source)},
                          "model_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/model.py"),
                          "preprocessing_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/preprocessing.py")}
            cache_dir = root / "prediction_cache"
            cache_dir.mkdir()
            metadata = {"schema": phase.CACHE_VERSION, "dataset": "APTOS2019", "split": "val",
                        "validation_count": 8, "cache_count": 8,
                        "manifest_sha256": spec["manifest_sha256"], "provenance": provenance,
                        "implementation_sha256": phase.sha256(phase.__file__),
                        "image_order_sha256": __import__("hashlib").sha256(
                            "\n".join(row["image_id"] for row in rows).encode()).hexdigest()}
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            cache_path = cache_dir / "cache.jsonl"
            cache_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = SimpleNamespace(dataset="APTOS2019", artifact_root=root, cache_dir=cache_dir,
                                   output_dir=root / "e1_output", high_confidence=0.9,
                                   bootstrap=40, seed=42)
            with patch.dict(phase.SPECS, {"APTOS2019": spec}):
                loaded, hashes = e1.load_verified_cache("APTOS2019", root, cache_dir)
                self.assertEqual(len(loaded), 8)
                self.assertEqual(hashes["cache_jsonl_sha256"], phase.sha256(cache_path))
                e1.run(args)
                result = json.loads((args.output_dir / "e1_results.json").read_text(encoding="utf-8"))
                self.assertEqual(result["sample_count"], 8)
                self.assertNotIn("image_id", (args.output_dir / "e1_results.json").read_text(encoding="utf-8"))
                self.assertIn("E1_COMPLETE", (args.output_dir / "e1_report.md").read_text(encoding="utf-8"))
                for figure in ("confusion_matrices.svg", "grade_transitions.svg",
                               "per_grade_accuracy_delta.svg"):
                    ElementTree.parse(args.output_dir / figure)
                with self.assertRaises(FileExistsError):
                    e1.run(args)
                cache_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    e1.load_verified_cache("APTOS2019", root, cache_dir)

    def test_reject_bad_threshold_or_too_few_groups(self):
        rows = synthetic_cache()
        with self.assertRaisesRegex(ValueError, "threshold"):
            e1.analyze(rows, "DDR", {}, high_confidence=0)
        for row in rows:
            row["group_id"] = "same"
        with self.assertRaisesRegex(ValueError, "two validation groups"):
            e1.analyze(rows, "DDR", {}, bootstrap=40)


if __name__ == "__main__":
    unittest.main()
