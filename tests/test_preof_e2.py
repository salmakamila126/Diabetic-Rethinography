"""Synthetic-only E2 extraction, cache integrity, and exploratory analysis tests."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
from PIL import Image
import torch

from novelty import preof_e2 as e2
from novelty import preof_phase1 as phase
from reproduction.cadvit.model import CADViTCandidate


def _probabilities(grade):
    p = np.full(5, 0.025)
    p[grade] = 0.9
    return p


class E2Tests(unittest.TestCase):
    def test_audited_hooks_capture_pooled_features_and_preserve_forward(self):
        torch.set_num_threads(1)
        model = CADViTCandidate().eval()
        inputs = torch.rand(1, 3, 64, 64)
        pooled, logits = e2.pooled_forward(model, inputs)
        with torch.inference_mode():
            baseline = model(inputs)[0].numpy()
        self.assertEqual(pooled.shape, (608,))
        self.assertEqual(pooled.dtype, np.float32)
        np.testing.assert_allclose(logits, baseline, rtol=0, atol=0)
        self.assertFalse(any(module._forward_hooks for module in (model.stem, model.blocks[3],
                                                                 model.blocks[8], model.head)))
        head = pooled[-320:]
        with torch.inference_mode():
            torch.testing.assert_close(model.classifier(torch.from_numpy(head)),
                                       torch.from_numpy(logits), rtol=1e-5, atol=1e-5)

    def test_drift_definitions_and_exploratory_controls(self):
        a = np.asarray([[1., 0.], [0., 0.], [1., 1.]])
        b = np.asarray([[1., 0.], [0., 0.], [-1., -1.]])
        cosine, l2 = e2.cosine_and_normalized_l2(a, b)
        np.testing.assert_allclose(cosine, [0, 0, 2])
        np.testing.assert_allclose(l2, [0, 0, 2])
        self.assertAlmostEqual(e2.linear_cka(a, a), 1)
        self.assertAlmostEqual(e2.spearman([1, 2, 3], [3, 2, 1]), -1)
        cached, vectors = [], np.empty((80, 4, e2.DIM), np.float32)
        rng = np.random.default_rng(42)
        for i in range(80):
            grade = i % 5
            native_v1 = (grade + int(i % 4 == 0)) % 5
            native_v2 = (grade + int(i % 5 == 0)) % 5
            predictions = {"p11": native_v1, "p12": (native_v1 + int(i % 3 == 0)) % 5,
                           "p21": (native_v2 + int(i % 3 == 1)) % 5, "p22": native_v2}
            row = {"true_grade": grade, "group_id": f"g{i // 2}"}
            for key, pred in predictions.items():
                row[key] = {"probabilities": _probabilities(pred).tolist()}
            cached.append(row)
            native = rng.normal(size=e2.DIM).astype(np.float32)
            vectors[i, 0] = native
            vectors[i, 1] = native + rng.normal(scale=0.2, size=e2.DIM)
            vectors[i, 3] = rng.normal(size=e2.DIM)
            vectors[i, 2] = vectors[i, 3] + rng.normal(scale=0.2, size=e2.DIM)
        report = e2.analyze_arrays(cached, vectors, "APTOS2019", {"synthetic": True},
                                   folds=3, bootstrap=30, seed=42)
        self.assertEqual(report["status"], "E2_EXPLORATORY_COMPLETE")
        self.assertEqual(len(report["incremental_error_detection"]), 8)
        self.assertTrue(any(item["status"] == "OK" for item in report["incremental_error_detection"]))
        for model in ("V1", "V2"):
            self.assertEqual(set(report["profiles"][model]["layers"]), {name for name, _, _ in e2.LAYERS})
        for item in report["incremental_error_detection"]:
            if item["status"] == "OK":
                self.assertEqual(item["conditions"]["B"]["dimension"], 6)
                self.assertEqual(item["conditions"]["B+random2"]["dimension"], 8)
                self.assertEqual(item["conditions"]["B+all_layers"]["dimension"], 14)
                self.assertEqual(item["conditions"]["B+random8"]["dimension"], 14)
                self.assertEqual(item["conditions"]["D"]["dimension"], 20)
        for svg in (e2.profile_svg(report, "mean_cosine_distance", "cosine"),
                    e2.profile_svg(report, "linear_cka", "CKA"),
                    e2.per_grade_svg(report), e2.incremental_svg(report)):
            ElementTree.fromstring(svg)
        self.assertIn("Exploratory validation", e2.report_markdown(report))

    def test_validation_only_feature_cache_resume_logits_and_corruption(self):
        class FrozenTiny(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

        def fake_forward(model, inputs):
            mean = float(inputs.mean())
            return np.full(e2.DIM, mean + model.bias, np.float32), \
                np.asarray([mean + model.bias + k * 0.1 for k in range(5)], np.float32)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "datasets/APTOS2019"
            dataset_root.mkdir(parents=True)
            rows = []
            for i in range(2):
                path = dataset_root / f"v{i}.png"
                Image.fromarray(np.full((48, 48, 3), 75 + 50*i, np.uint8)).save(path)
                rows.append({"image_id": f"v{i}", "label": str(i), "group_id": f"g{i}",
                             "image_path": path.name, "sha256": phase.sha256(path), "split": "val"})
            rows.append({"image_id": "test", "label": "4", "group_id": "test_group",
                         "image_path": "missing_test.png", "sha256": "f" * 64, "split": "test"})
            manifest = root / "cad_vit_runs/manifests/APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            spec = {**phase.SPECS["APTOS2019"], "val_count": 2,
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
            prediction_root = root / "prediction"
            prediction_root.mkdir()
            metadata = {"schema": phase.CACHE_VERSION, "dataset": "APTOS2019", "split": "val",
                        "validation_count": 2, "cache_count": 2, "manifest_sha256": spec["manifest_sha256"],
                        "implementation_sha256": "a" * 64,
                        "image_order_sha256": hashlib.sha256("v0\nv1".encode()).hexdigest(),
                        "provenance": provenance}
            (prediction_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            prediction = []
            models = {"v1": FrozenTiny(0.0), "v2": FrozenTiny(0.2)}
            from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2
            for row in rows[:2]:
                with Image.open(dataset_root / row["image_path"]) as image:
                    pixels = {"v1": preprocess_fundus(image), "v2_bg10": preprocess_fundus_v2(image)}
                item = {"image_id": row["image_id"], "true_grade": int(row["label"]),
                        "group_id": row["group_id"], "split": "val", "image_sha256": row["sha256"]}
                for condition, model_name, variant in e2.CONDITIONS:
                    value = float(pixels[variant].astype(np.float32).mean() / 255) + models[model_name].bias
                    logits = np.asarray([value + k*0.1 for k in range(5)], np.float32)
                    exp = np.exp(logits - logits.max())
                    probs = exp / exp.sum()
                    item[condition] = {"logits": logits.tolist(), "probabilities": probs.tolist()}
                prediction.append(item)
            (prediction_root / "cache.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in prediction), encoding="utf-8")
            args = SimpleNamespace(dataset="APTOS2019", artifact_root=root,
                                   prediction_cache_dir=prediction_root,
                                   preprocess_cache_root=root / "preprocessed", v2_source=source,
                                   trust_checkpoints=True, output_dir=root / "features",
                                   device="cpu", limit=None, resume=False, atol=1e-4, rtol=1e-4)
            shared_preprocessing = phase.PreprocessingCache(args.preprocess_cache_root, dataset_root)
            with self.assertRaisesRegex(ValueError, "is miss"):
                e2._existing_preprocessed(shared_preprocessing, rows[0]["sha256"], "v1")
            self.assertFalse(args.preprocess_cache_root.exists())
            for row in rows[:2]:
                image_path = dataset_root / row["image_path"]
                for variant in ("v1", "v2_bg10"):
                    shared_preprocessing.get(image_path, row["sha256"], variant)
            with patch.dict(phase.SPECS, {"APTOS2019": spec}), \
                 patch.object(phase, "checked_models", return_value=(models, provenance)), \
                 patch.object(e2, "_layer_modules", return_value=()), \
                 patch.object(e2, "pooled_forward", side_effect=fake_forward):
                e2.extract(args)
                features, vectors, _ = e2.load_complete_features("APTOS2019", root, prediction_root, args.output_dir)
                self.assertEqual(len(features), 2)
                self.assertEqual(vectors.shape, (2, 4, e2.DIM))
                full_root = args.output_dir
                args.output_dir = root / "smoke"
                args.limit = 1
                e2.extract(args)
                self.assertEqual(len((args.output_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()), 1)
                with self.assertRaisesRegex(ValueError, "Incomplete|partial"):
                    e2.load_complete_features("APTOS2019", root, prediction_root, args.output_dir)
                args.output_dir = full_root
                args.limit = None
                args.resume = True
                with patch.object(phase, "checked_models", side_effect=AssertionError("model loaded")):
                    e2.extract(args)
                path = args.output_dir / "rows.jsonl"
                lines = path.read_text(encoding="utf-8").splitlines()
                path.write_text(lines[0] + "\n", encoding="utf-8")
                e2.extract(args)
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
                entry = json.loads(lines[0])
                vector_path = e2._feature_path(args.output_dir, entry["feature_key"])
                self.assertFalse(any(".tmp" in p.name for p in vector_path.parent.iterdir()))
                vector_path.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "digest differs"):
                    e2.extract(args)
                args.output_dir = root / "mismatched_logits"
                args.resume = False
                with patch.object(e2, "pooled_forward", side_effect=lambda m, x: (
                        fake_forward(m, x)[0], fake_forward(m, x)[1] + 0.1)):
                    with self.assertRaisesRegex(ValueError, "logits differ"):
                        e2.extract(args)


if __name__ == "__main__":
    unittest.main()
