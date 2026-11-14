"""Synthetic-only safety and statistical checks for PReOF Phase 1."""

import csv
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from novelty import preof_phase1 as phase


def write_manifest_metadata(manifest, dataset_root):
    manifest.with_suffix(".json").write_text(json.dumps({
        "dataset": "APTOS2019", "seed": 42,
        "manifest_file_sha256": phase.sha256(manifest), "root": str(dataset_root.resolve())
    }), encoding="utf-8")


class PreofPhase1Tests(unittest.TestCase):
    def test_preprocessing_cache_exact_pixels_variants_and_invalidation(self):
        from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2

        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image_path = images / "example.png"
            pixels = np.zeros((64, 64, 3), dtype=np.uint8)
            pixels[8:56, 8:56] = (175, 90, 35)
            Image.fromarray(pixels).save(image_path)
            digest = phase.sha256(image_path)
            cache = phase.PreprocessingCache(root / "preprocessed", images)
            with Image.open(image_path) as image:
                direct = (preprocess_fundus(image), preprocess_fundus_v2(image))
            first = (cache.get(image_path, digest, "v1"),
                     cache.get(image_path, digest, "v2_bg10"))
            second = (cache.get(image_path, digest, "v1"),
                      cache.get(image_path, digest, "v2_bg10"))
            for expected, generated, loaded in zip(direct, first, second):
                self.assertEqual(generated.shape, (512, 512, 3))
                self.assertEqual(generated.dtype, np.uint8)
                np.testing.assert_array_equal(generated, expected)
                np.testing.assert_array_equal(loaded, expected)
            self.assertEqual(cache.stats["misses"], 2)
            self.assertEqual(cache.stats["hits"], 2)
            self.assertEqual(cache.stats["writes"], 2)
            v1_key, payload = cache.key(digest, "v1")
            v2_key, _ = cache.key(digest, "v2_bg10")
            self.assertNotEqual(v1_key, v2_key)
            self.assertNotIn("checkpoint", json.dumps(payload).lower())
            cache.config["v1"] = {"base": {"changed": True}}
            self.assertNotEqual(v1_key, cache.key(digest, "v1")[0])
            cache.config["v1"] = payload["config"]
            cache.source_sha256 = "b" * 64
            self.assertNotEqual(v1_key, cache.key(digest, "v1")[0])
            cache.source_sha256 = payload["preprocessing_source_sha256"]
            cache.versions = {**cache.versions, "opencv": "different"}
            self.assertNotEqual(v1_key, cache.key(digest, "v1")[0])
            cache.versions = payload["libraries"]
            pixels[10, 10] = (1, 2, 3)
            Image.fromarray(pixels).save(image_path)
            changed_digest = phase.sha256(image_path)
            self.assertNotEqual(digest, changed_digest)
            self.assertNotEqual(v1_key, cache.key(changed_digest, "v1")[0])
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                cache.get(image_path, digest, "v1")

    def test_preprocessing_cache_corruption_and_atomic_replacement(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image_path = images / "example.png"
            Image.fromarray(np.full((48, 48, 3), 110, dtype=np.uint8)).save(image_path)
            digest = phase.sha256(image_path)
            cache = phase.PreprocessingCache(root / "preprocessed", images)
            original_replace = phase.os.replace
            replacements = []

            def observed_replace(source, target):
                self.assertTrue(Path(source).is_file())
                replacements.append(Path(target).suffix)
                return original_replace(source, target)

            with patch.object(phase.os, "replace", side_effect=observed_replace):
                expected = cache.get(image_path, digest, "v1")
            self.assertEqual(replacements, [".npy", ".json"])
            data_path, meta_path, _, _ = cache._paths(digest, "v1")
            self.assertFalse(any(".tmp" in path.name for path in data_path.parent.iterdir()))
            data_path.write_bytes(b"invalid npy")
            np.testing.assert_array_equal(cache.get(image_path, digest, "v1"), expected)
            meta_path.write_text("{bad json", encoding="utf-8")
            np.testing.assert_array_equal(cache.get(image_path, digest, "v1"), expected)
            self.assertEqual(cache.stats["invalid_regenerated"], 2)
            self.assertEqual(cache.stats["writes"], 3)
            self.assertGreater(cache.disk_bytes(), 0)

    def test_prediction_metadata_transport_change_is_compatible(self):
        expected = {"schema": phase.CACHE_VERSION, "provenance": {"checkpoint_sha256": "a"},
                    "implementation_sha256": "a" * 64}
        old = {**expected, "implementation_sha256": "b" * 64}
        self.assertTrue(phase._prediction_meta_compatible(old, expected))
        self.assertFalse(phase._prediction_meta_compatible({**old, "provenance": {}}, expected))
        self.assertFalse(phase._prediction_meta_compatible({**old, "implementation_sha256": "bad"}, expected))

    def test_complete_prediction_cache_static_preflight_checks_checkpoint_and_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {**phase.SPECS["APTOS2019"], "val_count": 1,
                    "v1_run": "checkpoints/v1", "v2_run": "checkpoints/v2"}
            for version in ("v1", "v2"):
                path = root / spec[f"{version}_run"] / "best.pt"
                path.parent.mkdir(parents=True)
                path.write_bytes(version.encode())
                spec[f"{version}_sha256"] = phase.sha256(path)
            source_path = phase.REPO / "reproduction/archive/CAD_ViT_original_105_cells.ipynb"
            source = phase.verify_v2_source(source_path)
            provenance = {"checkpoint_sha256": {v: spec[f"{v}_sha256"] for v in ("v1", "v2")},
                          "checkpoint_epoch": {v: spec[f"{v}_epoch"] for v in ("v1", "v2")},
                          "v2_historical_source": source,
                          "model_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/model.py"),
                          "preprocessing_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/preprocessing.py")}
            row = {"image_id": "v0", "label": "0", "group_id": "g0", "sha256": "a" * 64}
            logits = np.arange(5, dtype=float)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            item = {"image_id": "v0", "true_grade": 0, "group_id": "g0",
                    "split": "val", "image_sha256": row["sha256"]}
            for key in phase.KEYS:
                item[key] = {"logits": logits.tolist(), "probabilities": probabilities.tolist()}
            out = root / "prediction_cache"
            out.mkdir()
            args = SimpleNamespace(dataset="APTOS2019", v2_source=source_path)
            with patch.dict(phase.SPECS, {"APTOS2019": spec}):
                metadata = phase._cache_meta("APTOS2019", [row], provenance, torch.device("cpu"))
                metadata["implementation_sha256"] = "b" * 64
                (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                (out / "cache.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
                self.assertTrue(phase._complete_prediction_cache(out, args, [row], torch.device("cpu"), root))
                (root / spec["v1_run"] / "best.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    phase._complete_prediction_cache(out, args, [row], torch.device("cpu"), root)

    def test_server_source_ast_audit_accepts_archived_definitions(self):
        source = phase.REPO / "reproduction/archive/CAD_ViT_original_105_cells.ipynb"
        report = phase.verify_v2_source(source)
        self.assertEqual(report["sha256"], phase.sha256(source))
        self.assertEqual(len(report["definitions"]), 11)

    def test_cumulative_boundaries_and_cross_preprocessing_features(self):
        base = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        np.testing.assert_allclose(phase.cumulative(base[None]), [[0.9, 0.7, 0.4, 0.2]])
        cache = []
        for index in range(4):
            item = {key: {"probabilities": base.tolist()} for key in phase.KEYS}
            item["p12"]["probabilities"] = np.roll(base, index % 2).tolist()
            item["p21"]["probabilities"] = np.roll(base, 1).tolist()
            cache.append(item)
        features = phase.feature_sets(cache, 1, "V1")
        self.assertEqual({key: value.shape for key, value in features.items()},
                         {"A": (4, 4), "B": (4, 6), "C": (4, 8), "D": (4, 20)})
        self.assertEqual(features["C"][0, -2], 0)
        self.assertGreater(features["C"][1, -2], 0)

    def test_group_folds_and_train_only_estimator(self):
        groups = ["a", "a", "b", "b", "c", "d", "e", "f", "g", "h"]
        folds = phase.grouped_folds(groups, count=4, seed=42)
        np.testing.assert_array_equal(folds, phase.grouped_folds(groups, count=4, seed=42))
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[2], folds[3])
        for fold in range(4):
            held = {g for g, f in zip(groups, folds) if f == fold}
            train = {g for g, f in zip(groups, folds) if f != fold}
            self.assertFalse(held & train)
        x = np.array([[-3], [-2], [-1], [1], [2], [3]], float)
        y = np.array([0, 0, 0, 1, 1, 1])
        pred = phase.fit_logistic(x, y, np.array([[-2], [2]], float))
        self.assertLess(pred[0], 0.5)
        self.assertGreater(pred[1], 0.5)

    def test_metrics_and_group_bootstrap_are_paired(self):
        y = np.array([0, 0, 1, 1] * 10)
        weak = np.array([0.4, 0.6, 0.4, 0.6] * 10)
        strong = np.array([0.1, 0.2, 0.8, 0.9] * 10)
        b, c = phase.metric_bundle(y, weak), phase.metric_bundle(y, strong)
        self.assertGreater(c["auroc"], b["auroc"])
        self.assertGreater(c["auprc"], b["auprc"])
        self.assertLess(c["brier"], b["brier"])
        interval = phase.paired_group_bootstrap(y, weak, strong,
                                                np.repeat(np.arange(10), 4), repeats=80)
        self.assertGreater(interval["auprc"]["ci95"][0], 0)
        self.assertLess(interval["brier"]["ci95"][1], 0)

    def test_cache_validation_rejects_stale_or_truncated_rows(self):
        row = {"image_id": "example", "label": "2", "group_id": "g", "sha256": "a" * 64}
        logits = np.arange(5, dtype=float)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        item = {"image_id": "example", "true_grade": 2, "group_id": "g",
                "split": "val", "image_sha256": "a" * 64}
        for key in phase.KEYS:
            item[key] = {"logits": logits.tolist(), "probabilities": probs.tolist()}
        phase.validate_cache_record(item, row)
        with self.assertRaises(ValueError):
            phase.validate_cache_record({**item, "split": "test"}, row)
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.jsonl"
            cache_path.write_bytes(json.dumps(item).encode())  # no final newline
            with self.assertRaisesRegex(ValueError, "Truncated"):
                phase.read_cache(directory, [row])

    def test_synthetic_validation_cache_smoke_and_resume(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, image):
                self.calls += 1
                average = image.mean(dim=(1, 2, 3))
                return torch.stack([average + 0.1 * grade for grade in range(5)], dim=1)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "datasets/APTOS2019"
            images.mkdir(parents=True)
            rows = []
            for i in range(3):
                pixels = np.zeros((48, 48, 3), dtype=np.uint8)
                pixels[5:-5, 5:-5] = (100 + i * 20, 60 + i * 20, 30 + i * 20)
                path = images / f"v{i}.png"
                Image.fromarray(pixels).save(path)
                rows.append({"image_id": f"v{i}", "label": str(i), "image_path": path.name,
                             "group_id": f"g{i}", "sha256": phase.sha256(path), "split": "val"})
            manifest = root / "cad_vit_runs/manifests/APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_manifest_metadata(manifest, images)
            spec = {**phase.SPECS["APTOS2019"], "manifest_sha256": phase.sha256(manifest), "val_count": 3}
            models = {"v1": TinyModel(), "v2": TinyModel()}
            args = SimpleNamespace(dataset="APTOS2019", artifact_root=root, v2_source="unused",
                                   trust_checkpoints=True, limit=3, command="cache", device="cpu",
                                   output_dir=root / "cache", resume=False,
                                   preprocess_cache_root=root / "preprocessed", uncached=False)
            with patch.dict(phase.SPECS, {"APTOS2019": spec}), \
                 patch.object(phase, "checked_models", return_value=(models, {"synthetic": True})):
                phase.cache_predictions(args)
                saved = phase.read_cache(args.output_dir, rows)
                self.assertEqual(len(saved), 3)
                self.assertEqual(models["v1"].calls, 6)
                self.assertEqual(models["v2"].calls, 6)
                args.resume = True
                with patch.object(phase, "_complete_prediction_cache", return_value=True):
                    phase.cache_predictions(args)
                self.assertEqual(models["v1"].calls, 6)
                self.assertEqual(models["v2"].calls, 6)
                args.output_dir = root / "uncached_cache"
                args.resume = False
                args.preprocess_cache_root = None
                args.uncached = True
                phase.cache_predictions(args)
                uncached = phase.read_cache(args.output_dir, rows)
                self.assertEqual(saved, uncached)
                self.assertEqual(models["v1"].calls, 12)
                self.assertEqual(models["v2"].calls, 12)
                benchmark_args = SimpleNamespace(dataset="APTOS2019", artifact_root=root,
                                                 preprocess_cache_root=root / "benchmark_preprocessed",
                                                 limit=3)
                output = io.StringIO()
                with redirect_stdout(output):
                    phase.benchmark_preprocessing(benchmark_args)
                benchmark = json.loads(output.getvalue())
                self.assertTrue(benchmark["pixel_exact"])
                self.assertEqual(benchmark["first_cache_pass"]["hit_rate"], 0)
                self.assertEqual(benchmark["warm_cache_pass"]["hit_rate"], 1)
                self.assertEqual(models["v1"].calls, 12)

    def test_manifest_parser_retains_validation_only(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            rows = [
                {"image_id": "v1", "label": "0", "image_path": "missing_v1.png", "group_id": "a", "sha256": "a" * 64, "split": "val"},
                {"image_id": "t1", "label": "4", "image_path": "missing_test.png", "group_id": "b", "sha256": "b" * 64, "split": "test"},
                {"image_id": "v2", "label": "2", "image_path": "missing_v2.png", "group_id": "c", "sha256": "c" * 64, "split": "val"},
            ]
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_manifest_metadata(manifest, Path(directory))
            test_spec = {**phase.SPECS["APTOS2019"], "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "val_count": 2}
            with patch.dict(phase.SPECS, {"APTOS2019": test_spec}):
                selected = phase.validation_rows(manifest, "APTOS2019", Path(directory))
            self.assertEqual([row["image_id"] for row in selected], ["v1", "v2"])

    def test_e0_writes_aggregate_report_from_complete_synthetic_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "datasets/APTOS2019").mkdir(parents=True)
            manifest = root / "cad_vit_runs/manifests/APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            rows = [{"image_id": f"v{i}", "label": str(i % 5), "image_path": f"v{i}.png",
                     "group_id": f"g{i // 2}", "sha256": f"{i % 16:x}" * 64, "split": "val"}
                    for i in range(40)]
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_manifest_metadata(manifest, root / "datasets/APTOS2019")
            spec = {**phase.SPECS["APTOS2019"], "manifest_sha256": phase.sha256(manifest), "val_count": 40}
            cache_dir = root / "cache"
            cache_dir.mkdir()
            saved = []
            for i, row in enumerate(rows):
                item = {"image_id": row["image_id"], "true_grade": int(row["label"]),
                        "group_id": row["group_id"], "split": "val", "image_sha256": row["sha256"]}
                for key, shift in (("p11", i % 3 == 0), ("p12", i % 3 == 1),
                                   ("p21", i % 4 == 1), ("p22", i % 4 == 0)):
                    p = np.full(5, 0.05)
                    p[(int(row["label"]) + int(shift)) % 5] = 0.8
                    item[key] = {"probabilities": p.tolist(), "logits": np.log(p).tolist()}
                saved.append(item)
            (cache_dir / "cache.jsonl").write_bytes(("\n".join(json.dumps(x) for x in saved) + "\n").encode())
            source = phase.REPO / "reproduction/archive/CAD_ViT_original_105_cells.ipynb"
            metadata = {"schema": phase.CACHE_VERSION, "dataset": "APTOS2019", "split": "val",
                        "validation_count": 40, "cache_count": 40, "manifest_sha256": spec["manifest_sha256"],
                        "implementation_sha256": phase.sha256(phase.__file__),
                        "image_order_sha256": hashlib.sha256("\n".join(r["image_id"] for r in rows).encode()).hexdigest(),
                        "provenance": {
                            "checkpoint_sha256": {v: spec[f"{v}_sha256"] for v in ("v1", "v2")},
                            "v2_historical_source": {"path": str(source), "sha256": phase.sha256(source)},
                            "model_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/model.py"),
                            "preprocessing_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/preprocessing.py")}}
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = SimpleNamespace(cache_dir=cache_dir, artifact_root=root, output_dir=root / "report",
                                   folds=5, bootstrap=30, seed=42)
            with patch.dict(phase.SPECS, {"APTOS2019": spec}):
                phase.analyze_e0(args)
            report = json.loads((root / "report/e0_report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["comparisons"]), 8)
            self.assertEqual(sum(report["grade_support"].values()), 40)
            self.assertEqual(sum(report["five_class_complementary_errors"].values()), 40)
            self.assertTrue((root / "report/e0_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
