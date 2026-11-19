"""Synthetic CPU-only E5 data/cache tests. No KC data or GPU is accessed."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from novelty import foundation_adaptation as fa
from novelty_integrated.e5 import config, input_pipeline, runner, validation_cache


class SyntheticProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        pixels = []
        for image in images:
            image = image.resize((224, 224), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
            pixels.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        return {"pixel_values": torch.stack(pixels)}


def rows_in(root, split, count=5):
    result = []
    for index in range(count):
        base = np.arange(40 * 40, dtype=np.uint16).reshape(40, 40)
        image = np.stack([((base + index * 10) % 255).astype(np.uint8),
                          ((base // 2 + 45) % 255).astype(np.uint8),
                          ((base // 3 + 80) % 255).astype(np.uint8)], axis=2)
        path = root / f"{split}_{index}.png"
        Image.fromarray(image).save(path)
        result.append({"split": split, "label": index % 5, "image_id": f"{split}_{index}",
                       "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


class E5InputTests(unittest.TestCase):
    def test_decoded_lru_exactness_bounds_and_source_invalidation(self):
        with TemporaryDirectory() as temp:
            rows = rows_in(Path(temp), "train")
            cache = input_pipeline.DecodedRGB(max_bytes=40 * 40 * 3)
            original = np.asarray(cache.get(rows[0])).copy()
            self.assertEqual(cache.misses, 1)
            self.assertLessEqual(cache.resident_bytes, 40 * 40 * 3)
            np.testing.assert_array_equal(np.asarray(cache.get(rows[0])), original)
            self.assertEqual(cache.hits, 1)
            cache.get(rows[1])
            self.assertLessEqual(cache.resident_bytes, 40 * 40 * 3)
            path = Path(rows[0]["path"])
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                cache.get(rows[0])
            with self.assertRaises(ValueError):
                cache.get({**rows[1], "split": "test"})

    def test_training_input_matches_b1_and_changes_by_epoch(self):
        with TemporaryDirectory() as temp:
            train = rows_in(Path(temp), "train")
            processor = SyntheticProcessor()
            loader = input_pipeline.InputLoaders(
                {"train": train, "val": rows_in(Path(temp), "val")}, processor,
                seed=42, micro_batch=3, decoded_cache_mib=1)
            outputs = []
            for epoch in (1, 2):
                batches = list(loader.train_epoch(epoch))
                order = np.random.default_rng(42 + epoch).permutation(len(train))
                for batch_no, (pixels, labels) in enumerate(batches):
                    selected = [train[i] for i in order[batch_no * 3:batch_no * 3 + 3]]
                    expected, target = fa._image_batch(selected, processor,
                                                       augment_seed=42, augment_epoch=epoch)
                    self.assertTrue(torch.equal(pixels, expected))
                    self.assertTrue(torch.equal(labels, target))
                outputs.append({train[i]["image_id"]: pixels[j] for batch_no, (pixels, _) in
                                enumerate(batches) for j, i in enumerate(order[batch_no * 3:batch_no * 3 + 3])})
            self.assertTrue(any(not torch.equal(outputs[0][key], outputs[1][key])
                                for key in outputs[0]))
            self.assertGreater(loader.cache_counts()["train_rgb_hits"], 0)

    def test_persistent_workers_keep_order_and_epoch_augmentation(self):
        with TemporaryDirectory() as temp:
            rows = {"train": rows_in(Path(temp), "train", 8),
                    "val": rows_in(Path(temp), "val", 5)}
            processor = SyntheticProcessor()
            loaders = input_pipeline.InputLoaders(
                rows, processor, seed=42, micro_batch=3, num_workers=2,
                persistent_workers=True, prefetch_factor=2, decoded_cache_mib=1)
            for epoch in (1, 2):
                batches = list(loaders.train_epoch(epoch))
                order = np.random.default_rng(42 + epoch).permutation(len(rows["train"]))
                for batch_no, (pixels, labels) in enumerate(batches):
                    selected = [rows["train"][i] for i in order[batch_no * 3:batch_no * 3 + 3]]
                    expected, target = fa._image_batch(selected, processor,
                                                       augment_seed=42, augment_epoch=epoch)
                    self.assertTrue(torch.equal(pixels, expected))
                    self.assertTrue(torch.equal(labels, target))
            self.assertEqual(sum(len(labels) for _, labels in loaders.validation()), 5)
            counts = loaders.cache_counts()
            self.assertEqual(counts["train_rgb_hits"] + counts["train_rgb_misses"], 16)
            self.assertEqual(counts["validation_rgb_misses"], 5)

    def test_validation_cache_exact_roundtrip_and_corruption(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            val = rows_in(root, "val")
            meta = {"schema": "foundation-official-input-v1", "e5_schema": validation_cache.SCHEMA,
                    "allowed_split": "val", "verification_batch_size": 3,
                    "source_sha256": "a" * 64}
            processor = SyntheticProcessor()
            destination = root / "cache"
            report = validation_cache.build(destination, meta, val, processor,
                                            max_gib=.01, reserve_gib=0)
            self.assertEqual(report["row_count"], 5)
            cache = validation_cache.open_complete(destination, meta, val)
            linear = torch.nn.Linear(3 * 224 * 224, 5)
            for row in val:
                expected, _ = fa._image_batch([row], processor)
                self.assertTrue(torch.equal(cache.read(row), expected[0]))
                self.assertTrue(torch.equal(linear(cache.read(row).flatten()[None]),
                                            linear(expected.flatten()[None])))
            with self.assertRaises(FileExistsError):
                validation_cache.build(destination, meta, val, processor,
                                       max_gib=.01, reserve_gib=0)
            with self.assertRaises(ValueError):
                validation_cache.open_complete(destination, {**meta, "source_sha256": "b" * 64}, val)
            data_path, _ = cache._paths(val[0])
            with data_path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                validation_cache.open_complete(destination, meta, val)

    def test_validation_cache_disk_guard(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            val = rows_in(root, "val")
            with self.assertRaises(ValueError):
                validation_cache.build(root / "cache", {"verification_batch_size": 3},
                                       val, SyntheticProcessor(), max_gib=.00001,
                                       reserve_gib=0)

    def test_actual_image_input_gate_is_cpu_only_and_train_val_only(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = {"train": rows_in(root, "train"), "val": rows_in(root, "val")}
            manifest = {"dataset": "APTOS2019", "manifest_sha256": "a" * 64}
            args = SimpleNamespace(output_dir=str(root / "input_preflight"),
                                   artifact_root=str(root), model_dir=str(root / "official"),
                                   validation_cache_root=None, dataset="APTOS2019",
                                   manifest_source_root=None, revision=config.OFFICIAL_SNAPSHOT["revision"],
                                   weight_sha256=config.OFFICIAL_SNAPSHOT["weight_sha256"],
                                   micro_batch=4, seed=42, decoded_cache_mib=1,
                                   sample_per_grade=1)
            with (patch.object(runner.data, "load_train_val", return_value=(rows, manifest)),
                  patch.object(fa, "audit_snapshot", return_value=config.OFFICIAL_SNAPSHOT.copy()),
                  patch.object(fa, "load_local_processor", return_value=SyntheticProcessor())):
                runner.verify_inputs(args)
            report = json.loads((root / "input_preflight" / "input_equivalence.json").read_text())
            self.assertEqual(report["status"], "ACTUAL_INPUT_EQUIVALENCE_PASS")
            self.assertEqual((report["train_samples"], report["validation_samples"]), (5, 5))
            self.assertEqual(report["final_test_status"], "NOT_ACCESSED")


if __name__ == "__main__":
    unittest.main()
