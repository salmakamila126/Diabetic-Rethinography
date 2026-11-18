"""CPU-only integrity and exact-equivalence tests for V2 deterministic cache."""
import json
import os
import subprocess
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novelty.cadvit_input_cache import (V2ImageCache, SHAPE, SCHEMA,
                                      ENTRY_BUDGET, FREE_RESERVE)  # noqa: E402
from novelty.cadvit_pilot import V2FundusSplit, _cache_from_record  # noqa: E402
from novelty.cadvit_training import file_sha256  # noqa: E402
from reproduction.cadvit.preprocessing import preprocess_fundus_v2  # noqa: E402


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "artifacts"
        self.output = self.root / "cached_runs"
        self.cache_root = self.root / "cache"
        self.image_dir = self.artifact / "datasets" / "APTOS2019"
        self.image_dir.mkdir(parents=True)
        self.rows = []
        for index in range(3):
            pixels = np.full((80, 88, 3), 25 + index * 17, dtype=np.uint8)
            pixels[8:70, 9:77] = [120 + index * 20, 45 + index * 17, 90]
            pixels[20:34, 30 + index:48 + index] = [230, 190, 50]
            path = self.image_dir / f"synthetic_{index}.png"
            Image.fromarray(pixels).save(path)
            self.rows.append({"path": str(path), "sha256": file_sha256(path),
                              "label": index, "split": "train" if index else "val",
                              "image_id": str(index)})
        self.cache = V2ImageCache(
            self.cache_root, self.artifact, "APTOS2019", "a" * 64, self.output)

    def test_exact_pixels_and_validation_tensors(self):
        row = self.rows[0]
        with Image.open(row["path"]) as image:
            direct = preprocess_fundus_v2(image)
        cached = self.cache.get(row)
        self.assertEqual(cached.shape, SHAPE)
        self.assertEqual(cached.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(cached, direct)
        split = {"val": [row]}
        uncached_tensor, _ = V2FundusSplit(split, "val")[0]
        cached_tensor, _ = V2FundusSplit(split, "val", self.cache)[0]
        torch.testing.assert_close(cached_tensor, uncached_tensor, atol=0, rtol=0)
        again, _ = V2FundusSplit(split, "val", self.cache)[0]
        torch.testing.assert_close(cached_tensor, again, atol=0, rtol=0)

    def test_augmentation_same_rng_and_stochastic_across_epochs(self):
        row = self.rows[1]
        split = {"train": [row]}
        direct = V2FundusSplit(split, "train")
        cached = V2FundusSplit(split, "train", self.cache)
        for seed in (13, 14, 15):
            torch.manual_seed(seed)
            expected, _ = direct[0]
            torch.manual_seed(seed)
            observed, _ = cached[0]
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        outputs = []
        for seed in (13, 14, 15):
            torch.manual_seed(seed)
            outputs.append(cached[0][0])
        self.assertTrue(any(not torch.equal(outputs[0], other)
                            for other in outputs[1:]))
        with Image.open(row["path"]) as image:
            np.testing.assert_array_equal(self.cache.get(row),
                                          preprocess_fundus_v2(image))

    def test_corrupt_data_and_metadata_are_repaired(self):
        row = self.rows[0]
        original = self.cache.get(row).copy()
        key, data, meta = self.cache._paths(row)
        self.assertEqual(json.loads(meta.read_text())["schema"], SCHEMA)
        data.write_bytes(b"corrupt")
        np.testing.assert_array_equal(self.cache.get(row), original)
        meta.write_bytes(bytes([255, 254, 253]))
        np.testing.assert_array_equal(self.cache.get(row), original)
        meta.write_text("[]", encoding="utf-8")
        np.testing.assert_array_equal(self.cache.get(row), original)
        data.write_bytes(b"")
        np.testing.assert_array_equal(self.cache.get(row), original)
        self.assertEqual(json.loads(meta.read_text())["key"], key)
        self.assertFalse(data.with_suffix(".lock").exists())

    def test_interrupted_pair_and_abandoned_writer_lock(self):
        row = self.rows[0]
        original = self.cache.get(row).copy()
        _, data, meta = self.cache._paths(row)
        lock = self.cache.directory.parent / ".writer.lock"
        self.assertTrue(lock.is_file())
        subprocess.run(
            [sys.executable, "-c",
             "import fcntl,os,sys; fd=os.open(sys.argv[1],os.O_RDWR); "
             "fcntl.flock(fd,fcntl.LOCK_EX); os._exit(0)", str(lock)],
            check=True)
        data.write_bytes(b"interrupted replacement")
        np.testing.assert_array_equal(self.cache.get(row), original)
        self.assertTrue(meta.is_file())
        self.assertTrue(lock.is_file())
        self.assertEqual(list(data.parent.glob(data.name + ".*.tmp")), [])

    def test_entry_budget_covers_allocated_files_and_temp_buffer(self):
        row = self.rows[0]
        self.cache.get(row)
        _, data, meta = self.cache._paths(row)
        allocated = (data.stat().st_blocks + meta.stat().st_blocks) * 512
        self.assertGreaterEqual(ENTRY_BUDGET, allocated + 4096)
        estimate = self.cache.ensure_space([row])
        self.assertEqual(estimate["missing_or_incomplete_entries"], 0)
        self.assertEqual(estimate["upper_bound_bytes"], ENTRY_BUDGET)

    def test_source_hash_guard_on_miss(self):
        row = dict(self.rows[0])
        pixels = np.full((80, 88, 3), 200, dtype=np.uint8)
        Image.fromarray(pixels).save(row["path"])
        with self.assertRaisesRegex(ValueError, "Source image differs"):
            self.cache.get(row)
        key, data, meta = self.cache._paths(row)
        self.assertFalse(data.exists())
        self.assertFalse(meta.exists())

    def test_bounded_warm_and_disk_guard(self):
        result = self.cache.warm(self.rows, 2, progress=2)
        self.assertEqual(result["checked"], 2)
        self.assertEqual(len(list(self.cache.directory.rglob("*.npy"))), 2)
        with self.assertRaises(ValueError):
            self.cache.warm(self.rows, 0)
        with self.assertRaises(ValueError):
            self.cache.warm(self.rows, 4)
        with self.assertRaises(ValueError):
            self.cache.warm(self.rows + [{"split": "test", "sha256": "b"*64}], 4)
        space = self.cache.ensure_space(self.rows)
        self.assertEqual(space["missing_or_incomplete_entries"], 1)
        with patch("novelty.cadvit_input_cache.shutil.disk_usage",
                   return_value=SimpleNamespace(free=FREE_RESERVE + 2 * ENTRY_BUDGET)):
            self.cache.ensure_space(self.rows)
        self.cache.get(self.rows[2])
        self.assertEqual(self.cache.ensure_space(self.rows)["missing_or_incomplete_entries"], 0)
        with patch("novelty.cadvit_input_cache.shutil.disk_usage",
                   return_value=SimpleNamespace(free=FREE_RESERVE + ENTRY_BUDGET - 1)):
            with self.assertRaises(OSError):
                self.cache.ensure_space(self.rows)

    def test_space_guard_uses_cache_filesystem_and_rejects_unwritable_root(self):
        # Artifact/output paths are intentionally independent of the cache root.
        external = self.root / "external-cache"
        external.mkdir()
        cache = V2ImageCache(external, self.artifact, "APTOS2019",
                             "a" * 64, self.output)
        row = self.rows[0]
        with patch("novelty.cadvit_input_cache.shutil.disk_usage",
                   return_value=SimpleNamespace(
                       free=FREE_RESERVE + 2 * ENTRY_BUDGET - 1)) as usage:
            with self.assertRaisesRegex(OSError, "Insufficient cache space"):
                cache.ensure_space([row])
        usage.assert_called_once_with(external)
        with patch("novelty.cadvit_input_cache.shutil.disk_usage",
                   return_value=SimpleNamespace(
                       free=FREE_RESERVE + 2 * ENTRY_BUDGET)) as usage:
            cache.ensure_space([row])
        usage.assert_called_once_with(external)
        missing = external / "not-yet-created"
        absent = V2ImageCache(missing, self.artifact, "APTOS2019",
                              "a" * 64, self.output)
        with patch("novelty.cadvit_input_cache.shutil.disk_usage",
                   return_value=SimpleNamespace(
                       free=FREE_RESERVE + 2 * ENTRY_BUDGET)) as usage:
            absent.ensure_space([row])
        usage.assert_called_once_with(external)
        self.assertFalse(missing.exists())
        with patch("novelty.cadvit_input_cache.os.access", return_value=False):
            with self.assertRaisesRegex(PermissionError, "not writable"):
                absent.ensure_space([row])

    def test_namespace_symlink_cannot_redirect_entries(self):
        external = self.root / "external-cache"
        outside = self.root / "outside-cache"
        external.mkdir()
        outside.mkdir()
        (external / "APTOS2019").symlink_to(outside, target_is_directory=True)
        cache = V2ImageCache(external, self.artifact, "APTOS2019",
                             "a" * 64, self.output)
        with self.assertRaisesRegex(ValueError, "escapes cache root"):
            cache.ensure_space([self.rows[0]])
        with self.assertRaisesRegex(ValueError, "escapes cache root"):
            cache.get(self.rows[0])
        self.assertEqual(list(outside.iterdir()), [])

    def test_cache_entry_lock_and_atomic_paths_stay_under_cache_root(self):
        external = self.root / "external-cache"
        external.mkdir()
        cache = V2ImageCache(external, self.artifact, "APTOS2019",
                             "a" * 64, self.output)
        with patch("novelty.cadvit_input_cache.os.replace",
                   wraps=os.replace) as replacements:
            cache.get(self.rows[0])
        self.assertEqual(replacements.call_count, 2)
        _, data, meta = cache._paths(self.rows[0])
        lock = cache.directory.parent / ".writer.lock"
        for path in (data, meta, lock):
            self.assertTrue(path.is_relative_to(external))
            self.assertEqual(path.stat().st_dev, external.stat().st_dev)
        for call in replacements.call_args_list:
            source, target = map(Path, call.args)
            self.assertTrue(source.is_relative_to(external))
            self.assertTrue(target.is_relative_to(external))
            self.assertEqual(source.parent, target.parent)

    def test_protocol_identity_and_paths(self):
        same = V2ImageCache(self.cache_root, self.artifact, "APTOS2019",
                            "a" * 64, self.output)
        changed = V2ImageCache(self.cache_root, self.artifact, "APTOS2019",
                               "b" * 64, self.output)
        self.assertEqual(self.cache.identity, same.identity)
        self.assertNotEqual(self.cache.identity, changed.identity)
        ddr = V2ImageCache(self.cache_root, self.artifact, "DDR",
                           "a" * 64, self.output)
        self.assertNotEqual(self.cache.identity, ddr.identity)
        self.assertNotEqual(self.cache.directory, ddr.directory)
        record = {"cache": self.cache.record(), "dataset": "APTOS2019",
                  "manifest": {"path": str(self.artifact / "cad_vit_runs"
                                              / "manifests" / "APTOS2019_seed42.csv"),
                               "manifest_sha256": "a" * 64},
                  "run_dir": str(self.output / "APTOS2019" / "arm")}
        self.assertEqual(_cache_from_record(record).identity, self.cache.identity)
        record["cache"]["protocol"]["versions"]["pillow"] = "mismatch"
        with self.assertRaisesRegex(ValueError, "Cache protocol"):
            _cache_from_record(record)
        with self.assertRaises(ValueError):
            V2ImageCache(ROOT / "cache", self.artifact, "APTOS2019",
                         "a" * 64, self.output)

    def test_concurrent_readers_write_one_entry(self):
        errors = []
        arrays = []
        def read():
            try:
                arrays.append(self.cache.get(self.rows[0]))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=read) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(arrays), 2)
        np.testing.assert_array_equal(arrays[0], arrays[1])
        self.assertEqual(len(list(self.cache.directory.rglob("*.npy"))), 1)
        self.assertTrue((self.cache.directory.parent / ".writer.lock").is_file())


if __name__ == "__main__":
    unittest.main()
