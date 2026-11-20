"""CPU-only decoded RGB cache integrity and split isolation tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

import torch
from novelty import foundation_adaptation as fa
from novelty_integrated.e6 import decoded_cache, input_pipeline


class TestDecodedCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source.png"
        image = Image.new("RGB", (8, 6), (40, 80, 120))
        image.putpixel((0, 0), (10, 20, 30))
        exif = Image.Exif(); exif[274] = 3
        image.save(self.source, exif=exif)
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.row = {"split": "train", "image_id": "synthetic_0", "path": str(self.source),
                    "sha256": digest, "label": 0}
        self.rows = [self.row]
        self.manifest = {"manifest_sha256": "a"*64, "manifest_sidecar_sha256": "b"*64}
        self.snapshot = {"processor_config_sha256": "c"*64}
        self.cache_root = root / "cache"
        self.meta = decoded_cache.metadata(self.manifest, self.rows, self.snapshot)

    def _build(self):
        return decoded_cache.build(self.cache_root, self.meta, self.rows,
                                   max_gib=0.01, reserve_gib=0)

    def test_exact_exif_rgb_roundtrip_and_reopen(self):
        report = self._build()
        self.assertEqual(report["row_count"], 1)
        cache = decoded_cache.open_complete(self.cache_root, self.meta, self.rows)
        with Image.open(self.source) as source:
            expected = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        actual = cache.read(self.row)
        self.assertIsInstance(actual, np.memmap)
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual.flags.writeable)
        before = (self.cache_root / "summary.json").read_bytes()
        self.assertEqual(self._build()["status"], "COMPLETE")
        self.assertEqual((self.cache_root / "summary.json").read_bytes(), before)

    def test_b1_train_tensors_equal_direct(self):
        self._build()
        class Processor:
            def __call__(self, *, images, return_tensors):
                arrays = [np.asarray(image.resize((224,224)), dtype=np.float32)
                          .transpose(2,0,1) / 255 for image in images]
                return {"pixel_values": torch.from_numpy(np.stack(arrays))}
        processor = Processor()
        epoch = input_pipeline.multiprocessing.Value("i", 1)
        cache = decoded_cache.open_complete(self.cache_root, self.meta, self.rows)
        collator = input_pipeline.TrainCollator(processor, cache, seed=42, epoch=epoch)
        for current in (1,2):
            epoch.value = current
            cached, labels = collator([self.row])
            direct, truth = fa._image_batch([self.row], processor,
                                            augment_seed=42, augment_epoch=current)
            self.assertTrue(torch.equal(cached, direct))
            self.assertTrue(torch.equal(labels, truth))

    def test_verified_worker_read_avoids_rehash_until_file_changes(self):
        self._build()
        cache = decoded_cache.open_complete(self.cache_root, self.meta, self.rows)
        with patch.object(decoded_cache.phase, "sha256", side_effect=AssertionError("rehash")):
            self.assertEqual(cache.read(self.row).dtype, np.uint8)
        file, _ = cache._paths(self.row)
        old = file.read_bytes()
        file.write_bytes(old[:-1] + bytes([old[-1] ^ 1]))
        with self.assertRaises(ValueError):
            cache.read(self.row)

    def test_corruption_fails_closed_and_no_repair(self):
        self._build()
        cache = decoded_cache.DecodedCache(self.cache_root, self.meta)
        file, _ = cache._paths(self.row)
        old = file.read_bytes()
        file.write_bytes(old[:-1] + bytes([old[-1] ^ 1]))
        with self.assertRaises(ValueError):
            decoded_cache.open_complete(self.cache_root, self.meta, self.rows)
        with self.assertRaises(ValueError):
            self._build()

    def test_test_row_rejected(self):
        with self.assertRaises(ValueError):
            decoded_cache.metadata(self.manifest, [{**self.row, "split":"test"}], self.snapshot)
        self._build()
        cache = decoded_cache.DecodedCache(self.cache_root, self.meta)
        with self.assertRaises(ValueError):
            cache.read({**self.row, "split":"test"})

    def test_metadata_mismatch_rejected(self):
        self._build()
        wrong = {**self.meta, "processor_config_sha256": "d"*64}
        with self.assertRaises(ValueError):
            decoded_cache.open_complete(self.cache_root, wrong, self.rows)

    def test_incomplete_entry_rebuilt_under_lock(self):
        self.cache_root.mkdir()
        decoded_cache.DecodedCache(self.cache_root, self.meta, create=True)
        cache = decoded_cache.DecodedCache(self.cache_root, self.meta)
        file, sidecar = cache._paths(self.row)
        file.write_bytes(b"interrupted")
        self.assertFalse(sidecar.exists())
        self.assertEqual(self._build()["entries_built"], 1)
        self.assertEqual(decoded_cache.open_complete(self.cache_root, self.meta, self.rows)
                         .read(self.row).dtype, np.uint8)

    def test_source_sha_guard(self):
        self.row["sha256"] = "0"*64
        self.meta = decoded_cache.metadata(self.manifest, self.rows, self.snapshot)
        with self.assertRaises(ValueError):
            self._build()

    def test_existing_lock_preserved(self):
        self.cache_root.mkdir()
        lock = self.cache_root / "builder.lock"
        lock.write_text("another builder")
        with self.assertRaises(FileExistsError):
            self._build()
        self.assertEqual(lock.read_text(), "another builder")


if __name__ == "__main__":
    unittest.main()
