"""CPU-only synthetic cache and provenance checks; no research images."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from novelty import preof_phase1 as phase
from novelty_integrated.e7 import cache, runner


class TestE7Cache(unittest.TestCase):
    def test_e6_cache_is_read_only_even_if_reuse_flag_is_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(artifact_root=root, model_dir=root / "model",
                train_cache_root=root / "foundation_e6" / "train",
                val_cache_root=root / "foundation_e7" / "validation", reuse_e6_train=False)
            with self.assertRaisesRegex(ValueError, "below foundation_e7"):
                runner._cache_paths(args)
            args.reuse_e6_train = True
            self.assertEqual(runner._cache_paths(args)[0], args.train_cache_root.resolve())
            args.val_cache_root = root / "foundation_e6" / "validation"
            with self.assertRaisesRegex(ValueError, "below foundation_e7"):
                runner._cache_paths(args)

    def test_train_reuse_and_isolated_validation_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "synthetic.png"
            rgb = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
            Image.fromarray(rgb, mode="RGB").save(image)
            digest = phase.sha256(image)
            train = [{"split": "train", "image_id": "synthetic-train", "sha256": digest,
                      "path": str(image), "label": 2}]
            val = [{"split": "val", "image_id": "synthetic-val", "sha256": digest,
                    "path": str(image), "label": 2}]
            manifest = {"manifest_sha256": "a" * 64, "manifest_sidecar_sha256": "b" * 64}
            snapshot = {"processor_config_sha256": "c" * 64}
            train_root, val_root = root / "train-cache", root / "val-cache"
            cache.build_train(train_root, manifest, train, snapshot,
                              max_gib=0.01, reserve_gib=0)
            before = phase.sha256(train_root / "metadata.json")
            train_cache = cache.open_train(train_root, manifest, train, snapshot)
            assert np.array_equal(np.asarray(train_cache.read(train[0])), rgb)
            expected = cache.val_metadata(manifest, val, snapshot)
            cache.build_val(val_root, expected, val, max_gib=0.01, reserve_gib=0)
            val_cache = cache.open_val(val_root, expected, val)
            assert np.array_equal(np.asarray(val_cache.read(val[0])), rgb)
            self.assertEqual(before, phase.sha256(train_root / "metadata.json"))
            with self.assertRaises(ValueError):
                val_cache.read({**val[0], "split": "test"})
            with self.assertRaises(ValueError):
                cache.open_val(val_root, {**expected, "manifest_sha256": "d" * 64}, val)
            entry, _ = val_cache._paths(val[0])
            with entry.open("r+b") as stream:
                stream.seek(-1, 2)
                last = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([last[0] ^ 1]))
            with self.assertRaises(ValueError):
                cache.open_val(val_root, expected, val)

    def test_val_builder_rejects_train_and_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "x.png"
            Image.new("RGB", (10, 10), (1, 2, 3)).save(image)
            row = {"split": "train", "image_id": "x", "sha256": phase.sha256(image),
                   "path": str(image), "label": 0}
            with self.assertRaises(ValueError):
                cache.val_metadata({"manifest_sha256": "a", "manifest_sidecar_sha256": "b"},
                                   [row], {"processor_config_sha256": "c"})
