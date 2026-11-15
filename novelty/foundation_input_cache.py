"""Verified, model-specific official-processor input cache for E4/E5.

Stores float32 CHW tensors, never backbone embeddings. No work runs on import.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from novelty import preof_phase1 as phase

SCHEMA = "foundation-official-input-v1"
SHAPE = (3, 224, 224)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metadata(snapshot, manifest, processor, batch_size, runner_sha256):
    import torch
    import transformers
    from PIL import __version__ as pillow_version

    if batch_size not in {1, 4, 8, 16, 32}:
        raise ValueError("Cache verification batch size must be 1/4/8/16/32")
    effective = processor.to_dict()
    return {"schema": SCHEMA, "dataset": manifest["dataset"],
            "manifest": manifest,
            "snapshot": {key: snapshot[key] for key in
                         ("hub_id", "model_kind", "revision", "config_sha256",
                          "processor_config_sha256", "weight_file", "weight_sha256")},
            "processor_effective": effective,
            "processor_effective_sha256": digest_json(effective),
            "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "numpy": np.__version__, "pillow": pillow_version},
            "dtype": "float32", "shape": list(SHAPE),
            "verification_batch_size": batch_size,
            "runner_sha256": runner_sha256}


class InputCache:
    def __init__(self, root, expected, *, create=False):
        self.root = phase._outside_repository(root)
        self.expected = expected
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            self._atomic_json(self.root / "metadata.json", expected)
        stored = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if stored != expected:
            raise ValueError("Official-input cache metadata differs; use a new cache directory")
        self.meta_sha256 = phase.sha256(self.root / "metadata.json")

    @staticmethod
    def _atomic_json(path, value):
        temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + chr(10)).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _paths(self, row):
        key = digest_json((row["split"], row["image_id"], row["sha256"]))
        base = self.root / row["split"] / key[:2] / key
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def exists(self, row):
        data, sidecar = self._paths(row)
        if data.exists() != sidecar.exists():
            raise ValueError("Interrupted or corrupt input-cache entry")
        return data.exists()

    def read(self, row):
        import torch

        data, sidecar = self._paths(row)
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = {key: row[key] for key in ("split", "image_id", "sha256")}
        if (any(record.get(key) != value for key, value in expected.items())
                or record.get("cache_metadata_sha256") != self.meta_sha256
                or record.get("exact_uncached_equivalence") is not True
                or record.get("file_sha256") != phase.sha256(data)):
            raise ValueError("Official-input cache provenance or file hash differs")
        array = np.load(data, allow_pickle=False)
        if (array.shape != SHAPE or array.dtype != np.float32
                or not array.flags.c_contiguous or not np.isfinite(array).all()
                or hashlib.sha256(array.tobytes()).hexdigest() != record.get("tensor_sha256")):
            raise ValueError("Official-input cache tensor integrity differs")
        return torch.from_numpy(array)

    def write_verified(self, row, uncached):
        import torch

        if not isinstance(uncached, torch.Tensor) or uncached.shape != SHAPE:
            raise ValueError("Uncached official tensor shape differs")
        array = np.ascontiguousarray(uncached.detach().cpu().numpy())
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Official tensor must be finite float32")
        data, sidecar = self._paths(row)
        data.parent.mkdir(parents=True, exist_ok=True)
        if data.exists() or sidecar.exists():
            raise FileExistsError("Input-cache entry already exists")
        temporary = data.with_name(data.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            reread = np.load(temporary, allow_pickle=False)
            if not np.array_equal(reread, array):
                raise ValueError("Input-cache write is not exactly equal to official tensor")
            os.replace(temporary, data)
            record = {key: row[key] for key in ("split", "image_id", "sha256")}
            record.update({"cache_metadata_sha256": self.meta_sha256,
                           "file_sha256": phase.sha256(data),
                           "tensor_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                           "exact_uncached_equivalence": True})
            self._atomic_json(sidecar, record)
        finally:
            temporary.unlink(missing_ok=True)

    def batch(self, rows):
        import torch

        return torch.stack([self.read(row) for row in rows]), torch.tensor(
            [row["label"] for row in rows], dtype=torch.long)


def prepare(cache, rows_by_split, batch_size, uncached_batch, *, verify_equivalence=False):
    """Cold writes compare exact tensors; warm reads verify hashes and optionally recompute."""
    import torch

    if batch_size != cache.expected["verification_batch_size"]:
        raise ValueError("Cache batch size differs")
    totals = {"cold_images": 0, "warm_images": 0, "exact_compared_images": 0,
              "cold_seconds": 0.0, "warm_seconds": 0.0}
    ordered = rows_by_split["train"] + rows_by_split["val"]
    for split in ("train", "val"):
        rows = rows_by_split[split]
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            existence = [cache.exists(row) for row in selected]
            if all(existence) and not verify_equivalence:
                tick = time.perf_counter()
                cache.batch(selected)
                totals["warm_images"] += len(selected)
                totals["warm_seconds"] += time.perf_counter() - tick
                continue
            tick = time.perf_counter()
            official, _ = uncached_batch(selected)
            if official.shape != (len(selected), *SHAPE) or official.dtype != torch.float32:
                raise ValueError("Official processor dtype/shape differs")
            for row, image, existed in zip(selected, official, existence):
                singleton, _ = uncached_batch([row])
                if singleton.shape != (1, *SHAPE) or not np.array_equal(
                        singleton[0].numpy(), image.numpy()):
                    raise ValueError("Official processor differs between singleton and batch")
                if existed:
                    cached = cache.read(row)
                    if not np.array_equal(cached.numpy(), image.numpy()):
                        raise ValueError("Cached and uncached official inputs differ exactly")
                    totals["warm_images"] += 1
                else:
                    cache.write_verified(row, image)
                    cached = cache.read(row)
                    if not np.array_equal(cached.numpy(), image.numpy()):
                        raise ValueError("Cold cache and uncached official inputs differ exactly")
                    totals["cold_images"] += 1
                totals["exact_compared_images"] += 1
            totals["cold_seconds"] += time.perf_counter() - tick
    if totals["warm_images"] + totals["cold_images"] != len(ordered):
        raise ValueError("Cache preparation count differs")
    disk_bytes = sum(path.stat().st_size for path in cache.root.rglob("*") if path.is_file())
    prior_path = cache.root / "summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    first_cold = prior.get("first_cold_measurement")
    if first_cold is None and totals["cold_images"]:
        first_cold = {"images": totals["cold_images"], "seconds": totals["cold_seconds"],
                      "images_per_second": totals["cold_images"] / totals["cold_seconds"]}
    report = {"schema": SCHEMA, "status": "COMPLETE", "metadata_sha256": cache.meta_sha256,
              "row_count": len(ordered), "counts": totals, "disk_bytes": disk_bytes,
              "first_cold_measurement": first_cold,
              "cold_images_per_second": (totals["cold_images"] / totals["cold_seconds"]
                                         if totals["cold_seconds"] else None),
              "warm_images_per_second": (totals["warm_images"] / totals["warm_seconds"]
                                         if totals["warm_seconds"] else None),
              "equivalence": "exact singleton/batch/roundtrip equality at creation; SHA-256 and metadata verified on every read"}
    cache._atomic_json(cache.root / "summary.json", report)
    return report
