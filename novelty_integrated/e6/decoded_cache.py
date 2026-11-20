"""Train-only disk-backed EXIF-corrected RGB cache; no processor tensors are stored.

Entry reads use read-only mmap and OS page cache. A complete entry is immutable;
corrupt complete pairs fail closed. Missing entries may be populated only by build().
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image, ImageOps, features, __version__ as pillow_version
from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase

SCHEMA = "e6-decoded-rgb-train-v1"


def _reject_symlinks(path):
    candidate = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("Decoded cache path traverses a symlink")


def metadata(manifest, rows, snapshot):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Only frozen training rows may enter decoded cache")
    return {"schema": SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
            "manifest_sidecar_sha256": manifest["manifest_sidecar_sha256"],
            "image_order_sha256": fa._digest_json([(r["image_id"], r["sha256"]) for r in rows]),
            "row_count": len(rows), "decode": "PIL.ImageOps.exif_transpose then convert RGB",
            "processor_config_sha256": snapshot["processor_config_sha256"],
            "cache_implementation_sha256": phase.sha256(Path(__file__)),
            "pillow": pillow_version, "numpy": np.__version__,
            "jpeg_decoder": features.version("jpg"), "zlib_decoder": features.version("zlib")}


def _atomic_json(path, value):
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class DecodedCache:
    def __init__(self, root, expected, *, create=False):
        _reject_symlinks(root)
        self.root = phase._outside_repository(root)
        if self.root.is_symlink():
            raise ValueError("Decoded cache root cannot be a symlink")
        self.expected = expected
        self._verified = {}  # Entry SHA checked once; workers retain only small stat metadata.
        self.meta_sha256 = fa._digest_json(expected)
        meta_path = self.root / "metadata.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not meta_path.exists():
                _atomic_json(meta_path, expected)
        if meta_path.is_symlink() or not meta_path.is_file() or json.loads(meta_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Decoded cache metadata missing or incompatible")
        self.meta_file_sha256 = phase.sha256(meta_path)

    def _paths(self, row):
        if row["split"] != "train":
            raise ValueError("Decoded cache is train-only")
        key = hashlib.sha256((row["image_id"] + "\0" + row["sha256"] + "\0" + self.meta_sha256).encode()).hexdigest()
        return self.root / f"{key}.npy", self.root / f"{key}.json"

    def read(self, row):
        file, sidecar = self._paths(row)
        if file.is_symlink() or sidecar.is_symlink():
            raise ValueError("Decoded cache entry cannot be a symlink")
        if not file.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Incomplete decoded cache entry: {row['image_id']}")
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if (info.get("schema") != SCHEMA or info.get("image_id") != row["image_id"]
                or info.get("source_sha256") != row["sha256"]
                or info.get("metadata_sha256") != self.meta_sha256):
            raise ValueError("Decoded cache entry provenance mismatch")
        fs, ss = file.stat(), sidecar.stat()
        fingerprint = (fs.st_size, fs.st_mtime_ns, ss.st_size, ss.st_mtime_ns)
        if self._verified.get(str(file)) != fingerprint:
            if info.get("file_sha256") != phase.sha256(file):
                raise ValueError("Decoded cache entry digest mismatch")
            self._verified[str(file)] = fingerprint
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3
                or list(array.shape) != info.get("shape") or info.get("dtype") != "uint8"):
            raise ValueError("Decoded cache shape or dtype mismatch")
        return array

    def write_missing(self, row):
        file, sidecar = self._paths(row)
        if file.exists() and sidecar.exists():
            self.read(row)  # Complete entries are immutable; corruption fails closed.
            return False
        if file.exists() != sidecar.exists():
            # Only the exclusive builder reaches here. Recover a torn, unpublished pair.
            file.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
        source = Path(row["path"])
        before = source.stat()
        phase.require_hash(source, row["sha256"])
        with Image.open(source) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            array = np.ascontiguousarray(np.asarray(rgb), dtype=np.uint8)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source image changed during decode")
        temp = file.with_name(file.name + f".{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            # Builder uses a single writer. An unexpected concurrent entry fails closed.
            if file.exists() or sidecar.exists():
                raise FileExistsError("Concurrent cache writer detected")
            os.rename(temp, file)  # Same-filesystem atomic rename under exclusive builder.lock.
            _atomic_json(sidecar, {"schema": SCHEMA, "image_id": row["image_id"],
                                   "source_sha256": row["sha256"],
                                   "metadata_sha256": self.meta_sha256,
                                   "file_sha256": phase.sha256(file),
                                   "shape": list(array.shape), "dtype": "uint8"})
            loaded = self.read(row)
            if not np.array_equal(array, loaded):
                raise ValueError("Decoded cache roundtrip mismatch")
            return True
        finally:
            temp.unlink(missing_ok=True)


def estimate_bytes(rows):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Estimate requires training rows only")
    total = 0
    for row in rows:
        with Image.open(row["path"]) as image:
            width, height = image.size
        total += width * height * 3 + 128 + 2048
    return total


def build(root, expected, rows, *, max_gib, reserve_gib):
    _reject_symlinks(root)
    root = phase._outside_repository(root)
    if max_gib <= 0 or reserve_gib < 0:
        raise ValueError("Cache budget and reserve must be nonnegative")
    summary_path = root / "summary.json"
    if summary_path.exists():
        if (root / "builder.lock").exists():
            raise FileExistsError("Decoded cache builder is active")
        open_complete(root, expected, rows)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    if parent.is_symlink() or root.is_symlink():
        raise ValueError("Cache path cannot traverse a symlinked ancestor")
    estimate = estimate_bytes(rows)
    if estimate > max_gib * 1024**3 or shutil.disk_usage(parent).free - estimate < reserve_gib * 1024**3:
        raise ValueError("Decoded RGB cache exceeds disk budget or free-space reserve")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "builder.lock"
    own_lock = False
    started = time.perf_counter()
    try:
        with lock.open("x") as stream:
            own_lock = True
            stream.write(str(os.getpid()))
        cache = DecodedCache(root, expected, create=True)
        built = 0
        for row in rows:
            built += int(cache.write_missing(row))
        ordered = [phase.sha256(cache._paths(row)[1]) for row in rows]
        summary = {"schema": SCHEMA, "status": "COMPLETE", "row_count": len(rows),
                   "metadata_sha256": cache.meta_sha256, "ordered_entry_digest": fa._digest_json(ordered),
                   "entries_built": built, "estimated_bytes": estimate,
                   "build_seconds": time.perf_counter() - started,
                   "actual_bytes": sum(p.stat().st_size for p in root.iterdir() if p.is_file())}
        if summary["actual_bytes"] > max_gib * 1024**3 or shutil.disk_usage(root).free < reserve_gib * 1024**3:
            raise ValueError("Actual cache exceeds disk budget or reserve")
        _atomic_json(root / "summary.json", summary)
        return summary
    finally:
        if own_lock:
            lock.unlink(missing_ok=True)


def open_complete(root, expected, rows):
    cache = DecodedCache(root, expected)
    summary_path = cache.root / "summary.json"
    if summary_path.is_symlink():
        raise ValueError("Decoded cache summary cannot be a symlink")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("schema") != SCHEMA or summary.get("status") != "COMPLETE"
            or summary.get("row_count") != len(rows) or summary.get("metadata_sha256") != cache.meta_sha256):
        raise ValueError("Decoded cache is incomplete or incompatible")
    ordered = [phase.sha256(cache._paths(row)[1]) for row in rows]
    if fa._digest_json(ordered) != summary.get("ordered_entry_digest"):
        raise ValueError("Decoded cache entry order differs")
    for row in rows:
        cache.read(row)
    return cache
