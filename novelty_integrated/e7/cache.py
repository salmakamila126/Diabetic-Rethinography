"""Read-only E6 train RGB reuse and isolated E7 validation RGB cache."""
import json
import hashlib
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image, features, __version__ as pillow_version
from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from novelty_integrated.e6 import decoded_cache as e6cache

VAL_SCHEMA = "e7-decoded-rgb-validation-v1"
reject_symlinks = e6cache._reject_symlinks


class ValidationRGB(e6cache.DecodedCache):
    def _paths(self, row):
        if row["split"] != "val":
            raise ValueError("E7 validation cache forbids train/test")
        key = hashlib.sha256((row["image_id"] + "\0" + row["sha256"] + "\0" + self.meta_sha256).encode()).hexdigest()
        return self.root / f"{key}.npy", self.root / f"{key}.json"


def val_metadata(manifest, rows, snapshot):
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("Validation cache accepts frozen val only")
    return {"schema": VAL_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
            "sidecar_sha256": manifest["manifest_sidecar_sha256"],
            "image_order_sha256": fa._digest_json([(r["image_id"], r["sha256"], r["label"]) for r in rows]),
            "row_count": len(rows), "processor_config_sha256": snapshot["processor_config_sha256"],
            "decode": "PIL EXIF transpose then RGB", "pillow": pillow_version,
            "numpy": np.__version__, "jpeg_decoder": features.version("jpg"),
            "zlib_decoder": features.version("zlib"),
            "cache_source_sha256": phase.sha256(__file__),
            "e6_decode_source_sha256": phase.sha256(e6cache.__file__)}


def _entry_digest(cache, rows):
    return fa._digest_json([phase.sha256(cache._paths(row)[1]) for row in rows])


def open_val(root, expected, rows):
    cache = ValidationRGB(root, expected)
    if (cache.root / "builder.lock").exists():
        raise FileExistsError("E7 validation cache builder is active")
    summary_path = cache.root / "summary.json"
    if summary_path.is_symlink():
        raise ValueError("Validation summary symlink forbidden")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("schema") != VAL_SCHEMA or summary.get("status") != "COMPLETE"
            or summary.get("metadata_sha256") != cache.meta_sha256
            or summary.get("row_count") != len(rows)
            or summary.get("ordered_entry_digest") != _entry_digest(cache, rows)):
        raise ValueError("E7 validation RGB cache is incomplete or incompatible")
    for row in rows:
        cache.read(row)
    return cache


def build_val(root, expected, rows, *, max_gib, reserve_gib):
    e6cache._reject_symlinks(root)
    root = phase._outside_repository(root)
    if (root / "summary.json").exists():
        if (root / "builder.lock").exists():
            raise FileExistsError("E7 validation cache builder is active")
        open_val(root, expected, rows)
        return json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if max_gib <= 0 or reserve_gib < 0:
        raise ValueError("Invalid E7 val cache disk budget")
    estimated = 0
    for row in rows:
        if row["split"] != "val":
            raise ValueError("Val cache cannot admit train/test")
        with Image.open(row["path"]) as image:
            estimated += image.width * image.height * 3 + 4096
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if estimated > max_gib * 1024**3 or shutil.disk_usage(ancestor).free - estimated < reserve_gib * 1024**3:
        raise ValueError("Val RGB cache exceeds budget or reserve")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "builder.lock"
    owned = False
    started = time.perf_counter()
    try:
        with lock.open("x") as stream:
            owned = True
            stream.write("E7 val-only builder")
        cache = ValidationRGB(root, expected, create=True)
        built = sum(int(cache.write_missing(row)) for row in rows)
        report = {"schema": VAL_SCHEMA, "status": "COMPLETE", "row_count": len(rows),
                  "metadata_sha256": cache.meta_sha256,
                  "ordered_entry_digest": _entry_digest(cache, rows),
                  "entries_built": built, "estimated_bytes": estimated,
                  "build_seconds": time.perf_counter() - started,
                  "actual_bytes": sum(p.stat().st_size for p in root.iterdir() if p.is_file())}
        if report["actual_bytes"] > max_gib * 1024**3 or shutil.disk_usage(root).free < reserve_gib * 1024**3:
            raise ValueError("Val RGB cache actual size exceeds budget or reserve")
        e6cache._atomic_json(root / "summary.json", report)
    finally:
        if owned:
            lock.unlink(missing_ok=True)
    open_val(root, expected, rows)
    return report


def open_train(root, manifest, rows, snapshot):
    if (Path(root) / "builder.lock").exists():
        raise FileExistsError("Training cache builder is active")
    return e6cache.open_complete(root, e6cache.metadata(manifest, rows, snapshot), rows)


def build_train(root, manifest, rows, snapshot, *, max_gib, reserve_gib):
    return e6cache.build(root, e6cache.metadata(manifest, rows, snapshot), rows,
                         max_gib=max_gib, reserve_gib=reserve_gib)
