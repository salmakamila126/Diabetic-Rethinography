"""Strict, optional E5 validation tensor cache using the audited foundation format."""

import json
import shutil

import numpy as np

from novelty import foundation_adaptation as fa
from novelty import foundation_input_cache as official_cache
from novelty import preof_phase1 as phase

SCHEMA = "e5-validation-official-input-v1"


def expected_metadata(snapshot, manifest, processor, micro_batch, source_hashes):
    # The existing cache format binds every entry to its complete metadata hash.
    source_digest = official_cache.digest_json(source_hashes)
    meta = official_cache.metadata(snapshot, manifest, processor, micro_batch, source_digest)
    meta["e5_schema"] = SCHEMA
    meta["allowed_split"] = "val"
    meta["e5_source_sha256"] = source_hashes
    return meta


def ordered_digest(rows, cache):
    values = []
    for row in rows:
        _, sidecar = cache._paths(row)
        values.append(phase.sha256(sidecar))
    return official_cache.digest_json(values)


def open_complete(root, expected, val_rows):
    cache = official_cache.InputCache(root, expected)
    summary = json.loads((cache.root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("schema") != SCHEMA or summary.get("status") != "COMPLETE"
            or summary.get("metadata_sha256") != cache.meta_sha256
            or summary.get("row_count") != len(val_rows)):
        raise ValueError("E5 validation cache is incomplete or incompatible")
    for row in val_rows:
        if row["split"] != "val":
            raise ValueError("Validation cache contains a non-validation row")
        cache.read(row)
    if ordered_digest(val_rows, cache) != summary.get("ordered_entry_digest"):
        raise ValueError("E5 validation entry order or hashes changed")
    return cache


def build(root, expected, val_rows, processor, *, max_gib=2.0, reserve_gib=5.0):
    """Create a new val-only cache; never repair existing entries or touch B1 cache."""
    if max_gib <= 0 or reserve_gib < 0:
        raise ValueError("Invalid validation cache disk budget")
    root = phase._outside_repository(root)
    if root.exists():
        raise FileExistsError("Use a new E5 validation cache directory")
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if root.is_symlink() or ancestor.is_symlink():
        raise ValueError("Validation cache destination must not be a symlink")
    estimated = len(val_rows) * (3 * 224 * 224 * 4 + 4096)
    if estimated > max_gib * (1024 ** 3):
        raise ValueError("Estimated validation cache exceeds explicit disk budget")
    free = shutil.disk_usage(ancestor).free
    if free - estimated < reserve_gib * (1024 ** 3):
        raise ValueError("Insufficient free space on validation cache filesystem")
    cache = official_cache.InputCache(root, expected, create=True)
    for start in range(0, len(val_rows), expected["verification_batch_size"]):
        selected = val_rows[start:start + expected["verification_batch_size"]]
        if any(row["split"] != "val" for row in selected):
            raise ValueError("Validation cache build forbids train/test rows")
        batch, _ = fa._image_batch(selected, processor)
        for row, tensor in zip(selected, batch):
            singleton, _ = fa._image_batch([row], processor)
            if not np.array_equal(tensor.numpy(), singleton[0].numpy()):
                raise ValueError("Official singleton and batch preprocessing differ")
            cache.write_verified(row, tensor)
            if not np.array_equal(cache.read(row).numpy(), tensor.numpy()):
                raise ValueError("Cached validation tensor differs from uncached")
    actual_bytes = sum(p.stat().st_size for p in cache.root.rglob("*") if p.is_file())
    if actual_bytes > max_gib * (1024 ** 3):
        raise ValueError("Actual validation cache exceeded explicit budget")
    report = {"schema": SCHEMA, "status": "COMPLETE", "row_count": len(val_rows),
              "metadata_sha256": cache.meta_sha256, "disk_bytes": actual_bytes,
              "ordered_entry_digest": ordered_digest(val_rows, cache),
              "equivalence": "exact batch/singleton/roundtrip float32 tensor equality"}
    cache._atomic_json(cache.root / "summary.json", report)
    open_complete(root, expected, val_rows)
    return report
