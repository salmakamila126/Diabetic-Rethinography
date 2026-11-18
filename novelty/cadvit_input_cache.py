"""Lossless deterministic V2 BG10 image cache for train/validation only.

No cache directory is created on import or construction. The source image is
verified against its frozen-manifest SHA before every new/repaired entry.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import cv2
import numpy as np
from PIL import Image
import PIL

from novelty.cadvit_training import file_sha256
from reproduction.cadvit.preprocessing import (
    BG_V2_CONFIG, PREPROCESS_CONFIG, preprocess_fundus_v2,
)

SCHEMA = "cadvit-v2-bg10-uint8-cache-v1"
SHAPE = (512, 512, 3)
DTYPE = np.dtype("uint8")
PIXEL_BYTES = int(np.prod(SHAPE))
MAX_NPY_BYTES = PIXEL_BYTES + 4096
# NPY allocated blocks + JSON block + directory/lock allowance per image.
ENTRY_BUDGET = PIXEL_BYTES + 12288
FREE_RESERVE = 8 * 2**30
MAX_WARM_LIMIT = 12000
REPO = Path(__file__).resolve().parents[1]


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8")).hexdigest()


def _validate_root(root, artifact_root, output_root=None):
    root = Path(root).expanduser().resolve()
    artifact = Path(artifact_root).expanduser().resolve()
    protected = [REPO, artifact / "datasets", artifact / "cad_vit_runs",
                 artifact / "experiment_archive_80ep_20261105",
                 artifact / "foundation_e4", artifact / "foundation_inputs",
                 artifact / "cadvit_novelty_pilots"]
    if output_root is not None:
        protected.append(Path(output_root).expanduser().resolve())
    for path in protected:
        path = path.resolve()
        if root == path or path in root.parents or root in path.parents:
            raise ValueError(f"Cache root overlaps protected path: {path}")
    return root


class V2ImageCache:
    """One namespace per manifest, exact preprocessing source and decoder stack."""

    def __init__(self, root, artifact_root, dataset, manifest_sha256, output_root=None):
        if dataset not in ("APTOS2019", "DDR"):
            raise ValueError("Only frozen APTOS2019/DDR train-validation data are supported")
        if len(manifest_sha256) != 64 or any(c not in "0123456789abcdef"
                                              for c in manifest_sha256):
            raise ValueError("Expected lowercase full manifest SHA-256")
        self.root = _validate_root(root, artifact_root, output_root)
        self.dataset = dataset
        self.protocol = {
            "schema": SCHEMA, "dataset": dataset, "manifest_sha256": manifest_sha256,
            "preprocessing": "V2_BG10_STABLE",
            "preprocess_config": PREPROCESS_CONFIG,
            "ben_graham_config": BG_V2_CONFIG,
            "preprocessing_source_sha256":
                file_sha256(REPO / "reproduction/cadvit/preprocessing.py"),
            "cache_source_sha256": file_sha256(Path(__file__)),
            "decoder": "PIL.Image.open -> ImageOps.exif_transpose -> RGB -> OpenCV",
            "versions": {"pillow": PIL.__version__, "opencv": cv2.__version__,
                         "numpy": np.__version__},
            "array": {"shape": list(SHAPE), "dtype": "uint8", "order": "C",
                      "format": "numpy-npy-uncompressed"},
        }
        self.identity = _digest(self.protocol)
        self.directory = self.root / dataset / self.identity / "entries"

    def record(self):
        return {"enabled": True, "root": str(self.root), "identity": self.identity,
                "protocol": self.protocol, "free_reserve_bytes": FREE_RESERVE,
                "entry_budget_bytes": int(ENTRY_BUDGET)}

    def _guard_entry_parent(self, parent):
        # Reject a symlink escape or a nested mount that would redirect entries
        # away from the filesystem selected by --cache-root.
        resolved = parent.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Cache entry path escapes cache root")
        root_anchor = self.root
        while not root_anchor.exists():
            root_anchor = root_anchor.parent
        entry_anchor = parent
        while not entry_anchor.exists():
            entry_anchor = entry_anchor.parent
        if not root_anchor.is_dir() or not entry_anchor.is_dir():
            raise NotADirectoryError("Cache path ancestor is not a directory")
        if entry_anchor.stat().st_dev != root_anchor.stat().st_dev:
            raise ValueError("Cache entry path crosses cache filesystem")

    def _paths(self, row):
        if row.get("split") not in ("train", "val"):
            raise ValueError("Cache accepts train/validation rows only")
        image_sha = row.get("sha256")
        if (not isinstance(image_sha, str) or len(image_sha) != 64
                or any(c not in "0123456789abcdef" for c in image_sha)):
            raise ValueError("Frozen row lacks verified lowercase image SHA-256")
        key = _digest({"protocol": self.identity, "source_image_sha256": image_sha})
        parent = self.directory / key[:2]
        self._guard_entry_parent(parent)
        return key, parent / (key + ".npy"), parent / (key + ".json")

    def _valid(self, key, data_path, meta_path, image_sha):
        try:
            if data_path.is_symlink() or meta_path.is_symlink():
                return None
            if not data_path.is_file() or not meta_path.is_file():
                return None
            if data_path.stat().st_size > MAX_NPY_BYTES or meta_path.stat().st_size > 16384:
                return None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                return None
            if (meta.get("schema") != SCHEMA or meta.get("key") != key
                    or meta.get("protocol_identity") != self.identity
                    or meta.get("source_image_sha256") != image_sha
                    or meta.get("shape") != list(SHAPE) or meta.get("dtype") != "uint8"):
                return None
            raw = data_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != meta.get("npy_sha256"):
                return None
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            if (array.shape != SHAPE or array.dtype != DTYPE
                    or not array.flags.c_contiguous
                    or hashlib.sha256(array.tobytes()).hexdigest()
                       != meta.get("pixels_sha256")):
                return None
            return array
        except (OSError, ValueError, TypeError, KeyError, UnicodeError, EOFError):
            return None

    def _source_pixels(self, row):
        path = Path(row["path"])
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Source image differs from frozen manifest")
        with Image.open(path) as image:
            pixels = preprocess_fundus_v2(image, PREPROCESS_CONFIG, BG_V2_CONFIG)
        if pixels.shape != SHAPE or pixels.dtype != DTYPE or not pixels.flags.c_contiguous:
            raise ValueError("V2 preprocessing did not produce exact uint8 RGB 512x512x3")
        return pixels

    def _write(self, key, data_path, meta_path, pixels, image_sha):
        data_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp_data = data_path.with_name(data_path.name + suffix)
        temp_meta = meta_path.with_name(meta_path.name + suffix)
        try:
            with temp_data.open("xb") as stream:
                np.save(stream, pixels, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            meta = {
                "schema": SCHEMA, "key": key, "protocol_identity": self.identity,
                "source_image_sha256": image_sha,
                "shape": list(SHAPE), "dtype": "uint8",
                "npy_sha256": file_sha256(temp_data),
                "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            }
            with temp_meta.open("x", encoding="utf-8") as stream:
                json.dump(meta, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_data, data_path)
            os.replace(temp_meta, meta_path)
        finally:
            temp_data.unlink(missing_ok=True)
            temp_meta.unlink(missing_ok=True)

    def get(self, row):
        key, data_path, meta_path = self._paths(row)
        cached = self._valid(key, data_path, meta_path, row["sha256"])
        if cached is not None:
            return cached
        # A stable OS advisory lock is released by the kernel if a writer dies.
        # The lock file itself may remain; it never blocks a later healthy run.
        lock = self.directory.parent / ".writer.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        deadline = time.monotonic() + 30.0
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Cache writer lock did not clear: {lock}")
                    time.sleep(0.1)
            cached = self._valid(key, data_path, meta_path, row["sha256"])
            if cached is not None:
                return cached
            pixels = self._source_pixels(row)
            self.ensure_space([row])
            self._write(key, data_path, meta_path, pixels, row["sha256"])
            checked = self._valid(key, data_path, meta_path, row["sha256"])
            if checked is None or not np.array_equal(checked, pixels):
                raise IOError("Atomic cache write failed integrity validation")
            return checked
        finally:
            os.close(fd)

    def ensure_space(self, rows, reserve_bytes=FREE_RESERVE):
        """Conservative bound for requested rows, including possible repairs."""
        unique_rows = {row["sha256"]: row for row in rows}
        missing = 0
        for row in unique_rows.values():
            _, data_path, meta_path = self._paths(row)
            if not data_path.is_file() or not meta_path.is_file():
                missing += 1
        # One extra entry covers a temporary write when replacing corruption.
        needed = int((missing + 1) * ENTRY_BUDGET)
        # The nearest existing cache-entry ancestor determines the filesystem,
        # including when the configured root has not been created yet.
        ancestor = self.directory
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise NotADirectoryError(f"Cache path ancestor is not a directory: {ancestor}")
        if not os.access(ancestor, os.W_OK | os.X_OK):
            raise PermissionError(f"Cache path is not writable/searchable: {ancestor}")
        free = shutil.disk_usage(ancestor).free
        if free < needed + reserve_bytes:
            raise OSError(f"Insufficient cache space: free={free}, upper_bound={needed}, "
                          f"reserve={reserve_bytes}")
        return {"rows": len(rows), "unique_image_hashes": len(unique_rows),
                "missing_or_incomplete_entries": missing,
                "upper_bound_bytes": needed, "free_bytes": free,
                "reserve_bytes": reserve_bytes}

    def warm(self, rows, limit, progress=25):
        if not isinstance(limit, int) or not 1 <= limit <= MAX_WARM_LIMIT:
            raise ValueError(f"Explicit warm-up limit must be 1..{MAX_WARM_LIMIT}")
        if limit > len(rows) or any(row.get("split") not in ("train", "val") for row in rows):
            raise ValueError("Warm-up must select existing train/validation rows only")
        selected = rows[:limit]
        space = self.ensure_space(selected)
        started = time.perf_counter()
        for count, row in enumerate(selected, 1):
            self.get(row)
            if count == 1 or count % progress == 0 or count == len(selected):
                print(f"CACHE_WARM {count}/{len(selected)} elapsed={time.perf_counter()-started:.2f}s",
                      flush=True)
        return {"checked": len(selected), "seconds": time.perf_counter()-started,
                "space_preflight": space}
