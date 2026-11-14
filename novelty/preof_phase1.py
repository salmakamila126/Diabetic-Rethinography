"""PReOF Phase 1: validation-only preprocessing/prediction caches and E0 analysis.

No work runs on import. Checkpoint loading requires an explicit trust flag and
the recorded SHA-256; this script never selects or opens test images.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import uuid

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CACHE_VERSION = "preof-phase1-cache-v1"
PREPROCESS_CACHE_VERSION = "preof-preprocessing-npy-v1"
KEYS = ("p11", "p12", "p21", "p22")
SPECS = {
    "APTOS2019": {
        "manifest_sha256": "73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854",
        "val_count": 366,
        "v1_run": "experiments/cadvit_tables_1_5_balanced_v1/aptos_r16",
        "v1_epoch": 24,
        "v1_best_f1": 0.6614213564213565,
        "v1_sha256": "6db01fbb840f4ed9641f9d8d3b9ac90e98d3b0772d57101be56aaabc60e05601",
        "v1_variant": "aptos_r16",
        "v2_run": "checkpoints/aptos_v2_bg10_stable_80ep_seed42",
        "v2_epoch": 69,
        "v2_best_f1": 0.6571274188819457,
        "v2_sha256": "e72573f0f1c26baac53e18d950076845952da503261e212021ff6320c9ae72d5",
    },
    "DDR": {
        "manifest_sha256": "7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1",
        "val_count": 1252,
        "v1_run": "experiments/cadvit_tables_1_5_balanced_fp32_math_mb8_diag/ddr_full",
        "v1_epoch": 69,
        "v1_best_f1": 0.6530693590977841,
        "v1_sha256": "84e3272d3f10d9179fd2dd9e2799b640962e1185981856f87bcb3876b9deace0",
        "v1_variant": "ddr_full",
        "v2_run": "checkpoints/ddr_v2_fp32_weighted_fp64clip_seed42",
        "v2_epoch": 72,
        "v2_best_f1": 0.6903234255707131,
        "v2_sha256": "844665f0c597142915286b727a21f1d2223f43e0f057d083b6c1dd035c180ac1",
    },
}
MODEL_NODES = ("ConvBNAct", "MultiScaleFeatureBlock", "CoordinateAttention",
               "InvertedResidualBlock", "EnhancedMobileViTBlock", "ARCH_CONFIG",
               "CADViTCandidate")
PREPROCESS_NODES = ("PREPROCESS_CONFIG", "preprocess_fundus", "BG_V2_CONFIG",
                    "preprocess_fundus_v2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _nodes(source):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in MODEL_NODES + PREPROCESS_NODES:
            if name in found:
                raise ValueError(f"Duplicate historical definition: {name}")
            found[name] = ast.dump(node, include_attributes=False)
    return found


def verify_v2_source(path):
    """Compare historical server definitions by AST, without executing them."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "CAD_ViT.ipynb"
    if not path.is_file() or path.suffix != ".ipynb":
        raise FileNotFoundError("Provide the historical server CAD_ViT.ipynb via --v2-source")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = {}
    markers = ("class ConvBNAct(", "class CADViTCandidate(", "def preprocess_fundus(")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(marker in source for marker in markers):
            for name, node in _nodes(source).items():
                if name in found:
                    raise ValueError(f"Duplicate server source definition: {name}")
                found[name] = node
    local = {}
    for relative in ("reproduction/cadvit/model.py", "reproduction/cadvit/preprocessing.py"):
        local.update(_nodes((REPO / relative).read_text(encoding="utf-8")))
    for name in MODEL_NODES + PREPROCESS_NODES:
        if name not in found or found[name] != local[name]:
            raise ValueError(f"Historical V2 source differs from local candidate: {name}")
    return {"path": str(path), "sha256": sha256(path), "definitions": list(MODEL_NODES + PREPROCESS_NODES)}


def validation_rows(manifest, dataset, dataset_root):
    """Hash the frozen manifest, retaining only validation rows; open no images."""
    spec = SPECS[dataset]
    manifest = Path(manifest).expanduser().resolve()
    require_hash(manifest, spec["manifest_sha256"])
    meta_path = manifest.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("dataset") != dataset or meta.get("seed") != 42
            or meta.get("manifest_file_sha256") != spec["manifest_sha256"]
            or Path(meta.get("root", "")).expanduser().resolve() != dataset_root):
        raise ValueError("Manifest metadata does not match the frozen dataset root")
    rows, ids, paths = [], set(), set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "label", "image_path", "group_id", "sha256", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Incomplete manifest columns")
        for row in reader:
            if row["split"] != "val":
                continue
            rel = PurePosixPath(row["image_path"])
            if (row["label"] not in {str(i) for i in range(5)} or not row["image_id"]
                    or not row["group_id"] or not rel.parts or rel.is_absolute()
                    or ".." in rel.parts or "\\" in row["image_path"]):
                raise ValueError("Invalid validation row")
            if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
                raise ValueError("Invalid image SHA-256 in manifest")
            if row["image_id"] in ids or row["image_path"] in paths:
                raise ValueError("Duplicate validation image ID/path")
            ids.add(row["image_id"])
            paths.add(row["image_path"])
            rows.append({key: row[key] for key in required})
    if len(rows) != spec["val_count"]:
        raise ValueError(f"Expected {spec['val_count']} validation rows, found {len(rows)}")
    return rows


def _same_json(a, b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def checked_models(dataset, artifact_root, v2_source, trust_checkpoints):
    """Fail closed on source, checkpoint, record, and strict state compatibility."""
    if not trust_checkpoints:
        raise ValueError("Trusted project checkpoints require --trust-checkpoints")
    import torch
    from reproduction.cadvit.model import ARCH_CONFIG, CADViTCandidate
    from reproduction.cadvit.preprocessing import PREPROCESS_CONFIG, BG_V2_CONFIG

    spec = SPECS[dataset]
    source = verify_v2_source(v2_source)
    paths = {version: artifact_root / spec[f"{version}_run"] / "best.pt" for version in ("v1", "v2")}
    for version, path in paths.items():
        require_hash(path, spec[f"{version}_sha256"])
    # Only the researcher-specified, hash-matched checkpoints reach pickle loading.
    states = {version: torch.load(path, map_location="cpu", weights_only=False)
              for version, path in paths.items()}
    v1, v2 = states["v1"], states["v2"]
    v1_record, v2_record = v1.get("record"), v2.get("record")
    if not isinstance(v1_record, dict) or not isinstance(v2_record, dict):
        raise ValueError("Checkpoint record missing")
    v1_config_path = paths["v1"].with_name("config.json")
    v1_config = json.loads(v1_config_path.read_text(encoding="utf-8"))
    if (v1_record != v1_config or v1_record.get("variant", {}).get("name") != spec["v1_variant"]
            or v1_record.get("variant", {}).get("dataset") != dataset
            or v1_record.get("variant", {}).get("task") != "five_class"
            or v1_record.get("manifest_sha256") != spec["manifest_sha256"]
            or v1_record.get("campaign") != "balanced_v1"
            or v1.get("epoch") != spec["v1_epoch"] - 1
            or not math.isclose(float(v1.get("best_f1", float("nan"))), spec["v1_best_f1"], rel_tol=0, abs_tol=1e-9)
            or v1_record.get("model_sha256") != sha256(REPO / "reproduction/cadvit/model.py")
            or v1_record.get("preprocessing_sha256") != sha256(REPO / "reproduction/cadvit/preprocessing.py")
            or v1_record.get("protocol", {}).get("checkpoint_metric") != "validation_macro_f1"
            or not _same_json(v1_record.get("model_config"), ARCH_CONFIG)):
        raise ValueError("V1 checkpoint, source, or run config differs")
    experiment = v2_record.get("experiment", {})
    if (v2_record.get("pipeline_version") != "V2_BG10_STABLE"
            or v2_record.get("manifest_sha256") != spec["manifest_sha256"]
            or v2.get("epoch") != spec["v2_epoch"] - 1
            or not math.isclose(float(v2.get("best_f1", float("nan"))), spec["v2_best_f1"], rel_tol=0, abs_tol=1e-9)
            or not _same_json(v2_record.get("model"), ARCH_CONFIG)
            or not _same_json(v2_record.get("preprocessing"),
                              {"base": PREPROCESS_CONFIG, "ben_graham": BG_V2_CONFIG})
            or experiment.get("dataset") != dataset or experiment.get("seed") != 42
            or experiment.get("epochs") != 80 or experiment.get("ablation") != "MFB+Attention"
            or experiment.get("physical_batch_size") != 16 or experiment.get("accumulation") != 1
            or experiment.get("balance") is not False or experiment.get("augment") is not True
            or not math.isclose(float(experiment.get("learning_rate", float("nan"))), 0.000186,
                                rel_tol=0, abs_tol=1e-12)
            or v2_record.get("parameter_count") != 5_282_101):
        raise ValueError("V2 checkpoint/source/preprocessing metadata differs")
    if dataset == "DDR":
        recovery = v2.get("recovery")
        if (not isinstance(recovery, dict)
                or recovery.get("manifest_sha256") != spec["manifest_sha256"]
                or recovery.get("loss") != "inverse_sqrt_frequency_weighted_cross_entropy"
                or recovery.get("precision_during_recovery") != "FP32"):
            raise ValueError("DDR V2 recovery metadata differs")
    models = {}
    for version, record in (("v1", v1_record), ("v2", v2_record)):
        config = record["model_config"] if version == "v1" else ARCH_CONFIG
        model = CADViTCandidate(config)
        if sum(p.numel() for p in model.parameters()) != 5_282_101:
            raise ValueError("Model parameter count differs")
        model.load_state_dict(states[version]["model"], strict=True)
        if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError("Non-finite checkpoint tensor")
        models[version] = model.eval()
    provenance = {
        "checkpoint_sha256": {version: spec[f"{version}_sha256"] for version in ("v1", "v2")},
        "checkpoint_epoch": {version: spec[f"{version}_epoch"] for version in ("v1", "v2")},
        "checkpoint_record_sha256": {version: hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for version, record in (("v1", v1_record), ("v2", v2_record))},
        "v2_historical_source": source,
        "model_source_sha256": sha256(REPO / "reproduction/cadvit/model.py"),
        "preprocessing_source_sha256": sha256(REPO / "reproduction/cadvit/preprocessing.py"),
    }
    return models, provenance


def _outside_repository(path):
    path = Path(path).expanduser().resolve()
    if any((directory / ".git").exists() for directory in (path, *path.parents)):
        raise ValueError("Caches and reports must be outside every Git worktree")
    return path


class PreprocessingCache:
    """Shared, content-addressed cache of deterministic uint8 RGB arrays."""

    def __init__(self, root, dataset_root):
        import cv2
        from PIL import __version__ as pillow_version
        from reproduction.cadvit.preprocessing import PREPROCESS_CONFIG, BG_V2_CONFIG

        self.root = _outside_repository(root)
        dataset_root = Path(dataset_root).resolve()
        if self.root == dataset_root or dataset_root in self.root.parents:
            raise ValueError("Preprocessing cache must be outside the raw dataset")
        self.config = {"v1": {"base": PREPROCESS_CONFIG},
                       "v2_bg10": {"base": PREPROCESS_CONFIG, "ben_graham": BG_V2_CONFIG}}
        self.versions = {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version}
        self.source_sha256 = sha256(REPO / "reproduction/cadvit/preprocessing.py")
        self.stats = {"hits": 0, "misses": 0, "invalid_regenerated": 0, "writes": 0,
                      "bytes_written": 0}

    def key(self, image_sha256, variant):
        if variant not in self.config or len(image_sha256) != 64 or any(c not in "0123456789abcdef" for c in image_sha256):
            raise ValueError("Invalid preprocessing variant or source image SHA-256")
        payload = {"schema": PREPROCESS_CACHE_VERSION, "image_sha256": image_sha256,
                   "variant": variant, "config": self.config[variant],
                   "preprocessing_source_sha256": self.source_sha256, "libraries": self.versions}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return key, payload

    def _paths(self, image_sha256, variant):
        key, payload = self.key(image_sha256, variant)
        base = self.root / variant / key[:2] / key
        return base.with_suffix(".npy"), base.with_suffix(".json"), key, payload

    @staticmethod
    def _valid_array(array):
        return isinstance(array, np.ndarray) and array.shape == (512, 512, 3) and array.dtype == np.uint8

    def _read(self, data_path, meta_path, key, payload):
        if not data_path.exists() and not meta_path.exists():
            return None, "miss"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") != key or meta.get("payload") != payload:
                raise ValueError("Preprocessing cache key metadata differs")
            array = np.load(data_path, allow_pickle=False)
            if not self._valid_array(array):
                raise ValueError("Preprocessing cache shape/dtype differs")
            if hashlib.sha256(array.tobytes(order="C")).hexdigest() != meta.get("pixel_sha256"):
                raise ValueError("Preprocessing cache pixel digest differs")
            return array, "hit"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None, "invalid"

    def _write(self, data_path, meta_path, key, payload, array):
        data_path.parent.mkdir(parents=True, exist_ok=True)
        token = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temporary_data = data_path.with_name(data_path.name + token)
        temporary_meta = meta_path.with_name(meta_path.name + token)
        try:
            with temporary_data.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            metadata = {"key": key, "payload": payload,
                        "pixel_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
            with temporary_meta.open("xb") as stream:
                stream.write((json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            # A concurrent reader may see one new file first. It detects the
            # digest/key mismatch and recomputes; it never returns bad pixels.
            os.replace(temporary_data, data_path)
            os.replace(temporary_meta, meta_path)
            self.stats["writes"] += 1
            self.stats["bytes_written"] += data_path.stat().st_size + meta_path.stat().st_size
        finally:
            temporary_data.unlink(missing_ok=True)
            temporary_meta.unlink(missing_ok=True)

    def get(self, image_path, image_sha256, variant, *, source_verified=False):
        """Load a verified hit or lazily regenerate an absent/corrupt entry."""
        from PIL import Image
        from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2

        image_path = Path(image_path)
        if not source_verified:
            require_hash(image_path, image_sha256)
        data_path, meta_path, key, payload = self._paths(image_sha256, variant)
        array, status = self._read(data_path, meta_path, key, payload)
        if status == "hit":
            self.stats["hits"] += 1
            return array
        self.stats["misses" if status == "miss" else "invalid_regenerated"] += 1
        with Image.open(image_path) as image:
            array = (preprocess_fundus(image) if variant == "v1" else preprocess_fundus_v2(image))
        if not self._valid_array(array):
            raise ValueError("Historical preprocessing returned unexpected array")
        self._write(data_path, meta_path, key, payload, array)
        return array

    def disk_bytes(self):
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()) if self.root.exists() else 0


def _runtime_info(device):
    import cv2
    import torch
    from PIL import __version__ as pillow_version

    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def _cache_meta(dataset, rows, provenance, device):
    return {
        "schema": CACHE_VERSION, "dataset": dataset, "split": "val",
        "manifest_sha256": SPECS[dataset]["manifest_sha256"],
        "validation_count": SPECS[dataset]["val_count"], "cache_count": len(rows),
        "image_order_sha256": hashlib.sha256("\n".join(r["image_id"] for r in rows).encode()).hexdigest(),
        "provenance": provenance,
        # The file hash is retained for audit but is not a scientific cache
        # key: loading identical uint8 arrays from disk changes this file.
        "implementation_sha256": sha256(__file__),
        "runtime": _runtime_info(device),
    }


def _prediction_meta_compatible(existing, expected):
    """Cache transport and edits to E0 do not alter saved prediction meaning."""
    fingerprint = existing.get("implementation_sha256") if isinstance(existing, dict) else None
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        return False
    return ({key: value for key, value in existing.items() if key != "implementation_sha256"}
            == {key: value for key, value in expected.items() if key != "implementation_sha256"})


def validate_cache_record(item, row):
    """Reject stale, reordered, non-finite, or inconsistent saved predictions."""
    expected = {"image_id": row["image_id"], "true_grade": int(row["label"]),
                "group_id": row["group_id"], "split": "val", "image_sha256": row["sha256"]}
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError("Cache row differs from frozen validation manifest")
    for key in KEYS:
        value = item.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Missing {key} prediction")
        logits = np.asarray(value.get("logits"), dtype=float)
        probs = np.asarray(value.get("probabilities"), dtype=float)
        if logits.shape != (5,) or probs.shape != (5,) or not np.isfinite(logits).all() or not np.isfinite(probs).all():
            raise ValueError(f"Invalid {key} logits/probabilities")
        if (np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1) > 1e-6):
            raise ValueError(f"Unnormalized {key} probabilities")
        shifted = logits - logits.max()
        expected_probs = np.exp(shifted) / np.exp(shifted).sum()
        if not np.allclose(probs, expected_probs, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{key} probabilities disagree with logits")


def read_cache(cache_dir, rows):
    path = Path(cache_dir) / "cache.jsonl"
    if not path.is_file():
        return []
    saved = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError("Truncated or blank cache line; inspect before resuming")
            if len(saved) >= len(rows):
                raise ValueError("Cache contains too many rows")
            item = json.loads(line)
            validate_cache_record(item, rows[len(saved)])
            saved.append(item)
    return saved


def _complete_prediction_cache(out, args, rows, device, artifact_root):
    """Verify a complete Level 2 cache without deserializing checkpoints."""
    metadata_path = out / "metadata.json"
    if not metadata_path.is_file():
        return False
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = meta.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Prediction cache provenance is malformed")
    spec = SPECS[args.dataset]
    expected_hashes = {version: spec[f"{version}_sha256"] for version in ("v1", "v2")}
    if provenance.get("checkpoint_sha256") != expected_hashes:
        raise ValueError("Prediction cache checkpoint hashes differ")
    if provenance.get("checkpoint_epoch") != {version: spec[f"{version}_epoch"] for version in ("v1", "v2")}:
        raise ValueError("Prediction cache checkpoint epochs differ")
    for version in ("v1", "v2"):
        require_hash(artifact_root / spec[f"{version}_run"] / "best.pt", expected_hashes[version])
    source = verify_v2_source(args.v2_source)
    recorded_source = provenance.get("v2_historical_source", {})
    if not isinstance(recorded_source, dict) or recorded_source.get("sha256") != source["sha256"]:
        raise ValueError("Prediction cache historical source differs")
    for name, key in (("model.py", "model_source_sha256"),
                      ("preprocessing.py", "preprocessing_source_sha256")):
        if provenance.get(key) != sha256(REPO / "reproduction/cadvit" / name):
            raise ValueError(f"Prediction cache {name} source differs")
    expected = _cache_meta(args.dataset, rows, provenance, device)
    if not _prediction_meta_compatible(meta, expected):
        raise ValueError("Existing prediction cache metadata/runtime differs")
    saved = read_cache(out, rows)
    return len(saved) == len(rows)


def cache_predictions(args):
    import torch
    from PIL import Image
    from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2

    dataset = args.dataset
    root = Path(args.artifact_root).expanduser().resolve()
    dataset_root = (root / "datasets" / dataset).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest = root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    all_rows = validation_rows(manifest, dataset, dataset_root)
    limit = len(all_rows) if args.limit is None else args.limit
    if not 1 <= limit <= len(all_rows):
        raise ValueError("--limit must select 1..validation_count images")
    rows = all_rows[:limit]
    if args.command == "preflight":
        models, provenance = checked_models(dataset, root, args.v2_source, args.trust_checkpoints)
        print(json.dumps({"status": "PASS", "dataset": dataset, "split": "val",
                          "validation_count": len(all_rows), "requested_count": limit,
                          "provenance": provenance}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    out = _outside_repository(args.output_dir)
    if out == dataset_root or dataset_root in out.parents:
        raise ValueError("Cache must not be written into raw dataset directory")
    if out.exists() and _complete_prediction_cache(out, args, rows, device, root):
        print(f"COMPLETE: {dataset} validation cache {len(rows)}/{len(rows)}; no checkpoint or preprocessing load")
        return
    if out.exists() and not args.resume:
        raise FileExistsError("Incomplete cache directory exists; use --resume for an identical cache")
    if not args.uncached and not args.preprocess_cache_root:
        raise ValueError("Specify --preprocess-cache-root or --uncached for incomplete prediction caches")
    preprocessing_cache = (None if args.uncached else
                           PreprocessingCache(args.preprocess_cache_root, dataset_root))
    models, provenance = checked_models(dataset, root, args.v2_source, args.trust_checkpoints)
    for model in models.values():
        model.to(device).eval()
    meta = _cache_meta(dataset, rows, provenance, device)
    metadata_path = out / "metadata.json"
    if out.exists():
        if not metadata_path.is_file():
            raise FileExistsError("Cache directory exists; use --resume only for an identical cache")
        if not _prediction_meta_compatible(json.loads(metadata_path.read_text(encoding="utf-8")), meta):
            raise ValueError("Existing cache provenance/runtime differs; refuse stale reuse")
    else:
        out.mkdir(parents=True)
        metadata_path.write_bytes((json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
    saved = read_cache(out, rows)
    if len(saved) == len(rows):
        print(f"COMPLETE: {dataset} validation cache {len(saved)}/{len(rows)}; no inference needed")
        return
    cache_path = out / "cache.jsonl"
    with cache_path.open("ab") as stream, torch.inference_mode():
        for row in rows[len(saved):]:
            rel = PurePosixPath(row["image_path"])
            image_path = (dataset_root / Path(*rel.parts)).resolve()
            if dataset_root not in image_path.parents:
                raise ValueError("Validation image path escapes dataset root")
            require_hash(image_path, row["sha256"])
            if preprocessing_cache is None:
                with Image.open(image_path) as image:
                    pixels_v1 = preprocess_fundus(image)
                    pixels_v2 = preprocess_fundus_v2(image)
            else:
                pixels_v1 = preprocessing_cache.get(image_path, row["sha256"], "v1", source_verified=True)
                pixels_v2 = preprocessing_cache.get(image_path, row["sha256"], "v2_bg10", source_verified=True)
            inputs = {}
            for key, pixels in (("v1", pixels_v1), ("v2", pixels_v2)):
                if pixels.shape != (512, 512, 3) or pixels.dtype != np.uint8:
                    raise ValueError("Historical preprocessing returned unexpected array")
                inputs[key] = torch.from_numpy(np.array(pixels, copy=True)).permute(2, 0, 1).float()[None].to(device) / 255
            item = {"image_id": row["image_id"], "true_grade": int(row["label"]),
                    "group_id": row["group_id"], "split": "val", "image_sha256": row["sha256"]}
            for key, model_key, input_key in (("p11", "v1", "v1"), ("p12", "v1", "v2"),
                                              ("p21", "v2", "v1"), ("p22", "v2", "v2")):
                logits = models[model_key](inputs[input_key])
                if logits.shape != (1, 5) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f"Non-finite or malformed {key} logits")
                item[key] = {"logits": logits[0].cpu().tolist(),
                             "probabilities": logits.softmax(1)[0].cpu().tolist()}
            validate_cache_record(item, row)
            stream.write((json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{dataset}: {len(saved) + 1}/{len(rows)} {row['image_id']}", flush=True)
            saved.append(item)
    assert len(read_cache(out, rows)) == len(rows)
    print(f"COMPLETE: {dataset} validation cache {len(rows)}/{len(all_rows)}")
    if preprocessing_cache is not None:
        stats = dict(preprocessing_cache.stats)
        requests = stats["hits"] + stats["misses"] + stats["invalid_regenerated"]
        stats["hit_rate"] = stats["hits"] / requests if requests else 0.0
        stats["disk_bytes"] = preprocessing_cache.disk_bytes()
        print(json.dumps({"preprocessing_cache": stats}, indent=2))


def cumulative(probabilities):
    """Five grade probabilities -> P(Y >= k), k=1..4."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("Expected N x 5 probabilities")
    return np.column_stack([probabilities[:, k:].sum(axis=1) for k in range(1, 5)])


def feature_sets(cache, boundary, native):
    probs = {key: np.asarray([row[key]["probabilities"] for row in cache], dtype=float) for key in KEYS}
    q = {key: cumulative(value)[:, boundary - 1] for key, value in probs.items()}
    own = "p11" if native == "V1" else "p22"
    own_p, own_q = probs[own], q[own]
    binary_entropy = -(own_q * np.log(np.clip(own_q, 1e-12, 1))
                       + (1 - own_q) * np.log(np.clip(1 - own_q, 1e-12, 1)))
    five_entropy = -(own_p * np.log(np.clip(own_p, 1e-12, 1))).sum(axis=1)
    a = np.column_stack([own_p.max(axis=1), five_entropy,
                         2 * np.abs(own_q - 0.5), binary_entropy])
    b = np.column_stack([a, np.abs(q["p11"] - q["p22"]),
                         (q["p11"] >= 0.5) != (q["p22"] >= 0.5)])
    c = np.column_stack([b, np.abs(q["p11"] - q["p12"]),
                         np.abs(q["p21"] - q["p22"])])
    d = np.column_stack([probs[key] for key in KEYS])
    return {"A": a, "B": b, "C": c, "D": d}


def grouped_folds(groups, count=5, seed=42):
    """One deterministic fold assignment per dataset, shared by every comparison."""
    groups = np.asarray(groups)
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < count or count < 2:
        raise ValueError("Too few distinct groups for grouped out-of-fold evaluation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = sorted(order, key=lambda i: -sizes[i])
    fold_sizes = np.zeros(count, dtype=int)
    assignment = np.empty(len(unique), dtype=int)
    for group_index in order:
        fold = int(np.argmin(fold_sizes))
        assignment[group_index] = fold
        fold_sizes[fold] += sizes[group_index]
    folds = assignment[inverse]
    if any(not np.any(folds == i) for i in range(count)):
        raise ValueError("Empty group fold")
    return folds


def _sigmoid(values):
    values = np.clip(values, -40, 40)
    return 1 / (1 + np.exp(-values))


def fit_logistic(train_x, train_y, test_x, penalty=1.0):
    """Train-only standardization and fixed-capacity L2 logistic estimator."""
    train_x, test_x = np.asarray(train_x, float), np.asarray(test_x, float)
    train_y = np.asarray(train_y, float)
    if set(np.unique(train_y)) != {0.0, 1.0}:
        raise ValueError("Fold training labels contain only one error class")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Non-finite E0 feature")
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_x, test_x = (train_x - mean) / scale, (test_x - mean) / scale
    train_x = np.column_stack([train_x, np.ones(len(train_x))])
    test_x = np.column_stack([test_x, np.ones(len(test_x))])
    regularization = np.full(train_x.shape[1], penalty)
    regularization[-1] = 0.0  # intercept
    prior = np.clip(train_y.mean(), 1e-4, 1 - 1e-4)
    weights = np.zeros(train_x.shape[1])
    weights[-1] = math.log(prior / (1 - prior))

    def objective(candidate):
        z = train_x @ candidate
        return (np.logaddexp(0, z) - train_y * z).sum() + 0.5 * (regularization * candidate**2).sum()

    for _ in range(80):
        probabilities = _sigmoid(train_x @ weights)
        gradient = train_x.T @ (probabilities - train_y) + regularization * weights
        variance = probabilities * (1 - probabilities)
        hessian = train_x.T @ (variance[:, None] * train_x) + np.diag(regularization)
        step = np.linalg.solve(hessian + np.eye(len(weights)) * 1e-8, gradient)
        if np.max(np.abs(step)) < 1e-8:
            break
        rate = 1.0
        previous = objective(weights)
        while rate > 1e-8 and objective(weights - rate * step) > previous:
            rate *= 0.5
        weights -= rate * step
        if rate <= 1e-8:
            break
    return _sigmoid(test_x @ weights)


def metric_bundle(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("AUROC/AUPRC require both error classes")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    auc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = fp = 0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        added = int(sorted_labels[start:end].sum())
        tp += added
        fp += end - start - added
        average_precision += added / positives * tp / (tp + fp)
        start = end
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    return {"auroc": float(auc), "auprc": float(average_precision),
            "brier": float(np.mean((scores - labels)**2)),
            "log_loss": float(np.mean(-labels * np.log(clipped) - (1 - labels) * np.log(1 - clipped)))}


def paired_group_bootstrap(labels, b_scores, c_scores, groups, repeats=500, seed=42):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("auroc", "auprc", "brier", "log_loss")}
    for _ in range(repeats):
        sample = np.concatenate([indices[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(labels[sample])) != 2:
            continue
        b = metric_bundle(labels[sample], b_scores[sample])
        c = metric_bundle(labels[sample], c_scores[sample])
        for name in draws:
            draws[name].append(c[name] - b[name])
    if len(draws["auroc"]) < max(30, repeats // 2):
        return {"status": "insufficient valid paired group resamples"}
    return {name: {"delta_C_minus_B": float(np.mean(values)),
                   "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])]}
            for name, values in draws.items()}


def analyze_e0(args):
    if args.bootstrap < 30:
        raise ValueError("At least 30 paired group bootstrap draws are required")
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = meta.get("dataset")
    if dataset not in SPECS or meta.get("schema") != CACHE_VERSION or meta.get("split") != "val":
        raise ValueError("Unknown or non-validation cache")
    spec = SPECS[dataset]
    if (meta.get("validation_count") != spec["val_count"] or meta.get("cache_count") != spec["val_count"]
            or meta.get("manifest_sha256") != spec["manifest_sha256"]
            or not isinstance(meta.get("implementation_sha256"), str)
            or len(meta["implementation_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in meta["implementation_sha256"])
            or meta.get("provenance", {}).get("checkpoint_sha256") !=
               {version: spec[f"{version}_sha256"] for version in ("v1", "v2")}):
        raise ValueError("Partial or stale cache provenance")
    source = meta["provenance"]["v2_historical_source"]
    require_hash(source["path"], source["sha256"])
    for name, key in (("model.py", "model_source_sha256"),
                      ("preprocessing.py", "preprocessing_source_sha256")):
        require_hash(REPO / "reproduction/cadvit" / name, meta["provenance"][key])
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    dataset_root = (artifact_root / "datasets" / dataset).resolve()
    manifest = artifact_root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    rows = validation_rows(manifest, dataset, dataset_root)
    if meta.get("image_order_sha256") != hashlib.sha256("\n".join(r["image_id"] for r in rows).encode()).hexdigest():
        raise ValueError("Cache order differs from frozen validation manifest")
    cache = read_cache(cache_dir, rows)
    if len(cache) != spec["val_count"]:
        raise ValueError("Incomplete validation prediction cache")
    groups = np.asarray([row["group_id"] for row in cache])
    grades = np.asarray([row["true_grade"] for row in cache])
    folds = grouped_folds(groups, args.folds, args.seed)
    native_pred = {name: np.argmax(np.asarray([row[key]["probabilities"] for row in cache]), axis=1)
                   for name, key in (("V1", "p11"), ("V2", "p22"))}
    native_wrong = {name: prediction != grades for name, prediction in native_pred.items()}
    result = {"status": "E0_COMPLETE", "dataset": dataset, "split": "val",
              "cache_metadata_sha256": sha256(cache_dir / "metadata.json"),
              "cache_sha256": sha256(cache_dir / "cache.jsonl"),
              "folds": args.folds, "seed": args.seed, "estimator": "train-only standardized L2 logistic; lambda=1",
              "grade_support": {str(k): int(np.sum(grades == k)) for k in range(5)},
              "five_class_complementary_errors": {
                  "both_correct": int(np.sum(~native_wrong["V1"] & ~native_wrong["V2"])),
                  "V1_only_wrong": int(np.sum(native_wrong["V1"] & ~native_wrong["V2"])),
                  "V2_only_wrong": int(np.sum(~native_wrong["V1"] & native_wrong["V2"])),
                  "both_wrong": int(np.sum(native_wrong["V1"] & native_wrong["V2"]))},
              "comparisons": []}
    for boundary in range(1, 5):
        truth = (grades >= boundary).astype(int)
        native_q = {native: cumulative(np.asarray([row[key]["probabilities"] for row in cache]))[:, boundary - 1]
                    for native, key in (("V1", "p11"), ("V2", "p22"))}
        errors = {native: ((q >= 0.5).astype(int) != truth).astype(int) for native, q in native_q.items()}
        complementary = {"both_correct": int(np.sum((errors["V1"] == 0) & (errors["V2"] == 0))),
                         "V1_only_wrong": int(np.sum((errors["V1"] == 1) & (errors["V2"] == 0))),
                         "V2_only_wrong": int(np.sum((errors["V1"] == 0) & (errors["V2"] == 1))),
                         "both_wrong": int(np.sum((errors["V1"] == 1) & (errors["V2"] == 1)))}
        for native in ("V1", "V2"):
            target = errors[native]
            item = {"boundary": boundary, "native_expert": native,
                    "support_ge_k": int(truth.sum()), "support_lt_k": int(len(truth) - truth.sum()),
                    "error_count": int(target.sum()), "complementary_errors": complementary}
            if len(np.unique(target)) != 2:
                item["status"] = "UNESTIMABLE: only one error class"
                result["comparisons"].append(item)
                continue
            features = feature_sets(cache, boundary, native)
            oof = {name: np.empty(len(cache), dtype=float) for name in features}
            try:
                for fold in range(args.folds):
                    train, held = folds != fold, folds == fold
                    if set(groups[train]) & set(groups[held]):
                        raise AssertionError("Group leakage")
                    for name, values in features.items():
                        oof[name][held] = fit_logistic(values[train], target[train], values[held])
            except (ValueError, np.linalg.LinAlgError) as exc:
                item["status"] = f"UNESTIMABLE: {exc}"
                result["comparisons"].append(item)
                continue
            item["status"] = "OK"
            item["metrics"] = {name: metric_bundle(target, scores) for name, scores in oof.items()}
            item["paired_group_bootstrap_C_minus_B"] = paired_group_bootstrap(
                target, oof["B"], oof["C"], groups, args.bootstrap, args.seed)
            result["comparisons"].append(item)
    if any(item["status"] != "OK" for item in result["comparisons"]):
        result["status"] = "E0_PARTIAL"
    out = _outside_repository(args.output_dir)
    if out.exists():
        raise FileExistsError("E0 output directory exists; results are never overwritten")
    out.mkdir(parents=True)
    (out / "e0_report.json").write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    lines = [f"# PReOF E0 — {dataset} validation only", "",
             f"Cache SHA-256: `{result['cache_sha256']}`. Grouped {args.folds}-fold OOF; fixed L2 logistic estimator.",
             "A=native confidence/entropy; B=+native disagreement; C=+cross-preprocessing response; D=all four distributions.",
             "Positive target: incorrect native boundary decision at P(Y >= k) >= 0.5. No test data used.", "",
             f"Grade support: `{result['grade_support']}`. Five-class complementary errors: `{result['five_class_complementary_errors']}`.", "",
             "| Boundary | Expert | Grade >= k | Errors | AUROC A/B/C/D | AUPRC A/B/C/D | Brier A/B/C/D | Log loss A/B/C/D |",
             "| ---: | --- | ---: | ---: | --- | --- | --- | --- |"]
    for item in result["comparisons"]:
        if item["status"] == "OK":
            m = item["metrics"]
            auc = "/".join(f"{m[k]['auroc']:.4f}" for k in "ABCD")
            ap = "/".join(f"{m[k]['auprc']:.4f}" for k in "ABCD")
            br = "/".join(f"{m[k]['brier']:.4f}" for k in "ABCD")
            ll = "/".join(f"{m[k]['log_loss']:.4f}" for k in "ABCD")
        else:
            auc = ap = br = ll = item["status"]
        lines.append(f"| {item['boundary']} | {item['native_expert']} | {item['support_ge_k']} | {item['error_count']} | {auc} | {ap} | {br} | {ll} |")
    lines += ["", "## Paired group bootstrap: C−B", "",
              "AUPRC/AUROC delta > 0 and Brier/log-loss delta < 0 favor C. Intervals reflect resampled validation groups.", "",
              "| Boundary | Expert | Δ AUPRC [95% CI] | Δ Brier [95% CI] |",
              "| ---: | --- | --- | --- |"]
    for item in result["comparisons"]:
        intervals = item.get("paired_group_bootstrap_C_minus_B", {})
        if item["status"] == "OK" and "auprc" in intervals:
            ap = intervals["auprc"]
            br = intervals["brier"]
            ap_text = f"{ap['delta_C_minus_B']:.4f} [{ap['ci95'][0]:.4f}, {ap['ci95'][1]:.4f}]"
            br_text = f"{br['delta_C_minus_B']:.4f} [{br['ci95'][0]:.4f}, {br['ci95'][1]:.4f}]"
        else:
            ap_text = br_text = item["status"]
        lines.append(f"| {item['boundary']} | {item['native_expert']} | {ap_text} | {br_text} |")
    lines += ["", "Exact per-boundary metrics, all four bootstrap intervals, and complementary error counts are in `e0_report.json`.",
              "The models differ in training protocol as well as preprocessing. Sensitivity alone does not identify an incorrect prediction or a lesion."]
    (out / "e0_report.md").write_bytes(("\n".join(lines) + "\n").encode())
    print(json.dumps({"status": result["status"], "dataset": dataset,
                      "comparison_rows": len(result["comparisons"]), "output_dir": str(out)}, indent=2))


def benchmark_preprocessing(args):
    """Time validation-only deterministic pixels; no model or inference."""
    from PIL import Image
    from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2

    root = Path(args.artifact_root).expanduser().resolve()
    dataset_root = (root / "datasets" / args.dataset).resolve()
    rows = validation_rows(root / "cad_vit_runs" / "manifests" / f"{args.dataset}_seed42.csv",
                           args.dataset, dataset_root)
    if not 1 <= args.limit <= len(rows):
        raise ValueError("--limit must select 1..validation_count images")
    rows = rows[:args.limit]
    cache = PreprocessingCache(args.preprocess_cache_root, dataset_root)
    samples = []
    for row in rows:
        rel = PurePosixPath(row["image_path"])
        path = (dataset_root / Path(*rel.parts)).resolve()
        if dataset_root not in path.parents:
            raise ValueError("Validation image path escapes dataset root")
        require_hash(path, row["sha256"])
        samples.append((path, row["sha256"]))
    direct = []
    started = time.perf_counter()
    for path, _ in samples:
        with Image.open(path) as image:
            direct.append((preprocess_fundus(image), preprocess_fundus_v2(image)))
    uncached_seconds = time.perf_counter() - started

    def pass_once():
        before = dict(cache.stats)
        started = time.perf_counter()
        arrays = [(cache.get(path, digest, "v1", source_verified=True),
                   cache.get(path, digest, "v2_bg10", source_verified=True)) for path, digest in samples]
        elapsed = time.perf_counter() - started
        hits = cache.stats["hits"] - before["hits"]
        total = 2 * len(samples)
        return arrays, {"seconds": elapsed, "hits": hits, "requests": total,
                        "hit_rate": hits / total}

    first, first_stats = pass_once()
    warm, warm_stats = pass_once()
    equivalent = all(np.array_equal(a, b) and np.array_equal(a, c)
                     for direct_pair, first_pair, warm_pair in zip(direct, first, warm)
                     for a, b, c in zip(direct_pair, first_pair, warm_pair))
    report = {"dataset": args.dataset, "split": "val", "images": len(rows),
              "uncached_seconds": uncached_seconds, "first_cache_pass": first_stats,
              "warm_cache_pass": warm_stats, "pixel_exact": equivalent,
              "cache_stats": cache.stats, "disk_bytes": cache.disk_bytes(),
              "note": "First pass may contain hits if a compatible cache already exists."}
    print(json.dumps(report, indent=2))
    if not equivalent:
        raise ValueError("Cached and uncached preprocessing pixels differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "cache"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--dataset", choices=tuple(SPECS), required=True)
        cmd.add_argument("--artifact-root", required=True)
        cmd.add_argument("--v2-source", required=True, help="Historical server CAD_ViT.ipynb or its repository directory")
        cmd.add_argument("--trust-checkpoints", action="store_true")
        cmd.add_argument("--limit", type=int, help="Validation-only prefix for a separate smoke cache")
        if name == "cache":
            cmd.add_argument("--output-dir", required=True)
            cmd.add_argument("--device", default="cpu")
            cmd.add_argument("--resume", action="store_true")
            mode = cmd.add_mutually_exclusive_group()
            mode.add_argument("--preprocess-cache-root")
            mode.add_argument("--uncached", action="store_true")
    cmd = commands.add_parser("benchmark", help="Validation-only preprocessing timing, no inference")
    cmd.add_argument("--dataset", choices=tuple(SPECS), required=True)
    cmd.add_argument("--artifact-root", required=True)
    cmd.add_argument("--preprocess-cache-root", required=True)
    cmd.add_argument("--limit", type=int, default=3)
    cmd = commands.add_parser("analyze")
    cmd.add_argument("--cache-dir", required=True)
    cmd.add_argument("--artifact-root", required=True)
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--folds", type=int, default=5)
    cmd.add_argument("--bootstrap", type=int, default=500)
    cmd.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command in {"preflight", "cache"}:
        cache_predictions(args)
    elif args.command == "benchmark":
        benchmark_preprocessing(args)
    else:
        analyze_e0(args)


if __name__ == "__main__":
    main()
