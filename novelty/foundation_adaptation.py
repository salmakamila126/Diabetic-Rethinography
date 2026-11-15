"""E4/E5 conventional foundation-backbone adaptation, train/validation only.

All model files must already exist in a pinned local Hugging Face snapshot.
This module never contacts the Hub, downloads weights, or reads test images.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import time
import uuid

import numpy as np

from novelty import preof_e1 as e1
from novelty import preof_phase1 as phase


SCHEMA = "foundation-adaptation-v1"
LEGACY_20_SOURCE_SHA256 = "289d50684fe9d8ca06c9237d5318ebc449346a6e51f75a659f3ecfcfc0e112b1"
CACHE_SCHEMA = "foundation-frozen-vectors-v1"
MODELS = {
    "dinov3_vits16": {"hub_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
                      "model_type": "dinov3_vit", "dimension": 384,
                      "weight_kind": "safetensors", "partial_blocks": 2},
    "convnext_tiny": {"hub_id": "facebook/convnext-tiny-224",
                      "model_type": "convnext", "dimension": 768,
                      "weight_kind": "bin", "partial_blocks": 2},
    "dinov3_convnext_tiny": {"hub_id": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                             "model_type": "dinov3_convnext", "dimension": 768,
                             "weight_kind": "safetensors", "partial_blocks": None},
}


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_out(path, protected):
    output = phase._outside_repository(path)
    for root in protected:
        root = Path(root).expanduser().resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Output must be separate from model snapshots, datasets, and caches")
    return output


def load_train_val_manifest(artifact_root, dataset, *, verify_images=True):
    """Verify frozen manifest, split separation, and train/val image integrity.

    The manifest file is hashed as bytes; test rows are skipped before their
    label/path fields are validated or used. Test images are never opened.
    """
    if dataset not in phase.SPECS:
        raise ValueError("Unknown dataset")
    root = Path(artifact_root).expanduser().resolve()
    dataset_root = (root / "datasets" / dataset).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest = root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    spec = phase.SPECS[dataset]
    phase.require_hash(manifest, spec["manifest_sha256"])
    metadata = json.loads(manifest.with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata.get("dataset") != dataset or metadata.get("seed") != 42
            or metadata.get("manifest_file_sha256") != spec["manifest_sha256"]
            or Path(metadata.get("root", "")).expanduser().resolve() != dataset_root):
        raise ValueError("Frozen manifest metadata differs")
    rows = {"train": [], "val": []}
    ids, paths, image_split, group_split = set(), set(), {}, {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "image_path", "label", "split", "group_id", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Frozen manifest columns incomplete")
        for raw in reader:
            split = raw["split"]
            if split not in rows:
                if split != "test":
                    raise ValueError("Unknown frozen-manifest split")
                continue
            image_id, relative = raw["image_id"], PurePosixPath(raw["image_path"])
            digest, group = raw["sha256"], raw["group_id"]
            if (raw["label"] not in {str(i) for i in range(5)} or not image_id or not group
                    or not relative.parts or relative.is_absolute() or ".." in relative.parts
                    or "\\" in raw["image_path"] or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("Invalid train/validation manifest row")
            if (image_id in ids or raw["image_path"] in paths
                    or (digest in image_split and image_split[digest] != split)
                    or (group in group_split and group_split[group] != split)):
                raise ValueError("Duplicate image or cross-split group/content leakage")
            ids.add(image_id)
            paths.add(raw["image_path"])
            image_split[digest], group_split[group] = split, split
            path = (dataset_root / Path(*relative.parts)).resolve()
            if dataset_root not in path.parents:
                raise ValueError("Image path escapes frozen dataset root")
            item = {"image_id": image_id, "image_path": raw["image_path"],
                    "path": str(path), "label": int(raw["label"]), "group_id": group,
                    "sha256": digest, "split": split}
            if verify_images:
                phase.require_hash(path, digest)
            rows[split].append(item)
    if len(rows["val"]) != spec["val_count"] or not rows["train"]:
        raise ValueError("Train/validation counts differ from frozen manifest")
    support = {split: [sum(item["label"] == grade for item in rows[split])
                       for grade in range(5)] for split in rows}
    if any(count == 0 for split in support.values() for count in split):
        raise ValueError("A grade is absent from train or validation")
    return rows, {"dataset": dataset, "manifest_sha256": spec["manifest_sha256"],
                  "dataset_root": str(dataset_root),
                  "counts": {split: len(part) for split, part in rows.items()},
                  "grade_support": support,
                  "group_count": {split: len({row["group_id"] for row in part})
                                  for split, part in rows.items()},
                  "image_order_sha256": _digest_json([(r["split"], r["image_id"], r["sha256"])
                                                       for split in ("train", "val") for r in rows[split]]),
                  "images_verified": verify_images}


def audit_snapshot(model_kind, model_dir, revision, weight_sha256):
    """Read local metadata and weight bytes only; no model import or download."""
    if model_kind not in MODELS:
        raise ValueError("Unknown model kind")
    if (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)
            or len(weight_sha256) != 64 or any(c not in "0123456789abcdef" for c in weight_sha256)):
        raise ValueError("Require full lowercase 40-char revision and 64-char weight SHA-256")
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir() or directory.name != revision:
        raise ValueError("Model directory must be a pinned HF snapshots/<revision> directory")
    phase._outside_repository(directory)
    config_path = directory / "config.json"
    processor_path = directory / "preprocessor_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    spec = MODELS[model_kind]
    if config.get("model_type") != spec["model_type"]:
        raise ValueError("Model config type does not match requested backbone")
    files = [path for path in directory.iterdir() if path.name in {"model.safetensors", "pytorch_model.bin"}]
    if len(files) != 1:
        raise ValueError("Require exactly one official single-file model weight")
    weight = files[0]
    if spec["weight_kind"] == "safetensors" and weight.suffix != ".safetensors":
        raise ValueError("DINOv3 weight must use safetensors")
    if spec["weight_kind"] == "bin" and weight.name != "pytorch_model.bin":
        raise ValueError("This official ConvNeXt snapshot expects pytorch_model.bin")
    phase.require_hash(weight, weight_sha256)
    return {"schema": SCHEMA, "hub_id": spec["hub_id"], "model_kind": model_kind,
            "model_type": spec["model_type"], "revision": revision,
            "revision_evidence": "snapshot directory name; record HF download receipt separately",
            "model_dir": str(directory), "config_sha256": phase.sha256(config_path),
            "processor_config_sha256": phase.sha256(processor_path),
            "weight_file": weight.name, "weight_sha256": weight_sha256,
            "weight_bytes": weight.stat().st_size,
            "declared_embedding_dimension": spec["dimension"],
            "config": config, "processor_config": processor}


def load_local_processor(snapshot):
    """Read the pinned official processor without loading model weights."""
    import transformers
    from transformers import AutoImageProcessor

    if transformers.__version__ != "4.57.3":
        raise ValueError("E4/E5 requires audited transformers==4.57.3")
    return AutoImageProcessor.from_pretrained(
        snapshot["model_dir"], local_files_only=True, trust_remote_code=False, use_fast=False)


def load_local_backbone(snapshot, *, trust_legacy_bin=False):
    """Load only hash-checked local official weights, never Hub or remote code."""
    if snapshot["weight_file"] == "pytorch_model.bin" and not trust_legacy_bin:
        raise ValueError("Legacy .bin loading requires --trust-legacy-bin after hash verification")
    from transformers import AutoModel

    processor = load_local_processor(snapshot)
    model_options = ({"attn_implementation": "eager"}
                     if snapshot["model_kind"] == "dinov3_vits16" else {})
    backbone = AutoModel.from_pretrained(
        snapshot["model_dir"], local_files_only=True, trust_remote_code=False,
        use_safetensors=snapshot["weight_file"] == "model.safetensors", **model_options)
    if backbone.config.model_type != snapshot["model_type"]:
        raise ValueError("Loaded model type changed")
    return backbone, processor


def pooled(backbone, pixels, model_kind):
    import torch

    output = backbone(pixel_values=pixels)
    vector = output.pooler_output
    expected = MODELS[model_kind]["dimension"]
    if vector is None or tuple(vector.shape) != (len(pixels), expected) or not torch.isfinite(vector).all():
        raise ValueError("Backbone pooled embedding shape/values differ from audited design")
    hidden = output.last_hidden_state
    if model_kind == "dinov3_vits16" and tuple(hidden.shape) != (len(pixels), 201, 384):
        raise ValueError("DINOv3 ViT-S token shape differs at 224x224")
    if model_kind == "convnext_tiny" and tuple(hidden.shape) != (len(pixels), 768, 7, 7):
        raise ValueError("ConvNeXt Tiny feature shape differs at 224x224")
    return vector


def configure_trainable(backbone, model_kind, mode):
    """Select full, partial, or frozen backbone parameters; PEFT remains separate."""
    if mode == "peft":
        raise NotImplementedError("PEFT requires a separate reviewed adapter protocol")
    if mode not in {"frozen", "partial", "full"}:
        raise ValueError("Mode must be frozen, partial, or full")
    if mode == "full":
        if model_kind not in {"dinov3_vits16", "convnext_tiny"}:
            raise ValueError("Full tuning is audited only for the two primary backbones")
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        if not any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("Full tuning selected no backbone parameters")
        return ["entire_backbone"]
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if mode == "frozen":
        return []
    if model_kind == "dinov3_vits16":
        # transformers==4.57.3 exposes the 12 ViT blocks directly as .layer.
        layers = backbone.layer
        if len(layers) != 12 or backbone.config.hidden_size != 384:
            raise ValueError("DINOv3 ViT-S layer layout differs")
        names = ["layer.10", "layer.11", "norm"]
        modules = [layers[10], layers[11], backbone.norm]
    elif model_kind == "convnext_tiny":
        stages = backbone.encoder.stages
        if len(stages) != 4 or len(stages[-1].layers) != 3 or backbone.config.hidden_sizes[-1] != 768:
            raise ValueError("ConvNeXt Tiny stage layout differs")
        names = ["encoder.stages.3.layers.1", "encoder.stages.3.layers.2", "layernorm"]
        modules = [stages[-1].layers[1], stages[-1].layers[2], backbone.layernorm]
    else:
        raise ValueError("Optional DINOv3 ConvNeXt partial tuning awaits module audit")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if not any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("Partial tuning selected no trainable parameters")
    return names


def _image_batch(rows, processor, *, augment_seed=None, augment_epoch=None):
    from PIL import Image, ImageOps
    import torch

    images = []
    for row in rows:
        if augment_epoch is not None and row.get("split") != "train":
            raise ValueError("Stochastic augmentation is train-only")
        phase.require_hash(row["path"], row["sha256"])
        with Image.open(row["path"]) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        if augment_epoch is not None:
            from novelty import foundation_augmentation as augment
            rgb, _ = augment.apply(rgb, seed=augment_seed, epoch=augment_epoch,
                                   image_id=row["image_id"])
        images.append(rgb)
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (len(rows), 3, 224, 224):
        raise ValueError("Official processor did not produce Bx3x224x224 pixels")
    if not torch.isfinite(pixels).all():
        raise ValueError("Non-finite processor output")
    return pixels, torch.tensor([row["label"] for row in rows], dtype=torch.long)


def _seed(seed):
    import torch

    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _precision(device, requested):
    import torch

    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported precision")
    if device.type != "cuda":
        if requested not in {"auto", "fp32"}:
            raise ValueError("Mixed precision requires CUDA")
        return "fp32"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    return requested


def _autocast(device, precision):
    import torch

    return (torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
            if device.type == "cuda" and precision != "fp32" else contextlib.nullcontext())


def _parameter_counts(backbone, head):
    return {"backbone_total": sum(p.numel() for p in backbone.parameters()),
            "backbone_trainable": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            "head_total": sum(p.numel() for p in head.parameters()),
            "total_trainable": sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            + sum(p.numel() for p in head.parameters())}


def _classification(truth, probabilities):
    base = e1.classification(np.asarray(truth, int), np.asarray(probabilities, float))
    support = np.asarray([item["support"] for item in base["per_grade"]], dtype=float)
    scores = np.asarray([item["f1"] for item in base["per_grade"]], dtype=float)
    base["weighted_f1"] = float(np.dot(support, scores) / support.sum())
    return base


def _metrics(truth, probabilities, loss):
    base = _classification(truth, probabilities)
    base["validation_loss"] = float(loss)
    return base


def _cache_metadata(snapshot, manifest_info, processor, count, backbone):
    import torch
    import transformers
    from PIL import __version__ as pillow_version

    return {"schema": CACHE_SCHEMA, "snapshot": {key: snapshot[key] for key in
            ("hub_id", "model_kind", "revision", "config_sha256", "processor_config_sha256",
             "weight_file", "weight_sha256")},
            "manifest": manifest_info, "row_count": count,
            "vector_dimension": MODELS[snapshot["model_kind"]]["dimension"],
            "vector_dtype": "float32", "pooling": "official AutoModel.pooler_output",
            "backbone_parameter_count": sum(p.numel() for p in backbone.parameters()),
            "processor_effective": processor.to_dict(),
            "processor_effective_sha256": _digest_json(processor.to_dict()),
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__, "pillow": pillow_version},
            "implementation_sha256": phase.sha256(__file__)}


def _read_feature_index(root, rows, metadata):
    root = Path(root)
    vector_path, index_path = root / "features.npy", root / "rows.jsonl"
    if not vector_path.is_file():
        raise FileNotFoundError(vector_path)
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(rows), metadata["vector_dimension"])
    if vectors.shape != expected_shape or vectors.dtype != np.float32:
        raise ValueError("Frozen feature cache shape/dtype differs")
    entries = []
    if index_path.exists():
        with index_path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n") or not line.strip() or len(entries) >= len(rows):
                    raise ValueError("Frozen feature index has truncated or extra row")
                entry = json.loads(line)
                source = rows[len(entries)]
                expected = {key: source[key] for key in
                            ("image_id", "split", "label", "group_id", "sha256")}
                if any(entry.get(key) != value for key, value in expected.items()):
                    raise ValueError("Frozen feature index differs from manifest")
                vector = np.asarray(vectors[len(entries)])
                if not np.isfinite(vector).all() or hashlib.sha256(vector.tobytes()).hexdigest() != entry.get("vector_sha256"):
                    raise ValueError("Frozen feature vector checksum/values differ")
                entries.append(entry)
    return vectors, entries


def _feature_cache(args, rows, manifest_info, snapshot):
    """Load a verified complete cache; never use this function for fine-tuning."""
    root = Path(args.feature_cache_dir).expanduser().resolve()
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_snapshot = {key: snapshot[key] for key in
                         ("hub_id", "model_kind", "revision", "config_sha256",
                          "processor_config_sha256", "weight_file", "weight_sha256")}
    if (meta.get("schema") != CACHE_SCHEMA or meta.get("snapshot") != expected_snapshot
            or meta.get("manifest") != manifest_info or meta.get("row_count") != len(rows)
            or meta.get("vector_dimension") != MODELS[snapshot["model_kind"]]["dimension"]
            or meta.get("vector_dtype") != "float32"):
        raise ValueError("Frozen feature cache provenance differs")
    vectors, entries = _read_feature_index(root, rows, meta)
    if len(entries) != len(rows):
        raise ValueError("Frozen feature cache incomplete")
    return vectors, meta


def extract_features(args):
    import torch

    rows_by_split, manifest_info = load_train_val_manifest(args.artifact_root, args.dataset)
    rows = rows_by_split["train"] + rows_by_split["val"]
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    root = _safe_out(args.feature_cache_dir, [(Path(args.artifact_root) / "datasets"), args.model_dir])
    if root.exists() and not args.resume:
        raise FileExistsError("Frozen feature output exists; use --resume with identical configuration")
    if root.exists():
        saved = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        expected_snapshot = {key: snapshot[key] for key in
                             ("hub_id", "model_kind", "revision", "config_sha256",
                              "processor_config_sha256", "weight_file", "weight_sha256")}
        if (saved.get("schema") != CACHE_SCHEMA or saved.get("snapshot") != expected_snapshot
                or saved.get("manifest") != manifest_info or saved.get("row_count") != len(rows)
                or saved.get("implementation_sha256") != phase.sha256(__file__)):
            raise ValueError("Frozen feature cache provenance differs; refuse resume")
        _, existing = _read_feature_index(root, rows, saved)
        if len(existing) == len(rows):
            print(f"COMPLETE: {args.dataset} frozen vectors {len(rows)}; no model load or extraction repeated")
            return
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    backbone, processor = load_local_backbone(snapshot, trust_legacy_bin=args.trust_legacy_bin)
    configure_trainable(backbone, args.model_kind, "frozen")
    backbone.to(device).eval()
    meta = _cache_metadata(snapshot, manifest_info, processor, len(rows), backbone)
    if root.exists():
        saved = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if saved != meta:
            raise ValueError("Frozen feature cache metadata differs; refuse resume")
        vectors, entries = _read_feature_index(root, rows, saved)
        del vectors
    else:
        root.mkdir(parents=True)
        _atomic_json(root / "metadata.json", meta)
        vector_file = np.lib.format.open_memmap(root / "features.npy", mode="w+",
                                                 dtype=np.float32,
                                                 shape=(len(rows), meta["vector_dimension"]))
        vector_file.flush()
        del vector_file
        entries = []
    vectors = np.lib.format.open_memmap(root / "features.npy", mode="r+")
    with (root / "rows.jsonl").open("ab") as stream, torch.inference_mode():
        for index in range(len(entries), len(rows)):
            row = rows[index]
            pixels, _ = _image_batch([row], processor)
            embedding = pooled(backbone, pixels.to(device), args.model_kind)[0].float().cpu().numpy()
            vectors[index] = embedding
            vectors.flush()
            entry = {key: row[key] for key in ("image_id", "split", "label", "group_id", "sha256")}
            entry["vector_sha256"] = hashlib.sha256(embedding.tobytes()).hexdigest()
            stream.write((json.dumps(entry, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f"{args.dataset} frozen vectors: {index + 1}/{len(rows)}", flush=True)
    del vectors
    _read_feature_index(root, rows, meta)
    print(f"COMPLETE: {args.dataset} frozen vectors {len(rows)}")


def _epochs_batches(mode, rows_by_split, processor, vectors, micro_batch, seed, epoch, input_cache=None, experiment="B0"):
    """Yield epoch batches in deterministic order; no validation batch shuffling."""
    import torch

    n_train = len(rows_by_split["train"])
    if mode == "frozen":
        generator = np.random.default_rng(seed + epoch)
        order = generator.permutation(n_train)
        for start in range(0, n_train, micro_batch):
            indices = order[start:start + micro_batch]
            yield torch.from_numpy(np.asarray(vectors[indices]).copy()), torch.tensor(
                [rows_by_split["train"][i]["label"] for i in indices], dtype=torch.long)
    else:
        generator = np.random.default_rng(seed + epoch)
        order = generator.permutation(n_train)
        for start in range(0, n_train, micro_batch):
            selected = [rows_by_split["train"][i] for i in order[start:start + micro_batch]]
            if experiment == "B1":
                yield _image_batch(selected, processor, augment_seed=seed, augment_epoch=epoch)
            else:
                yield input_cache.batch(selected) if input_cache is not None else _image_batch(selected, processor)


def _timed_batches(batches, timings):
    """Measure synchronous input loading without a per-batch CUDA sync."""
    iterator = iter(batches)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        timings["input_seconds"] += time.perf_counter() - started
        yield batch


def _process_peak_rss_bytes():
    # Linux ru_maxrss is KiB; it is a process-lifetime peak, not an epoch peak.
    if os.name != "posix":
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _validation_batches(mode, rows_by_split, processor, vectors, micro_batch, input_cache=None):
    import torch

    offset = len(rows_by_split["train"])
    for start in range(0, len(rows_by_split["val"]), micro_batch):
        selected = rows_by_split["val"][start:start + micro_batch]
        if mode == "frozen":
            data = np.asarray(vectors[offset + start:offset + start + len(selected)]).copy()
            yield torch.from_numpy(data), torch.tensor([row["label"] for row in selected], dtype=torch.long)
        else:
            yield input_cache.batch(selected) if input_cache is not None else _image_batch(selected, processor)


def _eval_epoch(backbone, head, mode, model_kind, batches, device, precision, progress=None):
    import torch
    import torch.nn.functional as F

    head.eval()
    if backbone is not None:
        backbone.eval()
    truths, probabilities, total_loss, count = [], [], 0.0, 0
    with torch.inference_mode():
        for batch_index, (inputs, labels) in enumerate(batches):
            inputs, labels = inputs.to(device), labels.to(device)
            with _autocast(device, precision):
                embedding = inputs if mode == "frozen" else pooled(backbone, inputs, model_kind)
                logits = head(embedding.float())
                loss = F.cross_entropy(logits.float(), labels)
            if logits.shape != (len(labels), 5) or not torch.isfinite(loss):
                raise ValueError("Non-finite or malformed validation output")
            total_loss += float(loss) * len(labels)
            count += len(labels)
            truths.append(labels.detach())
            probabilities.append(logits.detach().float().softmax(1))
            if progress is not None:
                progress.update(batch_index + 1, count, total_loss / count)
    if count == 0:
        raise ValueError("Empty validation set")
    return _metrics(torch.cat(truths).cpu().numpy(),
                    torch.cat(probabilities).cpu().numpy(), total_loss / count)


def _train_epoch(backbone, head, mode, model_kind, batches, optimizer, scaler,
                 device, precision, accumulation, max_grad_norm, total_batches,
                 total_images, micro_batch, progress=None, class_weights=None):
    """Each image contributes equally, including incomplete accumulation windows."""
    import torch
    import torch.nn.functional as F

    head.train()
    if backbone is not None:
        backbone.train() if mode == "full" else backbone.eval()
    if total_images < 1 or total_batches != math.ceil(total_images / micro_batch):
        raise ValueError("Accumulation batch/image budget differs")
    if class_weights is not None:
        if len(class_weights) != 5 or any(not math.isfinite(w) or w <= 0 for w in class_weights):
            raise ValueError("Five finite, positive training class weights are required")
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        weights = None
    total_loss, count, steps, window_images = 0.0, 0, 0, []
    truths, probabilities = [], []
    optimizer.zero_grad(set_to_none=True)
    for index, (inputs, labels) in enumerate(batches):
        inputs, labels = inputs.to(device), labels.to(device)
        window_start = (index // accumulation) * accumulation * micro_batch
        window_size = min(accumulation * micro_batch, total_images - window_start)
        expected_batch = min(micro_batch, total_images - index * micro_batch)
        if len(labels) != expected_batch or window_size < len(labels):
            raise ValueError("Invalid accumulation window/image count")
        with _autocast(device, precision):
            embedding = inputs if mode == "frozen" else pooled(backbone, inputs, model_kind)
            logits = head(embedding.float())
            if weights is None:
                loss = F.cross_entropy(logits.float(), labels)
            else:
                per_image = F.cross_entropy(logits.float(), labels, reduction="none")
                loss = (per_image * weights[labels]).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        scaled_loss = loss * (len(labels) / window_size)
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total_loss += float(loss.detach()) * len(labels)
        count += len(labels)
        truths.append(labels.detach())
        probabilities.append(logits.detach().float().softmax(1))
        if progress is not None:
            progress.update(index + 1, count, total_loss / count)
        if (index + 1) % accumulation == 0 or index + 1 == total_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise ValueError("Non-finite training gradient")
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            window_images.append(window_size)
    if count != total_images or index + 1 != total_batches or sum(window_images) != total_images:
        raise ValueError("Train iterator image/batch count differs")
    metrics = _classification(torch.cat(truths).cpu().numpy(),
                              torch.cat(probabilities).cpu().numpy())
    return {"loss": total_loss / count, "images": count, "optimizer_steps": steps,
            "accumulation_window_images": window_images,
            "last_window_images": window_images[-1], **metrics}


def _class_weights(train_rows):
    """Inverse sqrt frequency, with mean training-image weight exactly one."""
    counts = np.bincount([row["label"] for row in train_rows], minlength=5)
    if len(counts) != 5 or np.any(counts <= 0):
        raise ValueError("All five grades must occur in the training split")
    inverse = 1.0 / np.sqrt(counts.astype(np.float64))
    weights = inverse / (np.dot(counts, inverse) / counts.sum())
    if not np.isclose(np.dot(counts, weights) / counts.sum(), 1.0, atol=1e-12):
        raise ValueError("Training class-weight normalization failed")
    return counts.tolist(), weights.tolist()


def _augmentation_metadata(experiment):
    if experiment != "B1":
        return None
    from PIL import __version__ as pillow_version
    from novelty import foundation_augmentation as augment
    return {"protocol": augment.PROTOCOL,
            "implementation_sha256": phase.sha256(augment.__file__),
            "pillow_version": pillow_version}


def _run_metadata(args, manifest_info, snapshot, precision, feature_meta=None, input_cache=None, class_weight_info=None):
    import torch

    experiment = getattr(args, "experiment", "B0")
    return {"schema": SCHEMA, "dataset": args.dataset, "mode": args.mode,
            "experiment": experiment,
            "class_weighting": class_weight_info,
            "training_loss": ("ordinary CE" if experiment != "B2" else
                              "sum_i weight[label_i]*CE_i / actual window image count; validation ordinary CE"),
            "model_kind": args.model_kind,
            "manifest": manifest_info,
            "snapshot": {key: snapshot[key] for key in ("hub_id", "revision", "config_sha256",
                         "processor_config_sha256", "weight_file", "weight_sha256")},
            "feature_cache": ({"metadata_sha256": phase.sha256(Path(args.feature_cache_dir) / "metadata.json"),
                               "rows_sha256": phase.sha256(Path(args.feature_cache_dir) / "rows.jsonl")}
                              if feature_meta is not None else None),
            "input_cache": ({"directory": str(input_cache.root),
                             "metadata_sha256": input_cache.meta_sha256}
                            if input_cache is not None else None),
            "classifier": "single trainable Linear(embedding_dim,5); cross entropy",
            "selection": "max validation macro-F1; tie lower validation loss; tie earliest epoch",
            "smoke_report_sha256": phase.sha256(args.smoke_report),
            "seed": args.seed, "epochs": args.epochs, "micro_batch": args.micro_batch,
            "accumulation": args.accumulation, "effective_batch": args.micro_batch * args.accumulation,
            "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
            "optimizer": "AdamW", "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm, "precision": precision,
            "augmentation": (_augmentation_metadata(experiment) if experiment == "B1" else
                             "none; deterministic official processor for train and validation"),
            "device_type": args.device.split(":")[0], "torch": torch.__version__,
            "implementation_sha256": phase.sha256(__file__)}


def _checkpoint_path(output, name):
    return Path(output) / f"{name}.pt"


def _save_checkpoint(path, state):
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _atomic_json(path.with_name(path.name + ".sha256.json"),
                     {"sha256": phase.sha256(path), "bytes": path.stat().st_size})
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path, *, trust):
    import torch

    if not trust:
        raise ValueError("Resuming project-owned pickle checkpoints requires --trust-run-checkpoint")
    path = Path(path)
    sidecar = json.loads(path.with_name(path.name + ".sha256.json").read_text(encoding="utf-8"))
    if sidecar != {"sha256": phase.sha256(path), "bytes": path.stat().st_size}:
        raise ValueError("Run checkpoint digest differs")
    return torch.load(path, map_location="cpu", weights_only=False)


def _rng_state():
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _best_is_new(metrics, prior):
    if prior is None:
        return True
    current_f1, best_f1 = metrics["macro_f1"], prior["macro_f1"]
    if current_f1 > best_f1 + 1e-12:
        return True
    if abs(current_f1 - best_f1) <= 1e-12 and metrics["validation_loss"] < prior["validation_loss"] - 1e-12:
        return True
    return False


def _audit_extension(args, new_record, new_smoke, output):
    """Verify a completed epoch-20 run without modifying any source artifact."""
    if not getattr(args, "trust_run_checkpoint", False):
        raise ValueError("Extension requires explicit --trust-run-checkpoint")
    if args.mode != "full" or args.epochs != 80:
        raise ValueError("Only completed 20-epoch full runs may extend to epoch 80")
    old_root = phase._outside_repository(args.extend_from_run)
    source_file = getattr(args, "source_file", None) or args.legacy_source_file
    source_smoke = getattr(args, "source_smoke_report", None) or args.legacy_smoke_report
    old_source = phase._outside_repository(source_file)
    old_smoke_path = phase._outside_repository(source_smoke)
    if (old_root == output or old_root in output.parents or output in old_root.parents
            or not old_root.is_dir()):
        raise ValueError("Extension output must be isolated from the completed source run")
    source_sha = phase.sha256(old_source)
    current_source = new_record["implementation_sha256"]
    legacy = source_sha == LEGACY_20_SOURCE_SHA256
    if not legacy and source_sha != current_source:
        raise ValueError("20-epoch runner source differs from both audited legacy and current source")
    if legacy and (new_record.get("experiment", "B0") != "B0"
                   or new_record.get("class_weighting") is not None):
        raise ValueError("Legacy vanilla 20-epoch run may only extend as B0")
    old_record_path = old_root / "run.json"
    old_record = json.loads(old_record_path.read_text(encoding="utf-8"))
    if (old_record.get("implementation_sha256") != source_sha
            or old_record.get("epochs") != 20 or old_record.get("mode") != "full"
            or new_record["epochs"] != 80 or new_record["mode"] != "full"):
        raise ValueError("Source run is not an audited completed 20-epoch full run")
    allowed_changes = {"epochs", "smoke_report_sha256"}
    if legacy:
        allowed_changes.add("implementation_sha256")
        # Older B0 records predate explicit experiment/loss fields and may add
        # a verified deterministic input transport without changing training.
        if old_record.get("augmentation") != new_record.get("augmentation"):
            raise ValueError("Legacy extension changes augmentation")
        for key, value in old_record.items():
            if key not in allowed_changes and new_record.get(key) != value:
                raise ValueError(f"Extension changes frozen training protocol: {key}")
    else:
        old_comparable = {key: value for key, value in old_record.items()
                          if key not in allowed_changes}
        new_comparable = {key: value for key, value in new_record.items()
                          if key not in allowed_changes}
        if old_comparable != new_comparable:
            raise ValueError("Extension changes B0/B1/B2 scientific or cache configuration")
    old_smoke_sha = phase.sha256(old_smoke_path)
    if old_smoke_sha != old_record.get("smoke_report_sha256"):
        raise ValueError("Original smoke report hash differs")
    old_smoke = json.loads(old_smoke_path.read_text(encoding="utf-8"))
    if (old_smoke.get("effective_processor") != new_smoke.get("effective_processor")
            or old_smoke.get("batch_size") != new_smoke.get("batch_size")
            or old_smoke.get("precision") != new_smoke.get("precision")
            or old_smoke.get("snapshot_weight_sha256") != new_smoke.get("snapshot_weight_sha256")
            or old_smoke.get("experiment", "B0") != new_smoke.get("experiment", "B0")
            or old_smoke.get("augmentation") != new_smoke.get("augmentation")
            or old_smoke.get("class_weighting") != new_smoke.get("class_weighting")):
        raise ValueError("Official processor or smoke protocol differs from original run")
    last_path, best_path = _checkpoint_path(old_root, "last"), _checkpoint_path(old_root, "best")
    old_last = _load_checkpoint(last_path, trust=True)
    old_best = _load_checkpoint(best_path, trust=True)
    history_path, summary_path = old_root / "history.json", old_root / "summary.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = None
    for row in history:
        validation = row["validation"]
        if _best_is_new(validation, selected["validation"] if selected else None):
            selected = {"epoch": row["epoch"], "validation": validation}
    if (selected != old_last.get("best")
            or old_last.get("record") != old_record or old_best.get("record") != old_record
            or old_last.get("epoch") != 20 or len(old_last.get("history", [])) != 20
            or history != old_last["history"] or [row["epoch"] for row in history] != list(range(1, 21))
            or old_last.get("best") != summary.get("best")
            or old_best.get("best") != old_last.get("best")
            or old_best.get("epoch") != old_last["best"]["epoch"]
            or summary.get("status") != "VALIDATION_COMPLETE"
            or summary.get("epochs_completed") != 20
            or summary.get("record") != old_record
            or old_last.get("optimizer") is None or old_last.get("rng") is None
            or old_last.get("backbone") is None or old_last.get("head") is None):
        raise ValueError("Original 20-epoch history/checkpoint/optimizer provenance differs")
    info = {"source_run": str(old_root), "source_run_json_sha256": phase.sha256(old_record_path),
            "source_last_sha256": phase.sha256(last_path),
            "source_best_sha256": phase.sha256(best_path),
            "source_history_sha256": phase.sha256(history_path),
            "source_summary_sha256": phase.sha256(summary_path),
            "source_smoke_sha256": old_smoke_sha, "source_smoke_path": str(old_smoke_path),
            "source_runner_sha256": source_sha, "source_runner_path": str(old_source),
            "completed_epochs": 20,
            "optimizer_and_rng_carried_forward": True}
    return info, old_last, old_best


def _initialize_extension(output, record, old_last, old_best):
    """Create an isolated continuation directory; source checkpoints remain untouched."""
    stage = output.with_name(output.name + f".extension-{os.getpid()}.{uuid.uuid4().hex}.tmp")
    stage.mkdir(parents=True, exist_ok=False)
    last = {**old_last, "record": record}
    best = {**old_best, "record": record}
    _atomic_json(stage / "run.json", record)
    _save_checkpoint(_checkpoint_path(stage, "last"), last)
    _save_checkpoint(_checkpoint_path(stage, "best"), best)
    _atomic_json(stage / "history.json", old_last["history"])
    _atomic_json(stage / "milestones.json", _milestones(old_last["history"]))
    source = Path(record["extension_from"]["source_run"])
    for name, key in (("run.json", "source_run_json_sha256"),
                      ("last.pt", "source_last_sha256"),
                      ("best.pt", "source_best_sha256"),
                      ("history.json", "source_history_sha256"),
                      ("summary.json", "source_summary_sha256")):
        if phase.sha256(source / name) != record["extension_from"][key]:
            raise ValueError("Source run changed during extension initialization")
    for path_key, hash_key in (("source_runner_path", "source_runner_sha256"),
                               ("source_smoke_path", "source_smoke_sha256")):
        if phase.sha256(record["extension_from"][path_key]) != record["extension_from"][hash_key]:
            raise ValueError("Source code or smoke changed during extension initialization")
    os.replace(stage, output)


def train(args):
    import torch
    from novelty import foundation_reporting as reporting

    experiment = getattr(args, "experiment", "B0")
    if experiment not in {"B0", "B1", "B2"} or (experiment != "B0" and args.mode != "full"):
        raise ValueError("B1/B2 are independent full-fine-tuning configurations only")
    if (args.mode not in {"frozen", "partial", "full"} or args.epochs < 1 or args.micro_batch < 1
            or args.accumulation < 1 or args.seed < 0 or args.head_lr <= 0
            or args.backbone_lr <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0):
        raise ValueError("Invalid prespecified training settings")
    if args.mode in {"partial", "full"} and args.feature_cache_dir:
        raise ValueError("Backbone fine-tuning must never consume frozen embeddings")
    if args.mode == "full" and (args.micro_batch not in {1, 4, 8, 16, 32}
                                or args.accumulation != 32 // args.micro_batch):
        raise ValueError("Full tuning requires micro-batch 1/4/8/16/32 and effective batch 32")
    if args.mode == "frozen" and not args.feature_cache_dir:
        raise ValueError("Frozen linear probe requires the persistent feature cache")
    input_cache_dir = getattr(args, "input_cache_dir", None)
    if input_cache_dir and args.mode != "full":
        raise ValueError("Official-input cache is supported only for full fine-tuning")
    rows_by_split, manifest_info = load_train_val_manifest(args.artifact_root, args.dataset)
    rows = rows_by_split["train"] + rows_by_split["val"]
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    protected = [Path(args.artifact_root) / "datasets", args.model_dir]
    if args.feature_cache_dir:
        protected.append(args.feature_cache_dir)
    if input_cache_dir:
        protected.append(input_cache_dir)
    extend_from = getattr(args, "extend_from_run", None)
    if extend_from:
        if (not (getattr(args, "source_file", None) or getattr(args, "legacy_source_file", None))
                or not (getattr(args, "source_smoke_report", None)
                        or getattr(args, "legacy_smoke_report", None))):
            raise ValueError("Extension requires source file and source smoke report")
        protected.append(extend_from)
    output = _safe_out(args.output_dir, protected)
    if output.exists() and not args.resume:
        raise FileExistsError("Run output exists; use --resume for identical configuration")
    if output.exists() and args.resume:
        existing_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if existing_record.get("extension_from") and not extend_from:
            raise ValueError("Extension resume requires the original --extend-from-run provenance flags")
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    precision = _precision(device, args.precision)
    vectors = processor = backbone = feature_meta = input_cache = None
    if args.mode == "frozen":
        vectors, feature_meta = _feature_cache(args, rows, manifest_info, snapshot)
    else:
        backbone, processor = load_local_backbone(snapshot, trust_legacy_bin=args.trust_legacy_bin)
        configure_trainable(backbone, args.model_kind, args.mode)
        backbone.to(device)
        if input_cache_dir:
            from novelty import foundation_input_cache as inputs
            expected = inputs.metadata(snapshot, manifest_info, processor, args.micro_batch,
                                       phase.sha256(__file__))
            input_cache = inputs.InputCache(input_cache_dir, expected)
            report = json.loads((input_cache.root / "summary.json").read_text(encoding="utf-8"))
            if (report.get("status") != "COMPLETE"
                    or report.get("metadata_sha256") != input_cache.meta_sha256
                    or report.get("row_count") != len(rows)):
                raise ValueError("Official-input cache is incomplete or differs")
            for split in ("train", "val"):
                for start in range(0, len(rows_by_split[split]), args.micro_batch):
                    input_cache.batch(rows_by_split[split][start:start + args.micro_batch])
    head = torch.nn.Linear(MODELS[args.model_kind]["dimension"], 5).to(device)
    groups = [{"params": list(head.parameters()), "lr": args.head_lr}]
    if backbone is not None:
        groups.insert(0, {"params": [p for p in backbone.parameters() if p.requires_grad],
                          "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16") if precision == "fp16" else None
    weight_counts, class_weights = (_class_weights(rows_by_split["train"])
                                    if experiment == "B2" else (None, None))
    class_weight_info = ({"formula": "w_c=(n_c^-0.5)/(sum_j n_j*n_j^-0.5 / N_train)",
                          "train_counts": weight_counts, "weights": class_weights}
                         if experiment == "B2" else None)
    record = _run_metadata(args, manifest_info, snapshot, precision, feature_meta,
                           input_cache, class_weight_info)
    smoke = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
    if (smoke.get("schema") != SCHEMA or smoke.get("status") != "BATCH_SMOKE_PASS"
            or smoke.get("optimizer_steps") != 2 or smoke.get("weights_saved") is not False
            or smoke.get("dataset") != args.dataset
            or smoke.get("model_kind") != args.model_kind or smoke.get("mode") != args.mode
            or smoke.get("snapshot_weight_sha256") != snapshot["weight_sha256"]
            or smoke.get("snapshot_revision") != snapshot["revision"]
            or smoke.get("processor_config_sha256") != snapshot["processor_config_sha256"]
            or smoke.get("manifest_sha256") != manifest_info["manifest_sha256"]
            or smoke.get("device_type") != device.type or smoke.get("batch_size") != args.micro_batch
            or smoke.get("precision") != precision
            or smoke.get("implementation_sha256") != phase.sha256(__file__)
            or smoke.get("experiment", "B0") != experiment
            or (experiment == "B2" and smoke.get("class_weighting") != class_weight_info)
            or (experiment == "B1" and smoke.get("augmentation") != record["augmentation"])):
        raise ValueError("Matching two-step smoke report is absent or differs from this run")
    if extend_from:
        info, old_last, old_best = _audit_extension(args, record, smoke, output)
        record["extension_from"] = info
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            _initialize_extension(output, record, old_last, old_best)
        elif not args.resume:
            raise FileExistsError("Extension output exists; use guarded --resume")
    if output.exists():
        if json.loads((output / "run.json").read_text(encoding="utf-8")) != record:
            raise ValueError("Run protocol/provenance differs; refuse resume")
        saved = _load_checkpoint(_checkpoint_path(output, "last"), trust=args.trust_run_checkpoint)
        if saved.get("record") != record or saved.get("epoch", 0) > args.epochs:
            raise ValueError("Last checkpoint record/epoch differs")
        head.load_state_dict(saved["head"], strict=True)
        if backbone is not None:
            backbone.load_state_dict(saved["backbone"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(saved["scaler"])
        _restore_rng(saved["rng"])
        history, best = saved["history"], saved["best"]
        start_epoch = saved["epoch"] + 1
        best_path = _checkpoint_path(output, "best")
        if best_path.is_file():
            selected_state = _load_checkpoint(best_path, trust=args.trust_run_checkpoint)
            if selected_state.get("record") != record or selected_state.get("epoch") != best["epoch"]:
                if best["epoch"] == saved["epoch"]:
                    _save_checkpoint(best_path, saved)  # crash after last, before best
                else:
                    raise ValueError("Best checkpoint is stale; inspect run before resuming")
        elif best["epoch"] == saved["epoch"]:
            _save_checkpoint(best_path, saved)
        else:
            raise FileNotFoundError("Best checkpoint missing; inspect run before resuming")
    else:
        output.mkdir(parents=True)
        _atomic_json(output / "run.json", record)
        history, best, start_epoch = [], None, 1
    logger = reporting.RunLogger(output)
    logger.line(f"run={args.dataset}/{args.model_kind}/{args.mode} "
                f"epochs={args.epochs} starting_epoch={start_epoch} "
                f"resume={bool(args.resume)} source_sha256={phase.sha256(__file__)}")
    n_batches = math.ceil(len(rows_by_split["train"]) / args.micro_batch)
    n_validation_batches = math.ceil(len(rows_by_split["val"]) / args.micro_batch)
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        cpu_start = time.process_time()
        train_input = {"input_seconds": 0.0}
        validation_input = {"input_seconds": 0.0}
        train_progress = reporting.BatchProgress(
            logger, "train", epoch, n_batches, len(rows_by_split["train"]))
        try:
            train_info = _train_epoch(backbone, head, args.mode, args.model_kind,
                                      _timed_batches(
                                          _epochs_batches(args.mode, rows_by_split, processor, vectors,
                                                          args.micro_batch, args.seed, epoch, input_cache,
                                                          experiment), train_input),
                                      optimizer, scaler, device, precision, args.accumulation,
                                      args.max_grad_norm, n_batches, len(rows_by_split["train"]),
                                      args.micro_batch, train_progress, class_weights)
        finally:
            train_progress.close()
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        validation_progress = reporting.BatchProgress(
            logger, "validation", epoch, n_validation_batches, len(rows_by_split["val"]))
        try:
            validation = _eval_epoch(
                backbone, head, args.mode, args.model_kind,
                _timed_batches(
                    _validation_batches(args.mode, rows_by_split, processor, vectors,
                                        args.micro_batch, input_cache), validation_input),
                device, precision, validation_progress)
        finally:
            validation_progress.close()
        validation_seconds = time.perf_counter() - start
        resource = {"train_seconds": train_seconds, "validation_seconds": validation_seconds,
                    "train_input_seconds": train_input["input_seconds"],
                    "validation_input_seconds": validation_input["input_seconds"],
                    "process_cpu_seconds": time.process_time() - cpu_start,
                    "process_peak_rss_bytes": _process_peak_rss_bytes(),
                    "validation_images_per_second": len(rows_by_split["val"]) / validation_seconds,
                    "train_images_per_second": train_info["images"] / train_seconds,
                    "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                 if device.type == "cuda" else None),
                    "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                                if device.type == "cuda" else None),
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "gpu_free_bytes_after_epoch": (torch.cuda.mem_get_info(device)[0]
                                                   if device.type == "cuda" else None),
                    "gpu_total_bytes": (torch.cuda.mem_get_info(device)[1]
                                        if device.type == "cuda" else None)}
        improved = _best_is_new(validation, best["validation"] if best else None)
        if improved:
            best = {"epoch": epoch, "validation": validation}
        learning_rates = {"head": optimizer.param_groups[-1]["lr"],
                          "backbone": optimizer.param_groups[0]["lr"] if backbone is not None else None}
        current = {"epoch": epoch, "train": train_info, "validation": validation,
                   "resource": resource, "learning_rates": learning_rates,
                   "best_epoch": best["epoch"], "best_updated": improved}
        history.append(current)
        state = {"record": record, "epoch": epoch, "head": head.state_dict(),
                 "backbone": backbone.state_dict() if backbone is not None else None,
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler is not None else None,
                 "rng": _rng_state(), "history": history, "best": best}
        _save_checkpoint(_checkpoint_path(output, "last"), state)
        if improved:
            _save_checkpoint(_checkpoint_path(output, "best"), state)
        _atomic_json(output / "history.json", history)
        _atomic_json(output / "milestones.json", _milestones(history))
        reporting.write_history_csv(output / "history.csv", history)
        for line in reporting.epoch_lines(current, best, improved, learning_rates):
            logger.line(line)
    if len(history) == args.epochs:
        _atomic_json(output / "summary.json", {"status": "VALIDATION_COMPLETE", "record": record,
                                                "best": best, "parameter_counts":
                                                _parameter_counts(backbone, head) if backbone is not None else
                                                {"backbone_total": feature_meta["backbone_parameter_count"],
                                                 "backbone_trainable": 0,
                                                 "head_total": sum(p.numel() for p in head.parameters()),
                                                 "total_trainable": sum(p.numel() for p in head.parameters())},
                                                "epochs_completed": len(history),
                                                "milestones": _milestones(history),
                                                "final_test_status": "NOT_ACCESSED"})
        reporting.learning_curves(output / "learning_curves.svg", history)
        logger.line(f"complete best_epoch={best['epoch']} "
                    f"best_validation_macro_f1={best['validation']['macro_f1']:.6f}")
    logger.close()


def smoke(args):
    """Measure two disposable training steps at the chosen micro-batch size.

    Step two includes AdamW state in the forward/backward memory peak.
    No updated weights are saved.
    """
    import torch
    import torch.nn.functional as F

    experiment = getattr(args, "experiment", "B0")
    if experiment not in {"B0", "B1", "B2"} or (experiment != "B0" and args.mode != "full"):
        raise ValueError("B1/B2 smoke requires full fine-tuning")
    if args.mode not in {"frozen", "partial", "full"}:
        raise ValueError("Smoke mode must be frozen, partial, or full")
    batch_size = getattr(args, "batch_size", 1)
    if batch_size not in {1, 4, 8, 16, 32}:
        raise ValueError("Smoke batch size must be 1, 4, 8, 16, or 32")
    rows_by_split, manifest_info = load_train_val_manifest(args.artifact_root, args.dataset)
    if len(rows_by_split["train"]) < batch_size:
        raise ValueError("Training manifest has fewer images than smoke batch")
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    output = _safe_out(args.output_dir, [Path(args.artifact_root) / "datasets", args.model_dir])
    if output.exists():
        raise FileExistsError("Smoke output exists")
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    precision = _precision(device, args.precision)
    backbone, processor = load_local_backbone(snapshot, trust_legacy_bin=args.trust_legacy_bin)
    selected = configure_trainable(backbone, args.model_kind, args.mode)
    backbone.to(device)
    backbone.train() if args.mode == "full" else backbone.eval()
    head = torch.nn.Linear(MODELS[args.model_kind]["dimension"], 5).to(device)
    parameters = list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16") if precision == "fp16" else None
    class_counts, class_weights = (_class_weights(rows_by_split["train"])
                                   if experiment == "B2" else (None, None))
    if experiment == "B1":
        pixels, labels = _image_batch(rows_by_split["train"][:batch_size], processor,
                                      augment_seed=args.seed, augment_epoch=1)
    else:
        pixels, labels = _image_batch(rows_by_split["train"][:batch_size], processor)
    pixels, labels = pixels.to(device), labels.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free_before, total_bytes = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        free_before = total_bytes = None
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            if args.mode == "frozen":
                with torch.no_grad():
                    embedding = pooled(backbone, pixels, args.model_kind)
            else:
                embedding = pooled(backbone, pixels, args.model_kind)
            logits = head(embedding.float()).float()
            if class_weights is None:
                loss = F.cross_entropy(logits, labels)
            else:
                weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
                loss = (F.cross_entropy(logits, labels, reduction="none") * weights[labels]).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite smoke loss")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("Non-finite smoke gradient")
        if args.mode == "full":
            missing_gradients = [name for name, parameter in backbone.named_parameters()
                                 if parameter.requires_grad and parameter.grad is None]
            allowed_missing = ({"embeddings.mask_token"}
                               if args.model_kind == "dinov3_vits16" else set())
            if set(missing_gradients) - allowed_missing:
                raise ValueError("Unexpected full-backbone parameters received no gradient: "
                                 + ", ".join(sorted(set(missing_gradients) - allowed_missing)))
            if any(parameter.grad is None for parameter in head.parameters()):
                raise ValueError("Full-tuning head received no gradient")
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = {"schema": SCHEMA, "status": "BATCH_SMOKE_PASS", "dataset": args.dataset,
              "model_kind": args.model_kind, "mode": args.mode,
              "experiment": experiment,
              "augmentation": _augmentation_metadata(experiment),
              "class_weighting": ({"formula": "w_c=(n_c^-0.5)/(sum_j n_j*n_j^-0.5 / N_train)",
                                   "train_counts": class_counts, "weights": class_weights}
                                  if experiment == "B2" else None),
              "snapshot_weight_sha256": snapshot["weight_sha256"],
              "snapshot_revision": snapshot["revision"],
              "manifest_sha256": manifest_info["manifest_sha256"],
              "processor_config_sha256": snapshot["processor_config_sha256"],
              "effective_processor": processor.to_dict(),
              "input_shape": list(pixels.shape), "embedding_shape": list(embedding.shape),
              "selected_backbone_modules": selected,
              "parameter_counts": _parameter_counts(backbone, head),
              "precision": precision, "batch_size": batch_size, "device_type": device.type,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
              "gpu_free_bytes_before": free_before,
              "gpu_free_bytes_after": torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None,
              "gpu_total_bytes": total_bytes,
              "elapsed_seconds": elapsed, "images_per_second": 2 * batch_size / elapsed,
              "optimizer_steps": 2, "weights_saved": False,
              "unused_trainable_backbone_parameters": missing_gradients if args.mode == "full" else [],
              "implementation_sha256": phase.sha256(__file__)}
    output.mkdir(parents=True)
    _atomic_json(output / "smoke.json", report)
    print(json.dumps({key: report[key] for key in ("status", "model_kind", "mode", "batch_size",
                                                   "peak_allocated_bytes", "peak_reserved_bytes",
                                                   "elapsed_seconds")}, indent=2))


def preflight(args):
    _, manifest = load_train_val_manifest(args.artifact_root, args.dataset)
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    print(json.dumps({"status": "PREFLIGHT_PASS", "split": "train+val", "manifest": manifest,
                      "snapshot": {key: snapshot[key] for key in
                                   ("hub_id", "model_kind", "revision", "config_sha256",
                                    "processor_config_sha256", "weight_file", "weight_sha256",
                                    "weight_bytes", "declared_embedding_dimension")}}, indent=2))


def sensitivity(args):
    """Later, manually evaluate a selected checkpoint on the exact E3 pilot rows.

    The standard official-processor input is an additional descriptive anchor;
    E3 variants originate from historical V1/V2 pixels and are not equivalent.
    """
    import torch
    from PIL import Image, ImageOps
    from novelty import preof_e3 as e3

    rows_by_split, manifest_info = load_train_val_manifest(args.artifact_root, args.dataset)
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    run_dir = phase._outside_repository(args.run_dir)
    e3_dir = phase._outside_repository(args.e3_pilot_dir)
    cache_root = Path(args.preprocess_cache_root).expanduser().resolve()
    output = _safe_out(args.output_dir, [Path(args.artifact_root) / "datasets", args.model_dir,
                                         run_dir, e3_dir, cache_root])
    if output.exists():
        raise FileExistsError("Sensitivity output exists")
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if (run_record.get("dataset") != args.dataset or run_record.get("model_kind") != args.model_kind
            or run_record.get("manifest") != manifest_info
            or run_record.get("snapshot") != {key: snapshot[key] for key in
                ("hub_id", "revision", "config_sha256", "processor_config_sha256",
                 "weight_file", "weight_sha256")}):
        raise ValueError("Trained run differs from frozen manifest or model snapshot")
    e3_meta = json.loads((e3_dir / "metadata.json").read_text(encoding="utf-8"))
    if (e3_meta.get("schema") != e3.SCHEMA or e3_meta.get("dataset") != args.dataset
            or e3_meta.get("manifest_sha256") != manifest_info["manifest_sha256"]):
        raise ValueError("E3 pilot metadata differs")
    selected, selection = e3.select_pilot(rows_by_split["val"], args.dataset,
                                          e3_meta["selection"]["groups_per_grade"],
                                          e3_meta["selection"]["seed"])
    if selection != e3_meta["selection"]:
        raise ValueError("E3 pilot group/grade selection differs")
    with (e3_dir / "rows.jsonl").open("r", encoding="utf-8") as stream:
        old_rows = [json.loads(line) for line in stream]
    if (len(old_rows) != len(selected) or any(old.get("image_id") != row["image_id"]
            or old.get("image_sha256") != row["sha256"]
            or old.get("true_grade") != row["label"]
            or old.get("group_id") != row["group_id"]
            or old.get("split") != "val" for old, row in zip(old_rows, selected))):
        raise ValueError("E3 pilot image order differs")
    checkpoint = _load_checkpoint(_checkpoint_path(run_dir, "best"), trust=args.trust_run_checkpoint)
    if checkpoint.get("record") != run_record or checkpoint["epoch"] != checkpoint["best"]["epoch"]:
        raise ValueError("Selected checkpoint differs from validation-selected best")
    _seed(run_record["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if _precision(device, run_record["precision"]) != run_record["precision"]:
        raise ValueError("Selected run precision differs on this device")
    backbone, processor = load_local_backbone(snapshot, trust_legacy_bin=args.trust_legacy_bin)
    head = torch.nn.Linear(MODELS[args.model_kind]["dimension"], 5)
    head.load_state_dict(checkpoint["head"], strict=True)
    if run_record["mode"] in {"partial", "full"}:
        backbone.load_state_dict(checkpoint["backbone"], strict=True)
    elif run_record["mode"] != "frozen" or checkpoint["backbone"] is not None:
        raise ValueError("Unknown or inconsistent trained mode")
    backbone.to(device).eval()
    head.to(device).eval()
    cache = phase.PreprocessingCache(cache_root, Path(manifest_info["dataset_root"]))
    condition_names = ("official_raw",) + e3.CONDITIONS
    observed = {name: [] for name in condition_names}
    truths = [row["label"] for row in selected]
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for row in selected:
            v1 = e3._read_pixels(cache, row["sha256"], "v1")
            v2 = e3._read_pixels(cache, row["sha256"], "v2_bg10")
            phase.require_hash(row["path"], row["sha256"])
            with Image.open(row["path"]) as image:
                standard = ImageOps.exif_transpose(image).convert("RGB")
            for name in condition_names:
                image = standard if name == "official_raw" else Image.fromarray(
                    e3.condition_pixels(v1, v2, name), "RGB")
                pixels = processor(images=image, return_tensors="pt")["pixel_values"]
                if tuple(pixels.shape) != (1, 3, 224, 224):
                    raise ValueError("Sensitivity processor shape differs")
                with _autocast(device, run_record["precision"]):
                    logits = head(pooled(backbone, pixels.to(device), args.model_kind).float())
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite sensitivity logits")
                observed[name].append(logits.float().softmax(1)[0].cpu().tolist())
    elapsed = time.perf_counter() - started
    results = {}
    for name, probs in observed.items():
        probabilities = np.asarray(probs, float)
        # Cross entropy is unavailable from probabilities alone without logits;
        # descriptive sensitivity uses classification metrics and no loss value.
        result = e1.classification(np.asarray(truths), probabilities)
        support = np.asarray([grade["support"] for grade in result["per_grade"]], float)
        result["weighted_f1"] = float(np.dot(support,
                                      [grade["f1"] for grade in result["per_grade"]]) / support.sum())
        results[name] = result
    report = {"schema": "foundation-e3-sensitivity-v1", "status": "EXPLORATORY_VALIDATION_PILOT",
              "dataset": args.dataset, "split": "val", "model_kind": args.model_kind,
              "trained_mode": run_record["mode"], "selected_checkpoint_epoch": checkpoint["epoch"],
              "selected_checkpoint_sha256": phase.sha256(_checkpoint_path(run_dir, "best")),
              "snapshot_weight_sha256": snapshot["weight_sha256"],
              "e3_pilot_metadata_sha256": phase.sha256(e3_dir / "metadata.json"),
              "e3_pilot_rows_sha256": phase.sha256(e3_dir / "rows.jsonl"),
              "manifest_sha256": manifest_info["manifest_sha256"], "selection": selection,
              "definition": "official_raw uses raw image plus official processor; other conditions apply E3 historical 512px pixel variants before the same official processor; same validation data selected the checkpoint.",
              "conditions": results,
              "resource": {"seconds": elapsed,
                           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                           if device.type == "cuda" else None,
                           "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                           if device.type == "cuda" else None}}
    output.mkdir(parents=True)
    _atomic_json(output / "sensitivity.json", report)
    print(f"COMPLETE: {args.dataset} exploratory E3 pilot sensitivity for {args.model_kind}")


def _milestones(history):
    return {str(row["epoch"]): {"train": row["train"], "validation": row["validation"],
                                "resource": row["resource"]}
            for row in history if row["epoch"] in {20, 40, 60, 80}}


def prepare_input_cache(args):
    """Manual cold/warm cache preparation; no model forward or test access."""
    from novelty import foundation_input_cache as inputs

    rows_by_split, manifest_info = load_train_val_manifest(args.artifact_root, args.dataset)
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    root = _safe_out(args.input_cache_dir, [Path(args.artifact_root) / "datasets", args.model_dir])
    if root.exists() and not args.resume:
        raise FileExistsError("Input cache exists; use --resume with identical provenance")
    processor = load_local_processor(snapshot)
    expected = inputs.metadata(snapshot, manifest_info, processor, args.batch_size,
                               phase.sha256(__file__))
    cache = inputs.InputCache(root, expected, create=not root.exists())
    report = inputs.prepare(cache, rows_by_split, args.batch_size,
                            lambda batch: _image_batch(batch, processor),
                            verify_equivalence=args.verify_equivalence)
    print(json.dumps(report, indent=2))


def inspect_augmentation(args):
    """Inspect B1 train-only transforms without loading model weights or writing in Git."""
    from novelty import foundation_augmentation as augment
    rows, manifest = load_train_val_manifest(args.artifact_root, args.dataset)
    snapshot = audit_snapshot(args.model_kind, args.model_dir, args.revision, args.weight_sha256)
    processor = load_local_processor(snapshot)
    report = augment.inspect_gallery(rows["train"], processor, args.output_dir,
                                     seed=args.seed, epoch=args.epoch,
                                     samples_per_grade=args.samples_per_grade)
    report["dataset"] = args.dataset
    report["manifest_sha256"] = manifest["manifest_sha256"]
    report["processor_config_sha256"] = snapshot["processor_config_sha256"]
    report["augmentation_implementation_sha256"] = phase.sha256(augment.__file__)
    _atomic_json(Path(args.output_dir) / "augmentation_gallery.json", report)
    print(json.dumps({"status": "COMPLETE", "images": len(report["rows"]),
                      "output_dir": str(args.output_dir)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "smoke", "extract", "train", "sensitivity", "prepare-input-cache", "inspect-augmentation"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", choices=tuple(phase.SPECS), required=True)
        sub.add_argument("--artifact-root", required=True)
        sub.add_argument("--model-kind", choices=tuple(MODELS), required=True)
        sub.add_argument("--model-dir", required=True)
        sub.add_argument("--revision", required=True)
        sub.add_argument("--weight-sha256", required=True)
        if name != "preflight":
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--trust-legacy-bin", action="store_true")
        if name in {"smoke", "train"}:
            sub.add_argument("--experiment", choices=("B0", "B1", "B2"), default="B0")
        if name == "smoke":
            sub.add_argument("--mode", choices=("full", "frozen", "partial"), required=True)
            sub.add_argument("--batch-size", type=int, default=1)
            sub.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
            sub.add_argument("--output-dir", required=True)
        elif name == "inspect-augmentation":
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--epoch", type=int, default=1)
            sub.add_argument("--samples-per-grade", type=int, default=3)
        elif name == "prepare-input-cache":
            sub.add_argument("--input-cache-dir", required=True)
            sub.add_argument("--batch-size", type=int, required=True)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--verify-equivalence", action="store_true")
        elif name == "extract":
            sub.add_argument("--feature-cache-dir", required=True)
            sub.add_argument("--resume", action="store_true")
        elif name == "train":
            sub.add_argument("--mode", choices=("full", "frozen", "partial"), required=True)
            sub.add_argument("--feature-cache-dir")
            sub.add_argument("--input-cache-dir")
            sub.add_argument("--smoke-report", required=True)
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--epochs", type=int, default=20)
            sub.add_argument("--extend-from-run")
            sub.add_argument("--legacy-source-file")
            sub.add_argument("--legacy-smoke-report")
            sub.add_argument("--source-file")
            sub.add_argument("--source-smoke-report")
            sub.add_argument("--micro-batch", type=int, default=1)
            sub.add_argument("--accumulation", type=int, default=32)
            sub.add_argument("--head-lr", type=float, default=1e-3)
            sub.add_argument("--backbone-lr", type=float, default=1e-5)
            sub.add_argument("--weight-decay", type=float, default=.01)
            sub.add_argument("--max-grad-norm", type=float, default=1.0)
            sub.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--trust-run-checkpoint", action="store_true")
        elif name == "sensitivity":
            sub.add_argument("--run-dir", required=True)
            sub.add_argument("--e3-pilot-dir", required=True)
            sub.add_argument("--preprocess-cache-root", required=True)
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--trust-run-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    {"preflight": preflight, "smoke": smoke, "extract": extract_features,
     "train": train, "sensitivity": sensitivity,
     "prepare-input-cache": prepare_input_cache,
     "inspect-augmentation": inspect_augmentation}[args.command](args)


if __name__ == "__main__":
    main()
