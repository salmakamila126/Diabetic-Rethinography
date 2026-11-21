"""Strict checkpoint loading and categorical inference contracts."""

import hashlib
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch import nn

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from novelty_integrated.e5 import data
from novelty_integrated.e5.model import E5Heads
from novelty_integrated.e7 import geometry
from novelty_integrated.e7.model import GlobalLocalModel

from .registry import CLASS_ORDER, INTEGRATED, ROOT, require_sources


class CategoricalModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, pixels):
        embedding = fa.pooled(self.backbone, pixels, "convnext_tiny")
        if isinstance(self.head, E5Heads):
            return self.head.predict_logits(embedding)
        return self.head(embedding)


class GlobalLocalCategorical(nn.Module):
    """Preserve the authentic preprocessing, shared-weight fusion and gate."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, global_pixels, patch_pixels):
        return self.model(global_pixels, patch_pixels)[0]


def reconstruct(family, backbone, state, train_counts=None):
    """Construct from authentic model classes. Caller must verify checkpoint identity."""
    if family == "hf_v1_global_local":
        selected = state["selection_weights"]
        if selected not in ("raw", "ema"):
            raise ValueError("E7 needs a verified raw or EMA selection")
        model = GlobalLocalModel(backbone, train_counts)
        weights = state["ema"]["shadow"] if selected == "ema" else state["model"]
        model.load_state_dict(weights, strict=True)
        wrapper = GlobalLocalCategorical(model)
    else:
        if family not in ("hf_v1_ordinal", "hf_v1_linear"):
            raise ValueError(f"No checkpoint adapter for {family}")
        ordinal = family == "hf_v1_ordinal"
        head = E5Heads(768, train_counts, ordinal=True) if ordinal else nn.Linear(768, 5)
        backbone.load_state_dict(state["backbone"], strict=True)
        head.load_state_dict(state["heads" if ordinal else "head"], strict=True)
        if ordinal and not bool((head.ordinal.thresholds().diff() > 0).all()):
            raise ValueError("Checkpoint ordinal thresholds are not ordered")
        wrapper = CategoricalModel(backbone, head)
    model = wrapper.cpu().float().eval()
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()
           if value.is_floating_point()):
        raise ValueError("Checkpoint contains non-finite inference state")
    return model


def verify_receipt(entry, checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_CHECKPOINT: {entry.dataset} {entry.arm}")
    if path.name != "best.pt" or path.is_relative_to(ROOT):
        raise ValueError("Use the original best.pt outside the repository")
    phase._outside_repository(path)
    phase.require_hash(path, entry.checkpoint_sha256)
    if "checkpoint_sidecars" in entry.evidence and path.stat().st_size != entry.evidence["checkpoint_sidecars"]["best.pt"]["bytes"]:
        raise ValueError("Checkpoint length differs from the published receipt")
    return path


def verify_record(entry, state):
    cfg = entry.configuration
    record = state["record"]
    baseline_arm = "B1" if entry.arm.startswith("B4-") else entry.arm
    if (state["epoch"] != entry.best["epoch"] or state["best"]["epoch"] != state["epoch"]
            or record["dataset"] != entry.dataset or record["schema"] != cfg["schema"]
            or record["seed"] != cfg["seed"] or (entry.arm != "E7" and record["model_kind"] != "convnext_tiny")
            or record["snapshot"] != cfg["snapshot"] or record.get("class_order", list(CLASS_ORDER)) != list(CLASS_ORDER)
            or any(record["manifest"][key] != cfg["manifest"][key]
                   for key in ("manifest_sha256", "counts", "grade_support", "image_order_sha256"))):
        raise ValueError("Checkpoint metadata differs from published validation evidence")
    if entry.arm == "E7":
        fields = ("source_sha256", "git_head", "class_order", "effective_processor_sha256", "train_counts",
                  "geometry", "init", "selection_weights", "ema_decay", "val_cache_metadata_sha256")
        if (any(record.get(key) != cfg[key] for key in fields) or state.get("best") != entry.best
                or state.get("selection_weights") != cfg["selection_weights"]):
            raise ValueError("E7 checkpoint selection or provenance differs")
    elif entry.arm in INTEGRATED:
        if record.get("source_sha256") != cfg["source_sha256"] or record.get("train_counts") != cfg["train_counts"]:
            raise ValueError("Checkpoint source or training counts differ")
        if record.get("effective_processor_sha256") != cfg["effective_processor_sha256"]:
            raise ValueError("Checkpoint effective processor differs")
        if entry.arm == "E5" and record.get("arm") != "integrated":
            raise ValueError("E5 checkpoint is outside the integrated arm")
    elif (record.get("mode") != "full" or record.get("experiment", "B0") != baseline_arm
          or record.get("implementation_sha256") != cfg["implementation_sha256"]
          or (entry.arm == "B3" and record.get("sampling") != cfg["sampling"])
          or (entry.arm.startswith("B4-") and record.get("b4") != cfg["b4"])):
        raise ValueError("Baseline checkpoint arm or historical source differs")


def load_original(entry, checkpoint, model_dir, *, trust_checkpoint=False,
                  trust_snapshot=False, external_source_dir=None):
    if not entry.evidence:
        require_sources(entry)
    path = verify_receipt(entry, checkpoint)
    sources = require_sources(entry, external_source_dir)
    state = fa._load_checkpoint(path, trust=trust_checkpoint)
    verify_record(entry, state)
    cfg = entry.configuration
    pinned = cfg["snapshot"]
    snap = fa.audit_snapshot("convnext_tiny", model_dir, pinned["revision"], pinned["weight_sha256"])
    if any(snap.get(key) != value for key, value in pinned.items()):
        raise ValueError("Pinned ConvNeXt snapshot metadata differs")
    backbone, processor = fa.load_local_backbone(snap, trust_legacy_bin=trust_snapshot)
    processor_sha = fa._digest_json(processor.to_dict())
    if cfg.get("effective_processor_sha256", processor_sha) != processor_sha:
        raise ValueError("Effective official processor differs from the historical record")
    model = reconstruct(entry.family, backbone, state, cfg.get("train_counts"))
    del state
    phase.require_hash(path, entry.checkpoint_sha256)
    return model, processor, {"source": sources, "snapshot": pinned, "effective_processor_sha256": processor_sha,
                              "checkpoint_sha256": entry.checkpoint_sha256, "checkpoint_bytes": path.stat().st_size,
                              "epoch": entry.best["epoch"]}


def load_rows(entry, artifact_root, source_root=None):
    rows, info = data.load_train_val(artifact_root, entry.dataset, manifest_source_root=source_root,
                                     verify_images=False)
    expected = entry.configuration["manifest"]
    if (info["manifest_sha256"] != expected["manifest_sha256"]
            or info["counts"]["val"] != expected["counts"]["val"]
            or info["grade_support"]["val"] != expected["grade_support"]["val"]):
        raise ValueError("Loaded validation split differs from the published receipt")
    if not entry.arm.startswith("B4-") and info["image_order_sha256"] != expected["image_order_sha256"]:
        raise ValueError("Frozen manifest order differs")
    return rows


def validation_batches(rows, *, batch_size, sample_count=None, seed=42):
    """Deterministic validation selection, preserving manifest order after selection."""
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("Optimization batches accept frozen validation rows only")
    count = len(rows) if sample_count is None else sample_count
    if batch_size < 1 or not 1 <= count <= len(rows) or count % batch_size:
        raise ValueError("Require a positive sample count divisible by the fixed batch size")
    indices = range(len(rows))
    if count < len(rows):
        ranked = sorted(indices, key=lambda i: hashlib.sha256(
            f"{seed}:{rows[i]['image_id']}:{rows[i]['sha256']}".encode()).digest())
        indices = sorted(ranked[:count])
    selected = [rows[i] for i in indices]
    if {r["label"] for r in selected} != set(CLASS_ORDER):
        raise ValueError("Quality comparison requires all five grades in the fixed validation subset")
    return [selected[i:i + batch_size] for i in range(0, count, batch_size)]


def verified_images(batches):
    for batch in batches:
        if any(row["split"] != "val" for row in batch):
            raise ValueError("Image verification accepts validation only")
        for row in batch:
            phase.require_hash(row["path"], row["sha256"])


def prepare_inputs(rows, processor, family):
    """Decode verified validation images without hashing inside the latency interval."""
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("Preprocessing accepts frozen validation images only")
    rgb = []
    for row in rows:
        with Image.open(row["path"]) as image:
            rgb.append(ImageOps.exif_transpose(image).convert("RGB"))
    if family == "hf_v1_global_local":
        views = [geometry.pil_views(image) for image in rgb]
        shared = {"do_resize": False, "return_tensors": "pt"}
        g = processor(images=[view[0] for view in views], **shared)["pixel_values"]
        p = processor(images=[patch for view in views for patch in view[1]], **shared)["pixel_values"]
        return g, p.reshape(len(rows), 2, 3, 224, 224)
    if family not in ("hf_v1_linear", "hf_v1_ordinal"):
        raise ValueError(f"Unknown preprocessing family: {family}")
    return (processor(images=rgb, return_tensors="pt")["pixel_values"],)


def verify_inputs(rows, processor, family):
    inputs = prepare_inputs(rows, processor, family)
    if family == "hf_v1_global_local":
        by_row = []
        for row in rows:
            phase.require_hash(row["path"], row["sha256"])
            with Image.open(row["path"]) as image:
                by_row.append(geometry.model_views(ImageOps.exif_transpose(image).convert("RGB"), processor)[:2])
        expected = tuple(torch.stack(values) for values in zip(*by_row))
    else:
        expected = (fa._image_batch(rows, processor)[0],)
    if any(x.dtype != torch.float32 or x.shape != y.shape or not torch.equal(x, y)
           or not bool(torch.isfinite(x).all()) for x, y in zip(inputs, expected, strict=True)):
        raise ValueError("Optimized input preparation does not exactly match authoritative validation tensors")
    return inputs
