"""Objectives and strict baseline-checkpoint transfer for the CAD-ViT prototype.

This module defines losses and CPU-safe compatibility operations only. It does
not train, infer on a dataset, save checkpoints, or touch experiment artifacts.
"""
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.torch_version import TorchVersion
from numpy._core.multiarray import _reconstruct

from novelty.cadvit_model import NoveltyFlags, NoveltyOutput
from reproduction.cadvit.model import CADViTCandidate


BOUNDARY_LOSS_WEIGHT = 0.20
ORDINAL_LOSS_WEIGHT = 0.20
_ALLOWED_PICKLE_GLOBALS = {
    "numpy._core.multiarray._reconstruct",
    "numpy.dtype",
    "numpy.ndarray",
    "torch.torch_version.TorchVersion",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restricted_checkpoint_state(path, expected_sha256):
    """Read a project-owned checkpoint with a pinned hash and narrow globals.

    No unrestricted pickle loading. Only the model mapping is returned; optimizer,
    RNG, and scaler state are not used by the prototype compatibility check.
    """
    path = Path(path)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("An expected full SHA-256 is required")
    observed = file_sha256(path)
    if observed != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch")
    referenced = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not referenced.issubset(_ALLOWED_PICKLE_GLOBALS):
        raise ValueError(f"Unexpected checkpoint pickle globals: {sorted(referenced - _ALLOWED_PICKLE_GLOBALS)}")
    allowed = [TorchVersion, _reconstruct, np.dtype, np.ndarray,
               type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("Expected checkpoint with a model state mapping")
    state = checkpoint["model"]
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("Model state must contain only named tensors")
    return state, {"path": str(path), "sha256": observed,
                   "checkpoint_epoch": checkpoint.get("epoch"),
                   "record": checkpoint.get("record"),
                   "recovery": checkpoint.get("recovery")}


def transfer_baseline_state(model: nn.Module, baseline_state: Mapping):
    """Require a complete exact baseline state; initialize new keys separately."""
    baseline = CADViTCandidate()
    expected = baseline.state_dict()
    if not isinstance(baseline_state, Mapping):
        raise TypeError("baseline_state must be a mapping")
    provided_keys = set(baseline_state)
    expected_keys = set(expected)
    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)
    shape_mismatch = sorted(
        key for key in expected_keys & provided_keys
        if not isinstance(baseline_state[key], torch.Tensor)
        or tuple(baseline_state[key].shape) != tuple(expected[key].shape)
        or baseline_state[key].dtype != expected[key].dtype
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError({
            "missing_original": missing,
            "unexpected_checkpoint": unexpected,
            "original_shape_or_dtype_mismatch": shape_mismatch,
        })
    target = model.state_dict()
    unavailable = sorted(expected_keys - set(target))
    target_mismatch = sorted(
        key for key in expected_keys & set(target)
        if target[key].shape != expected[key].shape or target[key].dtype != expected[key].dtype
    )
    if unavailable or target_mismatch:
        raise ValueError({"missing_original_in_target": unavailable,
                          "target_shape_or_dtype_mismatch": target_mismatch})
    new_keys = sorted(set(target) - expected_keys)
    for key in expected:
        target[key] = baseline_state[key]
    result = model.load_state_dict(target, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError("Strict target load returned incompatible keys")
    return {
        "original_tensors_loaded": len(expected),
        "original_parameters_loaded": sum(p.numel() for p in baseline.parameters()),
        "newly_initialized_tensors": new_keys,
        "missing_original_tensors": [],
        "unexpected_checkpoint_tensors": [],
        "mismatched_original_tensors": [],
    }


def ordinal_targets(labels):
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional tensor")
    if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("labels must be integer grades")
    if labels.numel() == 0 or bool(((labels < 0) | (labels > 4)).any()):
        raise ValueError("labels must contain grades 0 through 4")
    return (labels[:, None] > torch.arange(4, device=labels.device)[None, :]).to(torch.float32)


def research_loss(output: NoveltyOutput, labels, flags: NoveltyFlags,
                  boundary_weight=BOUNDARY_LOSS_WEIGHT,
                  ordinal_weight=ORDINAL_LOSS_WEIGHT):
    """CE + optional mean four-site boundary BCE + optional global ordinal BCE."""
    if not isinstance(output, NoveltyOutput) or not isinstance(flags, NoveltyFlags):
        raise TypeError("Expected NoveltyOutput and NoveltyFlags")
    for weight in (boundary_weight, ordinal_weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Loss weights must be finite and nonnegative")
    target = ordinal_targets(labels)
    if output.logits.shape != (labels.numel(), 5):
        raise ValueError("Primary logits must have shape [B,5]")
    ce = F.cross_entropy(output.logits, labels.long())
    zero = ce.new_zeros(())
    if flags.p3_boundary_probe:
        if len(output.boundary_logits) != 4 or any(z.shape != target.shape for z in output.boundary_logits):
            raise ValueError("Expected four [B,4] boundary outputs")
        boundary = torch.stack([
            F.binary_cross_entropy_with_logits(z, target) for z in output.boundary_logits
        ]).mean()
    else:
        if output.boundary_logits:
            raise ValueError("Boundary outputs present when P3 is disabled")
        boundary = zero
    if flags.p4_ordinal_head:
        if output.ordinal_logits is None or output.ordinal_logits.shape != target.shape:
            raise ValueError("Expected [B,4] ordinal head output")
        ordinal = F.binary_cross_entropy_with_logits(output.ordinal_logits, target)
    else:
        if output.ordinal_logits is not None:
            raise ValueError("Ordinal output present when P4 is disabled")
        ordinal = zero
    total = ce + (boundary_weight * boundary if flags.p3_boundary_probe else zero)
    total = total + (ordinal_weight * ordinal if flags.p4_ordinal_head else zero)
    if not all(bool(torch.isfinite(value)) for value in (total, ce, boundary, ordinal)):
        raise FloatingPointError("Non-finite CAD-ViT research loss")
    return {"total": total, "cross_entropy": ce,
            "boundary_bce": boundary, "ordinal_bce": ordinal}
