"""FP32 categorical and auxiliary ordinal losses for E5."""

import torch
from torch.nn import functional as F

from novelty_integrated.e5 import config


def ordinal_targets(labels):
    if labels.ndim != 1 or labels.dtype != torch.long or bool(((labels < 0) | (labels > 4)).any()):
        raise ValueError("Require five-grade integer labels")
    return (labels[:, None] > torch.arange(4, device=labels.device)[None, :]).float()


def loss_components(categorical_logits, ordinal_logits, labels, *, class_weights=None,
                    ordinal_coefficient=config.ORDINAL_COEFFICIENT):
    """Return per-image-mean losses; caller handles actual accumulation windows."""
    if categorical_logits.shape != (len(labels), 5) or len(labels) == 0:
        raise ValueError("Expected nonempty Nx5 categorical logits")
    if ordinal_coefficient != config.ORDINAL_COEFFICIENT:
        raise ValueError("E5 pilot ordinal coefficient is locked at 0.25")
    ce = F.cross_entropy(categorical_logits.float(), labels, reduction="none")
    if class_weights is not None:
        if len(class_weights) != 5:
            raise ValueError("Expected five class weights")
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=labels.device)
        if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError("Class weights must be finite and positive")
        ce = ce * weights[labels]
    categorical = ce.mean()
    if ordinal_logits is None:
        ordinal = categorical.new_zeros(())
    else:
        if ordinal_logits.shape != (len(labels), 4):
            raise ValueError("Expected Nx4 ordinal threshold logits")
        ordinal = F.binary_cross_entropy_with_logits(
            ordinal_logits.float(), ordinal_targets(labels))
    total = categorical + (ordinal_coefficient * ordinal if ordinal_logits is not None else ordinal)
    if not bool(torch.isfinite(total)):
        raise ValueError("Non-finite E5 objective")
    return {"total": total, "categorical": categorical, "ordinal": ordinal}
