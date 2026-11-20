"""FP32 LDAM-DRW plus E5 ordinal and explicit adjacent-grade objective."""
import torch
from torch.nn import functional as F
from novelty_integrated.e5.objective import ordinal_targets
from novelty_integrated.e6 import config


def loss_components(categorical_logits, ordinal_logits, labels, *, train_counts, epoch):
    counts = config.require_counts(train_counts)
    if (labels.ndim != 1 or labels.dtype != torch.long or not len(labels)
            or categorical_logits.shape != (len(labels), 5)
            or ordinal_logits is None or ordinal_logits.shape != (len(labels), 4)
            or bool(((labels < 0) | (labels > 4)).any())):
        raise ValueError("E6 requires Nx5 categorical, Nx4 ordinal logits and five-grade labels")
    logits = categorical_logits.float()
    margins = torch.tensor(config.ldam_margins(counts), device=logits.device, dtype=torch.float32)
    adjusted = logits - F.one_hot(labels, 5).float() * margins[labels, None]
    per_image = F.cross_entropy(config.LDAM_SCALE * adjusted, labels, reduction="none")
    weights = config.weights_for_epoch(counts, epoch)
    if weights is not None:
        per_image = per_image * torch.tensor(weights, device=logits.device)[labels]
    categorical = per_image.mean()
    ordinal = F.binary_cross_entropy_with_logits(ordinal_logits.float(), ordinal_targets(labels))
    # Grade 0: compare 1; grade 1: compare 0 and 2; grade 2: compare 1.
    # Each eligible image contributes its mean neighboring softplus hinge.
    left = F.softplus(config.LDAM_SCALE * (logits[:, 0] - logits[:, 1])) / config.LDAM_SCALE
    right = F.softplus(config.LDAM_SCALE * (logits[:, 2] - logits[:, 1])) / config.LDAM_SCALE
    g0 = F.softplus(config.LDAM_SCALE * (logits[:, 1] - logits[:, 0])) / config.LDAM_SCALE
    g2 = F.softplus(config.LDAM_SCALE * (logits[:, 1] - logits[:, 2])) / config.LDAM_SCALE
    adjacent_per_image = torch.where(labels == 0, g0, torch.where(labels == 1, (left + right) / 2,
                                                                    torch.where(labels == 2, g2, torch.zeros_like(g0))))
    adjacent = adjacent_per_image.mean()
    total = categorical + config.ORDINAL_COEFFICIENT * ordinal + config.ADJACENT_COEFFICIENT * adjacent
    if not bool(torch.isfinite(total)):
        raise ValueError("Non-finite E6 objective")
    return {"total": total, "categorical": categorical, "ordinal": ordinal, "adjacent": adjacent}
