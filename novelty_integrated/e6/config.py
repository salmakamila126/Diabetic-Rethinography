"""Prespecified E6 pilot constants, aligned with the verified E5/B1 protocol."""
import math
from novelty_integrated.e5 import config as e5

SCHEMA = "convnext-e6-boundary-v1"
OFFICIAL_SNAPSHOT = e5.OFFICIAL_SNAPSHOT
PILOT_EPOCHS = e5.PILOT_EPOCHS
WEIGHT_START_EPOCH = e5.WEIGHT_START_EPOCH
ORDINAL_COEFFICIENT = e5.ORDINAL_COEFFICIENT
EFFECTIVE_BATCH = e5.EFFECTIVE_BATCH
BACKBONE_LR = e5.BACKBONE_LR
HEAD_LR = e5.HEAD_LR
WEIGHT_DECAY = e5.WEIGHT_DECAY
MAX_GRAD_NORM = e5.MAX_GRAD_NORM
LDAM_MAX_MARGIN = 0.5
LDAM_SCALE = 30.0
EFFECTIVE_NUMBER_BETA = 0.9999
ADJACENT_COEFFICIENT = 0.1
validate_batch = e5.validate_batch
require_official_snapshot = e5.require_official_snapshot


def require_counts(counts):
    if len(counts) != 5 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("Five positive train-only class counts required")
    return tuple(counts)


def ldam_margins(counts):
    counts = require_counts(counts)
    raw = [n ** (-0.25) for n in counts]
    return tuple(LDAM_MAX_MARGIN * value / max(raw) for value in raw)


def effective_number_weights(counts):
    counts = require_counts(counts)
    beta = EFFECTIVE_NUMBER_BETA
    # Stable at beta close to one. Mean training-image weight is exactly one.
    raw = [-math.expm1(math.log(beta)) / -math.expm1(n * math.log(beta)) for n in counts]
    denominator = sum(n * value for n, value in zip(counts, raw)) / sum(counts)
    return tuple(value / denominator for value in raw)


def weights_for_epoch(counts, epoch):
    if not isinstance(epoch, int) or not 1 <= epoch <= PILOT_EPOCHS:
        raise ValueError("Pilot epoch must be in 1..20")
    return effective_number_weights(counts) if epoch >= WEIGHT_START_EPOCH else None
