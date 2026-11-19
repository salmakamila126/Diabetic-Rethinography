"""Locked 20-epoch E5 pilot settings and four controlled arms."""

import math

SCHEMA = "convnext-e5-integrated-v1"
OFFICIAL_SNAPSHOT = {
    "hub_id": "facebook/convnext-tiny-224",
    "revision": "6166b7613034066690a621d8bf25ffdf181a34f0",
    "config_sha256": "441f982e839a9ea96df7c1678422dc4e65cd5632731a6957f6aa03728195423d",
    "processor_config_sha256": "fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741",
    "weight_file": "pytorch_model.bin",
    "weight_sha256": "1fe15b2eba9e355de14c4e244e6365bf61ffef84da7d3442c0afbb2d8564f477",
}
PILOT_EPOCHS = 20
WEIGHT_START_EPOCH = 11
ORDINAL_COEFFICIENT = 0.25
EFFECTIVE_BATCH = 32
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# (ordinal auxiliary task, deferred class weighting)
ARMS = {
    "control": (False, False),
    "ordinal": (True, False),
    "deferred": (False, True),
    "integrated": (True, True),
}


def arm_components(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown E5 arm: {arm}")
    return ARMS[arm]


def train_class_weights(counts):
    """B2's inverse-sqrt weights, normalized to mean train-image weight one."""
    if len(counts) != 5 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                               for n in counts):
        raise ValueError("Require positive integer training counts for all five grades")
    raw = [1.0 / math.sqrt(n) for n in counts]
    denominator = sum(n * w for n, w in zip(counts, raw)) / sum(counts)
    return tuple(w / denominator for w in raw)


def weights_for_epoch(counts, arm, epoch):
    """None denotes ordinary CE; only training counts may enter this function."""
    _, deferred = arm_components(arm)
    if not isinstance(epoch, int) or epoch < 1 or epoch > PILOT_EPOCHS:
        raise ValueError("E5 pilot epoch outside 1..20")
    return train_class_weights(counts) if deferred and epoch >= WEIGHT_START_EPOCH else None


def validate_batch(micro_batch, accumulation):
    if micro_batch not in (1, 4, 8, 16, 32) or accumulation != EFFECTIVE_BATCH // micro_batch:
        raise ValueError("Require a B1-compatible micro-batch and effective batch 32")


def require_official_snapshot(snapshot):
    if any(snapshot.get(key) != value for key, value in OFFICIAL_SNAPSHOT.items()):
        raise ValueError("E5 requires the exact official ConvNeXt snapshot used by B1")
