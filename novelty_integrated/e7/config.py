"""Frozen starting protocol; E7 is exploratory and test-free."""
from novelty_integrated.e5 import config as e5

SCHEMA = "convnext-e7-global-local-v1"
OFFICIAL_SNAPSHOT = e5.OFFICIAL_SNAPSHOT
EPOCHS = 20
SEED = 42
GLOBAL_SIZE = 384
PATCH_SIZE = 224
PATCH_COUNT = 2
PATCH_WIDTH_FRACTION = 0.55  # Also applied to crop height; local scale > global scale.
FOV_THRESHOLD = 8
FOV_MARGIN_FRACTION = 0.05
EFFECTIVE_BATCH = e5.EFFECTIVE_BATCH
BACKBONE_LR = e5.BACKBONE_LR
HEAD_LR = e5.HEAD_LR
WEIGHT_DECAY = e5.WEIGHT_DECAY
MAX_GRAD_NORM = e5.MAX_GRAD_NORM
ORDINAL_COEFFICIENT = e5.ORDINAL_COEFFICIENT
WEIGHT_START_EPOCH = e5.WEIGHT_START_EPOCH

def validate_batch(micro_batch, accumulation):
    if micro_batch not in (1, 2, 4, 8, 16, 32) or micro_batch * accumulation != EFFECTIVE_BATCH:
        raise ValueError("E7 requires effective batch 32")
