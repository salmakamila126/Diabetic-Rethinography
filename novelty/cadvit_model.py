"""Checkpoint-compatible ablations around the immutable CADViTCandidate baseline.

Five-class logits remain the primary inference output. No data or checkpoint I/O.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from reproduction.cadvit.model import (
    ARCH_CONFIG, CADViTCandidate, CoordinateAttention,
    EnhancedMobileViTBlock, InvertedResidualBlock, MultiScaleFeatureBlock,
)


@dataclass(frozen=True)
class NoveltyFlags:
    p1_detail: bool = False
    p2_adaptive_mfb: bool = False
    p3_boundary_probe: bool = False
    p3_modulation: bool = False
    p4_ordinal_head: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise TypeError("All novelty flags must be booleans")
        if self.p3_modulation and not self.p3_boundary_probe:
            raise ValueError("P3 modulation requires a boundary probe")

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "baseline": NoveltyFlags(),
    "p1_only": NoveltyFlags(p1_detail=True),
    "p2_only": NoveltyFlags(p2_adaptive_mfb=True),
    "ordinary_gate_control": NoveltyFlags(p2_adaptive_mfb=True),
    "p3_probe_only": NoveltyFlags(p3_boundary_probe=True),
    "p3_only": NoveltyFlags(p3_boundary_probe=True, p3_modulation=True),
    "p4_only": NoveltyFlags(p4_ordinal_head=True),
    "p2_p3_probe": NoveltyFlags(p2_adaptive_mfb=True, p3_boundary_probe=True),
    "p2_p3_full": NoveltyFlags(p2_adaptive_mfb=True, p3_boundary_probe=True, p3_modulation=True),
    "p123": NoveltyFlags(p1_detail=True, p2_adaptive_mfb=True, p3_boundary_probe=True, p3_modulation=True),
    "p1234": NoveltyFlags(p1_detail=True, p2_adaptive_mfb=True, p3_boundary_probe=True,
                           p3_modulation=True, p4_ordinal_head=True),
}


class DetailResidual(nn.Module):
    """Project high-pass local contrast to the stem grid, initially zero."""

    def __init__(self, stem_channels):
        super().__init__()
        self.projection = nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, image):
        local_mean = F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)
        return self.projection(image - local_mean)


class AdaptiveMultiScaleFeatureBlock(MultiScaleFeatureBlock):
    """Sample-conditioned weights over original dilation branches and fusion."""

    def __init__(self, original):
        if not isinstance(original, MultiScaleFeatureBlock):
            raise TypeError("Expected original MultiScaleFeatureBlock")
        channels = original.branches[0].in_channels
        super().__init__(channels)
        if tuple(self.rates) != (3, 5, 7, 9):
            raise AssertionError("Baseline dilation rates changed")
        self.branches = original.branches
        self.fuse = original.fuse
        self.gate = nn.Linear(channels, 4)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def branch_coefficients(self, x):
        return 4 * F.softmax(self.gate(x.mean(dim=(2, 3))), dim=1)

    def forward(self, x):
        alpha = self.branch_coefficients(x)
        features = [
            branch(x) * alpha[:, i].reshape(-1, 1, 1, 1)
            for i, branch in enumerate(self.branches)
        ]
        if any(feature.shape != x.shape for feature in features):
            raise ValueError("Multi-scale branch shape changed")
        return self.fuse(torch.cat(features, dim=1))


class BoundaryCoordinateAttention(CoordinateAttention):
    """Original coordinate attention plus a pre-modulation ordinal probe."""

    def __init__(self, original, modulate=True):
        if not isinstance(original, CoordinateAttention):
            raise TypeError("Expected original CoordinateAttention")
        channels = original.height_gate.out_channels
        reduced = original.shared[0].out_channels
        super().__init__(channels, max(1, channels // reduced))
        self.shared = original.shared
        self.height_gate = original.height_gate
        self.width_gate = original.width_gate
        self.modulate = modulate
        self.severity_score = nn.Linear(channels, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.5))
        self.threshold_steps_raw = nn.Parameter(torch.full((3,), math.log(math.expm1(1.0))))
        if modulate:
            self.boundary_modulation = nn.Linear(4, channels)
            nn.init.zeros_(self.boundary_modulation.weight)
            nn.init.zeros_(self.boundary_modulation.bias)

    def ordered_thresholds(self):
        steps = F.softplus(self.threshold_steps_raw)
        return torch.cat((self.threshold_start.reshape(1),
                          self.threshold_start + torch.cumsum(steps, dim=0)))

    def forward_with_boundary(self, x):
        score = self.severity_score(x.mean(dim=(2, 3)))
        logits = score - self.ordered_thresholds().unsqueeze(0)
        baseline = super().forward(x)
        if not self.modulate:
            return baseline, logits
        delta = torch.tanh(self.boundary_modulation(torch.sigmoid(logits)))
        return baseline * (1 + delta[:, :, None, None]), logits

    def forward(self, x):
        return self.forward_with_boundary(x)[0]


@dataclass
class NoveltyOutput:
    logits: torch.Tensor
    boundary_logits: tuple[torch.Tensor, ...]
    ordinal_logits: torch.Tensor | None


class NovelCADViT(CADViTCandidate):
    """Preserve every baseline module key while adding optional research paths."""

    def __init__(self, flags: NoveltyFlags = NoveltyFlags()):
        if not isinstance(flags, NoveltyFlags):
            raise TypeError("flags must be NoveltyFlags")
        super().__init__(config=ARCH_CONFIG)
        self.flags = flags
        if flags.p1_detail:
            self.detail_path = DetailResidual(ARCH_CONFIG["stem_channels"])
        if flags.p2_adaptive_mfb:
            count = 0
            for block in self.blocks:
                if isinstance(block, EnhancedMobileViTBlock):
                    block.mfb = AdaptiveMultiScaleFeatureBlock(block.mfb)
                    count += 1
            if count != 3:
                raise AssertionError(f"Expected 3 MFB instances; found {count}")
        if flags.p3_boundary_probe:
            count = 0
            for block in self.blocks:
                if isinstance(block, InvertedResidualBlock) and isinstance(block.attention, CoordinateAttention):
                    block.attention = BoundaryCoordinateAttention(block.attention, modulate=flags.p3_modulation)
                    count += 1
            if count != 4:
                raise AssertionError(f"Expected 4 coordinate-attention instances; found {count}")
        if flags.p4_ordinal_head:
            self.ordinal_head = nn.Linear(ARCH_CONFIG["head_channels"], 4)
            nn.init.zeros_(self.ordinal_head.weight)
            nn.init.zeros_(self.ordinal_head.bias)

    def _features_and_boundaries(self, image, record_shapes=False):
        shapes = {}
        x = self.stem(image)
        if self.flags.p1_detail:
            detail = self.detail_path(image)
            if detail.shape != x.shape:
                raise ValueError("Detail residual does not match stem shape")
            x = x + detail
        if record_shapes:
            shapes["stem"] = tuple(x.shape)
        boundaries = []
        for name, block in zip(self.block_names, self.blocks):
            if isinstance(block, InvertedResidualBlock) and isinstance(block.attention, BoundaryCoordinateAttention):
                y = block.depthwise(block.expand(x))
                y, logits = block.attention.forward_with_boundary(y)
                y = block.project(y)
                x = x + y if block.use_residual else y
                boundaries.append(logits)
            else:
                x = block(x)
            if record_shapes:
                shapes[name] = tuple(x.shape)
        x = self.head(x)
        if record_shapes:
            shapes["head"] = tuple(x.shape)
        return x, tuple(boundaries), shapes

    def forward_features(self, x, record_shapes=False):
        features, _, shapes = self._features_and_boundaries(x, record_shapes)
        return (features, shapes) if record_shapes else features

    def forward_with_aux(self, image):
        features, boundaries, _ = self._features_and_boundaries(image)
        pooled = self.pool(features).flatten(1)
        return NoveltyOutput(
            logits=self.classifier(pooled),
            boundary_logits=boundaries,
            ordinal_logits=self.ordinal_head(pooled) if self.flags.p4_ordinal_head else None,
        )

    def forward(self, image):
        return self.forward_with_aux(image).logits

    def configuration(self):
        return self.flags.to_dict()


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
