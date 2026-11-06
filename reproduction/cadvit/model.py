"""CAD-ViT candidate architecture extracted from the historical notebook.
No architectural modifications have been made.
"""

import torch
from torch import nn
from torch.nn import functional as F

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, dilation=1, activate=True):
        padding = dilation * (kernel_size // 2)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=False))
        super().__init__(*layers)

class MultiScaleFeatureBlock(nn.Module):
    # Pers. (1)-(2); concat diprioritaskan atas simbol sum Gbr. 2.
    def __init__(self, channels):
        super().__init__()
        self.rates = (3, 5, 7, 9)
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate, bias=False)
            for rate in self.rates
        ])
        self.fuse = nn.Conv2d(4 * channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        assert all(feature.shape == x.shape for feature in features)
        return self.fuse(torch.cat(features, dim=1))

class CoordinateAttention(nn.Module):
    # Pers. (3)-(6). Pembulatan C/r ke bawah dengan minimum 1 adalah assumption.
    def __init__(self, channels, reduction=16):
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction harus positif")
        reduced = max(1, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.SiLU(inplace=False),
        )
        self.height_gate = nn.Conv2d(reduced, channels, 1)
        self.width_gate = nn.Conv2d(reduced, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        shared = self.shared(torch.cat([pooled_h, pooled_w], dim=2))
        feat_h, feat_w = torch.split(shared, [height, width], dim=2)
        gate_h = self.height_gate(feat_h).sigmoid()
        gate_w = self.width_gate(feat_w).sigmoid().transpose(2, 3)
        return x * gate_h * gate_w

class InvertedResidualBlock(nn.Module):
    # IRB/IRB+A dari subdiagram Gbr. 3; expansion dan stride adalah assumption.
    def __init__(self, in_channels, out_channels, stride=1,
                 expansion=2, with_attention=False, reduction=16):
        super().__init__()
        hidden = in_channels * expansion
        self.expand = ConvBNAct(in_channels, hidden, 1)
        self.depthwise = ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden)
        self.attention = CoordinateAttention(hidden, reduction) if with_attention else nn.Identity()
        self.project = ConvBNAct(hidden, out_channels, 1, activate=False)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        y = self.project(self.attention(self.depthwise(self.expand(x))))
        return x + y if self.use_residual else y

class EnhancedMobileViTBlock(nn.Module):
    # Gbr. 3: MFB-unfold-Transformer-fold-C1+BS-residual-C3+BS.
    def __init__(self, channels, patch_size=2, depth=2, heads=4, mlp_ratio=2, use_mfb=True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels harus habis dibagi heads")
        self.patch_size = patch_size
        self.mfb = MultiScaleFeatureBlock(channels) if use_mfb else nn.Identity()
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=channels, nhead=heads,
                dim_feedforward=channels * mlp_ratio,
                dropout=0.0, activation="gelu", batch_first=True
            )
            for _ in range(depth)
        ])
        self.project = ConvBNAct(channels, channels, 1)
        self.fuse = ConvBNAct(channels, channels, 3)

    def unfold_tokens(self, features):
        # MobileViT v1: [B,C,H,W] -> [B*P,N,C], P=patch pixels, N=patch count.
        batch, channels, height, width = features.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError("H dan W harus kelipatan patch_size")
        patch_area = patch * patch
        num_patches = (height // patch) * (width // patch)
        columns = F.unfold(features, kernel_size=patch, stride=patch)
        return columns.reshape(batch, channels, patch_area, num_patches).permute(0, 2, 3, 1).reshape(
            batch * patch_area, num_patches, channels
        )

    def fold_tokens(self, tokens, output_size, batch):
        height, width = output_size
        patch = self.patch_size
        patch_area = patch * patch
        num_patches = (height // patch) * (width // patch)
        if tokens.shape != (batch * patch_area, num_patches, self.project[0].in_channels):
            raise ValueError("Shape token tidak cocok dengan ukuran fitur")
        columns = tokens.reshape(batch, patch_area, num_patches, -1).permute(0, 3, 1, 2).reshape(
            batch, -1, num_patches
        )
        return F.fold(columns, output_size=output_size, kernel_size=patch, stride=patch)

    def forward(self, x):
        batch, _, height, width = x.shape
        tokens = self.unfold_tokens(self.mfb(x))
        for layer in self.transformer:
            tokens = layer(tokens)
        global_features = self.fold_tokens(tokens, (height, width), batch)
        return self.fuse(x + self.project(global_features))

ARCH_CONFIG = {
    "stem_channels": 32,
    "stage_channels": (32, 64, 64, 96, 96, 128, 128, 160, 160),
    "expansion": 2,
    "reduction": 16,
    "patch_size": 2,
    "transformer_depth": 2,
    "transformer_heads": 4,
    "mlp_ratio": 2,
    "head_channels": 320,
    "num_classes": 5,
    "use_mfb": True,
    "use_attention": True,
}

class CADViTCandidate(nn.Module):
    # Strukturnya mengikuti Gbr. 3; angka internal berasal dari ARCH_CONFIG.
    def __init__(self, config=ARCH_CONFIG):
        super().__init__()
        c = config["stage_channels"]
        self.stem = ConvBNAct(3, config["stem_channels"], 3, stride=2)
        self.block_names = (
            "IRB-1", "IRB+A-1", "IRB-2", "IRB+A-2", "EMB-1",
            "IRB+A-3", "EMB-2", "IRB+A-4", "EMB-3"
        )
        def irb(in_c, out_c, stride, attention):
            return InvertedResidualBlock(
                in_c, out_c, stride=stride, expansion=config["expansion"],
                with_attention=attention and config["use_attention"], reduction=config["reduction"]
            )
        def emb(channels):
            return EnhancedMobileViTBlock(
                channels, patch_size=config["patch_size"],
                depth=config["transformer_depth"],
                heads=config["transformer_heads"],
                mlp_ratio=config["mlp_ratio"], use_mfb=config["use_mfb"]
            )
        self.blocks = nn.ModuleList([
            irb(config["stem_channels"], c[0], 1, False),
            irb(c[0], c[1], 2, True),
            irb(c[1], c[2], 1, False),
            irb(c[2], c[3], 2, True),
            emb(c[4]),
            irb(c[4], c[5], 2, True),
            emb(c[6]),
            irb(c[6], c[7], 2, True),
            emb(c[8]),
        ])
        self.head = ConvBNAct(c[-1], config["head_channels"], 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(config["head_channels"], config["num_classes"])

    def forward_features(self, x, record_shapes=False):
        shapes = {}
        x = self.stem(x)
        if record_shapes:
            shapes["stem"] = tuple(x.shape)
        for name, block in zip(self.block_names, self.blocks):
            x = block(x)
            if record_shapes:
                shapes[name] = tuple(x.shape)
        x = self.head(x)
        if record_shapes:
            shapes["head"] = tuple(x.shape)
        return (x, shapes) if record_shapes else x

    def forward(self, x):
        features = self.forward_features(x)
        return self.classifier(self.pool(features).flatten(1))
