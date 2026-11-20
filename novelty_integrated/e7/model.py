"""Shared ConvNeXt trunk with global and two original-resolution local views."""
import copy
import torch
from torch import nn
from novelty_integrated.e5.model import E5Heads
from novelty_integrated.e7 import config


def pooled_convnext(backbone, pixels, expected_hw: int):
    result = backbone(pixel_values=pixels)
    hidden, vector = result.last_hidden_state, result.pooler_output
    if (tuple(hidden.shape) != (len(pixels), 768, expected_hw, expected_hw)
            or vector is None or tuple(vector.shape) != (len(pixels), 768)):
        raise ValueError("ConvNeXt output differs from expected E7 spatial layout")
    return vector


class GlobalLocalModel(nn.Module):
    def __init__(self, backbone: nn.Module, train_counts):
        super().__init__()
        self.backbone = backbone
        self.heads = E5Heads(768, train_counts, ordinal=True)
        self.gate = nn.Linear(1536, 1)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(-2.0)

    def forward(self, global_pixels, patch_pixels):
        batch = len(global_pixels)
        if (tuple(global_pixels.shape) != (batch, 3, config.GLOBAL_SIZE, config.GLOBAL_SIZE)
                or tuple(patch_pixels.shape) != (batch, config.PATCH_COUNT, 3,
                                                  config.PATCH_SIZE, config.PATCH_SIZE)):
            raise ValueError("E7 requires Bx3x384x384 and Bx2x3x224x224")
        global_embedding = pooled_convnext(self.backbone, global_pixels, 12)
        local = patch_pixels.reshape(batch * config.PATCH_COUNT, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        local_embeddings = pooled_convnext(self.backbone, local, 7).reshape(batch, config.PATCH_COUNT, 768)
        local_mean = local_embeddings.mean(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat((global_embedding.float(), local_mean.float()), dim=1)))
        fused = (1 - gate) * global_embedding.float() + gate * local_mean.float()
        categorical, ordinal = self.heads(fused)
        return categorical, ordinal, gate


class ModelEMA:
    """Optional fixed-decay shadow; selection mode must be fixed before training."""
    def __init__(self, model: nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        current = model.state_dict()
        shadow = self.shadow.state_dict()
        if current.keys() != shadow.keys():
            raise ValueError("EMA model parameters differ")
        for name, value in current.items():
            if torch.is_floating_point(value):
                shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                shadow[name].copy_(value.detach())
        self.updates += 1

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state):
        if state.get("decay") != self.decay or state.get("updates", -1) < 0:
            raise ValueError("EMA protocol differs")
        self.shadow.load_state_dict(state["shadow"], strict=True)
        self.updates = state["updates"]
