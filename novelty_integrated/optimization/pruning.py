"""Expansion-only structured pruning and forward-based MAC accounting."""

import copy
import math

import torch
from torch import nn

RATIOS = (0.1, 0.3, 0.5)


def _rank(proj, ratio):
    weight = proj.weight.detach().abs().flatten(1)
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Cannot rank non-finite expansion channels")
    count = weight.shape[0] - math.floor(weight.shape[0] * ratio)
    order = torch.argsort(weight.sum(1), descending=True, stable=True)
    return order[:count].sort().values


def _linear(original, inputs, outputs):
    weight = original.weight.detach()
    layer = nn.Linear(len(inputs), len(outputs), bias=original.bias is not None,
                      device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight[outputs][:, inputs])
        if original.bias is not None:
            layer.bias.copy_(original.bias[outputs])
    layer.requires_grad_(original.weight.requires_grad)
    if original.bias is not None:
        layer.bias.requires_grad_(original.bias.requires_grad)
    return layer.train(original.training)


def _linear_pair(first, second):
    if (type(first) is not nn.Linear or type(second) is not nn.Linear
            or first.out_features != second.in_features
            or first.in_features != second.out_features
            or first.out_features <= first.in_features):
        raise ValueError("Expected a compatible width-preserving Linear expansion pair")


def _replace_pair(parent, first_name, second_name, kept):
    first, second = getattr(parent, first_name), getattr(parent, second_name)
    _linear_pair(first, second)
    width = torch.arange(first.in_features, device=first.weight.device)
    setattr(parent, first_name, _linear(first, width, kept))
    setattr(parent, second_name, _linear(second, kept, width))


def _hf_blocks(backbone):
    if getattr(getattr(backbone, "config", None), "model_type", None) != "convnext":
        raise ValueError("HF V1 pruning requires the ConvNeXt model type")
    for stage_index, stage in enumerate(backbone.encoder.stages):
        for block_index, block in enumerate(stage.layers):
            if hasattr(block, "grn") or not hasattr(block, "dwconv"):
                raise ValueError("Expected HF ConvNeXt V1 depthwise residual block without GRN")
            _linear_pair(block.pwconv1, block.pwconv2)
            yield f"encoder.stages.{stage_index}.layers.{block_index}", block


def _timm_blocks(model):
    from timm.layers import GlobalResponseNormMlp
    from timm.models.convnext import ConvNeXt, ConvNeXtBlock

    if not isinstance(model, ConvNeXt):
        raise ValueError("Expected a timm ConvNeXtV2 model")
    for name, block in model.named_modules():
        if isinstance(block, ConvNeXtBlock):
            if not isinstance(block.mlp, GlobalResponseNormMlp) or block.mlp.grn.spatial_dim != (1, 2):
                raise ValueError("Only channels-last timm V2 GRN expansion is supported")
            _linear_pair(block.mlp.fc1, block.mlp.fc2)
            yield name, block.mlp


def _prune(model, ratio, *, timm=False, topology=None):
    if ratio not in RATIOS:
        raise ValueError(f"Choose one of the prespecified pruning ratios {RATIOS}")
    result = copy.deepcopy(model)
    blocks = list(_timm_blocks(result) if timm else _hf_blocks(result.backbone))
    if not blocks:
        raise ValueError("No compatible expansion blocks")
    if topology is not None and [r["layer"] for r in topology] != [name for name, _ in blocks]:
        raise ValueError("Recorded topology does not match the architecture")
    layers = []
    for index, (name, block) in enumerate(blocks):
        first_name, second_name = ("fc1", "fc2") if timm else ("pwconv1", "pwconv2")
        first = getattr(block, first_name)
        if topology is None:
            kept = _rank(first, ratio)
        else:
            entry = topology[index]
            channels = entry["kept_indices"]
            width = first.out_features
            if (entry["original_width"] != width or len(channels) != width - math.floor(width * ratio)
                    or not channels or channels != sorted(set(channels))
                    or channels[0] < 0 or channels[-1] >= width):
                raise ValueError("Recorded expansion indices are invalid")
            kept = torch.tensor(channels, device=first.weight.device, dtype=torch.long)
        _replace_pair(block, first_name, second_name, kept)
        if timm:
            grn = block.grn
            if grn.weight.shape != (first.out_features,) or grn.bias.shape != grn.weight.shape:
                raise ValueError("Unrecognized timm V2 GRN parameter layout")
            grn.weight = nn.Parameter(grn.weight.detach()[kept].clone(), requires_grad=grn.weight.requires_grad)
            grn.bias = nn.Parameter(grn.bias.detach()[kept].clone(), requires_grad=grn.bias.requires_grad)
        layers.append({"layer": name, "original_width": first.out_features,
                       "pruned_width": len(kept), "kept_indices": kept.tolist()})
    return result.eval(), layers


def prune_hf(model, ratio, *, topology=None):
    """Prune the shared HF V1 backbone once. Head and fusion dimensions stay fixed."""
    return _prune(model, ratio, topology=topology)


def prune_timm_v2(model, ratio, *, topology=None):
    """Separate optional adapter for a verified timm V2 model with channels-last GRN."""
    return _prune(model, ratio, timm=True, topology=topology)


@torch.inference_mode()
def profile(model, inputs):
    """Count Conv2d and Linear MACs on the actual categorical forward path."""
    counts = {"conv2d": 0, "linear": 0}

    def count_conv(module, args, output):
        kh, kw = module.kernel_size
        counts["conv2d"] += output.numel() * (module.in_channels // module.groups) * kh * kw

    def count_linear(module, args, output):
        counts["linear"] += output.numel() * module.in_features

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(count_conv))
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(count_linear))
    try:
        result = model(*inputs)
        images = len(inputs[0])
        if tuple(result.shape) != (images, 5) or not bool(torch.isfinite(result).all()):
            raise ValueError("Expected finite categorical logits in Nx5 grade order")
    finally:
        for hook in hooks:
            hook.remove()
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "macs_per_image": sum(counts.values()) // images,
            "conv2d_macs_per_image": counts["conv2d"] // images,
            "linear_macs_per_image": counts["linear"] // images,
            "flops_per_image_two_per_mac": 2 * sum(counts.values()) // images,
            "accounting": "Executed Conv2d and Linear only; no biases, normalization, activation, "
                          "pooling, gate elementwise operations or memory traffic. Shared E7 calls all count."}
