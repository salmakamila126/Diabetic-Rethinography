"""Post-hoc Grad-CAM for the unchanged CADViTCandidate architecture.

Target ``blocks.5.attention`` is an implementation assumption: CAD-ViT Figure 8
does not identify a specific layer. No function here participates in training.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from reproduction.cadvit.evaluation import file_sha256
from reproduction.cadvit.experiments import (MODEL_SOURCE, PREPROCESS_SOURCE, VARIANTS,
                                            campaign_root, load_protocol)
from reproduction.cadvit.model import CADViTCandidate, CoordinateAttention


DEFAULT_TARGET_LAYER = "blocks.5.attention"  # ASSUMED; CA at a 32x32 map for 512px input


def gradcam(model, inputs, target_layer=DEFAULT_TARGET_LAYER, target_class=None):
    """Compute one heatmap from synthetic or explicitly approved non-test input."""
    if inputs.ndim != 4 or inputs.shape[0] != 1 or inputs.shape[1] != 3:
        raise ValueError("Grad-CAM expects one RGB tensor [1,3,H,W]")
    layer = model.get_submodule(target_layer)
    if not isinstance(layer, CoordinateAttention):
        raise ValueError("Target layer must be a CoordinateAttention module")
    captured = {}

    def capture(_module, _args, output):
        captured["activation"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(capture)
    was_training = model.training
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(inputs)
            prediction = int(logits.argmax(dim=1).item())
            chosen = prediction if target_class is None else int(target_class)
            if chosen < 0 or chosen >= logits.shape[1]:
                raise ValueError("Target class outside classifier output")
            logits[0, chosen].backward()
            activation = captured["activation"]
            gradient = activation.grad
            if gradient is None:
                raise RuntimeError("Attention activation did not receive gradients")
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = F.relu((weights * activation).sum(dim=1, keepdim=True))
            heatmap = F.interpolate(heatmap, size=inputs.shape[-2:], mode="bilinear",
                                    align_corners=False)[0, 0]
            minimum, maximum = heatmap.min(), heatmap.max()
            heatmap = ((heatmap - minimum) / (maximum - minimum + 1e-12)).detach().cpu()
            confidence = float(logits.softmax(dim=1)[0, prediction].detach().cpu())
        return {"heatmap": heatmap, "predicted_class": prediction,
                "target_class": chosen, "confidence": confidence,
                "target_layer": target_layer,
                "status": "SYNTHETIC_TEST or explicitly approved non-test image"}
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def load_controlled_best(artifact_root, campaign, variant_name, *, trust_checkpoint=False):
    """Load only a trusted campaign checkpoint after config/hash validation."""
    if not trust_checkpoint:
        raise ValueError("Explicit trust_checkpoint=True required for torch.load")
    if variant_name not in VARIANTS:
        raise ValueError("Unknown variant")
    run_dir = campaign_root(artifact_root, campaign) / variant_name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    protocol, protocol_sha = load_protocol(artifact_root, campaign)
    if (config.get("protocol_sha256") != protocol_sha
            or config.get("model_sha256") != file_sha256(MODEL_SOURCE)
            or config.get("preprocessing_sha256") != file_sha256(PREPROCESS_SOURCE)
            or config.get("variant", {}).get("name") != variant_name):
        raise ValueError("Checkpoint source/config/protocol mismatch")
    if campaign == "balanced_v1" and config.get("campaign") != campaign:
        raise ValueError("Balanced checkpoint campaign mismatch")
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if state.get("record") != config or not 0 <= state.get("epoch", -1) < protocol.epochs:
        raise ValueError("Best checkpoint record/epoch mismatch")
    model = CADViTCandidate(VARIANTS[variant_name].model_config()).eval()
    model.load_state_dict(state["model"], strict=True)
    return model, {"campaign": campaign, "variant": variant_name,
                   "best_epoch": state["epoch"] + 1,
                   "checkpoint_sha256": file_sha256(run_dir / "best.pt"),
                   "protocol_sha256": protocol_sha,
                   "manifest_sha256": config["manifest_sha256"],
                   "model_source_sha256": config["model_sha256"]}


def plot_gradcam(model_input_rgb, result, true_grade, provenance):
    """Display model-input RGB, aligned Grad-CAM heatmap, and overlay without training.

    Pass the exact preprocessed RGB pixels converted to the model tensor, NOT
    the uncropped source photograph: cropping/resizing changes coordinates.
    """
    import matplotlib.pyplot as plt

    original = np.asarray(model_input_rgb)
    if original.ndim != 3 or original.shape[2] != 3 or original.dtype != np.uint8:
        raise ValueError("Model input image must be RGB uint8")
    if tuple(result["heatmap"].shape) != tuple(original.shape[:2]):
        raise ValueError("Grad-CAM heatmap and model-input geometry differ")
    heat = result["heatmap"].detach().float()[None, None]
    heat = F.interpolate(heat, size=original.shape[:2], mode="bilinear",
                         align_corners=False)[0, 0].numpy()
    heat = np.clip(heat, 0.0, 1.0)
    colored = plt.get_cmap("turbo")(heat)[..., :3]
    overlay = np.clip(0.55 * original.astype(float) / 255 + 0.45 * colored, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, image, title in zip(axes, (original, heat, overlay),
                                ("Model input (RGB)", "Grad-CAM heatmap", "Aligned overlay")):
        ax.imshow(image, cmap="turbo" if title == "Grad-CAM heatmap" else None,
                  vmin=0 if title == "Grad-CAM heatmap" else None,
                  vmax=1 if title == "Grad-CAM heatmap" else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f'Grade {true_grade}; predicted {result["predicted_class"]}; '
                 f'confidence {result["confidence"]:.3f}; layer {result["target_layer"]}\n'
                 f'checkpoint {provenance.get("checkpoint_sha256", "SYNTHETIC_TEST")[:16]}')
    fig.tight_layout()
    return fig
