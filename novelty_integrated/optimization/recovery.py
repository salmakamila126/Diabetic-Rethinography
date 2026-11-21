"""Explicit Phase C recovery for disposable E5/E6 derivatives."""

import time

import torch

from novelty import foundation_adaptation as fa
from novelty_integrated.e5 import config, runner as e5runner
from novelty_integrated.e6 import runner as e6runner


def recover(model, entry, rows, processor, *, epochs, micro_batch, device="cpu"):
    if entry.arm not in ("E5", "E6") or not 1 <= epochs <= 3:
        raise ValueError("Recovery is implemented only for E5/E6, with a fixed budget of 1–3 epochs")
    if micro_batch not in (1, 4, 8, 16, 32):
        raise ValueError("Recovery requires a B1-compatible micro-batch")
    cfg = entry.configuration
    counts = cfg["train_counts"]
    if any(row["split"] != "train" for row in rows["train"]) or [
            sum(row["label"] == grade for row in rows["train"]) for grade in range(5)] != counts:
        raise ValueError("Recovery requires exactly the original frozen training grade counts")
    device = torch.device(device)
    model.to(device)
    fa.configure_trainable(model.backbone, "convnext_tiny", "full")
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["backbone_lr"]},
        {"params": model.head.parameters(), "lr": cfg["head_lr"]},
    ], weight_decay=cfg["weight_decay"])
    fa._seed(cfg["seed"])
    accumulation = config.EFFECTIVE_BATCH // micro_batch
    selected_epoch = entry.best["epoch"]
    common = dict(device=device, precision="fp32", total_images=len(rows["train"]),
                  micro_batch=micro_batch, accumulation=accumulation)
    history = []
    for index in range(1, epochs + 1):
        augmentation_epoch = selected_epoch + index
        began = time.perf_counter()
        batches = fa._epochs_batches("full", rows, processor, None, micro_batch,
                                     cfg["seed"], augmentation_epoch, experiment="B1")
        if entry.arm == "E5":
            weights = config.weights_for_epoch(counts, "integrated", selected_epoch)
            summary = e5runner._train_batches(model.backbone, model.head, optimizer, batches,
                                             class_weights=weights, **common)
            e5runner._assert_thresholds(model.head)
        else:
            summary = e6runner._train_batches(model.backbone, model.head, optimizer, batches,
                                             train_counts=counts, epoch=selected_epoch, **common)
            e6runner._assert_thresholds(model.head)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"recovery_epoch": index, "augmentation_epoch": augmentation_epoch,
                        "parent_weight_schedule_epoch": selected_epoch,
                        "train_seconds": time.perf_counter() - began, "train": summary})
    model.cpu().float().eval()
    return {"objective": cfg["objective"], "history": history, "seed": cfg["seed"],
            "device": str(device), "precision": "fp32", "effective_batch": config.EFFECTIVE_BATCH,
            "micro_batch": micro_batch, "selection": "fixed final recovery epoch; no validation tuning",
            "schedule": "weight activation stays at the parent selected epoch",
            "train_preprocessing": "original B1 augmentation, uncached; all source hashes verified",
            "optimizer": {"name": "fresh AdamW", "backbone_lr": cfg["backbone_lr"],
                          "head_lr": cfg["head_lr"], "weight_decay": cfg["weight_decay"]}}
