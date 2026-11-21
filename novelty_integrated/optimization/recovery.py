"""Train-split-only recovery for independently prepared E5, E6 and E7 arms."""

import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
import torch

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from novelty_integrated.e5 import config, runner as e5runner
from novelty_integrated.e5.input_pipeline import EpochBatches
from novelty_integrated.e6 import runner as e6runner
from novelty_integrated.e7 import runner as e7runner
from novelty_integrated.e7.input_pipeline import ViewCollator
from novelty_integrated.e7.model import ModelEMA

ARMS = ("E5", "E6", "E7")
EPOCHS = 3


class _SourceRGB:
    @staticmethod
    def read(row):
        if row["split"] != "train":
            raise ValueError("E7 recovery decoding requires a frozen training row")
        phase.require_hash(row["path"], row["sha256"])
        with Image.open(row["path"]) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _e7_batches(rows, processor, *, seed, epoch, micro_batch):
    sampler = EpochBatches(len(rows), micro_batch, seed, training=True)
    sampler.epoch = epoch
    collate = ViewCollator(processor, _SourceRGB(), training=True, seed=seed, epoch=SimpleNamespace(value=epoch))
    for indices in sampler:
        yield collate([rows[i] for i in indices])


def _safe_rng():
    state = fa._rng_state()
    name, keys, position, has_gauss, cached = state["numpy"]
    state["numpy"] = (name, keys.tolist(), position, has_gauss, cached)
    return state


def _restore_rng(state):
    name, keys, position, has_gauss, cached = state["numpy"]
    fa._restore_rng({**state, "numpy": (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)})


def recover(model, entry, rows, processor, *, epochs, micro_batch, device="cpu", resume=None, save_epoch=None):
    if entry.arm not in ARMS or not 1 <= epochs <= EPOCHS:
        raise ValueError("Recovery requires E5, E6 or E7 with a supported recovery budget")
    if micro_batch not in (1, 4, 8, 16, 32):
        raise ValueError("Recovery requires a B1-compatible micro-batch")
    cfg = entry.configuration
    counts = cfg["train_counts"]
    train_rows = rows["train"]
    if any(r["split"] != "train" or r["label"] not in range(5) for r in train_rows) or [
            sum(row["label"] == grade for row in train_rows) for grade in range(5)] != counts:
        raise ValueError("Recovery requires exactly the original frozen training grade counts")
    device = torch.device(device)
    model.to(device)
    fa.configure_trainable(model.backbone, "convnext_tiny", "full")
    multiview = entry.arm == "E7"
    inner = model.model if multiview else None
    head_parameters = (*inner.heads.parameters(), *inner.gate.parameters()) if multiview else model.head.parameters()
    selection = cfg["selection_weights"] if multiview else "raw"
    if selection not in ("raw", "ema"):
        raise ValueError("Unsupported recovery weight selection")
    ema = ModelEMA(inner, cfg["ema_decay"]) if multiview and cfg["ema_decay"] is not None else None
    if selection == "ema" and ema is None:
        raise ValueError("E7 EMA selection requires the verified EMA decay")
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["backbone_lr"]},
        {"params": head_parameters, "lr": cfg["head_lr"]},
    ], weight_decay=cfg["weight_decay"])
    fa._seed(cfg["seed"])
    accumulation = config.EFFECTIVE_BATCH // micro_batch
    selected_epoch = entry.best["epoch"]
    common = dict(device=device, precision="fp32", total_images=len(train_rows),
                  micro_batch=micro_batch, accumulation=accumulation)
    history = []
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        if (ema is None) != (resume["ema"] is None):
            raise ValueError("Recovery EMA checkpoint differs from the recovery configuration")
        if ema is not None:
            ema.load_state_dict(resume["ema"])
        history = resume["history"]
        _restore_rng(resume["rng"])
    for index in range(len(history) + 1, epochs + 1):
        augmentation_epoch = selected_epoch + index
        began = time.perf_counter()
        print(f"Recovery epoch {index}/{epochs}: {len(train_rows)} frozen train rows", flush=True)
        if multiview:
            batches = _e7_batches(train_rows, processor, seed=cfg["seed"], epoch=augmentation_epoch,
                                  micro_batch=micro_batch)
            summary = e7runner._train_batches(inner, optimizer, ema, batches, counts=counts,
                                              epoch=selected_epoch, **common)
            e5runner._assert_thresholds(inner.heads)
        else:
            batches = fa._epochs_batches("full", {"train": train_rows}, processor, None, micro_batch,
                                         cfg["seed"], augmentation_epoch, experiment="B1")
        if entry.arm == "E5":
            weights = config.weights_for_epoch(counts, "integrated", selected_epoch)
            summary = e5runner._train_batches(model.backbone, model.head, optimizer, batches,
                                             class_weights=weights, **common)
            e5runner._assert_thresholds(model.head)
        elif entry.arm == "E6":
            summary = e6runner._train_batches(model.backbone, model.head, optimizer, batches,
                                             train_counts=counts, epoch=selected_epoch, **common)
            e6runner._assert_thresholds(model.head)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"recovery_epoch": index, "augmentation_epoch": augmentation_epoch,
                        "parent_weight_schedule_epoch": selected_epoch,
                        "train_seconds": time.perf_counter() - began, "train": summary})
        if save_epoch is not None:
            save_epoch({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None,
                        "history": history, "rng": _safe_rng()})
        print(f"Recovery epoch {index}/{epochs} complete: train loss {summary.get('loss', 'unavailable')}", flush=True)
    if selection == "ema":
        inner.load_state_dict(ema.shadow.state_dict(), strict=True)
        e5runner._assert_thresholds(inner.heads)
    model.cpu().float().eval()
    return {"objective": cfg["objective"], "history": history, "seed": cfg["seed"],
            "device": str(device), "precision": "fp32", "effective_batch": config.EFFECTIVE_BATCH,
            "micro_batch": micro_batch, "selection": "fixed final recovery epoch; no validation tuning",
            "export_weights": selection,
            "schedule": "weight activation stays at the parent selected epoch",
            "train_preprocessing": ("E7 ViewCollator with original RGB decoding and keyed B1 before geometry"
                                    if multiview else "original B1 augmentation, uncached"),
            "optimizer": {"name": "fresh AdamW", "backbone_lr": cfg["backbone_lr"],
                          "head_lr": cfg["head_lr"], "weight_decay": cfg["weight_decay"]}}
