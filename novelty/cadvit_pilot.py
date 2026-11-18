"""Validation-only CAD-ViT novelty pilot runner. No work starts on import.

New runs are independent of historical reproduction and foundation experiments.
No test split can be instantiated by this module.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import PIL
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from novelty.cadvit_model import ARCH_CONFIG, NovelCADViT, NoveltyOutput, PRESETS, parameter_counts
from novelty.cadvit_training import (
    BOUNDARY_LOSS_WEIGHT, ORDINAL_LOSS_WEIGHT, file_sha256,
    research_loss, restricted_checkpoint_state, transfer_baseline_state,
)
from novelty.foundation_adaptation import load_train_val_manifest
from reproduction.cadvit import experiments
from reproduction.cadvit.model import CADViTCandidate
from reproduction.cadvit.evaluation import classification_metrics
from reproduction.cadvit.preprocessing import (
    BG_V2_CONFIG, PREPROCESS_CONFIG, preprocess_fundus_v2,
)


SCHEMA = "cadvit-novelty-pilot-v1"
REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "APTOS2019": {
        "relative": "experiment_archive_80ep_20261105/APTOS/best.pt",
        "sha256": "e72573f0f1c26baac53e18d950076845952da503261e212021ff6320c9ae72d5",
        "zero_based_epoch": 68,
    },
    "DDR": {
        "relative": "experiment_archive_80ep_20261105/DDR/best.pt",
        "sha256": "844665f0c597142915286b727a21f1d2223f43e0f057d083b6c1dd035c180ac1",
        "zero_based_epoch": 71,
    },
}
PRIMARY_ABLATIONS = (
    "baseline", "p1_only", "p2_only", "p3_probe_only", "p3_only",
    "p4_only", "p2_p3_probe", "p2_p3_full", "p1234",
)
SEED = 42
LEARNING_RATE = 0.000186
WEIGHT_DECAY = 0.0
EFFECTIVE_BATCH = 16
GRAD_CLIP = 1.0


def _json_normal(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_identity():
    def read(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args],
                                       text=True).strip()
    branch = read("branch", "--show-current")
    if branch != "preof-kc":
        raise ValueError("Pilot runner requires the preof-kc development branch")
    return {"branch": branch, "commit": read("rev-parse", "HEAD")}


def _source_hashes():
    files = (
        "novelty/cadvit_model.py", "novelty/cadvit_training.py",
        "novelty/cadvit_pilot.py", "novelty/foundation_adaptation.py",
        "reproduction/cadvit/model.py", "reproduction/cadvit/preprocessing.py",
        "reproduction/cadvit/experiments.py",
    )
    return {name: file_sha256(REPO / name) for name in files}


def _environment():
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "opencv": cv2.__version__,
        "pillow": PIL.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _make_model(ablation):
    if ablation not in PRIMARY_ABLATIONS:
        raise ValueError("Unknown primary ablation")
    return CADViTCandidate() if ablation == "baseline" else NovelCADViT(PRESETS[ablation])


def _forward_output(model, x):
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(x)
    return NoveltyOutput(model(x), (), None)


def _run_dir(artifact_root, output_root, dataset, ablation, micro_batch, epochs):
    if dataset not in CHECKPOINTS or ablation not in PRIMARY_ABLATIONS:
        raise ValueError("Dataset or primary ablation not recognized")
    if micro_batch < 1 or EFFECTIVE_BATCH % micro_batch:
        raise ValueError("Micro batch must divide the fixed effective batch 16")
    if not 1 <= epochs <= 5:
        raise ValueError("This Phase 2 runner permits only a 1-5 epoch pilot")
    artifact_root = Path(artifact_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    run_dir = output_root / dataset / f"{ablation}_seed42_e{epochs}_mb{micro_batch}_eff16_fp32"
    protected = (
        REPO, artifact_root / "datasets", artifact_root / "cad_vit_runs",
        artifact_root / "experiment_archive_80ep_20261105",
        artifact_root / "foundation_e4", artifact_root / "foundation_inputs",
    )
    for root in protected:
        root = root.resolve()
        if run_dir == root or root in run_dir.parents or run_dir in root.parents:
            raise ValueError(f"Run directory overlaps protected path: {root}")
    return run_dir


def _checkpoint_preflight(dataset, artifact_root, manifest_sha):
    spec = CHECKPOINTS[dataset]
    path = Path(artifact_root).expanduser().resolve() / spec["relative"]
    state, metadata = restricted_checkpoint_state(path, spec["sha256"])
    record = metadata["record"]
    if not isinstance(record, dict) or not isinstance(record.get("experiment"), dict):
        raise ValueError("Checkpoint record is missing")
    experiment = record["experiment"]
    expected_model = _json_normal(ARCH_CONFIG)
    expected_preprocessing = {
        "base": PREPROCESS_CONFIG, "ben_graham": BG_V2_CONFIG,
    }
    if (metadata["checkpoint_epoch"] != spec["zero_based_epoch"]
            or record.get("pipeline_version") != "V2_BG10_STABLE"
            or record.get("manifest_sha256") != manifest_sha
            or _json_normal(record.get("model")) != expected_model
            or record.get("preprocessing") != expected_preprocessing
            or experiment.get("dataset") != dataset
            or experiment.get("seed") != SEED
            or experiment.get("learning_rate") != LEARNING_RATE
            or experiment.get("physical_batch_size") != EFFECTIVE_BATCH):
        raise ValueError("Checkpoint protocol differs from pinned CAD-ViT V2")
    recovery = metadata["recovery"]
    if dataset == "DDR":
        if (not isinstance(recovery, dict)
                or recovery.get("loss") != "inverse_sqrt_frequency_weighted_cross_entropy"
                or recovery.get("manifest_sha256") != manifest_sha):
            raise ValueError("DDR mixed-protocol recovery metadata missing")
    elif recovery is not None:
        raise ValueError("Unexpected recovery metadata in APTOS checkpoint")
    baseline = CADViTCandidate()
    transfer = transfer_baseline_state(baseline, state)
    return state, {
        "path": str(path), "sha256": metadata["sha256"],
        "zero_based_epoch": metadata["checkpoint_epoch"],
        "source_protocol": "V2_BG10_STABLE",
        "mixed_protocol_recovery": bool(recovery),
        "recovery": recovery,
        "original_weight_transfer": transfer,
    }


def preflight(artifact_root, output_root, dataset, ablation, micro_batch=1, epochs=2):
    """Hash train/val source images and the exact checkpoint; never open test."""
    run_dir = _run_dir(artifact_root, output_root, dataset, ablation, micro_batch, epochs)
    rows, manifest = load_train_val_manifest(artifact_root, dataset, verify_images=True)
    if set(rows) != {"train", "val"} or not rows["train"] or not rows["val"]:
        raise ValueError("Expected nonempty train/validation only")
    state, checkpoint = _checkpoint_preflight(
        dataset, artifact_root, manifest["manifest_sha256"])
    manifest_path = Path(artifact_root).expanduser().resolve() / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    record = _json_normal({
        "schema": SCHEMA, "dataset": dataset, "ablation": ablation,
        "flags": PRESETS[ablation].to_dict(),
        "model_parameters": parameter_counts(_make_model(ablation)),
        "source": {"git": _git_identity(), "sha256": _source_hashes()},
        "manifest": {
            **manifest, "path": str(manifest_path),
            "sidecar_sha256": file_sha256(manifest_path.with_suffix(".json")),
        },
        "checkpoint": checkpoint,
        "preprocessing": {
            "name": "V2_BG10_STABLE",
            "base": PREPROCESS_CONFIG, "ben_graham": BG_V2_CONFIG,
            "train_augmentation": [
                "RandomHorizontalFlip(p=0.5)",
                "RandomVerticalFlip(p=0.5)",
                "RandomAffine(degrees=15,shear=10,bilinear,fill=0)",
            ],
            "validation_augmentation": "none",
            "tensor": "contiguous uint8 RGB CHW -> float32 / 255",
        },
        "optimization": {
            "seed": SEED, "epochs": epochs, "micro_batch": micro_batch,
            "effective_batch": EFFECTIVE_BATCH,
            "accumulation": EFFECTIVE_BATCH // micro_batch,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "precision": "fp32",
            "grad_clip_norm": GRAD_CLIP,
            "primary_loss": "unweighted five-class cross entropy",
            "boundary_loss_weight": BOUNDARY_LOSS_WEIGHT,
            "ordinal_loss_weight": ORDINAL_LOSS_WEIGHT,
            "checkpoint_selection": "validation macro-F1 desc, primary CE loss asc, earlier epoch",
        },
        "environment": _environment(),
        "run_dir": str(run_dir),
        "scope": "train/validation only; no test split access",
    })
    return record, rows, state, run_dir


class V2FundusSplit(Dataset):
    """Exact deterministic V2 BG10 preprocessing, then train-only augmentation."""

    def __init__(self, rows, split):
        if split not in ("train", "val"):
            raise ValueError("Only train and val splits are permitted")
        self.rows = rows[split]
        self.split = split
        self.augment = (transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, shear=10,
                                    interpolation=InterpolationMode.BILINEAR, fill=0),
        ]) if split == "train" else None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row["split"] != self.split:
            raise ValueError("Split row mismatch")
        with Image.open(row["path"]) as image:
            pixels = preprocess_fundus_v2(image, PREPROCESS_CONFIG, BG_V2_CONFIG)
        if self.augment is not None:
            pixels = np.asarray(self.augment(Image.fromarray(pixels)))
        tensor = torch.from_numpy(np.array(pixels, copy=True, order="C")).permute(2, 0, 1).float() / 255
        return tensor, row["label"]


def make_loader(rows, split, micro_batch, epoch=0, device="cpu"):
    if split not in ("train", "val"):
        raise ValueError("No test loader exists")
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(
        V2FundusSplit(rows, split), batch_size=micro_batch,
        shuffle=split == "train", generator=generator, num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )


def _ordinal_metrics(metrics):
    cm = metrics["confusion_matrix"]
    n = metrics["n"]
    actual = [sum(row) for row in cm]
    predicted = [sum(cm[i][j] for i in range(5)) for j in range(5)]
    observed = sum(cm[i][j] * (i-j)**2 for i in range(5) for j in range(5))
    expected = sum(actual[i] * predicted[j] * (i-j)**2
                   for i in range(5) for j in range(5)) / n
    metrics["qwk"] = 1 - observed / expected if expected else None
    metrics["grade_mae"] = sum(cm[i][j] * abs(i-j)
                               for i in range(5) for j in range(5)) / n
    return metrics


def _seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    kind, values, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((kind, np.array(values, dtype=np.uint32),
                         position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if state["cuda"] is None:
            raise ValueError("CUDA RNG state absent on CUDA resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_epoch(model, loader, optimizer, device, flags, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n = len(loader.dataset)
    truth, predicted = [], []
    total_sum = 0.0
    ce_sum = 0.0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        output = _forward_output(model, x)
        losses = research_loss(output, y, flags)
        window_start = (step // accumulation) * accumulation
        window_end = min(window_start + accumulation, len(loader))
        window_examples = min(n - window_start * loader.batch_size,
                              (window_end - window_start) * loader.batch_size)
        (losses["total"] * len(y) / window_examples).backward()
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError("Non-finite training loss")
        total_sum += float(losses["total"].detach()) * len(y)
        ce_sum += float(losses["cross_entropy"].detach()) * len(y)
        truth.extend(int(v) for v in y.cpu().tolist())
        predicted.extend(int(v) for v in output.logits.argmax(1).detach().cpu().tolist())
        if (step + 1) % accumulation and step + 1 != len(loader):
            continue
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Missing or non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if len(truth) != n:
        raise ValueError("Incomplete training epoch")
    metrics = classification_metrics(truth, predicted)
    metrics.update(experiments.support_weighted_metrics(metrics))
    metrics["loss"] = total_sum / n
    metrics["primary_ce_loss"] = ce_sum / n
    return _ordinal_metrics(metrics)


def _validate(model, loader, device):
    metrics = experiments.score(model, loader, device, "five_class")
    return _ordinal_metrics(metrics)


def _history(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _restore_run(run_dir, record, model, optimizer, device):
    if json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != record:
        raise ValueError("Run config changed; deterministic resume refused")
    history = _history(run_dir / "history.jsonl")
    state = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if (state.get("record") != record or best.get("record") != record
            or [item["epoch"] for item in history] != list(range(1, state["epoch"] + 2))
            or best["epoch"] + 1 != state["best_epoch"]
            or best["best_f1"] != state["best_f1"]
            or best["best_loss"] != state["best_loss"]):
        raise ValueError("Checkpoint/history/best provenance mismatch")
    selected = max(history, key=lambda item: (
        item["val"]["f1_macro"], -item["val"]["loss"], -item["epoch"]))
    if (selected["epoch"] != state["best_epoch"]
            or selected["val"]["f1_macro"] != state["best_f1"]
            or selected["val"]["loss"] != state["best_loss"]):
        raise ValueError("Saved best differs from validation-selection rule")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("last_epoch") != state["epoch"] + 1:
        raise ValueError("Summary and last checkpoint disagree")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"])
    return state["epoch"] + 1, state["best_f1"], state["best_loss"], state["best_epoch"]


def pilot(record, rows, baseline_state, run_dir, *, resume=False, device_name="cuda"):
    if device_name not in ("cuda", "cpu"):
        raise ValueError("Explicit device must be cuda or cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    device = torch.device(device_name)
    flags = PRESETS[record["ablation"]]
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    elif run_dir.exists():
        raise FileExistsError("Run already exists; select resume or a new protocol")
    _seed_all()
    model = _make_model(record["ablation"])
    transfer_baseline_state(model, baseline_state)
    model.to(device)
    settings = record["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    if resume:
        start, best_f1, best_loss, best_epoch = _restore_run(
            run_dir, record, model, optimizer, device)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(run_dir / "config.json", record)
        start, best_f1, best_loss, best_epoch = 0, -1.0, math.inf, None
    for epoch in range(start, settings["epochs"]):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        train_loader = make_loader(rows, "train", settings["micro_batch"], epoch, device)
        val_loader = make_loader(rows, "val", settings["micro_batch"], epoch, device)
        train_metrics = _train_epoch(model, train_loader, optimizer, device,
                                     flags, settings["accumulation"])
        train_seconds = time.perf_counter() - start_time
        val_metrics = _validate(model, val_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - start_time - train_seconds
        improved = (val_metrics["f1_macro"] > best_f1 or
                    (val_metrics["f1_macro"] == best_f1 and val_metrics["loss"] < best_loss))
        if improved:
            best_f1, best_loss, best_epoch = val_metrics["f1_macro"], val_metrics["loss"], epoch + 1
        entry = {
            "epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
            "train_seconds": train_seconds, "validation_seconds": val_seconds,
            "peak_gpu_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                             if device.type == "cuda" else None),
            "best_epoch": best_epoch, "is_best": improved,
        }
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "record": record, "rng": _rng_state(),
            "best_f1": best_f1, "best_loss": best_loss, "best_epoch": best_epoch,
        }
        experiments._atomic_save(run_dir / "last.pt", state)
        if improved:
            experiments._atomic_save(run_dir / "best.pt", state)
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_json(run_dir / "summary.json", {
            "schema": SCHEMA, "dataset": record["dataset"], "ablation": record["ablation"],
            "last_epoch": epoch + 1, "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1, "best_validation_primary_ce_loss": best_loss,
            "selection": settings["checkpoint_selection"],
        })
        print(json.dumps({"run_dir": str(run_dir), **entry}, sort_keys=True), flush=True)
    return run_dir


def synthetic_smoke(record, baseline_state, device_name="cuda"):
    if device_name not in ("cuda", "cpu"):
        raise ValueError("Explicit device must be cuda or cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed_all()
    device = torch.device(device_name)
    model = _make_model(record["ablation"])
    transfer_baseline_state(model, baseline_state)
    model.to(device).train()
    x = torch.randn(1, 3, PREPROCESS_CONFIG["image_size"],
                    PREPROCESS_CONFIG["image_size"], device=device)
    labels = torch.tensor([2], device=device)
    output = _forward_output(model, x)
    loss = research_loss(output, labels, PRESETS[record["ablation"]])["total"]
    loss.backward()
    if not bool(torch.isfinite(loss)) or not all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters()
            if p.grad is not None):
        raise FloatingPointError("Synthetic smoke produced non-finite loss/gradient")
    return {"status": "PASS", "synthetic_only": True,
            "input_shape": list(x.shape), "logit_shape": list(output.logits.shape),
            "peak_gpu_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                             if device.type == "cuda" else None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "pilot", "resume"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--ablation", choices=PRIMARY_ABLATIONS, required=True)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    record, rows, state, run_dir = preflight(
        args.artifact_root, args.output_root, args.dataset,
        args.ablation, args.micro_batch, args.epochs)
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS", "run_dir": str(run_dir),
                          "record": record}, indent=2, sort_keys=True), flush=True)
    elif args.mode == "smoke":
        print(json.dumps(synthetic_smoke(record, state, args.device), indent=2), flush=True)
    else:
        pilot(record, rows, state, run_dir,
              resume=args.mode == "resume", device_name=args.device)


if __name__ == "__main__":
    main()
