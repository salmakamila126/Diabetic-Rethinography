"""E7 global/local ConvNeXt experiment. Frozen train/validation only."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import time
import uuid

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from novelty import foundation_adaptation as fa
from novelty import foundation_reporting as reporting
from novelty import preof_phase1 as phase
from novelty_integrated.e5 import config as e5config, data as e5data
from novelty_integrated.e5.objective import loss_components
from novelty_integrated.e7 import cache, config, geometry, inference, input_pipeline
from novelty_integrated.e7.model import GlobalLocalModel, ModelEMA


def _hashes():
    root = Path(__file__).resolve().parents[2]
    names = ("novelty/foundation_adaptation.py", "novelty/foundation_augmentation.py",
             "novelty_integrated/e5/config.py", "novelty_integrated/e5/model.py",
             "novelty/preof_phase1.py", "novelty/preof_e1.py", "novelty/foundation_reporting.py",
             "novelty_integrated/e5/objective.py", "novelty_integrated/e5/data.py",
             "novelty_integrated/e5/input_pipeline.py", "novelty_integrated/e7/inference.py",
             "novelty_integrated/e6/decoded_cache.py",
             "novelty_integrated/e7/config.py", "novelty_integrated/e7/geometry.py",
             "novelty_integrated/e7/model.py", "novelty_integrated/e7/cache.py",
             "novelty_integrated/e7/input_pipeline.py", "novelty_integrated/e7/runner.py")
    return {name: phase.sha256(root / name) for name in names}


def _safe_path(path, args):
    return fa._safe_out(path, [Path(args.artifact_root) / "datasets",
        Path(args.artifact_root) / "cad_vit_runs" / "manifests", args.model_dir])


def _e7_path(path, args):
    candidate = _safe_path(path, args)
    root = Path(args.artifact_root).expanduser().resolve() / "foundation_e7"
    if root not in candidate.parents:
        raise ValueError("E7 writes require a dedicated path below foundation_e7")
    return candidate


def _data(args, *, verify_images):
    if args.seed != config.SEED:
        raise ValueError("E7 seed is frozen at 42")
    rows, manifest = e5data.load_train_val(args.artifact_root, args.dataset,
        manifest_source_root=args.manifest_source_root, verify_images=verify_images)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 config.OFFICIAL_SNAPSHOT["revision"],
                                 config.OFFICIAL_SNAPSHOT["weight_sha256"])
    e5config.require_official_snapshot(snapshot)
    processor = fa.load_local_processor(snapshot)
    return rows, manifest, snapshot, processor


def _cache_paths(args):
    for path in (args.train_cache_root, args.val_cache_root):
        cache.reject_symlinks(path)
    train = (_safe_path if args.reuse_e6_train else _e7_path)(args.train_cache_root, args)
    val = _e7_path(args.val_cache_root, args)
    if train == val or train in val.parents or val in train.parents:
        raise ValueError("Train/validation RGB caches must be separate")
    return train, val


def estimate_cache(args):
    rows, manifest, snapshot, _ = _data(args, verify_images=False)
    train, val = _cache_paths(args)
    def size(part):
        result = 0
        for row in part:
            with Image.open(row["path"]) as image:
                result += image.width * image.height * 3 + 4096
        return result
    def free(path):
        parent = path
        while not parent.exists():
            parent = parent.parent
        return shutil.disk_usage(parent).free
    print(json.dumps({"status": "READ_ONLY_ESTIMATE_UNVERIFIED_IMAGES",
        "dataset": args.dataset, "manifest_sha256": manifest["manifest_sha256"],
        "train_rows": len(rows["train"]), "val_rows": len(rows["val"]),
        "train_decoded_bytes": size(rows["train"]), "val_decoded_bytes": size(rows["val"]),
        "train_cache_free_bytes": free(train), "val_cache_free_bytes": free(val),
        "e6_train_cache_exists": (train / "summary.json").is_file(),
        "final_test_status": "NOT_ACCESSED"}, indent=2))


def prepare_cache(args):
    rows, manifest, snapshot, _ = _data(args, verify_images=False)
    train_root, val_root = _cache_paths(args)
    if args.reuse_e6_train:
        if not (train_root / "summary.json").is_file():
            raise ValueError("Requested E6 train cache is absent")
        train = cache.open_train(train_root, manifest, rows["train"], snapshot)
        train_report = {"status": "REUSED_READ_ONLY", "metadata_sha256": train.meta_sha256,
                        "row_count": len(rows["train"])}
    else:
        train_report = cache.build_train(train_root, manifest, rows["train"], snapshot,
            max_gib=args.train_cache_max_gib, reserve_gib=args.reserve_gib)
    val_meta = cache.val_metadata(manifest, rows["val"], snapshot)
    val_report = cache.build_val(val_root, val_meta, rows["val"],
        max_gib=args.val_cache_max_gib, reserve_gib=args.reserve_gib)
    print(json.dumps({"status": "CACHE_PREPARED_CPU_ONLY", "train": train_report,
                      "validation": val_report, "final_test_status": "NOT_ACCESSED"}, indent=2))


def _caches(args, rows, manifest, snapshot):
    train_root, val_root = _cache_paths(args)
    train = cache.open_train(train_root, manifest, rows["train"], snapshot)
    val = cache.open_val(val_root, cache.val_metadata(manifest, rows["val"], snapshot), rows["val"])
    return train, val


def verify_inputs(args):
    rows, manifest, snapshot, processor = _data(args, verify_images=True)
    train, val = _caches(args, rows, manifest, snapshot)
    if not 1 <= args.sample_per_grade <= 5:
        raise ValueError("Choose 1..5 samples per grade")
    checked = {}
    changed_between_epochs = 0
    for split, data_cache in (("train", train), ("val", val)):
        examples = [r for grade in range(5) for r in
                    [x for x in rows[split] if x["label"] == grade][:args.sample_per_grade]]
        if len(examples) != 5 * args.sample_per_grade:
            raise ValueError("A grade lacks input verification samples")
        for row in examples:
            with Image.open(row["path"]) as source:
                direct = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            cached = np.asarray(data_cache.read(row))
            if not np.array_equal(direct, cached):
                raise ValueError("Cached original RGB differs from verified source")
        previous = {}
        for epoch in ((1, 2) if split == "train" else (None,)):
            collate = input_pipeline.ViewCollator(processor, data_cache,
                training=split == "train", seed=args.seed, epoch=SimpleNamespace(value=epoch))
            for start in range(0, len(examples), args.micro_batch):
                batch = examples[start:start + args.micro_batch]
                g, p, labels, ids = collate(batch)
                if ids != [row["image_id"] for row in batch] or labels.tolist() != [row["label"] for row in batch]:
                    raise ValueError("E7 cache changes label or sample order")
                for index, row in enumerate(batch):
                    identity = row["image_id"]
                    with Image.open(row["path"]) as opened:
                        source = ImageOps.exif_transpose(opened).convert("RGB")
                    direct_g, direct_p, boxes = geometry.model_views(source, processor,
                        training=split == "train", seed=args.seed, epoch=epoch, image_id=identity)
                    if any(not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height)
                           for x0, y0, x1, y1 in boxes):
                        raise ValueError("E7 crop exceeds original image")
                    if not torch.equal(g[index], direct_g) or not torch.equal(p[index], direct_p):
                        raise ValueError("E7 cache collator changes global or patch tensors")
                    if epoch == 1:
                        previous[identity] = fa._digest_json([
                            hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                            for v in (direct_g, direct_p)])
                    elif epoch == 2:
                        digest = fa._digest_json([
                            hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                            for v in (direct_g, direct_p)])
                        changed_between_epochs += int(digest != previous[identity])
        checked[split] = len(examples)
    report = {"schema": config.SCHEMA, "status": "ACTUAL_INPUT_EQUIVALENCE_PASS",
        "dataset": args.dataset, "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": _hashes(), "train_cache_metadata_sha256": train.meta_file_sha256,
        "val_cache_metadata_sha256": val.meta_file_sha256,
        "samples_per_split": checked, "epochs_checked": [1,2],
        "train_samples_changed_between_epochs": changed_between_epochs,
        "equality": "source RGB and official normalized global/patch tensors exactly equal",
        "final_test_status": "NOT_ACCESSED"}
    out = _e7_path(args.output_dir, args)
    _separate_output(out, train, val)
    if out.exists():
        raise FileExistsError("E7 input preflight already exists")
    out.mkdir(parents=True)
    fa._atomic_json(out / "input_equivalence.json", report)
    print(json.dumps(report, indent=2))

def _record(args, manifest, snapshot, processor, train, val, counts, init_identity, precision):
    import transformers, PIL
    git = subprocess.run(["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=False)
    return {"schema": config.SCHEMA, "status": "EXPLORATORY_VALIDATION_ONLY",
        "dataset": args.dataset, "manifest": manifest, "class_order": list(inference.CLASS_ORDER),
        "snapshot": {k:snapshot[k] for k in ("hub_id","revision","config_sha256",
            "processor_config_sha256","weight_file","weight_sha256")},
        "effective_processor": processor.to_dict(),
        "effective_processor_sha256": fa._digest_json(processor.to_dict()),
        "init": init_identity, "train_counts": counts,
        "geometry": {"global_size": 384, "patch_size": 224, "patch_count": 2,
                     "patch_width_and_height_fraction": config.PATCH_WIDTH_FRACTION,
                     "fov_threshold": config.FOV_THRESHOLD,
                     "fov_margin_fraction": config.FOV_MARGIN_FRACTION,
                     "global": "full original RGB letterboxed; no center crop",
                     "local": "geometry-only central-height FOV left/right crops, then letterbox"},
        "augmentation": fa._augmentation_metadata("B1"),
        "objective": "E5 integrated: categorical CE + 0.25 ordinal BCE; train-only inverse-sqrt weights epochs 11..20",
        "seed": args.seed, "epochs": config.EPOCHS,
        "micro_batch": args.micro_batch, "accumulation": args.accumulation,
        "effective_batch": config.EFFECTIVE_BATCH,
        "optimizer": "AdamW", "backbone_lr": config.BACKBONE_LR,
        "head_lr": config.HEAD_LR, "weight_decay": config.WEIGHT_DECAY,
        "max_grad_norm": config.MAX_GRAD_NORM, "scheduler": None,
        "ema_decay": args.ema_decay, "selection_weights": args.selection_weights,
        "checkpoint_selection": "max validation macro-F1; tie lower raw categorical CE; tie earliest epoch",
        "train_cache_root": str(train.root), "train_cache_metadata_sha256": train.meta_file_sha256,
        "val_cache_root": str(val.root), "val_cache_metadata_sha256": val.meta_file_sha256,
        "workers": args.num_workers, "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor, "pin_memory": args.pin_memory,
        "precision": precision, "device_type": torch.device(args.device).type,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": PIL.__version__, "numpy": np.__version__,
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": _hashes(), "final_test_status": "NOT_ACCESSED"}


def _load_e5_initialization(args, model, manifest, counts):
    if args.init == "official":
        if args.e5_checkpoint or args.e5_checkpoint_sha256:
            raise ValueError("Official init cannot also name an E5 checkpoint")
        return {"mode": "OFFICIAL_PRETRAINED", "weight_sha256": config.OFFICIAL_SNAPSHOT["weight_sha256"]}
    if not args.e5_checkpoint or not args.e5_checkpoint_sha256:
        raise ValueError("E5 init requires immutable best checkpoint and explicit SHA-256")
    path = _safe_path(args.e5_checkpoint, args)
    if path.name != "best.pt":
        raise ValueError("E5 initialization requires selected best.pt")
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "VALIDATION_COMPLETE" or summary.get("epochs_completed") != 20:
        raise ValueError("E5 checkpoint run is not complete")
    phase.require_hash(path, args.e5_checkpoint_sha256)
    state = fa._load_checkpoint(path, trust=args.trust_e5_checkpoint)
    source = state.get("record", {})
    if (source.get("schema") != e5config.SCHEMA
            or source.get("dataset") != args.dataset or source.get("arm") != "integrated"
            or source.get("manifest", {}).get("manifest_sha256") != manifest["manifest_sha256"]
            or source.get("snapshot") != {k: config.OFFICIAL_SNAPSHOT[k]
                                          for k in config.OFFICIAL_SNAPSHOT}
            or source.get("train_counts") != counts
            or state.get("epoch") != summary.get("best", {}).get("epoch")
            or state.get("best") != summary.get("best")):
        raise ValueError("E5 best checkpoint protocol, labels or lineage differs")
    model.backbone.load_state_dict(state["backbone"], strict=True)
    model.heads.load_state_dict(state["heads"], strict=True)
    return {"mode": "E5_BEST_WARMSTART", "checkpoint_sha256": args.e5_checkpoint_sha256,
            "checkpoint_epoch": state["epoch"], "summary_sha256": phase.sha256(summary_path),
            "source_git_head": source.get("source_git_head")}


def _prepare(args):
    config.validate_batch(args.micro_batch, args.accumulation)
    if args.seed != config.SEED or args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("E7 seed/worker configuration invalid")
    if args.persistent_workers and not args.num_workers:
        raise ValueError("Persistent workers require num_workers > 0")
    if args.selection_weights not in ("raw", "ema") or (args.selection_weights == "ema" and args.ema_decay is None):
        raise ValueError("EMA selection requires a prespecified decay")
    rows, manifest, snapshot, processor = _data(args, verify_images=True)
    train_cache, val_cache = _caches(args, rows, manifest, snapshot)
    counts = [sum(row["label"] == grade for row in rows["train"]) for grade in range(5)]
    if min(counts) < 1:
        raise ValueError("Frozen train split lacks a grade")
    out = _e7_path(args.output_dir, args)
    _separate_output(out, train_cache, val_cache)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Selected CUDA device unavailable")
    precision = fa._precision(device, args.precision)
    fa._seed(args.seed)
    backbone, loaded_processor = fa.load_local_backbone(snapshot,
        trust_legacy_bin=args.trust_legacy_bin)
    if fa._digest_json(processor.to_dict()) != fa._digest_json(loaded_processor.to_dict()):
        raise ValueError("Official processor changed during model load")
    fa.configure_trainable(backbone, "convnext_tiny", "full")
    model = GlobalLocalModel(backbone, counts)
    init = _load_e5_initialization(args, model, manifest, counts)
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": config.BACKBONE_LR},
        {"params": list(model.heads.parameters()) + list(model.gate.parameters()),
         "lr": config.HEAD_LR}], weight_decay=config.WEIGHT_DECAY)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay is not None else None
    record = _record(args, manifest, snapshot, processor, train_cache, val_cache,
                     counts, init, precision)
    if args.command in ("smoke","benchmark","train"):
        if not args.input_preflight_report:
            raise ValueError("Actual E7 input equivalence report required")
        path = _safe_path(args.input_preflight_report, args)
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "ACTUAL_INPUT_EQUIVALENCE_PASS"
                or report.get("dataset") != args.dataset
                or report.get("manifest_sha256") != manifest["manifest_sha256"]
                or report.get("source_sha256") != _hashes()
                or report.get("train_cache_metadata_sha256") != train_cache.meta_file_sha256
                or report.get("val_cache_metadata_sha256") != val_cache.meta_file_sha256):
            raise ValueError("E7 input preflight differs")
        record["input_preflight_sha256"] = phase.sha256(path)
    return rows, model, processor, optimizer, ema, record, out, device, precision, train_cache, val_cache


def _separate_output(out, *caches):
    for item in caches:
        root = item.root
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("E7 output overlaps a cache")

def _loaders(args, rows, processor, train_cache, val_cache):
    return input_pipeline.InputLoaders(rows, processor, train_cache, val_cache,
        seed=args.seed, micro_batch=args.micro_batch,
        num_workers=args.num_workers, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and torch.device(args.device).type == "cuda")


def _check_gradients(model, optimizer):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError("E7 parameter has missing or nonfinite gradient")
    torch.nn.utils.clip_grad_norm_(parameters, config.MAX_GRAD_NORM, error_if_nonfinite=True)


def _train_batches(model, optimizer, ema, batches, *, device, precision, counts, epoch,
                   total_images, micro_batch, accumulation, progress=None, non_blocking=False):
    model.train(); optimizer.zero_grad(set_to_none=True)
    num_batches = math.ceil(total_images / micro_batch)
    truths, probabilities, windows = [], [], []
    total_loss = cat_loss = ord_loss = 0.0
    seen = steps = 0
    weights = e5config.weights_for_epoch(counts, "integrated", epoch)
    for index, (global_pixels, patches, labels, _) in enumerate(batches):
        expected = min(micro_batch, total_images - index * micro_batch)
        if index >= num_batches or len(labels) != expected:
            raise ValueError("E7 training batch count differs from frozen split")
        global_pixels = global_pixels.to(device, non_blocking=non_blocking)
        patches = patches.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        start = (index // accumulation) * accumulation * micro_batch
        window_size = min(accumulation * micro_batch, total_images - start)
        with fa._autocast(device, precision):
            logits, ordinal, _ = model(global_pixels, patches)
        losses = loss_components(logits, ordinal, labels, class_weights=weights)
        (losses["total"] * len(labels) / window_size).backward()
        total_loss += float(losses["total"].detach()) * len(labels)
        cat_loss += float(losses["categorical"].detach()) * len(labels)
        ord_loss += float(losses["ordinal"].detach()) * len(labels)
        seen += len(labels)
        truths.append(labels.detach().cpu())
        probabilities.append(logits.detach().float().softmax(1).cpu())
        if progress is not None:
            progress.update(index + 1, seen, total_loss / seen)
        if (index + 1) % accumulation == 0 or index + 1 == num_batches:
            _check_gradients(model, optimizer)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            steps += 1; windows.append(window_size)
    if seen != total_images or sum(windows) != total_images:
        raise ValueError("E7 training iterator did not cover all frozen rows")
    metrics = fa._classification(torch.cat(truths).numpy(), torch.cat(probabilities).numpy())
    return {"loss": total_loss / seen, "categorical_loss": cat_loss / seen,
            "ordinal_loss": ord_loss / seen, "images": seen, "optimizer_steps": steps,
            "accumulation_window_images": windows, **metrics}


def _evaluate(model, batches, *, device, precision, total_images, progress=None,
              non_blocking=False):
    model.eval()
    truths, probabilities = [], []
    loss_sum = count = 0
    with torch.inference_mode():
        for index, (global_pixels, patches, labels, _) in enumerate(batches):
            global_pixels = global_pixels.to(device, non_blocking=non_blocking)
            patches = patches.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            with fa._autocast(device, precision):
                logits, _, _ = model(global_pixels, patches)
            loss = F.cross_entropy(logits.float(), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite E7 validation loss")
            loss_sum += float(loss) * len(labels)
            count += len(labels)
            truths.append(labels.cpu())
            probabilities.append(logits.float().softmax(1).cpu())
            if progress is not None:
                progress.update(index + 1, count, loss_sum / count)
    if count != total_images:
        raise ValueError("E7 validation iterator did not cover frozen split")
    metrics = fa._metrics(torch.cat(truths).numpy(), torch.cat(probabilities).numpy(),
                          loss_sum / count)
    _add_macro_metrics(metrics)
    cm = metrics["confusion_matrix"]
    metrics["grade1_transitions"] = {"correct": cm[1][1], "support": sum(cm[1]),
        "to_grade0": cm[1][0], "to_grades2_4": sum(cm[1][2:])}
    metrics["grade3_transitions"] = {"correct": cm[3][3], "support": sum(cm[3]),
        "to_grades0_2": sum(cm[3][:3]), "to_grade4": cm[3][4]}
    return metrics


def _add_macro_metrics(metrics):
    for name in ("precision", "recall"):
        metrics[f"macro_{name}"] = sum(grade[name] for grade in metrics["per_grade"]) / 5


def _write_history_csv(path, history):
    fields = ("epoch","train_loss","train_macro_f1","validation_ce","validation_accuracy",
              "validation_macro_f1","validation_weighted_f1","validation_qwk",
              "validation_grade_mae","grade1_recall","grade3_recall",
              "train_seconds","validation_seconds","best_epoch")
    temp = Path(path).with_name(Path(path).name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in history:
                tr, val, resource = row["train"], row["validation"], row["resource"]
                writer.writerow({"epoch":row["epoch"], "train_loss":tr["loss"],
                    "train_macro_f1":tr["macro_f1"], "validation_ce":val["validation_loss"],
                    "validation_accuracy":val["accuracy"], "validation_macro_f1":val["macro_f1"],
                    "validation_weighted_f1":val["weighted_f1"],
                    "validation_qwk":val["quadratic_weighted_kappa"],
                    "validation_grade_mae":val["grade_mae"],
                    "grade1_recall":val["per_grade"][1]["recall"] if isinstance(val["per_grade"],list) else val["per_grade"]["1"]["recall"],
                    "grade3_recall":val["per_grade"][3]["recall"] if isinstance(val["per_grade"],list) else val["per_grade"]["3"]["recall"],
                    "train_seconds":resource["train_seconds"],
                    "validation_seconds":resource["validation_seconds"],
                    "best_epoch":row["best_epoch"]})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def _check_report(path, status, record, args):
    file = _safe_path(path, args)
    report = json.loads(file.read_text(encoding="utf-8"))
    if report.get("status") != status or report.get("record") != record:
        raise ValueError(f"E7 {status} report differs from current protocol")
    return phase.sha256(file), report


def smoke(args):
    rows, model, processor, optimizer, ema, record, out, device, precision, tr_cache, va_cache = _prepare(args)
    if out.exists():
        raise FileExistsError("E7 smoke output already exists")
    if device.type != "cuda" or not args.acknowledge_shared_gpu:
        raise ValueError("E7 GPU smoke requires explicit resource acknowledgement")
    loaders = _loaders(args, rows, processor, tr_cache, va_cache)
    batch = next(loaders.train_epoch(1))
    global_pixels, patches, labels, _ = batch
    global_pixels = global_pixels.to(device); patches = patches.to(device); labels = labels.to(device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with fa._autocast(device, precision):
        logits, ordinal, gate = model(global_pixels, patches)
    loss = loss_components(logits, ordinal, labels,
        class_weights=e5config.weights_for_epoch(record["train_counts"], "integrated", 1))["total"]
    loss.backward(); _check_gradients(model, optimizer); optimizer.step()
    if ema is not None: ema.update(model)
    torch.cuda.synchronize(device)
    report = {"status": "E7_SMOKE_PASS", "record": record,
        "input_global_shape": list(global_pixels.shape), "input_patch_shape": list(patches.shape),
        "logits_shape": list(logits.shape), "ordinal_shape": list(ordinal.shape),
        "gate_shape": list(gate.shape), "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "model_weights_saved": False, "final_test_status": "NOT_ACCESSED"}
    out.mkdir(parents=True); fa._atomic_json(out / "smoke.json", report)
    print(json.dumps({"status":report["status"],"peak_gpu_reserved_bytes":report["peak_gpu_reserved_bytes"]},indent=2))


def benchmark(args):
    import itertools
    rows, model, processor, optimizer, ema, record, out, device, precision, tr_cache, va_cache = _prepare(args)
    if out.exists(): raise FileExistsError("E7 benchmark output already exists")
    if device.type != "cuda" or not args.acknowledge_shared_gpu:
        raise ValueError("E7 benchmark requires explicit GPU resource acknowledgement")
    smoke_hash, smoke_report = _check_report(args.smoke_report, "E7_SMOKE_PASS", record, args)
    if not 4 <= args.bench_batches <= 16:
        raise ValueError("Choose bounded 4..16 benchmark batches")
    loaders = _loaders(args, rows, processor, tr_cache, va_cache)
    train_images = args.bench_batches * args.micro_batch
    if train_images > len(rows["train"]): raise ValueError("Benchmark exceeds train split")
    free_before, total_gpu = torch.cuda.mem_get_info(device)
    cache_bytes = 0
    cache_started = time.perf_counter()
    for row in rows["train"][:min(16, train_images)]:
        pixels = tr_cache.read(row)
        copied = pixels.copy()  # Fault pages in, matching the collator's materialization.
        cache_bytes += copied.nbytes
        del copied, pixels
    cache_seconds = time.perf_counter() - cache_started
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    _train_batches(model, optimizer, ema, itertools.islice(loaders.train_epoch(1), args.bench_batches),
        device=device, precision=precision, counts=record["train_counts"], epoch=1,
        total_images=train_images, micro_batch=args.micro_batch, accumulation=args.accumulation,
        non_blocking=args.pin_memory)
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    val_batches = min(args.bench_batches, math.ceil(len(rows["val"]) / args.micro_batch))
    val_images = min(val_batches * args.micro_batch, len(rows["val"]))
    started = time.perf_counter()
    _evaluate(model, itertools.islice(loaders.validation(), val_batches), device=device,
              precision=precision, total_images=val_images, non_blocking=args.pin_memory)
    torch.cuda.synchronize(device)
    val_seconds = time.perf_counter() - started
    epoch_estimate = (len(rows["train"]) / train_images * train_seconds
                      + len(rows["val"]) / val_images * val_seconds)
    report = {"status":"E7_BOUNDED_BENCHMARK", "record":record,
        "smoke_report_sha256":smoke_hash, "train_images":train_images,"val_images":val_images,
        "train_seconds":train_seconds,"validation_seconds":val_seconds,
        "train_images_per_second": train_images / train_seconds,
        "validation_images_per_second": val_images / val_seconds,
        "cache_read": {"bytes": cache_bytes, "seconds": cache_seconds,
            "mib_per_second": cache_bytes / (1024**2 * cache_seconds),
            "scope": "bounded serial mmap copy after integrity scan; OS cache state uncontrolled"},
        "estimated_epoch_seconds":epoch_estimate,
        "estimated_20_epoch_seconds":config.EPOCHS*epoch_estimate,
        "estimate_limitations":"short subset; excludes checkpoint I/O and shared load",
        "gpu_free_before_bytes":free_before,"gpu_total_bytes":total_gpu,
        "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "parent_peak_rss_bytes":fa._process_peak_rss_bytes(),
        "final_test_status":"NOT_ACCESSED"}
    out.mkdir(parents=True); fa._atomic_json(out / "benchmark.json", report)
    print(json.dumps({"status":report["status"],
                      "estimated_20_epoch_seconds":report["estimated_20_epoch_seconds"]},indent=2))


def _resume(out, record, model, optimizer, ema, *, trust):
    if json.loads((out / "run.json").read_text(encoding="utf-8")) != record:
        raise ValueError("E7 resume run metadata differs")
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != record:
        raise ValueError("E7 resume configuration differs")
    last = fa._load_checkpoint(out / "last.pt", trust=trust)
    history = last.get("history")
    best = last.get("best")
    if (last.get("record") != record or not isinstance(history,list)
            or not 1 <= len(history) <= config.EPOCHS
            or last.get("epoch") != len(history)
            or [row.get("epoch") for row in history] != list(range(1,len(history)+1))
            or not isinstance(best,dict) or not 1 <= best.get("epoch",0) <= len(history)):
        raise ValueError("E7 last checkpoint/history is inconsistent")
    history_file = out / "history.json"
    published = json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else []
    if not isinstance(published,list) or published != history[:len(published)]:
        raise ValueError("E7 published history diverges from last valid checkpoint")
    best_file = out / "best.pt"
    if not best_file.exists() and best["epoch"] == last["epoch"]:
        # Crash between atomic last/best writes on an improved epoch.
        fa._save_checkpoint(best_file, last)
    selected = fa._load_checkpoint(best_file, trust=trust)
    if (selected.get("record") == record and best["epoch"] == last["epoch"]
            and isinstance(selected.get("epoch"), int)
            and selected["epoch"] < last["epoch"]
            and isinstance(selected.get("best"), dict)
            and selected["best"].get("epoch") == selected["epoch"]):
        fa._save_checkpoint(best_file, last)
        selected = fa._load_checkpoint(best_file, trust=trust)
    if selected.get("record") != record or selected.get("epoch") != best["epoch"] or selected.get("best") != best:
        raise ValueError("E7 best checkpoint differs from selected epoch")
    if published != history:
        fa._atomic_json(history_file, history)
        _write_history_csv(out / "history.csv", history)
    model.load_state_dict(last["model"], strict=True)
    optimizer.load_state_dict(last["optimizer"])
    if ema is not None:
        if last.get("ema") is None: raise ValueError("EMA state missing")
        ema.load_state_dict(last["ema"])
    elif last.get("ema") is not None:
        raise ValueError("Unexpected EMA state")
    fa._restore_rng(last["rng"])
    return history, best, len(history)+1


@contextmanager
def _run_lock(out):
    lock = out / "training.lock"
    with lock.open("x", encoding="utf-8") as stream:
        try:
            stream.write(str(os.getpid()) + "\n")
            stream.flush()
            yield
        finally:
            stream.close()
            lock.unlink()


def train(args):
    if not args.acknowledge_shared_gpu or torch.device(args.device).type != "cuda":
        raise ValueError("E7 training requires an explicitly approved CUDA allocation")
    out = _e7_path(args.output_dir, args)
    for root in _cache_paths(args):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("E7 output overlaps a cache")
    if args.resume:
        if not out.is_dir():
            raise FileNotFoundError("E7 resume directory absent")
    else:
        if out.exists():
            raise FileExistsError("E7 run output exists; never overwrite it")
        # Complete read-only gates before reserving a new run directory.
    prepared = _prepare(args)
    if not args.resume:
        # mkdir is exclusive across competing launches.
        out.mkdir(parents=True)
    with _run_lock(out):
        _train_run(args, prepared)


def _train_run(args, prepared):
    rows, model, processor, optimizer, ema, record, out, device, precision, train_cache, val_cache = prepared
    if device.type != "cuda" or not args.acknowledge_shared_gpu:
        raise ValueError("E7 training requires an explicitly approved CUDA allocation")
    smoke_hash, _ = _check_report(args.smoke_report, "E7_SMOKE_PASS", record, args)
    bench_hash, bench = _check_report(args.benchmark_report, "E7_BOUNDED_BENCHMARK", record, args)
    if bench["smoke_report_sha256"] != smoke_hash:
        raise ValueError("E7 benchmark and smoke provenance differ")
    if args.resume:
        if not out.is_dir():
            raise FileNotFoundError("E7 resume directory absent")
        if (out / "summary.json").exists():
            raise FileExistsError("E7 run is already complete; do not overwrite it")
        history, best, first_epoch = _resume(out, record, model, optimizer, ema,
                                              trust=args.trust_run_checkpoint)
    else:
        fa._atomic_json(out / "run.json", record)
        fa._atomic_json(out / "config.json", record)
        history, best, first_epoch = [], None, 1
    loaders = _loaders(args, rows, processor, train_cache, val_cache)
    logger = reporting.RunLogger(out)
    try:
        logger.line(f"E7 dataset={args.dataset} start_epoch={first_epoch} init={record['init']['mode']}")
        for epoch in range(first_epoch, config.EPOCHS + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            progress = reporting.BatchProgress(logger, "train", epoch,
                math.ceil(len(rows["train"]) / args.micro_batch), len(rows["train"]))
            try:
                train_info = _train_batches(model, optimizer, ema, loaders.train_epoch(epoch),
                    device=device, precision=precision, counts=record["train_counts"],
                    epoch=epoch, total_images=len(rows["train"]), micro_batch=args.micro_batch,
                    accumulation=args.accumulation, progress=progress,
                    non_blocking=args.pin_memory)
            finally:
                progress.close()
            torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            selected = ema.shadow if args.selection_weights == "ema" else model
            progress = reporting.BatchProgress(logger, "validation", epoch,
                math.ceil(len(rows["val"]) / args.micro_batch), len(rows["val"]))
            try:
                val_info = _evaluate(selected, loaders.validation(), device=device,
                    precision=precision, total_images=len(rows["val"]), progress=progress,
                    non_blocking=args.pin_memory)
            finally:
                progress.close()
            torch.cuda.synchronize(device)
            val_seconds = time.perf_counter() - started
            thresholds = model.heads.ordinal.thresholds().detach()
            if not bool(torch.isfinite(thresholds).all() and (thresholds.diff() > 0).all()):
                raise ValueError("E7 ordinal thresholds must be finite and strictly ordered")
            improved = fa._best_is_new(val_info, best["validation"] if best else None)
            if improved:
                best = {"epoch": epoch, "validation": val_info}
            resource = {"train_seconds": train_seconds, "validation_seconds": val_seconds,
                "train_images_per_second": len(rows["train"]) / train_seconds,
                "validation_images_per_second": len(rows["val"]) / val_seconds,
                "process_peak_rss_bytes": fa._process_peak_rss_bytes(),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            current = {"epoch": epoch, "train": train_info, "validation": val_info,
                "resource": resource, "best_epoch": best["epoch"],
                "best_updated": improved,
                "class_weight_active": epoch >= config.WEIGHT_START_EPOCH,
                "learning_rates": {"backbone": config.BACKBONE_LR, "head": config.HEAD_LR}}
            history.append(current)
            state = {"record": record, "epoch": epoch, "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(), "rng": fa._rng_state(),
                "history": history, "best": best, "scheduler": None,
                "scaler": None, "selection_weights": args.selection_weights}
            fa._save_checkpoint(out / "last.pt", state)
            if improved:
                fa._save_checkpoint(out / "best.pt", state)
            fa._atomic_json(out / "history.json", history)
            _write_history_csv(out / "history.csv", history)
            for line in reporting.epoch_lines(current, best, improved, current["learning_rates"]):
                logger.line(line)
        if len(history) == config.EPOCHS:
            fa._atomic_json(out / "summary.json", {"status": "VALIDATION_COMPLETE",
                "final_test_status": "NOT_ACCESSED", "record": record, "best": best,
                "epochs_completed": len(history), "smoke_report_sha256": smoke_hash,
                "benchmark_report_sha256": bench_hash,
                "parameter_counts": {"total": sum(p.numel() for p in model.parameters()),
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}})
    finally:
        logger.close()


def evaluate(args):
    """Re-evaluate the selected E7 checkpoint on frozen validation only."""
    if torch.device(args.device).type == "cuda" and not args.acknowledge_shared_gpu:
        raise ValueError("E7 GPU evaluation requires resource acknowledgement")
    rows, manifest, snapshot, processor = _data(args, verify_images=True)
    run = _e7_path(args.run_dir, args)
    out = _e7_path(args.output_dir, args)
    if out.exists() or run == out or run in out.parents or out in run.parents:
        raise ValueError("Choose a new evaluation output outside the training run")
    _, val_root = _cache_paths(args)
    val_cache = cache.open_val(val_root, cache.val_metadata(manifest, rows["val"], snapshot), rows["val"])
    train_root, _ = _cache_paths(args)
    if out == train_root or train_root in out.parents or out in train_root.parents:
        raise ValueError("E7 evaluation output overlaps the training cache")
    _separate_output(out, val_cache)
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    best_path = run / "best.pt"
    phase.require_hash(best_path, args.checkpoint_sha256)
    best = fa._load_checkpoint(best_path, trust=args.trust_run_checkpoint)
    if (summary.get("status") != "VALIDATION_COMPLETE"
            or summary.get("epochs_completed") != config.EPOCHS
            or record.get("schema") != config.SCHEMA
            or record.get("dataset") != args.dataset or record.get("manifest") != manifest
            or record.get("class_order") != list(inference.CLASS_ORDER)
            or record.get("source_sha256") != _hashes()
            or record.get("snapshot") != config.OFFICIAL_SNAPSHOT
            or record.get("effective_processor_sha256") != fa._digest_json(processor.to_dict())
            or record.get("val_cache_metadata_sha256") != val_cache.meta_file_sha256
            or summary.get("record") != record or best.get("record") != record
            or best.get("best") != summary.get("best")
            or not isinstance(best.get("best"), dict) or best["best"].get("epoch") != best.get("epoch")
            or record.get("selection_weights") not in ("raw", "ema")
            or best.get("selection_weights") != record["selection_weights"]):
        raise ValueError("E7 selected checkpoint, source, or frozen validation provenance differs")
    backbone, _ = fa.load_local_backbone(snapshot, trust_legacy_bin=args.trust_legacy_bin)
    model = GlobalLocalModel(backbone, record["train_counts"])
    state = best["ema"]["shadow"] if record["selection_weights"] == "ema" else best["model"]
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    precision = fa._precision(device, args.precision)
    model.to(device).eval()
    values = []
    with torch.inference_mode(), fa._autocast(device, precision):
        if args.tta == "horizontal":
            for row in rows["val"]:
                rgb = Image.fromarray(np.asarray(val_cache.read(row)).copy(), mode="RGB")
                values.append(inference.tta_probabilities(model, processor, rgb, device)[None])
        else:
            collate = input_pipeline.ViewCollator(processor, val_cache, training=False,
                                                  seed=args.seed, epoch=None)
            for start in range(0, len(rows["val"]), args.micro_batch):
                g, p, _, _ = collate(rows["val"][start:start + args.micro_batch])
                logits, _, _ = model(g.to(device), p.to(device))
                values.append(logits.float().softmax(1).cpu())
    labels = [row["label"] for row in rows["val"]]
    probabilities = inference._probabilities(torch.cat(values).numpy())
    metrics = fa._classification(labels, probabilities)
    _add_macro_metrics(metrics)
    artifact = {"dataset": args.dataset, "split": "val", "class_order": list(inference.CLASS_ORDER),
        "manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": args.checkpoint_sha256,
        "processor_sha256": record["effective_processor_sha256"], "source_sha256": _hashes(),
        "geometry": record["geometry"], "selection_weights": record["selection_weights"],
        "checkpoint_epoch": best["epoch"], "tta": args.tta, "precision": precision,
        "image_ids": [row["image_id"] for row in rows["val"]], "labels": labels,
        "probabilities": probabilities.tolist(), "metrics": metrics,
        "final_test_status": "NOT_ACCESSED"}
    out.mkdir(parents=True)
    fa._atomic_json(out / "validation_predictions.json", artifact)
    print(json.dumps({"dataset": args.dataset, "tta": args.tta, "rows": len(labels),
                      "metrics": artifact["metrics"]}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate-cache", "prepare-cache", "verify-inputs", "smoke", "benchmark", "train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", choices=("APTOS2019", "DDR"), required=True)
        sub.add_argument("--artifact-root", required=True)
        sub.add_argument("--manifest-source-root")
        sub.add_argument("--model-dir", required=True)
        sub.add_argument("--train-cache-root", required=True)
        sub.add_argument("--val-cache-root", required=True)
        sub.add_argument("--reuse-e6-train", action="store_true")
        sub.add_argument("--seed", type=int, default=config.SEED)
        sub.add_argument("--output-dir")
        sub.add_argument("--train-cache-max-gib", type=float, default=160)
        sub.add_argument("--val-cache-max-gib", type=float, default=20)
        sub.add_argument("--reserve-gib", type=float, default=20)
        sub.add_argument("--sample-per-grade", type=int, default=2)
        sub.add_argument("--init", choices=("e5", "official"), default="e5")
        sub.add_argument("--e5-checkpoint")
        sub.add_argument("--e5-checkpoint-sha256")
        sub.add_argument("--trust-e5-checkpoint", action="store_true")
        sub.add_argument("--trust-legacy-bin", action="store_true")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        sub.add_argument("--micro-batch", type=int, default=4)
        sub.add_argument("--accumulation", type=int, default=8)
        sub.add_argument("--num-workers", type=int, default=0)
        sub.add_argument("--persistent-workers", action="store_true")
        sub.add_argument("--prefetch-factor", type=int, default=2)
        sub.add_argument("--pin-memory", action="store_true")
        sub.add_argument("--ema-decay", type=float)
        sub.add_argument("--selection-weights", choices=("raw", "ema"), default="raw")
        sub.add_argument("--input-preflight-report")
        sub.add_argument("--smoke-report")
        sub.add_argument("--benchmark-report")
        sub.add_argument("--bench-batches", type=int, default=4)
        sub.add_argument("--acknowledge-shared-gpu", action="store_true")
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--trust-run-checkpoint", action="store_true")
        if name == "evaluate":
            sub.add_argument("--run-dir", required=True)
            sub.add_argument("--checkpoint-sha256", required=True)
            sub.add_argument("--tta", choices=("none", "horizontal"), default="none")
    args = parser.parse_args(argv)
    if args.command in ("verify-inputs", "smoke", "benchmark", "train", "evaluate") and not args.output_dir:
        parser.error(f"{args.command} requires --output-dir")
    if args.command == "train" and not args.benchmark_report:
        parser.error("train requires --benchmark-report")
    if args.command in ("benchmark", "train") and not args.smoke_report:
        parser.error(f"{args.command} requires --smoke-report")
    {"estimate-cache": estimate_cache, "prepare-cache": prepare_cache,
     "verify-inputs": verify_inputs, "smoke": smoke,
     "benchmark": benchmark, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
