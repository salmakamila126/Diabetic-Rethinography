"""Isolated E5 ConvNeXt-Tiny pilot. Train/validation only; no test command."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from novelty import foundation_adaptation as fa
from novelty import foundation_reporting as reporting
from novelty import preof_phase1 as phase
from novelty_integrated.e5 import config
from novelty_integrated.e5 import data
from novelty_integrated.e5 import input_pipeline
from novelty_integrated.e5 import validation_cache
from novelty_integrated.e5.model import E5Heads
from novelty_integrated.e5.objective import loss_components


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = (
        "novelty/foundation_adaptation.py", "novelty/foundation_augmentation.py",
        "novelty/foundation_input_cache.py",
        "novelty/foundation_reporting.py", "novelty/preof_phase1.py",
        "novelty/preof_e1.py",
        "novelty_integrated/e5/config.py", "novelty_integrated/e5/model.py",
        "novelty_integrated/e5/data.py",
        "novelty_integrated/e5/input_pipeline.py",
        "novelty_integrated/e5/validation_cache.py",
        "novelty_integrated/e5/objective.py", "novelty_integrated/e5/runner.py",
    )
    return {name: phase.sha256(root / name) for name in paths}


def _git_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _output_path(path, protected):
    output = fa._safe_out(path, protected)
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("E5 output must stay outside every Git worktree")
    return output


def _cache_root(args):
    root = getattr(args, "validation_cache_root", None)
    if not root:
        return None
    artifact = Path(args.artifact_root)
    return fa._safe_out(root, [artifact / "datasets", args.model_dir,
                               artifact / "cad_vit_runs" / "manifests"])


def _record(args, manifest, snapshot, processor, train_counts, precision):
    from PIL import __version__ as pillow_version
    import transformers

    ordinal, deferred = config.arm_components(args.arm)
    return {
        "schema": config.SCHEMA, "research_status": "EXPLORATORY_VALIDATION_ONLY",
        "arm": args.arm, "dataset": args.dataset, "model_kind": "convnext_tiny",
        "official_pretrained_initialization": True,
        "manifest": manifest,
        "snapshot": {key: snapshot[key] for key in
                     ("hub_id", "revision", "config_sha256", "processor_config_sha256",
                      "weight_file", "weight_sha256")},
        "effective_processor": processor.to_dict(),
        "effective_processor_sha256": fa._digest_json(processor.to_dict()),
        "train_counts": train_counts,
        "b1_augmentation": fa._augmentation_metadata("B1"),
        "objective": {"categorical": "ordinary CE; train-only inverse-sqrt weights from epoch 11 if deferred",
                      "ordinal": ordinal, "deferred_weighting": deferred,
                      "ordinal_coefficient": config.ORDINAL_COEFFICIENT,
                      "weight_start_epoch": config.WEIGHT_START_EPOCH,
                      "weights": list(config.train_class_weights(train_counts)) if deferred else None,
                      "validation_loss": "ordinary categorical CE for every arm"},
        "seed": args.seed, "epochs": config.PILOT_EPOCHS,
        "micro_batch": args.micro_batch, "accumulation": args.accumulation,
        "effective_batch": config.EFFECTIVE_BATCH,
        "optimizer": "AdamW", "backbone_lr": config.BACKBONE_LR,
        "head_lr": config.HEAD_LR, "weight_decay": config.WEIGHT_DECAY,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "precision": precision, "device_type": torch.device(args.device).type,
        "checkpoint_selection": "max validation macro-F1; tie lower validation CE; tie earliest epoch",
        "input_transport": {
            "train": "B1 EXIF-transposed RGB then fresh train-only augmentation then official processor",
            "decoded_ram_mib_per_worker": getattr(args, "decoded_cache_mib", 0),
            "validation": "verified official float32 tensor cache" if getattr(args, "validation_cache_root", None)
                          else "uncached deterministic official processor",
            "validation_cache_root": str(_cache_root(args)) if _cache_root(args) else None,
            "num_workers": getattr(args, "num_workers", 0),
            "persistent_workers": getattr(args, "persistent_workers", False),
            "prefetch_factor": getattr(args, "prefetch_factor", 2),
            "pin_memory": getattr(args, "pin_memory", False)},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": pillow_version, "numpy": np.__version__,
        "source_git_head": _git_head(), "source_sha256": _source_hashes(),
        "final_test_status": "NOT_ACCESSED",
    }


def _prepare(args):
    config.arm_components(args.arm)
    config.validate_batch(args.micro_batch, args.accumulation)
    if (getattr(args, "num_workers", 0) < 0
            or getattr(args, "decoded_cache_mib", 0) < 0
            or getattr(args, "prefetch_factor", 2) < 1
            or (getattr(args, "persistent_workers", False)
                and not getattr(args, "num_workers", 0))):
        raise ValueError("Invalid E5 input worker/cache configuration")
    if args.seed < 0 or args.precision not in ("bf16", "fp32"):
        raise ValueError("E5 requires nonnegative seed and explicit BF16 or FP32")
    if (args.revision != config.OFFICIAL_SNAPSHOT["revision"]
            or args.weight_sha256 != config.OFFICIAL_SNAPSHOT["weight_sha256"]):
        raise ValueError("E5 must use the B1 official pretrained revision and weight hash")
    rows, manifest = data.load_train_val(
        args.artifact_root, args.dataset,
        manifest_source_root=getattr(args, "manifest_source_root", None),
        verify_images=True)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    cache_root = _cache_root(args)
    if cache_root is not None:
        processor_only = fa.load_local_processor(snapshot)
        meta = validation_cache.expected_metadata(
            snapshot, manifest, processor_only, args.micro_batch, _source_hashes())
        cache = validation_cache.open_complete(
            cache_root, meta, rows["val"])
        args._e5_cache_metadata_sha256 = cache.meta_sha256
    output = _output_path(args.output_dir, [Path(args.artifact_root) / "datasets",
                                            args.model_dir,
                                            Path(args.artifact_root) / "cad_vit_runs" / "manifests"])
    if cache_root is not None and (output == cache_root or output in cache_root.parents
                                   or cache_root in output.parents):
        raise ValueError("E5 cache and run output paths must be independent")
    fa._seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Selected CUDA device is unavailable")
    precision = fa._precision(device, args.precision)
    backbone, processor = fa.load_local_backbone(
        snapshot, trust_legacy_bin=args.trust_legacy_bin)
    fa.configure_trainable(backbone, "convnext_tiny", "full")
    counts = [sum(row["label"] == grade for row in rows["train"]) for grade in range(5)]
    if min(counts) < 1:
        raise ValueError("Training split lacks a grade")
    ordinal, _ = config.arm_components(args.arm)
    heads = E5Heads(768, counts, ordinal=ordinal)
    backbone.to(device)
    heads.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(backbone.parameters()), "lr": config.BACKBONE_LR},
        {"params": list(heads.parameters()), "lr": config.HEAD_LR},
    ], weight_decay=config.WEIGHT_DECAY)
    record = _record(args, manifest, snapshot, processor, counts, precision)
    if getattr(args, "validation_cache_root", None):
        record["input_transport"]["validation_cache_metadata_sha256"] = (
            args._e5_cache_metadata_sha256)
    accelerated_input = bool(getattr(args, "num_workers", 0)
                             or getattr(args, "decoded_cache_mib", 0)
                             or getattr(args, "validation_cache_root", None)
                             or getattr(args, "pin_memory", False))
    if accelerated_input:
        preflight_path = getattr(args, "input_preflight_report", None)
        if not preflight_path:
            raise ValueError("Optional E5 input transport requires actual-image preflight")
        preflight_path = phase._outside_repository(preflight_path)
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (preflight.get("status") != "ACTUAL_INPUT_EQUIVALENCE_PASS"
                or preflight.get("dataset") != args.dataset
                or preflight.get("manifest_sha256") != manifest["manifest_sha256"]
                or preflight.get("processor_sha256") != snapshot["processor_config_sha256"]
                or preflight.get("source_sha256") != _source_hashes()
                or preflight.get("decoded_cache_mib") != getattr(args, "decoded_cache_mib", 0)
                or preflight.get("validation_cache_metadata_sha256")
                != record["input_transport"].get("validation_cache_metadata_sha256")):
            raise ValueError("Actual-image input preflight differs from this E5 run")
        record["input_transport"]["input_preflight_sha256"] = phase.sha256(preflight_path)
    return rows, backbone, processor, heads, optimizer, record, output, device, precision


def prepare_cache(args):
    """Verify source images and create val-only official tensors; no model or GPU."""
    if not args.validation_cache_root:
        raise ValueError("--validation-cache-root is required for prepare-cache")
    rows, manifest = data.load_train_val(
        args.artifact_root, args.dataset,
        manifest_source_root=args.manifest_source_root, verify_images=True)
    if (args.revision != config.OFFICIAL_SNAPSHOT["revision"]
            or args.weight_sha256 != config.OFFICIAL_SNAPSHOT["weight_sha256"]):
        raise ValueError("Official snapshot identity differs")
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    processor = fa.load_local_processor(snapshot)
    meta = validation_cache.expected_metadata(
        snapshot, manifest, processor, args.micro_batch, _source_hashes())
    report = validation_cache.build(
        _cache_root(args), meta, rows["val"], processor,
        max_gib=args.cache_max_gib, reserve_gib=args.cache_reserve_gib)
    print(json.dumps(report, indent=2))


def verify_inputs(args):
    """CPU-only actual-image equivalence gate; no model, gradients, or test rows."""
    output = _output_path(args.output_dir, [Path(args.artifact_root) / "datasets",
                                            args.model_dir,
                                            Path(args.artifact_root) / "cad_vit_runs" / "manifests"])
    cache_root = _cache_root(args)
    if cache_root is not None and (output == cache_root or output in cache_root.parents
                                   or cache_root in output.parents):
        raise ValueError("Input preflight output and cache paths must be independent")
    if output.exists():
        raise FileExistsError("Input preflight output already exists")
    rows, manifest = data.load_train_val(
        args.artifact_root, args.dataset,
        manifest_source_root=args.manifest_source_root, verify_images=True)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    processor = fa.load_local_processor(snapshot)
    if args.sample_per_grade < 1 or args.sample_per_grade > 5:
        raise ValueError("Choose 1..5 train/validation samples per grade")

    samples = {}
    for split in ("train", "val"):
        selected = []
        for grade in range(5):
            candidates = [r for r in rows[split] if r["label"] == grade]
            if len(candidates) < args.sample_per_grade:
                raise ValueError("Frozen split lacks preflight grade samples")
            selected.extend(candidates[:args.sample_per_grade])
        samples[split] = selected
    epoch = input_pipeline.multiprocessing.Value("i", 1)
    train_collator = input_pipeline.B1Collator(
        processor, training=True, seed=args.seed, epoch=epoch,
        decoded_max_bytes=args.decoded_cache_mib * 1048576)
    previous = None
    changed = 0
    for number in (1, 2):
        epoch.value = number
        cached, labels = train_collator(samples["train"])
        uncached, expected_labels = fa._image_batch(
            samples["train"], processor, augment_seed=args.seed, augment_epoch=number)
        if not torch.equal(cached, uncached) or not torch.equal(labels, expected_labels):
            raise ValueError("Decoded cache changes actual B1 training tensors")
        if previous is not None:
            changed = int((cached != previous).flatten(1).any(dim=1).sum())
        previous = cached
    cached_val = None
    if args.validation_cache_root:
        meta = validation_cache.expected_metadata(
            snapshot, manifest, processor, args.micro_batch, _source_hashes())
        cached_val = validation_cache.open_complete(cache_root, meta, rows["val"])
        val_pixels, val_labels = cached_val.batch(samples["val"])
    else:
        val_pixels, val_labels = fa._image_batch(samples["val"], processor)
    direct, direct_labels = fa._image_batch(samples["val"], processor)
    if not torch.equal(val_pixels, direct) or not torch.equal(val_labels, direct_labels):
        raise ValueError("Validation cache changes official processor tensors")
    report = {"schema": config.SCHEMA, "status": "ACTUAL_INPUT_EQUIVALENCE_PASS",
              "dataset": args.dataset, "manifest_sha256": manifest["manifest_sha256"],
              "processor_sha256": snapshot["processor_config_sha256"],
              "source_sha256": _source_hashes(),
              "decoded_cache_mib": args.decoded_cache_mib,
              "train_samples": len(samples["train"]), "validation_samples": len(samples["val"]),
              "train_epochs_checked": [1, 2], "train_images_changed_between_epochs": changed,
              "validation_cache_metadata_sha256": cached_val.meta_sha256 if cached_val else None,
              "equality": "torch.equal for pixels and labels against original B1 _image_batch",
              "final_test_status": "NOT_ACCESSED"}
    output.mkdir(parents=True)
    fa._atomic_json(output / "input_equivalence.json", report)
    print(json.dumps(report, indent=2))


def _check_gradients(backbone, heads, optimizer):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError("Non-finite E5 gradient")
    if any(p.grad is None for p in heads.categorical.parameters()):
        raise ValueError("Categorical head lacks gradients")
    if heads.ordinal is not None and any(p.grad is None for p in heads.ordinal.parameters()):
        raise ValueError("Ordinal head lacks gradients")
    if any(p.grad is None for p in backbone.parameters()):
        raise ValueError("Full ConvNeXt backbone lacks gradients")
    torch.nn.utils.clip_grad_norm_(parameters, config.MAX_GRAD_NORM)


def _train_batches(backbone, heads, optimizer, batches, *, device, precision,
                   total_images, micro_batch, accumulation, class_weights,
                   progress=None, non_blocking=False):
    """Accumulation uses actual image counts, including the final short window."""
    backbone.train()
    heads.train()
    total_batches = math.ceil(total_images / micro_batch)
    optimizer.zero_grad(set_to_none=True)
    truths, probabilities = [], []
    sums = {"total": 0.0, "categorical": 0.0, "ordinal": 0.0}
    count = steps = 0
    windows = []
    for index, (pixels, labels) in enumerate(batches):
        if index >= total_batches:
            raise ValueError("Training iterator exceeded frozen split")
        pixels = pixels.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        expected = min(micro_batch, total_images - index * micro_batch)
        if len(labels) != expected:
            raise ValueError("Training batch size differs from frozen split")
        start = (index // accumulation) * accumulation * micro_batch
        window_size = min(accumulation * micro_batch, total_images - start)
        with fa._autocast(device, precision):
            embedding = fa.pooled(backbone, pixels, "convnext_tiny")
            categorical, ordinal = heads(embedding)
        losses = loss_components(categorical, ordinal, labels, class_weights=class_weights)
        (losses["total"] * (len(labels) / window_size)).backward()
        for key in sums:
            sums[key] += float(losses[key].detach()) * len(labels)
        count += len(labels)
        truths.append(labels.detach().cpu())
        probabilities.append(categorical.detach().float().softmax(1).cpu())
        if progress is not None:
            progress.update(index + 1, count, sums["total"] / count)
        if (index + 1) % accumulation == 0 or index + 1 == total_batches:
            _check_gradients(backbone, heads, optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            windows.append(window_size)
    if count != total_images or index + 1 != total_batches or sum(windows) != total_images:
        raise ValueError("Training iterator did not cover the frozen split exactly")
    metrics = fa._classification(torch.cat(truths).numpy(), torch.cat(probabilities).numpy())
    return {"loss": sums["total"] / count, "categorical_loss": sums["categorical"] / count,
            "ordinal_loss": sums["ordinal"] / count, "images": count,
            "optimizer_steps": steps, "accumulation_window_images": windows,
            "last_window_images": windows[-1], **metrics}


def _eval_batches(backbone, heads, batches, *, device, precision, total_images,
                  progress=None, non_blocking=False):
    backbone.eval()
    heads.eval()
    truths, probabilities = [], []
    categorical_sum = ordinal_sum = count = 0
    with torch.inference_mode():
        for index, (pixels, labels) in enumerate(batches):
            pixels = pixels.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            with fa._autocast(device, precision):
                embedding = fa.pooled(backbone, pixels, "convnext_tiny")
                categorical, ordinal = heads(embedding)
            cat_loss = F.cross_entropy(categorical.float(), labels)
            ord_loss = loss_components(categorical, ordinal, labels)["ordinal"]
            categorical_sum += float(cat_loss) * len(labels)
            ordinal_sum += float(ord_loss) * len(labels)
            count += len(labels)
            truths.append(labels.cpu())
            probabilities.append(categorical.float().softmax(1).cpu())
            if progress is not None:
                progress.update(index + 1, count, categorical_sum / count)
    if count != total_images:
        raise ValueError("Validation iterator did not cover the frozen split exactly")
    metrics = fa._metrics(torch.cat(truths).numpy(), torch.cat(probabilities).numpy(),
                          categorical_sum / count)
    metrics["categorical_loss"] = categorical_sum / count
    metrics["ordinal_loss"] = ordinal_sum / count
    return metrics


def _write_history_csv(path, history):
    """Flatten key metrics while retaining full per-grade/matrix JSON columns."""
    fields = ("epoch", "train_total_loss", "train_categorical_loss", "train_ordinal_loss",
              "train_accuracy", "train_macro_f1", "validation_categorical_ce",
              "validation_ordinal_loss", "validation_accuracy", "validation_macro_f1",
              "validation_weighted_f1", "validation_qwk", "validation_grade_mae",
              "class_weight_active", "best_epoch", "train_seconds", "validation_seconds",
              "train_confusion_matrix", "validation_confusion_matrix",
              "train_per_grade", "validation_per_grade")
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in history:
                train, val, resource = row["train"], row["validation"], row["resource"]
                writer.writerow({
                    "epoch": row["epoch"], "train_total_loss": train["loss"],
                    "train_categorical_loss": train["categorical_loss"],
                    "train_ordinal_loss": train["ordinal_loss"],
                    "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
                    "validation_categorical_ce": val["validation_loss"],
                    "validation_ordinal_loss": val["ordinal_loss"],
                    "validation_accuracy": val["accuracy"],
                    "validation_macro_f1": val["macro_f1"],
                    "validation_weighted_f1": val["weighted_f1"],
                    "validation_qwk": val["quadratic_weighted_kappa"],
                    "validation_grade_mae": val["grade_mae"],
                    "class_weight_active": row["class_weight_active"],
                    "best_epoch": row["best_epoch"],
                    "train_seconds": resource["train_seconds"],
                    "validation_seconds": resource["validation_seconds"],
                    "train_confusion_matrix": json.dumps(train["confusion_matrix"]),
                    "validation_confusion_matrix": json.dumps(val["confusion_matrix"]),
                    "train_per_grade": json.dumps(train["per_grade"]),
                    "validation_per_grade": json.dumps(val["per_grade"]),
                })
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_thresholds(heads):
    if heads.ordinal is not None:
        thresholds = heads.ordinal.thresholds().detach()
        if not bool(torch.isfinite(thresholds).all()) or not bool(
                torch.all(thresholds[1:] > thresholds[:-1])):
            raise ValueError("Ordinal thresholds are not finite and strictly ordered")


def profile(args):
    """Disposable one-batch HPC timing probe; no checkpoint or research result."""
    rows, backbone, processor, heads, optimizer, record, output, device, precision = _prepare(args)
    if output.exists():
        raise FileExistsError("Profile output already exists")
    if device.type != "cuda":
        raise ValueError("E5 A100 profile requires an explicit CUDA device")
    if len(rows["train"]) < args.micro_batch or not rows["val"]:
        raise ValueError("Insufficient frozen train/validation rows for profile")
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)

    def measured(action):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = action()
        torch.cuda.synchronize(device)
        return value, time.perf_counter() - start

    loaders = None
    if (args.num_workers or args.decoded_cache_mib or args.validation_cache_root
            or args.pin_memory):
        cache = None
        if args.validation_cache_root:
            snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                         args.revision, args.weight_sha256)
            meta = validation_cache.expected_metadata(
                snapshot, record["manifest"], processor, args.micro_batch, _source_hashes())
            cache = validation_cache.open_complete(_cache_root(args), meta, rows["val"])
        loaders = input_pipeline.InputLoaders(
            rows, processor, seed=args.seed, micro_batch=args.micro_batch,
            num_workers=args.num_workers, persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor, pin_memory=args.pin_memory,
            decoded_cache_mib=args.decoded_cache_mib, validation_cache=cache)
    train_batches = (loaders.train_epoch(1) if loaders else
                     fa._epochs_batches("full", rows, processor, None,
                                        args.micro_batch, args.seed, 1, experiment="B1"))
    (pixels, labels), train_input_s = measured(lambda: next(iter(train_batches)))
    (pixels, labels), h2d_s = measured(lambda: (
        pixels.to(device, non_blocking=args.pin_memory),
        labels.to(device, non_blocking=args.pin_memory)))
    backbone.train()
    heads.train()
    optimizer.zero_grad(set_to_none=True)

    def forward():
        with fa._autocast(device, precision):
            embedding = fa.pooled(backbone, pixels, "convnext_tiny")
            categorical, ordinal = heads(embedding)
        return loss_components(categorical, ordinal, labels)["total"]

    loss, forward_s = measured(forward)
    _, backward_s = measured(loss.backward)
    _, optimizer_s = measured(lambda: (_check_gradients(backbone, heads, optimizer),
                                       optimizer.step()))
    _assert_thresholds(heads)
    val_batches = (loaders.validation() if loaders else
                   fa._validation_batches("full", rows, processor, None,
                                          args.micro_batch))
    (val_pixels, val_labels), val_input_s = measured(lambda: next(iter(val_batches)))
    (val_pixels, val_labels), val_h2d_s = measured(lambda: (
        val_pixels.to(device, non_blocking=args.pin_memory),
        val_labels.to(device, non_blocking=args.pin_memory)))
    backbone.eval()
    heads.eval()

    def validation_forward():
        with torch.inference_mode(), fa._autocast(device, precision):
            embedding = fa.pooled(backbone, val_pixels, "convnext_tiny")
            categorical, _ = heads(embedding)
        return categorical.shape

    _, val_forward_s = measured(validation_forward)
    props = torch.cuda.get_device_properties(device)
    report = {"schema": config.SCHEMA, "status": "DISPOSABLE_PROFILE_ONLY",
              "record": record, "training_images": len(labels),
              "validation_images": len(val_labels),
              "parent_peak_rss_bytes": fa._process_peak_rss_bytes(),
              "input_cache_counts": loaders.cache_counts() if loaders else None,
              "device": {"name": props.name, "compute_capability": [props.major, props.minor],
                         "total_bytes": total_bytes, "free_before_bytes": free_before,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
              "seconds": {"train_input_wait": train_input_s, "train_h2d": h2d_s,
                          "forward_and_loss": forward_s, "backward": backward_s,
                          "clip_and_optimizer": optimizer_s,
                          "validation_input_wait": val_input_s,
                          "validation_h2d": val_h2d_s,
                          "validation_forward": val_forward_s},
              "timing_scope": "one cold/disposable batch, explicit CUDA sync at each boundary; not epoch throughput"}
    output.mkdir(parents=True)
    fa._atomic_json(output / "profile.json", report)
    print(json.dumps({"status": report["status"], "device": report["device"],
                      "seconds": report["seconds"]}, indent=2))


def smoke(args):
    """Two disposable train-image optimizer steps; no checkpoint or validation inference."""
    rows, backbone, processor, heads, optimizer, record, output, device, precision = _prepare(args)
    if output.exists():
        raise FileExistsError("Smoke output already exists")
    if len(rows["train"]) < args.micro_batch:
        raise ValueError("Insufficient training rows for smoke micro-batch")
    pixels, labels = fa._image_batch(rows["train"][:args.micro_batch], processor,
                                     augment_seed=args.seed, augment_epoch=1)
    pixels, labels = pixels.to(device), labels.to(device)
    if device.type == "cuda":
        free_before, total_bytes = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        free_before = total_bytes = None
    started = time.perf_counter()
    for step in range(2):
        backbone.train()
        heads.train()
        optimizer.zero_grad(set_to_none=True)
        with fa._autocast(device, precision):
            embedding = fa.pooled(backbone, pixels, "convnext_tiny")
            categorical, ordinal = heads(embedding)
        weights = config.weights_for_epoch(record["train_counts"], args.arm,
                                           1 if step == 0 else config.WEIGHT_START_EPOCH)
        loss = loss_components(categorical, ordinal, labels, class_weights=weights)["total"]
        loss.backward()
        _check_gradients(backbone, heads, optimizer)
        optimizer.step()
        _assert_thresholds(heads)
        if any(not bool(torch.isfinite(p).all()) for p in (*backbone.parameters(), *heads.parameters())):
            raise ValueError("Non-finite E5 parameter after smoke update")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {"schema": config.SCHEMA, "status": "E5_SMOKE_PASS", "record": record,
              "optimizer_steps": 2, "weights_saved": False,
              "input_shape": list(pixels.shape), "embedding_shape": list(embedding.shape),
              "categorical_shape": list(categorical.shape),
              "ordinal_shape": list(ordinal.shape) if ordinal is not None else None,
              "parameter_counts": fa._parameter_counts(backbone, heads),
              "thresholds": heads.ordinal.thresholds().detach().cpu().tolist()
              if heads.ordinal is not None else None,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "gpu_free_bytes_before": free_before, "gpu_total_bytes": total_bytes,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
              if device.type == "cuda" else None,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
              if device.type == "cuda" else None,
              "elapsed_seconds": time.perf_counter() - started}
    output.mkdir(parents=True)
    fa._atomic_json(output / "smoke.json", report)
    print(json.dumps({"status": report["status"], "dataset": args.dataset,
                      "arm": args.arm, "gpu_peak_reserved_bytes": report["gpu_peak_reserved_bytes"]},
                     indent=2))


def _check_smoke(path, record, micro_batch):
    path = Path(path).expanduser().resolve()
    phase._outside_repository(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != config.SCHEMA or report.get("status") != "E5_SMOKE_PASS"
            or report.get("record") != record or report.get("optimizer_steps") != 2
            or report.get("weights_saved") is not False
            or report.get("input_shape") != [micro_batch, 3, 224, 224]
            or report.get("categorical_shape") != [micro_batch, 5]
            or report.get("ordinal_shape") != ([micro_batch, 4] if record["objective"]["ordinal"] else None)):
        raise ValueError("A matching E5 two-step smoke report is required")
    return phase.sha256(path)


def _resume(output, record, backbone, heads, optimizer, *, trust):
    if json.loads((output / "run.json").read_text(encoding="utf-8")) != record:
        raise ValueError("E5 resume protocol or provenance differs")
    saved = fa._load_checkpoint(output / "last.pt", trust=trust)
    history = saved.get("history")
    best = saved.get("best")
    if (saved.get("record") != record or not isinstance(history, list)
            or saved.get("epoch") != len(history) or not 1 <= len(history) <= config.PILOT_EPOCHS
            or [row.get("epoch") for row in history] != list(range(1, len(history) + 1))
            or not isinstance(best, dict) or not 1 <= best.get("epoch", 0) <= len(history)):
        raise ValueError("E5 last checkpoint history is invalid")
    if json.loads((output / "history.json").read_text(encoding="utf-8")) != history:
        raise ValueError("E5 history file differs from last checkpoint; inspect interrupted write")
    selected = fa._load_checkpoint(output / "best.pt", trust=trust)
    if (selected.get("record") != record or selected.get("epoch") != best["epoch"]
            or selected.get("best") != best):
        raise ValueError("E5 best checkpoint differs from recorded selection")
    backbone.load_state_dict(saved["backbone"], strict=True)
    heads.load_state_dict(saved["heads"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    fa._restore_rng(saved["rng"])
    _assert_thresholds(heads)
    return history, best, len(history) + 1


def train(args, *, stop_after_epoch=None):
    """Run the locked pilot; stop_after_epoch is an internal CPU-test interruption hook."""
    if not args.smoke_report:
        raise ValueError("Training requires a matching E5 smoke report")
    rows, backbone, processor, heads, optimizer, base_record, output, device, precision = _prepare(args)
    record = {**base_record,
              "smoke_report_sha256": _check_smoke(args.smoke_report, base_record, args.micro_batch)}
    if output.exists():
        if not args.resume:
            raise FileExistsError("E5 output exists; use guarded --resume")
        history, best, start_epoch = _resume(output, record, backbone, heads, optimizer,
                                             trust=args.trust_run_checkpoint)
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a nonexistent E5 run")
        output.mkdir(parents=True)
        fa._atomic_json(output / "run.json", record)
        history, best, start_epoch = [], None, 1
    logger = reporting.RunLogger(output)
    try:
        use_loader = bool(getattr(args, "num_workers", 0)
                          or getattr(args, "decoded_cache_mib", 0)
                          or getattr(args, "validation_cache_root", None)
                          or getattr(args, "pin_memory", False))
        loaders = None
        if use_loader:
            cache = None
            if getattr(args, "validation_cache_root", None):
                snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                             args.revision, args.weight_sha256)
                meta = validation_cache.expected_metadata(
                    snapshot, base_record["manifest"], processor,
                    args.micro_batch, _source_hashes())
                cache = validation_cache.open_complete(
                    _cache_root(args), meta, rows["val"])
            loaders = input_pipeline.InputLoaders(
                rows, processor, seed=args.seed, micro_batch=args.micro_batch,
                num_workers=getattr(args, "num_workers", 0),
                persistent_workers=getattr(args, "persistent_workers", False),
                prefetch_factor=getattr(args, "prefetch_factor", 2),
                pin_memory=getattr(args, "pin_memory", False) and device.type == "cuda",
                decoded_cache_mib=getattr(args, "decoded_cache_mib", 0),
                validation_cache=cache)
        end_epoch = min(config.PILOT_EPOCHS, stop_after_epoch or config.PILOT_EPOCHS)
        for epoch in range(start_epoch, end_epoch + 1):
            cache_before = loaders.cache_counts() if loaders else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            counts = record["train_counts"]
            weights = config.weights_for_epoch(counts, args.arm, epoch)
            train_batches = (loaders.train_epoch(epoch) if loaders is not None else
                             fa._epochs_batches("full", rows, processor, None,
                                                args.micro_batch, args.seed, epoch,
                                                experiment="B1"))
            train_input = {"input_seconds": 0.0}
            train_batches = fa._timed_batches(train_batches, train_input)
            n_batches = math.ceil(len(rows["train"]) / args.micro_batch)
            progress = reporting.BatchProgress(logger, "train", epoch, n_batches,
                                               len(rows["train"]))
            try:
                train_info = _train_batches(backbone, heads, optimizer, train_batches,
                                            device=device, precision=precision,
                                            total_images=len(rows["train"]),
                                            micro_batch=args.micro_batch,
                                            accumulation=args.accumulation,
                                            class_weights=weights, progress=progress,
                                            non_blocking=getattr(args, "pin_memory", False)
                                            and device.type == "cuda")
            finally:
                progress.close()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            val_batches = (loaders.validation() if loaders is not None else
                           fa._validation_batches("full", rows, processor, None,
                                                  args.micro_batch))
            val_input = {"input_seconds": 0.0}
            val_batches = fa._timed_batches(val_batches, val_input)
            val_progress = reporting.BatchProgress(
                logger, "validation", epoch,
                math.ceil(len(rows["val"]) / args.micro_batch), len(rows["val"]))
            try:
                val_info = _eval_batches(backbone, heads, val_batches, device=device,
                                         precision=precision, total_images=len(rows["val"]),
                                         progress=val_progress,
                                         non_blocking=getattr(args, "pin_memory", False)
                                         and device.type == "cuda")
            finally:
                val_progress.close()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            val_seconds = time.perf_counter() - started
            _assert_thresholds(heads)
            improved = fa._best_is_new(val_info, best["validation"] if best else None)
            if improved:
                best = {"epoch": epoch, "validation": val_info}
            resource = {"train_seconds": train_seconds, "validation_seconds": val_seconds,
                        "train_input_seconds": train_input["input_seconds"],
                        "validation_input_seconds": val_input["input_seconds"],
                        "train_images_per_second": len(rows["train"]) / train_seconds,
                        "validation_images_per_second": len(rows["val"]) / val_seconds,
                        "process_peak_rss_bytes": fa._process_peak_rss_bytes(),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda" else None,
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda" else None}
            if loaders is not None:
                cache_after = loaders.cache_counts()
                resource["input_cache_counts"] = {
                    key: cache_after[key] - cache_before[key] for key in cache_after}
            current = {"epoch": epoch, "train": train_info, "validation": val_info,
                       "resource": resource, "class_weight_active": weights is not None,
                       "learning_rates": {"backbone": config.BACKBONE_LR, "head": config.HEAD_LR},
                       "best_epoch": best["epoch"], "best_updated": improved}
            history.append(current)
            state = {"record": record, "epoch": epoch,
                     "backbone": backbone.state_dict(), "heads": heads.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": fa._rng_state(),
                     "history": history, "best": best}
            fa._save_checkpoint(output / "last.pt", state)
            if improved:
                fa._save_checkpoint(output / "best.pt", state)
            fa._atomic_json(output / "history.json", history)
            _write_history_csv(output / "history.csv", history)
            logger.line(f"epoch={epoch} arm={args.arm} train_total={train_info['loss']:.6f} "
                        f"train_ce={train_info['categorical_loss']:.6f} "
                        f"train_ordinal={train_info['ordinal_loss']:.6f} "
                        f"val_ce={val_info['validation_loss']:.6f} "
                        f"val_ordinal={val_info['ordinal_loss']:.6f} "
                        f"macro_f1={val_info['macro_f1']:.6f} best_epoch={best['epoch']}")
            for line in reporting.epoch_lines(current, best, improved, current["learning_rates"]):
                logger.line(line)
        if len(history) == config.PILOT_EPOCHS:
            fa._atomic_json(output / "summary.json", {
                "status": "VALIDATION_COMPLETE", "final_test_status": "NOT_ACCESSED",
                "record": record, "best": best, "epochs_completed": len(history),
                "parameter_counts": fa._parameter_counts(backbone, heads)})
    finally:
        logger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "train", "prepare-cache", "verify-inputs", "profile"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", choices=("APTOS2019", "DDR"), required=True)
        sub.add_argument("--arm", choices=tuple(config.ARMS), required=True)
        sub.add_argument("--artifact-root", required=True)
        sub.add_argument("--manifest-source-root",
                         help="Exact original sidecar dataset root when files moved to a new host")
        sub.add_argument("--model-dir", required=True)
        sub.add_argument("--revision", required=True)
        sub.add_argument("--weight-sha256", required=True)
        sub.add_argument("--trust-legacy-bin", action="store_true")
        if name != "prepare-cache":
            sub.add_argument("--output-dir", required=True)
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--micro-batch", type=int, default=32)
        sub.add_argument("--accumulation", type=int, default=1)
        sub.add_argument("--validation-cache-root",
                         help="Isolated E5 deterministic validation tensor cache, outside Git")
        if name == "prepare-cache":
            sub.add_argument("--cache-max-gib", type=float, default=2.0)
            sub.add_argument("--cache-reserve-gib", type=float, default=5.0)
        else:
            sub.add_argument("--num-workers", type=int, default=0)
            sub.add_argument("--persistent-workers", action="store_true")
            sub.add_argument("--prefetch-factor", type=int, default=2)
            sub.add_argument("--pin-memory", action="store_true")
            sub.add_argument("--decoded-cache-mib", type=int, default=0,
                             help="Bounded EXIF-corrected RGB RAM per worker; zero disables")
            if name == "verify-inputs":
                sub.add_argument("--sample-per-grade", type=int, default=2)
            else:
                sub.add_argument("--input-preflight-report",
                                 help="Required actual-image equivalence JSON for accelerated transport")
        if name == "train":
            sub.add_argument("--smoke-report", required=True)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--trust-run-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    {"smoke": smoke, "train": train, "prepare-cache": prepare_cache,
     "verify-inputs": verify_inputs, "profile": profile}[args.command](args)


if __name__ == "__main__":
    main()
