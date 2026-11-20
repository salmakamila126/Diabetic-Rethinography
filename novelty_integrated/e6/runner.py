"""Isolated E6 boundary-aware pilot. Train/validation only; no test command."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from novelty import foundation_adaptation as fa
from novelty import foundation_reporting as reporting
from novelty import preof_phase1 as phase
from novelty_integrated.e6 import config
from novelty_integrated.e5 import data
from novelty_integrated.e6 import input_pipeline
from novelty_integrated.e5 import validation_cache
from novelty_integrated.e5 import runner as e5runner
from novelty_integrated.e6 import decoded_cache
from novelty_integrated.e5.model import E5Heads
from novelty_integrated.e6.objective import loss_components


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
        "novelty_integrated/e6/config.py", "novelty_integrated/e6/objective.py",
        "novelty_integrated/e6/decoded_cache.py", "novelty_integrated/e6/input_pipeline.py",
        "novelty_integrated/e6/runner.py",
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
        raise ValueError("E6 output must stay outside every Git worktree")
    return output


def _cache_root(args):
    if not args.train_cache_root:
        raise ValueError("E6 requires external train cache")
    root = fa._safe_out(args.train_cache_root, [Path(args.artifact_root) / "datasets",
        Path(args.artifact_root) / "cad_vit_runs" / "manifests", args.model_dir])
    if args.validation_cache_root:
        validation = phase._outside_repository(args.validation_cache_root)
        if root == validation or root in validation.parents or validation in root.parents:
            raise ValueError("E6 train cache and E5 validation cache must be separate")
    return root


def _validation_root(args):
    return phase._outside_repository(args.validation_cache_root) if args.validation_cache_root else None


def _validation_cache(args, rows, manifest, snapshot, processor):
    if not args.validation_cache_root:
        return None
    expected = validation_cache.expected_metadata(snapshot, manifest, processor,
                                                   args.micro_batch, e5runner._source_hashes())
    return validation_cache.open_complete(_validation_root(args), expected, rows["val"])

def _record(args, manifest, snapshot, processor, train_counts, precision, train_cache, val_cache):
    from PIL import __version__ as pillow_version
    import transformers
    return {
        "schema": config.SCHEMA, "research_status": "EXPLORATORY_VALIDATION_ONLY",
        "dataset": args.dataset, "model_kind": "convnext_tiny", "official_pretrained_initialization": True,
        "manifest": manifest,
        "snapshot": {key: snapshot[key] for key in
                     ("hub_id", "revision", "config_sha256", "processor_config_sha256",
                      "weight_file", "weight_sha256")},
        "effective_processor": processor.to_dict(),
        "effective_processor_sha256": fa._digest_json(processor.to_dict()),
        "train_counts": train_counts, "b1_augmentation": fa._augmentation_metadata("B1"),
        "objective": {"categorical": "LDAM true-class margin scaled by 30",
                      "ldam_margins": list(config.ldam_margins(train_counts)),
                      "effective_number_beta": config.EFFECTIVE_NUMBER_BETA,
                      "effective_number_weights": list(config.effective_number_weights(train_counts)),
                      "weight_start_epoch": config.WEIGHT_START_EPOCH,
                      "ordinal": True, "ordinal_coefficient": config.ORDINAL_COEFFICIENT,
                      "adjacent_coefficient": config.ADJACENT_COEFFICIENT,
                      "adjacent_pairs": [[0, 1], [1, 2]],
                      "validation_loss": "ordinary categorical CE on raw logits"},
        "seed": args.seed, "epochs": config.PILOT_EPOCHS,
        "micro_batch": args.micro_batch, "accumulation": args.accumulation,
        "effective_batch": config.EFFECTIVE_BATCH,
        "optimizer": "AdamW", "backbone_lr": config.BACKBONE_LR,
        "head_lr": config.HEAD_LR, "weight_decay": config.WEIGHT_DECAY,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "precision": precision, "device_type": torch.device(args.device).type,
        "checkpoint_selection": "max validation macro-F1; tie lower validation CE; tie earliest epoch",
        "input_transport": {
            "train": "verified mmap decoded RGB, stochastic B1 augmentation each epoch",
            "train_cache_root": str(_cache_root(args)),
            "train_cache_metadata_sha256": train_cache.meta_file_sha256,
            "validation": "read-only E5 official tensor cache" if val_cache else "uncached official processor",
            "validation_cache_root": str(_validation_root(args)) if val_cache else None,
            "validation_cache_metadata_sha256": val_cache.meta_sha256 if val_cache else None,
            "num_workers": args.num_workers, "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor, "pin_memory": args.pin_memory},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": pillow_version, "numpy": np.__version__,
        "source_git_head": _git_head(), "source_sha256": _source_hashes(),
        "final_test_status": "NOT_ACCESSED",
    }

def _prepare(args):
    config.validate_batch(args.micro_batch, args.accumulation)
    if args.seed != 42 or args.num_workers < 0 or args.prefetch_factor < 1 or (args.persistent_workers and not args.num_workers):
        raise ValueError("E6 requires seed 42 and valid worker setup")
    if args.precision not in ("bf16", "fp32"):
        raise ValueError("Explicit BF16/FP32 required")
    if (args.revision != config.OFFICIAL_SNAPSHOT["revision"]
            or args.weight_sha256 != config.OFFICIAL_SNAPSHOT["weight_sha256"]):
        raise ValueError("E6 requires exact B1 official pretrained revision and weight hash")
    rows, manifest = data.load_train_val(
        args.artifact_root, args.dataset, manifest_source_root=args.manifest_source_root,
        verify_images=True)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    processor_only = fa.load_local_processor(snapshot)
    train_meta = decoded_cache.metadata(manifest, rows["train"], snapshot)
    train_cache = decoded_cache.open_complete(_cache_root(args), train_meta, rows["train"])
    val_cache = _validation_cache(args, rows, manifest, snapshot, processor_only)
    output = _output_path(args.output_dir, [Path(args.artifact_root) / "datasets",
        Path(args.artifact_root) / "cad_vit_runs" / "manifests", args.model_dir])
    for root in (_cache_root(args), _validation_root(args)):
        if root and (output == root or output in root.parents or root in output.parents):
            raise ValueError("E6 cache and output paths must be independent")
    if _validation_root(args) and _validation_root(args) == _cache_root(args):
        raise ValueError("E6 train cache and E5 validation cache must be separate")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Selected CUDA device unavailable")
    precision = fa._precision(device, args.precision)
    fa._seed(args.seed)
    backbone, processor = fa.load_local_backbone(snapshot, trust_legacy_bin=args.trust_legacy_bin)
    if fa._digest_json(processor.to_dict()) != fa._digest_json(processor_only.to_dict()):
        raise ValueError("Loaded processor differs from audited official processor")
    fa.configure_trainable(backbone, "convnext_tiny", "full")
    counts = [sum(row["label"] == grade for row in rows["train"]) for grade in range(5)]
    config.require_counts(counts)
    heads = E5Heads(768, counts, ordinal=True)
    backbone.to(device); heads.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(backbone.parameters()), "lr": config.BACKBONE_LR},
        {"params": list(heads.parameters()), "lr": config.HEAD_LR},
    ], weight_decay=config.WEIGHT_DECAY)
    record = _record(args, manifest, snapshot, processor, counts, precision, train_cache, val_cache)
    if args.command in ("smoke", "benchmark", "train"):
        if not args.input_preflight_report:
            raise ValueError("E6 requires actual-image input preflight")
        preflight_path = phase._outside_repository(args.input_preflight_report)
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (preflight.get("status") != "ACTUAL_INPUT_EQUIVALENCE_PASS"
                or preflight.get("dataset") != args.dataset
                or preflight.get("manifest_sha256") != manifest["manifest_sha256"]
                or preflight.get("source_sha256") != _source_hashes()
                or preflight.get("train_cache_metadata_sha256") != train_cache.meta_file_sha256
                or preflight.get("validation_cache_metadata_sha256") != (val_cache.meta_sha256 if val_cache else None)):
            raise ValueError("E6 input equivalence preflight differs")
        record["input_transport"]["input_preflight_sha256"] = phase.sha256(preflight_path)
    return rows, backbone, processor, heads, optimizer, record, output, device, precision, train_cache, val_cache

def estimate_cache(args):
    """CPU-only header sizing; source hashes are checked before cache admission."""
    rows, manifest = data.load_train_val(args.artifact_root, args.dataset,
        manifest_source_root=args.manifest_source_root, verify_images=False)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    fa.load_local_processor(snapshot)
    root = _cache_root(args)
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    required = decoded_cache.estimate_bytes(rows["train"])
    print(json.dumps({"schema": config.SCHEMA, "status": "READ_ONLY_ESTIMATE_UNVERIFIED_IMAGES",
        "dataset": args.dataset, "train_rows": len(rows["train"]),
        "decoded_cache_estimated_bytes": required,
        "cache_filesystem_free_bytes": shutil.disk_usage(ancestor).free,
        "manifest_sha256": manifest["manifest_sha256"],
        "final_test_status": "NOT_ACCESSED"}, indent=2))


def prepare_cache(args):
    """CPU-only full train RGB build; no model, GPU, validation or test writes."""
    if (args.revision != config.OFFICIAL_SNAPSHOT["revision"]
            or args.weight_sha256 != config.OFFICIAL_SNAPSHOT["weight_sha256"]):
        raise ValueError("Official snapshot identity differs")
    rows, manifest = data.load_train_val(args.artifact_root, args.dataset,
        manifest_source_root=args.manifest_source_root, verify_images=False)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    fa.load_local_processor(snapshot)
    meta = decoded_cache.metadata(manifest, rows["train"], snapshot)
    result = decoded_cache.build(_cache_root(args), meta, rows["train"],
                                 max_gib=args.cache_max_gib, reserve_gib=args.cache_reserve_gib)
    print(json.dumps(result, indent=2))

def verify_inputs(args):
    """CPU-only exact actual-image B1 equivalence gate."""
    if args.sample_per_grade not in range(1, 6):
        raise ValueError("Choose 1..5 samples per grade")
    rows, manifest = data.load_train_val(args.artifact_root, args.dataset,
        manifest_source_root=args.manifest_source_root, verify_images=True)
    snapshot = fa.audit_snapshot("convnext_tiny", args.model_dir,
                                 args.revision, args.weight_sha256)
    config.require_official_snapshot(snapshot)
    processor = fa.load_local_processor(snapshot)
    meta = decoded_cache.metadata(manifest, rows["train"], snapshot)
    cache = decoded_cache.open_complete(_cache_root(args), meta, rows["train"])
    val_cache = _validation_cache(args, rows, manifest, snapshot, processor)
    samples = {}
    for split in ("train", "val"):
        samples[split] = [r for grade in range(5) for r in
                          [row for row in rows[split] if row["label"] == grade][:args.sample_per_grade]]
        if len(samples[split]) != 5 * args.sample_per_grade:
            raise ValueError("Frozen split lacks all preflight grades")
    epoch = input_pipeline.multiprocessing.Value("i", 1)
    collator = input_pipeline.TrainCollator(processor, cache, seed=args.seed, epoch=epoch)
    previous = None; changed = 0
    for number in (1, 2):
        epoch.value = number
        cached, labels = collator(samples["train"])
        direct, expected = fa._image_batch(samples["train"], processor,
            augment_seed=args.seed, augment_epoch=number)
        if not torch.equal(cached, direct) or not torch.equal(labels, expected):
            raise ValueError("E6 cache changes B1 training tensors")
        if previous is not None:
            changed = int((cached != previous).flatten(1).any(dim=1).sum())
        previous = cached
    direct, expected = fa._image_batch(samples["val"], processor)
    if val_cache:
        cached, labels = val_cache.batch(samples["val"])
        if not torch.equal(cached, direct) or not torch.equal(labels, expected):
            raise ValueError("E5 validation cache differs from B1 direct input")
    report = {"schema": config.SCHEMA, "status": "ACTUAL_INPUT_EQUIVALENCE_PASS",
              "dataset": args.dataset, "manifest_sha256": manifest["manifest_sha256"],
              "source_sha256": _source_hashes(),
              "train_cache_metadata_sha256": cache.meta_file_sha256,
              "validation_cache_metadata_sha256": val_cache.meta_sha256 if val_cache else None,
              "train_samples": len(samples["train"]), "validation_samples": len(samples["val"]),
              "train_epochs_checked": [1, 2], "train_images_changed_between_epochs": changed,
              "equality": "torch.equal pixels and labels against B1 uncached processing",
              "final_test_status": "NOT_ACCESSED"}
    output = _output_path(args.output_dir, [Path(args.artifact_root) / "datasets",
        Path(args.artifact_root) / "cad_vit_runs" / "manifests", args.model_dir])
    for root in (_cache_root(args), _validation_root(args)):
        if root and (output == root or output in root.parents or root in output.parents):
            raise ValueError("E6 preflight output overlaps a cache")
    if output.exists():
        raise FileExistsError("E6 preflight output already exists")
    output.mkdir(parents=True)
    fa._atomic_json(output / "input_equivalence.json", report)
    print(json.dumps(report, indent=2))

def _check_gradients(backbone, heads, optimizer):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError("Non-finite E6 gradient")
    if any(p.grad is None for p in heads.categorical.parameters()):
        raise ValueError("Categorical head lacks gradients")
    if heads.ordinal is not None and any(p.grad is None for p in heads.ordinal.parameters()):
        raise ValueError("Ordinal head lacks gradients")
    if any(p.grad is None for p in backbone.parameters()):
        raise ValueError("Full ConvNeXt backbone lacks gradients")
    torch.nn.utils.clip_grad_norm_(parameters, config.MAX_GRAD_NORM)


def _train_batches(backbone, heads, optimizer, batches, *, device, precision,
                   total_images, micro_batch, accumulation, train_counts, epoch,
                   progress=None, non_blocking=False):
    """FP32 E6 loss and actual-image-count normalization in the final window."""
    backbone.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
    total_batches = math.ceil(total_images / micro_batch)
    truths, probabilities, windows = [], [], []
    sums = {"total": 0.0, "categorical": 0.0, "ordinal": 0.0, "adjacent": 0.0}
    count = steps = 0
    for index, (pixels, labels) in enumerate(batches):
        if index >= total_batches:
            raise ValueError("Training iterator exceeded frozen split")
        expected = min(micro_batch, total_images - index * micro_batch)
        if len(labels) != expected:
            raise ValueError("Unexpected micro-batch size")
        pixels = pixels.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        start = (index // accumulation) * accumulation * micro_batch
        window_size = min(accumulation * micro_batch, total_images - start)
        with fa._autocast(device, precision):
            categorical, ordinal = heads(fa.pooled(backbone, pixels, "convnext_tiny"))
        losses = loss_components(categorical, ordinal, labels, train_counts=train_counts, epoch=epoch)
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
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            steps += 1; windows.append(window_size)
    if count != total_images or sum(windows) != total_images:
        raise ValueError("Training iterator did not cover frozen rows exactly")
    metrics = fa._classification(torch.cat(truths).numpy(), torch.cat(probabilities).numpy())
    return {"loss": sums["total"] / count, "categorical_loss": sums["categorical"] / count,
            "ordinal_loss": sums["ordinal"] / count, "adjacent_loss": sums["adjacent"] / count,
            "images": count, "optimizer_steps": steps, "accumulation_window_images": windows,
            "last_window_images": windows[-1], **metrics}

def _eval_batches(backbone, heads, batches, **kwargs):
    """Selection uses raw categorical CE, unchanged from E5/B1."""
    return e5runner._eval_batches(backbone, heads, batches, **kwargs)

def _write_history_csv(path, history):
    """Flatten key metrics while retaining full per-grade/matrix JSON columns."""
    fields = ("epoch", "train_total_loss", "train_categorical_loss", "train_ordinal_loss", "train_adjacent_loss",
              "train_accuracy", "train_macro_f1", "validation_categorical_ce",
              "validation_ordinal_loss", "validation_accuracy", "validation_macro_f1",
              "validation_weighted_f1", "validation_qwk", "validation_grade_mae",
              "validation_grade1_transitions", "validation_grade3_transitions",
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
                    "train_adjacent_loss": train["adjacent_loss"],
                    "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
                    "validation_categorical_ce": val["validation_loss"],
                    "validation_ordinal_loss": val["ordinal_loss"],
                    "validation_accuracy": val["accuracy"],
                    "validation_macro_f1": val["macro_f1"],
                    "validation_weighted_f1": val["weighted_f1"],
                    "validation_qwk": val["quadratic_weighted_kappa"],
                    "validation_grade_mae": val["grade_mae"],
                    "validation_grade1_transitions": json.dumps(val["grade1_transitions"]),
                    "validation_grade3_transitions": json.dumps(val["grade3_transitions"]),
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


def _loaders(args, rows, processor, train_cache, val_cache):
    return input_pipeline.InputLoaders(rows, processor, train_cache,
        seed=args.seed, micro_batch=args.micro_batch, num_workers=args.num_workers,
        persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and torch.device(args.device).type == "cuda",
        validation_cache=val_cache)


def benchmark(args):
    """Short disposable end-to-end pilot; no checkpoint or E5 resource use."""
    import itertools
    rows, backbone, processor, heads, optimizer, record, output, device, precision, train_cache, val_cache = _prepare(args)
    if output.exists():
        raise FileExistsError("Benchmark output exists")
    if device.type != "cuda" or not 1 <= args.bench_batches <= 16:
        raise ValueError("Explicit CUDA and 1..16 benchmark batches required")
    loaders = _loaders(args, rows, processor, train_cache, val_cache)
    def gpu_utilization():
        try:
            sample = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=5, check=False)
            return sample.stdout.strip() if sample.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    gpu_util_before = gpu_utilization()
    cpu_load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    train_images = args.bench_batches * args.micro_batch
    if train_images > len(rows["train"]):
        raise ValueError("Benchmark exceeds frozen training split")
    import gc
    # Same B1 augmentation and worker settings, with E5's uncached RGB path.
    baseline_loaders = e5runner.input_pipeline.InputLoaders(
        rows, processor, seed=args.seed, micro_batch=args.micro_batch,
        num_workers=args.num_workers, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor, pin_memory=args.pin_memory,
        decoded_cache_mib=0, validation_cache=val_cache)
    start = time.perf_counter()
    for _ in itertools.islice(baseline_loaders.train_epoch(1), args.bench_batches):
        pass
    b1_uncached_input_seconds = time.perf_counter() - start
    del baseline_loaders
    gc.collect()
    start = time.perf_counter()
    for _ in itertools.islice(loaders.train_epoch(1), args.bench_batches):
        pass
    e6_cache_input_seconds = time.perf_counter() - start
    # One disposable stage profile with explicit sync, outside the throughput timing.
    def timed(action):
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        result = action()
        torch.cuda.synchronize(device)
        return result, time.perf_counter() - begin
    (profile_pixels, profile_labels), input_wait = timed(lambda: next(loaders.train_epoch(1)))
    (profile_pixels, profile_labels), h2d = timed(lambda: (
        profile_pixels.to(device, non_blocking=args.pin_memory),
        profile_labels.to(device, non_blocking=args.pin_memory)))
    backbone.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
    def profile_forward():
        with fa._autocast(device, precision):
            logits, ordinal = heads(fa.pooled(backbone, profile_pixels, "convnext_tiny"))
        return loss_components(logits, ordinal, profile_labels,
            train_counts=record["train_counts"], epoch=1)["total"]
    profile_loss, forward_loss = timed(profile_forward)
    _, backward = timed(profile_loss.backward)
    _, optimizer_step = timed(lambda: (_check_gradients(backbone, heads, optimizer), optimizer.step()))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total = torch.cuda.mem_get_info(device)
    shm_before = shutil.disk_usage("/dev/shm").free if Path("/dev/shm").exists() else None
    start = time.perf_counter()
    _train_batches(backbone, heads, optimizer,
        itertools.islice(loaders.train_epoch(1), args.bench_batches),
        device=device, precision=precision, total_images=train_images,
        micro_batch=args.micro_batch, accumulation=args.accumulation,
        train_counts=record["train_counts"], epoch=1, non_blocking=args.pin_memory)
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    val_batches = min(args.bench_batches, math.ceil(len(rows["val"]) / args.micro_batch))
    val_images = min(val_batches * args.micro_batch, len(rows["val"]))
    start = time.perf_counter()
    _eval_batches(backbone, heads, itertools.islice(loaders.validation(), val_batches),
        device=device, precision=precision, total_images=val_images,
        non_blocking=args.pin_memory)
    torch.cuda.synchronize(device)
    val_seconds = time.perf_counter() - start
    estimate = (len(rows["train"]) / train_images * train_seconds
                + len(rows["val"]) / val_images * val_seconds)
    report = {"schema": config.SCHEMA, "status": "DISPOSABLE_BENCHMARK_ONLY",
        "record": record, "train_images": train_images, "val_images": val_images,
        "train_seconds": train_seconds, "validation_seconds": val_seconds,
        "input_comparison": {"e5_b1_uncached_seconds": b1_uncached_input_seconds,
                             "e6_decoded_cache_seconds": e6_cache_input_seconds,
                             "images": train_images, "same_workers": args.num_workers},
        "stage_seconds_one_batch": {"input_wait": input_wait, "h2d": h2d,
            "forward_and_loss": forward_loss, "backward": backward, "clip_optimizer": optimizer_step},
        "shm_free_before_bytes": shm_before,
        "shm_free_after_bytes": shutil.disk_usage("/dev/shm").free if shm_before is not None else None,
        "cpu_load_before": cpu_load_before,
        "cpu_load_after": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "gpu_utilization_before": gpu_util_before,
        "gpu_utilization_after": gpu_utilization(),
        "train_cache_disk_bytes": json.loads((_cache_root(args) / "summary.json").read_text())["actual_bytes"],
        "estimated_epoch_seconds": estimate, "estimated_20_epoch_seconds": 20 * estimate,
        "estimate_scope": "short subset extrapolation; excludes checkpoint I/O and shared-server variance",
        "gpu_free_before_bytes": free_before, "gpu_total_bytes": total,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "parent_peak_rss_bytes": fa._process_peak_rss_bytes(), "final_test_status": "NOT_ACCESSED"}
    output.mkdir(parents=True); fa._atomic_json(output / "benchmark.json", report)
    print(json.dumps({k: report[k] for k in
        ("status", "estimated_20_epoch_seconds", "gpu_peak_reserved_bytes")}, indent=2))

def smoke(args):
    """Two disposable optimizer steps, no checkpoint or validation inference."""
    rows, backbone, processor, heads, optimizer, record, output, device, precision, train_cache, val_cache = _prepare(args)
    if output.exists():
        raise FileExistsError("Smoke output exists")
    loaders = _loaders(args, rows, processor, train_cache, val_cache)
    pixels, labels = next(loaders.train_epoch(1))
    pixels, labels = pixels.to(device), labels.to(device)
    start = time.perf_counter()
    for epoch in (1, config.WEIGHT_START_EPOCH):
        backbone.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        with fa._autocast(device, precision):
            categorical, ordinal = heads(fa.pooled(backbone, pixels, "convnext_tiny"))
        loss = loss_components(categorical, ordinal, labels,
            train_counts=record["train_counts"], epoch=epoch)["total"]
        loss.backward(); _check_gradients(backbone, heads, optimizer)
        optimizer.step(); _assert_thresholds(heads)
        if any(not bool(torch.isfinite(p).all()) for p in (*backbone.parameters(), *heads.parameters())):
            raise ValueError("Non-finite E6 parameter")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {"schema": config.SCHEMA, "status": "E6_SMOKE_PASS", "record": record,
        "optimizer_steps": 2, "weights_saved": False,
        "input_shape": list(pixels.shape), "categorical_shape": list(categorical.shape),
        "ordinal_shape": list(ordinal.shape),
        "parameter_counts": fa._parameter_counts(backbone, heads),
        "elapsed_seconds": time.perf_counter() - start,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "final_test_status": "NOT_ACCESSED"}
    output.mkdir(parents=True); fa._atomic_json(output / "smoke.json", report)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))

def _check_smoke(path, record, micro_batch):
    path = phase._outside_repository(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != config.SCHEMA or report.get("status") != "E6_SMOKE_PASS"
            or report.get("record") != record or report.get("optimizer_steps") != 2
            or report.get("weights_saved") is not False
            or report.get("input_shape") != [micro_batch, 3, 224, 224]
            or report.get("categorical_shape") != [micro_batch, 5]
            or report.get("ordinal_shape") != [micro_batch, 4]):
        raise ValueError("Matching E6 smoke report required")
    return phase.sha256(path)

def _resume(output, record, backbone, heads, optimizer, *, trust):
    if json.loads((output / "run.json").read_text(encoding="utf-8")) != record:
        raise ValueError("E6 resume protocol or provenance differs")
    saved = fa._load_checkpoint(output / "last.pt", trust=trust)
    history = saved.get("history")
    best = saved.get("best")
    if (saved.get("record") != record or not isinstance(history, list)
            or saved.get("epoch") != len(history) or not 1 <= len(history) <= config.PILOT_EPOCHS
            or [row.get("epoch") for row in history] != list(range(1, len(history) + 1))
            or not isinstance(best, dict) or not 1 <= best.get("epoch", 0) <= len(history)):
        raise ValueError("E6 last checkpoint history is invalid")
    if json.loads((output / "history.json").read_text(encoding="utf-8")) != history:
        raise ValueError("E6 history file differs from last checkpoint; inspect interrupted write")
    selected = fa._load_checkpoint(output / "best.pt", trust=trust)
    if (selected.get("record") != record or selected.get("epoch") != best["epoch"]
            or selected.get("best") != best):
        raise ValueError("E6 best checkpoint differs from recorded selection")
    backbone.load_state_dict(saved["backbone"], strict=True)
    heads.load_state_dict(saved["heads"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    fa._restore_rng(saved["rng"])
    _assert_thresholds(heads)
    return history, best, len(history) + 1


def train(args, *, stop_after_epoch=None):
    """Run only after source/cache/smoke/benchmark and resource gates pass."""
    if not args.smoke_report or not args.benchmark_report:
        raise ValueError("E6 requires smoke and benchmark reports")
    rows, backbone, processor, heads, optimizer, base_record, output, device, precision, train_cache, val_cache = _prepare(args)
    report_path = phase._outside_repository(args.benchmark_report)
    benchmark_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (benchmark_report.get("schema") != config.SCHEMA
            or benchmark_report.get("status") != "DISPOSABLE_BENCHMARK_ONLY"
            or benchmark_report.get("record") != base_record
            or benchmark_report.get("train_images", 0) < 8 * args.micro_batch
            or not math.isfinite(benchmark_report.get("estimated_20_epoch_seconds", float("inf")))
            or args.max_pilot_seconds <= 0 or args.disk_reserve_gib < 0
            or benchmark_report["estimated_20_epoch_seconds"] * 1.5 > args.max_pilot_seconds):
        raise ValueError("E6 benchmark is incompatible or exceeds 20-epoch time budget")
    if device.type != "cuda":
        raise ValueError("E6 pilot requires an explicit CUDA device")
    free_gpu, _ = torch.cuda.mem_get_info(device)
    if free_gpu < benchmark_report["gpu_peak_reserved_bytes"] * 1.3:
        raise ValueError("GPU free memory below 1.3x benchmark peak reserve")
    output_parent = output.parent
    while not output_parent.exists():
        output_parent = output_parent.parent
    if shutil.disk_usage(output_parent).free < args.disk_reserve_gib * 1024**3:
        raise ValueError("Insufficient output-filesystem free-space reserve")
    record = {**base_record,
              "smoke_report_sha256": _check_smoke(args.smoke_report, base_record, args.micro_batch),
              "benchmark_report_sha256": phase.sha256(report_path),
              "max_pilot_seconds": args.max_pilot_seconds,
              "disk_reserve_gib": args.disk_reserve_gib}
    if output.exists():
        if not args.resume:
            raise FileExistsError("E6 output exists; use guarded --resume")
        history, best, start_epoch = _resume(output, record, backbone, heads, optimizer,
                                             trust=args.trust_run_checkpoint)
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a nonexistent E6 run")
        output.mkdir(parents=True)
        fa._atomic_json(output / "run.json", record)
        history, best, start_epoch = [], None, 1
    logger = reporting.RunLogger(output)
    try:
        loaders = _loaders(args, rows, processor, train_cache, val_cache)
        end_epoch = min(config.PILOT_EPOCHS, stop_after_epoch or config.PILOT_EPOCHS)
        for epoch in range(start_epoch, end_epoch + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            counts = record["train_counts"]
            weights = config.weights_for_epoch(counts, epoch)
            train_batches = loaders.train_epoch(epoch)
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
                                            train_counts=counts, epoch=epoch, progress=progress,
                                            non_blocking=getattr(args, "pin_memory", False)
                                            and device.type == "cuda")
            finally:
                progress.close()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            val_batches = loaders.validation()
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
            cm = val_info["confusion_matrix"]
            val_info["grade1_transitions"] = {
                "support": int(sum(cm[1])), "correct": int(cm[1][1]),
                "undergraded_to_0": int(cm[1][0]),
                "overgraded_to_2_4": int(sum(cm[1][2:]))}
            val_info["grade3_transitions"] = {
                "support": int(sum(cm[3])), "correct": int(cm[3][3]),
                "undergraded_to_0_2": int(sum(cm[3][:3])),
                "overgraded_to_4": int(cm[3][4])}
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
            logger.line(f"epoch={epoch} train_total={train_info['loss']:.6f} "
                        f"train_ldam={train_info['categorical_loss']:.6f} "
                        f"train_ordinal={train_info['ordinal_loss']:.6f} "
                        f"train_adjacent={train_info['adjacent_loss']:.6f} "
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
    for name in ("estimate-cache", "prepare-cache", "verify-inputs", "smoke", "benchmark", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", choices=("APTOS2019", "DDR"), required=True)
        sub.add_argument("--artifact-root", required=True)
        sub.add_argument("--manifest-source-root",
                         help="Exact source dataset root recorded in frozen sidecar, if remapped")
        sub.add_argument("--model-dir", required=True)
        sub.add_argument("--revision", default=config.OFFICIAL_SNAPSHOT["revision"])
        sub.add_argument("--weight-sha256", default=config.OFFICIAL_SNAPSHOT["weight_sha256"])
        sub.add_argument("--trust-legacy-bin", action="store_true")
        sub.add_argument("--train-cache-root", required=True)
        sub.add_argument("--validation-cache-root", help="Read-only E5 validation tensor cache")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--micro-batch", type=int, default=32)
        sub.add_argument("--accumulation", type=int, default=1)
        sub.add_argument("--num-workers", type=int, default=2)
        sub.add_argument("--persistent-workers", action="store_true")
        sub.add_argument("--prefetch-factor", type=int, default=2)
        sub.add_argument("--pin-memory", action="store_true")
        if name == "prepare-cache":
            sub.add_argument("--cache-max-gib", type=float, required=True)
            sub.add_argument("--cache-reserve-gib", type=float, default=10.0)
        else:
            sub.add_argument("--output-dir", required=True)
        if name == "verify-inputs":
            sub.add_argument("--sample-per-grade", type=int, default=2)
        if name in ("smoke", "benchmark", "train"):
            sub.add_argument("--input-preflight-report", required=True)
        if name == "benchmark":
            sub.add_argument("--bench-batches", type=int, default=8)
        if name == "train":
            sub.add_argument("--smoke-report", required=True)
            sub.add_argument("--benchmark-report", required=True)
            sub.add_argument("--max-pilot-seconds", type=float, default=21600.0)
            sub.add_argument("--disk-reserve-gib", type=float, default=10.0)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--trust-run-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    {"estimate-cache": estimate_cache, "prepare-cache": prepare_cache, "verify-inputs": verify_inputs,
     "smoke": smoke, "benchmark": benchmark, "train": train}[args.command](args)

if __name__ == "__main__":
    main()
