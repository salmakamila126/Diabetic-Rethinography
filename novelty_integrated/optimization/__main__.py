"""Explicit entry point for future read-only evaluation of original checkpoints."""

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import sys

import torch

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase

from . import backends, benchmark, models, pruning, registry, reporting


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("registry", help="List eligibility and installed backend candidates without loading models")
    run = commands.add_parser("run", help="Generate a plan; execution requires explicit authorization")
    run.add_argument("--dataset", required=True, choices=registry.DATASETS)
    run.add_argument("--arm", required=True, choices=("B0", "B1", "B2", "B3", "B4", "B4-Q", "B4-R", "E5", "E6", "E7"))
    for name in ("artifact-root", "model-dir", "checkpoint", "output"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--manifest-source-root")
    run.add_argument("--external-source-dir", type=Path)
    run.add_argument("--backend", nargs="+", choices=backends.BACKENDS, default=["eager", "torchscript"])
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--sample-count", type=int)
    run.add_argument("--sample-seed", type=int, default=42)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--atol", type=float, default=1e-4)
    run.add_argument("--rtol", type=float, default=1e-4)
    run.add_argument("--prune", nargs="+", choices=(0.0, *pruning.RATIOS), type=float, default=[])
    run.add_argument("--recovery-epochs", type=int, choices=(1, 2, 3))
    run.add_argument("--recovery-micro-batch", type=int, choices=(1, 4, 8, 16, 32), default=4)
    run.add_argument("--recovery-device", default="cpu", help="Recovery only. All benchmarks run on CPU.")
    run.add_argument("--authorize-recovery", action="store_true")
    run.add_argument("--acknowledge-shared-gpu", action="store_true")
    run.add_argument("--trust-checkpoint", action="store_true")
    run.add_argument("--trust-legacy-snapshot", action="store_true")
    run.add_argument("--execute", action="store_true")
    return parser


def _entry(args, entries):
    try:
        return entries[args.dataset, args.arm]
    except KeyError:
        if args.arm == "B4":
            raise ValueError("APTOS2019 B4 consists of independent B4-Q and B4-R checkpoints") from None
        raise ValueError("No dataset-specific evidence for this experiment") from None


def _output(args):
    artifact = args.artifact_root.expanduser().resolve()
    protected = [artifact / name for name in ("datasets", "cad_vit_runs", "foundation_e4",
        "foundation_e5", "foundation_e6", "foundation_e7", "foundation_inputs", "hf_foundation")]
    protected += [args.model_dir, args.checkpoint.expanduser().resolve().parent]
    if args.external_source_dir:
        protected.append(args.external_source_dir)
    return fa._safe_out(args.output, protected)


def _plan(args, entry):
    import math

    if (args.threads < 1 or args.batch_size < 1 or args.warmup < 1 or args.repeats < 2
            or any(not math.isfinite(x) or x < 0 for x in (args.atol, args.rtol))):
        raise ValueError("Invalid CPU benchmarking configuration")
    if args.recovery_epochs:
        if not args.authorize_recovery or not args.prune or entry.arm not in ("E5", "E6"):
            raise ValueError("Recovery needs separate authorization, explicit ratios and an E5/E6 arm")
        device = torch.device(args.recovery_device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not args.acknowledge_shared_gpu):
            raise ValueError("CUDA recovery requires shared-GPU acknowledgement")
    elif 0.0 in args.prune or args.authorize_recovery:
        raise ValueError("A 0% recovery control or authorization flag requires --recovery-epochs")
    return {"status": "PLAN_ONLY", "experiment": entry.describe(), "split": "val",
            "class_order": list(registry.CLASS_ORDER), "device": "cpu", "precision": "float32",
            "backends": list(dict.fromkeys(["eager", *args.backend])),
            "backend_candidates": backends.capabilities(), "prune": list(dict.fromkeys(args.prune)),
            "recovery_epochs": args.recovery_epochs, "recovery_device": args.recovery_device if args.recovery_epochs else None,
            "threads": args.threads, "batch_size": args.batch_size, "sample_count": args.sample_count,
            "sample_seed": args.sample_seed, "warmup": args.warmup, "repeats": args.repeats,
            "parity_tolerances": {"atol": args.atol, "rtol": args.rtol}}


def _measure_variant(name, model, batches, prepare, plan, output, tensor_digest, *, reference=None,
                     checked_logits=None):
    output.mkdir()
    example = prepare(batches[0])
    profile = pruning.profile(model, example)
    results = {}
    onnx = None
    for backend in plan["backends"]:
        info = plan["backend_candidates"][backend]
        if info["missing_packages"]:
            results[backend] = info
            continue
        initial_memory = benchmark.memory()
        try:
            candidate = backends.build(backend, model, example, output, threads=plan["threads"], shared_onnx=onnx)
            if backend == "onnxruntime":
                names = ["pixels"] if len(example) == 1 else ["global_pixels", "patch_pixels"]
                onnx = (candidate.artifact, names)
            if backend == "eager" and checked_logits is not None:
                logits, digest = checked_logits, tensor_digest
            else:
                logits, digest = benchmark.collect_logits(candidate.infer, batches, prepare)
            if digest != tensor_digest:
                raise ValueError("Backend input digest differs from the fixed validation tensors")
            if backend == "eager":
                eager_logits = logits
            parity = benchmark.parity(eager_logits, logits, **plan["parity_tolerances"])
            result = {"status": "PARITY_FAILED", "parity": parity, "settings": candidate.settings,
                      "quality": benchmark.quality(logits, batches),
                      "artifact": benchmark.artifact_receipt(candidate.artifact)}
            if parity["passed"]:
                timing = {scope: benchmark.time_backend(candidate.infer, batches, prepare,
                           warmup=plan["warmup"], repeats=plan["repeats"], scope=scope)
                          for scope in ("model_only", "end_to_end")}
                result.update(status="VALIDATED", timing=timing, memory_before=initial_memory,
                              memory_after=benchmark.memory())
            results[backend] = result
            del candidate
        except Exception as exc:
            if backend == "eager":
                raise
            print(f"{backend}: {type(exc).__name__}: {exc}", file=sys.stderr)
            results[backend] = {"status": "BACKEND_FAILED", "error_type": type(exc).__name__}
    variant = {"name": name, "profile": profile, "backends": results}
    if reference:
        parent = reference["profile"]
        variant["delta_vs_original"] = {"parameters": profile["parameters"] - parent["parameters"],
            "parameter_reduction_pct": 100 * (1 - profile["parameters"] / parent["parameters"]),
            "macs_per_image": profile["macs_per_image"] - parent["macs_per_image"],
            "mac_reduction_pct": 100 * (1 - profile["macs_per_image"] / parent["macs_per_image"]),
            "same_backend": {backend: _compare(reference["backends"][backend], result)
                             for backend, result in results.items() if result["status"] == "VALIDATED"
                             and reference["backends"][backend]["status"] == "VALIDATED"}}
    return variant


def _compare(parent, derived):
    delta = {}
    for scope, timed in derived["timing"].items():
        new = timed["median_ms_per_image"]
        old = parent["timing"][scope]["median_ms_per_image"]
        delta[scope] = {"median_ms_per_image": new - old, "speedup_parent_over_derived": old / new,
                        "images_per_second": timed["images_per_second"] - parent["timing"][scope]["images_per_second"]}
    return {"timing": delta, "artifact_bytes": derived["artifact"]["bytes"] - parent["artifact"]["bytes"],
            "quality": benchmark.quality_delta(parent["quality"], derived["quality"])}


def execute(args, entry, plan):
    if not entry.evidence:
        registry.require_sources(entry)
    checkpoint = models.verify_receipt(entry, args.checkpoint)
    if not args.trust_checkpoint or not args.trust_legacy_snapshot:
        raise ValueError("Both project checkpoint and pinned legacy snapshot trust must be explicit")
    output = _output(args)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Choose a fresh external run directory with an existing parent")
    rows = models.load_rows(entry, args.artifact_root, args.manifest_source_root)
    batches = models.validation_batches(rows["val"], batch_size=args.batch_size,
                                        sample_count=args.sample_count, seed=args.sample_seed)
    if args.recovery_epochs:
        for row in rows["train"]:
            phase.require_hash(row["path"], row["sha256"])
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    original, processor, receipt = models.load_original(
        entry, checkpoint, args.model_dir, trust_checkpoint=args.trust_checkpoint,
        trust_snapshot=args.trust_legacy_snapshot, external_source_dir=args.external_source_dir)
    prepare = partial(models.prepare_inputs, processor=processor, family=entry.family)
    validated = partial(models.verify_inputs, processor=processor, family=entry.family)
    checked_logits, tensor_digest = benchmark.collect_logits(original, batches, validated)
    ordered = [r for batch in batches for r in batch]
    positions = {r["image_id"]: i for i, r in enumerate(rows["val"])}
    source = {p.name: phase.sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    receipt["evidence"] = {"path": entry.evidence_path, "sha256": phase.sha256(registry.ROOT / entry.evidence_path)}
    source_manifest = args.artifact_root / "cad_vit_runs" / "manifests" / f"{entry.dataset}_seed42.csv"
    receipt["manifest"] = {"sha256": entry.configuration["manifest"]["manifest_sha256"],
                           "sidecar_sha256": phase.sha256(source_manifest.with_suffix(".json"))}
    output.mkdir()
    fa._atomic_json(output / "plan.json", {**plan, "status": "RUN_STARTED", "receipt": receipt})
    runtime = benchmark.runtime()
    parent = _measure_variant("original", original, batches, prepare, plan, output / "original", tensor_digest,
                              checked_logits=checked_logits)
    variants = [parent]
    for ratio in plan["prune"]:
        name = f"pruned_{round(100 * ratio):02d}" if ratio else "unpruned_recovery_control"
        if ratio:
            derived, topology = pruning.prune_hf(original, ratio)
            pruned = _measure_variant(f"{name}_without_recovery", derived, batches, prepare, plan,
                                     output / f"{name}_without_recovery", tensor_digest, reference=parent)
            if pruned["profile"]["parameters"] >= parent["profile"]["parameters"]:
                raise ValueError("Pruning yielded no parameter reduction")
            pruned.update(ratio=ratio, treatment="pruned_without_recovery", topology=topology)
            variants.append(pruned)
        else:
            derived, topology = copy.deepcopy(original), []
        if args.recovery_epochs:
            from .recovery import recover

            recovery = recover(derived, entry, rows, processor, epochs=args.recovery_epochs,
                               micro_batch=args.recovery_micro_batch, device=args.recovery_device)
            after = _measure_variant(f"{name}_with_recovery", derived, batches, prepare, plan,
                                     output / f"{name}_with_recovery", tensor_digest, reference=parent)
            after.update(ratio=ratio, treatment="pruned_with_recovery" if ratio else "unpruned_recovery_control",
                         topology=topology, recovery=recovery)
            if ratio:
                after["delta_vs_pruned_before_recovery"] = {
                    name: _compare(pruned["backends"][name], result)
                    for name, result in after["backends"].items() if result["status"] == "VALIDATED"
                    and pruned["backends"][name]["status"] == "VALIDATED"}
            variants.append(after)
        del derived

    models.verify_receipt(entry, checkpoint)
    models.verified_images(batches)
    phase.require_hash(source_manifest, receipt["manifest"]["sha256"])
    phase.require_hash(source_manifest.with_suffix(".json"), receipt["manifest"]["sidecar_sha256"])
    report = {"schema": "convnext-optimization-v1", "status": "MEASUREMENT_COMPLETE",
              "dataset": entry.dataset, "arm": entry.arm, "plan": plan, "runtime": runtime,
              "framework_source_sha256": source, "receipt": receipt, "variants": variants,
              "sample": {"split": "val", "count": len(ordered), "class_order": list(registry.CLASS_ORDER),
                         "frozen_validation_positions_zero_based": [positions[r["image_id"]] for r in ordered],
                         "ordered_preprocessed_tensor_sha256": tensor_digest,
                         "grade_support": [sum(r["label"] == grade for r in ordered) for grade in registry.CLASS_ORDER]},
              "historical_validation_reference": {"epoch": entry.best["epoch"],
                  "metrics": entry.best["validation"], "scope": "original historical full validation"},
              "limitations": ["Validation also selected the parent checkpoint; optimization comparisons are exploratory.",
                  "Model-only excludes preparation; end-to-end includes local decode, transforms and inference.",
                  "Hashes are outside timing. File/page cache is warm or uncontrolled; no cold-start claim.",
                  "Backend calls are synchronous CPU FP32 and timed sequentially in reported order.",
                  "RSS is process-wide; high-water mark is not an isolated backend peak.",
                  "Original PyTorch FP32 results here are newly measured on the aligned sample sequence."]}
    reporting.write_report(output, report)
    print(f"Report written to {output / 'report.json'}")


def main(argv=None):
    args = _parser().parse_args(argv)
    entries = registry.registry()
    if args.command == "registry":
        print(json.dumps({"experiments": [r.describe() for r in entries.values()],
                          "capabilities": backends.capabilities()}, indent=2))
        return
    entry = _entry(args, entries)
    plan = _plan(args, entry)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    execute(args, entry, plan)


if __name__ == "__main__":
    main()
