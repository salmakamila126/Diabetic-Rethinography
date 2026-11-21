"""Resumable preparation, untimed validation, and atomic CPU measurement."""

import copy
from functools import partial
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from . import backends, benchmark, models, pruning, registry, reporting
from .__main__ import _compare, _evaluate_backends, _measure_variant, _output


# Git blob SHA-256 for the changed modules at c101b1a5f72ee027bbb3adff8dd260536187c3e5.
# The complete predecessor map must match, including every unchanged source file.
_PRE_VALIDATION_SOURCE = {
    "__main__.py": "66469efd350cddf85a9e7fa2be336af5cbfeb499b287691d66ab42ba6c4a925a",
    "hpc.py": "8531a85494b30c8fd69219861d9b4f464b97ef49d0aca2fbaec8d0c05081b61d",
    "reporting.py": "cf2c0e04195ee3a241a46639bc56136109f8fbdefe7919c029381dec1439cc1e",
    "workflow.py": "5b1fa03ff9a75013e930e2cf103084e69afcb96c2720c88eff3843e993711916",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source():
    return {p.name: phase.sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _publish(directory, make):
    """A completed stage is one immutable directory. Interrupted scratch is never resumed."""
    with TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as scratch:
        candidate = Path(scratch) / "artifact"
        candidate.mkdir()
        make(candidate)
        if directory.exists():
            raise FileExistsError(directory)
        candidate.rename(directory)


def _prepared(directory, protocol, ratio, parent=None, recovery_policy=None):
    receipt = read(directory / "prepared.json")
    allowed = {"eager": "inference_state_dict.pt", "torchscript": "model.torchscript.pt",
               "onnxruntime": "model.onnx", "openvino": "model.xml"}
    if (receipt["protocol_sha256"] != protocol or receipt["ratio"] != ratio
            or receipt["parent_prepare_sha256"] != parent or receipt["recovery_policy"] != recovery_policy):
        raise ValueError("Prepared artifact receipt does not belong to this run")
    for backend, artifact in receipt["backends"].items():
        if artifact["status"] == "PREPARED":
            if (artifact["file"] != allowed[backend] or
                    benchmark.artifact_receipt(directory / artifact["file"]) != artifact["artifact"]):
                raise ValueError(f"Invalid prepared artifact: {directory.name}/{backend}")
    if receipt["backends"]["eager"]["status"] != "PREPARED":
        raise ValueError("Prepared eager state is required")
    return receipt


def _export(directory, model, inputs, plan):
    exports, onnx = {}, None
    for name in plan["backends"]:
        capability = plan["backend_candidates"][name]
        if capability["missing_packages"]:
            exports[name] = capability
            continue
        try:
            backend = backends.build(name, model, inputs, directory, threads=plan["threads"],
                                     shared_onnx=onnx)
            if name == "onnxruntime":
                names = ["pixels"] if len(inputs) == 1 else ["global_pixels", "patch_pixels"]
                onnx = (backend.artifact, names)
            exports[name] = {"status": "PREPARED", "file": backend.artifact.name,
                             "artifact": benchmark.artifact_receipt(backend.artifact)}
            del backend
        except Exception as exc:
            if name == "eager":
                raise
            print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            exports[name] = {"status": "BACKEND_FAILED", "error_type": type(exc).__name__}
    return exports


def _restore(original, directory, receipt):
    ratio = receipt["ratio"]
    if ratio:
        model, _ = pruning.prune_hf(original, ratio, topology=receipt["topology"])
    else:
        model = copy.deepcopy(original)
    model.load_state_dict(torch.load(directory / "inference_state_dict.pt",
                                     map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def _name(ratio):
    return f"pruned_{round(100 * ratio):02d}" if ratio else "original"


def _prepare_one(directory, *, original, ratio, inputs, plan, protocol, parent=None, recovery=None):
    parent_hash = phase.sha256(parent[0] / "prepared.json") if parent else None
    policy = recovery[0] if recovery else None
    def make(scratch):
        started = perf_counter()
        if parent:
            model = _restore(original, *parent)
            topology = parent[1]["topology"]
        elif ratio:
            model, topology = pruning.prune_hf(original, ratio)
        else:
            model, topology = original, []
        history = recovery[1](model) if recovery else None
        profile = pruning.profile(model, inputs)
        if ratio and profile["parameters"] >= sum(p.numel() for p in original.parameters()):
            raise ValueError("Pruning did not reduce parameter count")
        exports = _export(scratch, model, inputs, plan)
        fa._atomic_json(scratch / "prepared.json", {
            "protocol_sha256": protocol, "ratio": ratio, "topology": topology, "profile": profile,
            "backends": exports, "parent_prepare_sha256": parent_hash, "recovery_policy": policy,
            "recovery": history, "preparation_seconds": perf_counter() - started,
        })
    if not directory.exists():
        print(f"Preparing {directory.parent.name}/{directory.name}", flush=True)
        _publish(directory, make)
    else:
        print(f"Verifying completed {directory.parent.name}/{directory.name}", flush=True)
    result = _prepared(directory, protocol, ratio, parent_hash, policy)
    if set(result["backends"]) != set(plan["backends"]):
        raise ValueError("Prepared backend set differs")
    return result


def _sample(rows, batches):
    indices = {row["image_id"]: i for i, row in enumerate(rows)}
    selected = [row for batch in batches for row in batch]
    return {"split": "val", "count": len(selected), "class_order": list(registry.CLASS_ORDER),
            "frozen_validation_positions_zero_based": [indices[row["image_id"]] for row in selected],
            "grade_support": [sum(row["label"] == g for row in selected) for g in registry.CLASS_ORDER]}


def _validation_outcome(variants):
    statuses = [backend["status"] for variant in variants for backend in variant["backends"].values()]
    failures = len(statuses) - statuses.count("VALIDATED")
    return {"outcome": "BACKEND_FAILURES" if failures else "PASS", "backend_rows": len(statuses), "failures": failures}


def _validate_prepared(target, run, source, digest, prepared, original, batches, validate, plan, entry,
                       allocation, recheck):
    identity = {"schema": "convnext-optimization-validation-v1", "stage": "validate", "status": "VALIDATION_COMPLETE",
                "dataset": entry.dataset, "arm": entry.arm, "protocol_sha256": digest,
                "framework_source_sha256": source, "prepared_framework_source_sha256": run["framework_source_sha256"],
                "prepared_receipt_sha256": {_name(ratio): fa._digest_json(receipt)
                                            for ratio, (_, receipt) in prepared.items()},
                "sample": run["sample"], "backends": plan["backends"],
                "parity_tolerances": plan["parity_tolerances"], "receipt": run["receipt"],
                "timing_performed": False}
    if target.exists():
        report = read(target / "report.json")
        if (any(report.get(key) != value for key, value in identity.items()) or
                [v["name"] for v in report["variants"]] != [_name(ratio) for ratio in prepared] or
                any(set(v["backends"]) != set(plan["backends"]) for v in report["variants"]) or
                any(report.get(key) != value for key, value in _validation_outcome(report["variants"]).items()) or
                (target / "summary.csv").read_bytes() != reporting.validation_summary(report).encode("utf-8")):
            raise ValueError("Existing validation report differs from the verified campaign")
        recheck()
        models.verified_images(batches)
        print(f"Verified existing validation report: {target / 'report.json'}", flush=True)
    else:
        names = ["global_pixels", "patch_pixels"] if entry.family == "hf_v1_global_local" else ["pixels"]
        tensor_digest, variants = None, []
        for ratio, (path, item) in prepared.items():
            model = _restore(original, path, item)
            results, tensor_digest = _evaluate_backends(
                _name(ratio), model, batches, validate, plan, path, tensor_digest,
                input_names=names, prepared=item["backends"], timed=False)
            variants.append({"name": _name(ratio), "ratio": ratio, "backends": results})
            del model
        recheck()
        models.verified_images(batches)
        report = {**identity, "ordered_preprocessed_tensor_sha256": tensor_digest, "variants": variants,
                  **_validation_outcome(variants),
                  "runtime": benchmark.runtime(), "allocation": allocation,
                  "limitations": ["Exploratory frozen validation subset. Validation selected the parent checkpoint.",
                                  "Shared CPU, FP32 quality and parity only."]}
        _publish(target, lambda directory: reporting.write_validation_report(directory, report))
        print(f"Validation report written to {target / 'report.json'}", flush=True)
    failures = report["failures"]
    print(f"{entry.arm}_{entry.dataset}: {failures} unsuccessful validation rows. No timing performed.")
    return 2 if failures else 0


def execute(stage, args, entry, plan, allocation):
    plan = {**plan, "prune": list(pruning.RATIOS)}
    output = _output(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    rows = models.load_rows(entry, args.artifact_root, args.manifest_source_root)
    batches = models.validation_batches(rows["val"], batch_size=plan["batch_size"],
                                        sample_count=plan["sample_count"], seed=plan["sample_seed"])
    original, processor, receipt = models.load_original(
        entry, args.checkpoint, args.model_dir, trust_checkpoint=args.trust_checkpoint,
        trust_snapshot=args.trust_legacy_snapshot, external_source_dir=args.external_source_dir)
    prepare = partial(models.prepare_inputs, processor=processor, family=entry.family)
    validate = partial(models.verify_inputs, processor=processor, family=entry.family)
    sample = _sample(rows["val"], batches)
    source = _source()
    receipt["evidence"] = {"path": entry.evidence_path, "sha256": phase.sha256(registry.ROOT / entry.evidence_path)}
    manifest = args.artifact_root / "cad_vit_runs" / "manifests" / f"{entry.dataset}_seed42.csv"
    receipt["manifest"] = {"sha256": entry.configuration["manifest"]["manifest_sha256"],
                           "sidecar_sha256": phase.sha256(manifest.with_suffix(".json"))}
    phases = {key: plan[key] for key in ("backends", "batch_size", "sample_count", "sample_seed",
                                         "parity_tolerances", "warmup", "repeats")}
    run = {"schema": "convnext-optimization-staged-v1", "dataset": entry.dataset, "arm": entry.arm,
           "receipt": receipt, "framework_source_sha256": source, "sample": sample, "protocol": phases,
           "versions": benchmark.runtime()["versions"], "ratios": list(pruning.RATIOS)}
    run_file = output / "protocol.json"
    if stage == "prepare":
        output.mkdir(exist_ok=True)
        if not run_file.exists():
            fa._atomic_json(run_file, run)
    recorded = read(run_file)
    predecessor = {**run, "framework_source_sha256": {**source, **_PRE_VALIDATION_SOURCE}}
    if recorded != run and recorded != predecessor:
        raise ValueError("Run protocol differs. Use a new campaign root for changed inputs, code or versions.")
    if recorded != run and stage == "recover":
        raise ValueError("Use the preparation revision or a new campaign root to write recovery artifacts")
    digest = fa._digest_json(recorded)
    native = output / "prepared"
    recovered = output / "recovered"
    ratios = (0.0, *pruning.RATIOS)

    def recheck():
        models.verify_receipt(entry, args.checkpoint)
        phase.require_hash(manifest, receipt["manifest"]["sha256"])
        phase.require_hash(manifest.with_suffix(".json"), receipt["manifest"]["sidecar_sha256"])
        if _source() != source:
            raise ValueError("Framework source changed during this run")

    if stage == "prepare":
        recheck()
        if recorded != run and not all((native / _name(ratio) / "prepared.json").is_file() for ratio in ratios):
            raise ValueError("Use the preparation revision or a new campaign root to finish preparing artifacts")
        native.mkdir(exist_ok=True)
        example = validate(batches[0])
        for ratio in ratios:
            _prepare_one(native / _name(ratio), original=original, ratio=ratio,
                         inputs=example, plan=plan, protocol=digest)
        recheck()
        print(f"Prepared {len(ratios)} variants in {native}. No latency measured.")
        return

    prepared = {r: (native / _name(r), _prepared(native / _name(r), digest, r)) for r in ratios}
    if any(set(item["backends"]) != set(plan["backends"]) for _, item in prepared.values()):
        raise ValueError("Prepared backend set differs")
    if stage == "validate":
        return _validate_prepared(output / "validation", recorded, source, digest, prepared, original, batches,
                                  validate, plan, entry, allocation, recheck)
    if stage in ("recover", "measure-recovered"):
        raw = output / "measurement/report.json"
        if not raw.is_file():
            raise ValueError("Complete and review the unrecovered measurement before recovery")
        prior = read(raw)
        if (prior.get("stage") != "measure" or prior.get("status") != "MEASUREMENT_COMPLETE"
                or prior.get("protocol_sha256") != digest or len(prior["variants"]) != len(ratios)
                or any(v["backends"]["eager"]["status"] != "VALIDATED" for v in prior["variants"])):
            raise ValueError("Recovery requires a completed unrecovered measurement for the same protocol")
    if stage == "recover":
        from .recovery import recover

        for row in rows["train"]:
            phase.require_hash(row["path"], row["sha256"])
        recovered.mkdir(exist_ok=True)
        policy = {"epochs": args.recovery_epochs, "micro_batch": args.recovery_micro_batch,
                  "device": args.recovery_device}
        train = partial(recover, entry=entry, rows=rows, processor=processor, **policy)
        example = validate(batches[0])
        for ratio in ratios:
            _prepare_one(recovered / _name(ratio), original=original, ratio=ratio, inputs=example,
                         plan=plan, protocol=digest, parent=prepared[ratio], recovery=(policy, train))
        recheck()
        print(f"Recovery outputs prepared in {recovered}. No latency measured.")
        return

    if stage not in ("measure", "measure-recovered"):
        raise ValueError("Unknown execution stage")
    extra, policy = {}, None
    if stage == "measure-recovered":
        policy = read(recovered / "original" / "prepared.json")["recovery_policy"]
        for ratio in ratios:
            path = recovered / _name(ratio)
            extra[ratio] = (path, _prepared(path, digest, ratio,
                            phase.sha256(prepared[ratio][0] / "prepared.json"), policy))

    target = output / ("measurement" if not extra else "post-recovery")
    if target.exists():
        raise FileExistsError(f"Completed measurement already exists: {target}")
    reference, data_sha = benchmark.collect_logits(original, batches, validate)
    sample["ordered_preprocessed_tensor_sha256"] = data_sha

    def measure(directory, ratio, treatment, *, parent=None):
        path, item = directory
        is_native = treatment == "original"
        model = original if is_native else _restore(original, path, item)
        name = treatment if is_native else f"{_name(ratio)}_{treatment}"
        result = _measure_variant(name, model, batches, prepare, plan, path, data_sha,
                                  reference=parent, checked_logits=reference if is_native else None,
                                  prepared=item["backends"])
        if result["profile"] != item["profile"]:
            raise ValueError("Restored profile differs from preparation")
        if not is_native:
            result.update(ratio=ratio, treatment=treatment, topology=item["topology"],
                          recovery=item["recovery"])
        return result

    original_result = measure(prepared[0], 0, "original")
    before = {r: measure(prepared[r], r, "pruned_without_recovery", parent=original_result)
              for r in pruning.RATIOS}
    after = []
    for ratio, item in extra.items():
        treatment = "pruned_with_recovery" if ratio else "unpruned_recovery_control"
        result = measure(item, ratio, treatment, parent=original_result)
        if ratio:
            old = before[ratio]["backends"]
            result["delta_vs_pruned_before_recovery"] = {
                name: _compare(old[name], backend) for name, backend in result["backends"].items()
                if backend["status"] == old[name]["status"] == "VALIDATED"}
        after.append(result)
    recheck()
    models.verified_images(batches)
    report = {"schema": "convnext-optimization-v1", "status": "MEASUREMENT_COMPLETE",
              "stage": stage, "protocol_sha256": digest, "dataset": entry.dataset, "arm": entry.arm,
              "recovery_policy": policy,
              "runtime": benchmark.runtime(), "allocation": allocation, "sample": sample,
              "receipt": receipt, "framework_source_sha256": source,
              "prepared_framework_source_sha256": recorded["framework_source_sha256"], "plan": plan,
              "variants": [original_result, *before.values(), *after],
              "historical_validation_reference": {"epoch": entry.best["epoch"], "metrics": entry.best["validation"],
                                                  "scope": "original historical full validation"},
              "limitations": ["Validation also selected the parent checkpoint.",
                  "CPU FP32, synchronous calls, warm or uncontrolled filesystem cache.",
                  "Model-only excludes preparation. End-to-end includes decode, transforms and inference.",
                  "RSS is process-wide. Scaling estimates exclude export and recovery."]}
    _publish(target, lambda directory: reporting.write_report(directory, report))
    print(f"Report written to {target / 'report.json'}")
