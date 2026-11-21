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
    "recovery.py": "e4f52847d7bb4b978bad4ad4a8613143d81c80f96f6f01215dcb7b790b363e03",
}
# f406ddf produced the existing untimed validation reports.
_PRE_RECOVERY_SOURCE = {
    "__main__.py": "b60ea55358fcc129617e169a1a815fe3e7df520475d4ec6bc37e9cdfa417978c",
    "hpc.py": "41fc2e9de8b4bcf3ddabf216c976e642a65da2c5d1982fea02c9331f03e48b43",
    "workflow.py": "9ddf87993184f78971f2e77bc81ed71fb2dd34cc985c0f118c852f457c489f1e",
    "recovery.py": _PRE_VALIDATION_SOURCE["recovery.py"],
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
        if recovery and any(backend["status"] != "PREPARED" for backend in exports.values()):
            raise RuntimeError("Recovered exports incomplete. Retry export from the saved final recovery epoch.")
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


def _receipt_digests(prepared):
    return {_name(ratio): fa._digest_json(receipt) for ratio, (_, receipt) in prepared.items()}


def _validation_identity(run, digest, prepared, *, stage="validate"):
    return {"schema": "convnext-optimization-validation-v1", "stage": stage, "status": "VALIDATION_COMPLETE",
            "dataset": run["dataset"], "arm": run["arm"], "protocol_sha256": digest,
            "prepared_framework_source_sha256": run["framework_source_sha256"],
            "prepared_receipt_sha256": _receipt_digests(prepared), "sample": run["sample"],
            "backends": run["protocol"]["backends"], "parity_tolerances": run["protocol"]["parity_tolerances"],
            "receipt": run["receipt"], "timing_performed": False}


def _read_validation(target, identity, sources):
    report = read(target / "report.json")
    if (any(report.get(key) != value for key, value in identity.items()) or
            report.get("framework_source_sha256") not in sources or
            [v["name"] for v in report["variants"]] != list(identity["prepared_receipt_sha256"]) or
            any(v["name"] != _name(v["ratio"]) or set(v["backends"]) != set(identity["backends"])
                for v in report["variants"]) or
            any(report.get(key) != value for key, value in _validation_outcome(report["variants"]).items()) or
            (target / "summary.csv").read_bytes() != reporting.validation_summary(report).encode("utf-8")):
        raise ValueError("Existing validation report differs from the verified campaign")
    return report


def _recovery_comparison(variants, before):
    pairs = []
    for after, prior in zip(variants, before["variants"], strict=True):
        delta = {name: benchmark.quality_delta(prior["backends"][name]["quality"], result["quality"])
                 for name, result in after["backends"].items() if result["status"] == "VALIDATED"}
        pairs.append({"ratio": after["ratio"], "quality_delta_vs_same_ratio_before_recovery": delta})
    eligible = [v for v in variants if v["ratio"] and v["backends"]["eager"]["status"] == "VALIDATED"]

    def rank(variant):
        quality = variant["backends"]["eager"]["quality"]
        return -quality["macro_f1"], quality["grade_mae"], variant["ratio"]

    ranking = [v["ratio"] for v in sorted(eligible, key=rank)]
    complete = _validation_outcome(variants)["failures"] == 0
    return {"paired_changes": pairs, "pruned_ratio_ranking": ranking if complete else [],
            "selected_pruned_ratio": ranking[0] if complete else None,
            "selection_rule": "Highest eager macro-F1, then lower grade MAE, then smaller pruning ratio. "
                              "Requires all backends to pass. The 0% control is excluded."}


def _validate_prepared(target, identity, sources, prepared, original, batches, validate, plan, entry,
                       allocation, recheck, *, before=None):
    if target.exists():
        report = _read_validation(target, identity, sources)
        if before:
            matched_inputs = any("quality" in item for v in report["variants"] for item in v["backends"].values())
            expected = before["ordered_preprocessed_tensor_sha256"] if matched_inputs else None
            if (report.get("comparison") != _recovery_comparison(report["variants"], before) or
                    report["ordered_preprocessed_tensor_sha256"] != expected):
                raise ValueError("Recovered comparison differs from the recorded validation metrics or inputs")
        recheck()
        models.verified_images(batches)
        print(f"Verified existing validation report: {target / 'report.json'}", flush=True)
    else:
        names = ["global_pixels", "patch_pixels"] if entry.family == "hf_v1_global_local" else ["pixels"]
        tensor_digest = before["ordered_preprocessed_tensor_sha256"] if before else None
        variants = []
        for ratio, (path, item) in prepared.items():
            model = _restore(original, path, item)
            results, tensor_digest = _evaluate_backends(
                _name(ratio), model, batches, validate, plan, path, tensor_digest,
                input_names=names, prepared=item["backends"], timed=False)
            variants.append({"name": _name(ratio), "ratio": ratio, "backends": results})
            del model
        recheck()
        models.verified_images(batches)
        report = {**identity, "framework_source_sha256": sources[0],
                  "ordered_preprocessed_tensor_sha256": tensor_digest, "variants": variants,
                  **_validation_outcome(variants),
                  "runtime": benchmark.runtime(), "allocation": allocation,
                  "limitations": ["Exploratory frozen validation subset. Validation selected the parent checkpoint.",
                                  "Shared CPU, FP32 quality and parity only."]}
        if before:
            report["comparison"] = _recovery_comparison(variants, before)
            if not any("quality" in item for v in variants for item in v["backends"].values()):
                report["ordered_preprocessed_tensor_sha256"] = None
        _publish(target, lambda directory: reporting.write_validation_report(directory, report))
        print(f"Validation report written to {target / 'report.json'}", flush=True)
    failures = report["failures"]
    print(f"{entry.arm}_{entry.dataset}: {failures} unsuccessful validation rows. No timing performed.")
    return 2 if failures else 0


def _arm_progress(directory, recovery_sha, parent_sha, ratio, epochs):
    directory.mkdir(parents=True, exist_ok=True)
    boundaries = sorted(p for p in directory.iterdir() if not p.name.startswith("."))
    expected = [directory / f"epoch_{i:02d}" for i in range(1, len(boundaries) + 1)]
    if boundaries != expected or len(boundaries) > epochs:
        raise ValueError("Recovery epoch checkpoints must be contiguous and within the budget")
    binding = {"recovery_protocol_sha256": recovery_sha, "parent_prepare_sha256": parent_sha, "ratio": ratio}
    state = None
    if boundaries:
        checkpoint = boundaries[-1] / "state.pt"
        sidecar = read(checkpoint.with_name("state.pt.sha256.json"))
        if sidecar != {"sha256": phase.sha256(checkpoint), "bytes": checkpoint.stat().st_size}:
            raise ValueError("Recovery epoch checkpoint digest differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["binding"] != binding or
                [epoch["recovery_epoch"] for epoch in state["history"]] != list(range(1, len(boundaries) + 1))):
            raise ValueError("Recovery epoch checkpoint does not belong to this arm and protocol")

    def save(state):
        epoch = len(state["history"])
        _publish(directory / f"epoch_{epoch:02d}", lambda scratch: fa._save_checkpoint(
            scratch / "state.pt", {**state, "binding": binding}))

    return state, save


def _recovery_record(path, binding, policy=None):
    if policy is not None and not path.is_file():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("Existing recovery outputs are from another protocol. Preserve them in place.")
        path.parent.mkdir(exist_ok=True)
        fa._atomic_json(path, {**binding, "policy": policy})
    record = read(path)
    if record != {**binding, "policy": record["policy"]} or (policy is not None and record["policy"] != policy):
        raise ValueError("Recovery protocol differs from the frozen parent, source, or requested configuration")
    return {**record["policy"], "recovery_protocol_sha256": fa._digest_json(record)}


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
    validation_sources = (source, {**source, **_PRE_RECOVERY_SOURCE})
    compatible_sources = (*validation_sources, {**source, **_PRE_VALIDATION_SOURCE})
    if not any(recorded == {**run, "framework_source_sha256": version} for version in compatible_sources):
        raise ValueError("Run protocol differs. Use a new campaign root for changed inputs, code or versions.")
    digest = fa._digest_json(recorded)
    native = output / "prepared"
    recovered = output / "recovered"
    if stage in ("recover", "validate-recovered", "measure-recovered") and recovered.is_symlink():
        raise ValueError("Recovery outputs must reside in the campaign, without a redirected recovery directory")
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
    validation_dir = output / "validation"
    identity = _validation_identity(recorded, digest, prepared)
    if stage == "validate":
        return _validate_prepared(validation_dir, identity, validation_sources, prepared, original, batches,
                                  validate, plan, entry, allocation, recheck)
    recovery_file = recovered / "recovery_protocol.json"
    policy, extra = None, {}
    if stage in ("recover", "validate-recovered") or (stage == "measure-recovered" and recovery_file.is_file()):
        if not (validation_dir / "report.json").is_file():
            raise ValueError("Recovery requires a completed, passing unrecovered validate report")
        validation_sha = phase.sha256(validation_dir / "report.json")
        before = _read_validation(validation_dir, identity, validation_sources)
        phase.require_hash(validation_dir / "report.json", validation_sha)
        if (before["outcome"] != "PASS" or not before.get("ordered_preprocessed_tensor_sha256")
                or any(row["parity"]["passed"] is not True for item in before["variants"]
                       for row in item["backends"].values())):
            raise ValueError("Recovery requires every unrecovered variant/backend to pass validation")
        binding = {"schema": "convnext-optimization-recovery-v1", "protocol_sha256": digest,
                   "validation_report_sha256": validation_sha, "framework_source_sha256": source,
                   "parent_receipts_sha256": {str(r): phase.sha256(path / "prepared.json")
                                              for r, (path, _) in prepared.items()}}
    if stage == "recover":
        from .recovery import ARMS, EPOCHS, recover

        if not args.authorize_recovery or args.recovery_epochs != EPOCHS or entry.arm not in ARMS:
            raise ValueError("Staged recovery requires authorization, E5/E6/E7 and exactly three epochs per arm")
        requested = getattr(args, "recovery_ratio", None)
        selected = ratios if requested is None else (requested,)
        if not set(selected) <= set(ratios):
            raise ValueError("Unknown recovery ratio")
        settings = {"epochs": EPOCHS, "micro_batch": args.recovery_micro_batch, "device": args.recovery_device}
        policy = _recovery_record(recovery_file, binding, {**settings, "threads": args.threads,
                                                         "gpu_uuid": allocation.get("gpu_uuid")})
        example = validate(batches[0])
        for ratio in selected:
            def train(model):
                parent_sha = binding["parent_receipts_sha256"][str(ratio)]
                path = recovered / "progress" / _name(ratio)
                if path.resolve() != path:
                    raise ValueError("Recovery progress must reside in the campaign")
                state, save = _arm_progress(path, policy["recovery_protocol_sha256"], parent_sha, ratio, EPOCHS)
                summary = recover(model, entry, {"train": rows["train"]}, processor,
                                  resume=state, save_epoch=save, **settings)
                final_progress = path / f"epoch_{EPOCHS:02d}/state.pt.sha256.json"
                return {**summary, "final_epoch_checkpoint": read(final_progress)}

            phase.require_hash(prepared[ratio][0] / "prepared.json", binding["parent_receipts_sha256"][str(ratio)])
            _prepare_one(recovered / _name(ratio), original=original, ratio=ratio, inputs=example,
                         plan=plan, protocol=digest, parent=prepared[ratio], recovery=(policy, train))
        recheck()
        phase.require_hash(validation_dir / "report.json", validation_sha)
        print(f"Recovery outputs prepared in {recovered}. No latency measured.")
        return

    if stage not in ("measure", "measure-recovered", "validate-recovered"):
        raise ValueError("Unknown execution stage")
    if stage in ("measure-recovered", "validate-recovered"):
        if recovery_file.is_file():
            policy = _recovery_record(recovery_file, binding)
        elif stage == "measure-recovered":
            policy = read(recovered / "original" / "prepared.json")["recovery_policy"]
        else:
            raise ValueError("A completed three-epoch recovery campaign is required")
        for ratio in ratios:
            path = recovered / _name(ratio)
            if not (path / "prepared.json").is_file():
                raise ValueError(f"Recovery arm incomplete: {_name(ratio)}. Complete 0%, 10%, 30% and 50% first.")
            extra[ratio] = (path, _prepared(path, digest, ratio,
                            phase.sha256(prepared[ratio][0] / "prepared.json"), policy))
        if any(set(item["backends"]) != set(plan["backends"]) for _, item in extra.values()):
            raise ValueError("Recovered backend set differs")
    if stage == "validate-recovered":
        identity = {**_validation_identity(recorded, digest, extra, stage=stage),
                    "recovery_policy": policy, "unrecovered_validation_report_sha256": validation_sha,
                    "expected_ordered_preprocessed_tensor_sha256": before["ordered_preprocessed_tensor_sha256"]}

        def recheck_recovery():
            recheck()
            phase.require_hash(validation_dir / "report.json", validation_sha)

        return _validate_prepared(output / "validation-recovered", identity, (source,), extra, original, batches,
                                  validate, plan, entry, allocation, recheck_recovery, before=before)

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
