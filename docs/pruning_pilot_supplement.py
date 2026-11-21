"""Verify and package the pilot's prepared profiles and E5 recovery evidence snapshot."""

import argparse
import json
from pathlib import Path

from docs import pruning_pilot_publication as pilot
from docs.e5_e6_publication import check_public, require
from docs.e7_publication import parse


ACCOUNTING = ("Executed Conv2d and Linear only; no biases, normalization, activation, pooling, "
              "gate elementwise operations or memory traffic. Shared E7 calls all count.")
E5 = "E5_APTOS2019"
RECOVERY_FILE = "e5_aptos2019_recovery_audit.json"
RECOVERY_REVISION = pilot.VALIDATE_COMMITS["E6_DDR"]
RECOVERY_PROTOCOL = "recovered/recovery_protocol.json"
EPOCH_ONE = "recovered/progress/pruned_10/epoch_01/state.pt.sha256.json"


def dataset_file(dataset):
    return f"prepared_{dataset.lower()}.json"


def member(raw):
    document = parse(raw)
    require(pilot.json_bytes(document) == raw, "Source JSON bytes do not match the recorded producer format")
    return {"source_bytes": len(raw), "source_sha256": pilot.sha(raw), "record": document}


def check_member(value):
    require(value == member(pilot.json_bytes(value["record"])), "Source byte receipt differs")
    return value["record"]


def validate_profiles(bundle, profiles):
    check_public(profiles)
    require(set(profiles) == set(pilot.JOBS), "Prepared jobs differ")
    for job in pilot.JOBS:
        report = bundle["jobs"][job]["validation"]
        receipts = profiles[job]
        require(set(receipts) == set(pilot.VARIANTS), f"{job}: prepared variant set differs")
        original = receipts["original"]["record"]["profile"]
        previous = None
        for arm in report["variants"]:
            name, ratio = arm["name"], arm["ratio"]
            label = f"{job}/{name}"
            data = check_member(receipts[name])
            require(pilot.canonical_sha(data) == report["prepared_receipt_sha256"][name] and
                    data["protocol_sha256"] == report["protocol_sha256"] and data["ratio"] == ratio and
                    all(data[k] is None for k in ("parent_prepare_sha256", "recovery_policy", "recovery")),
                    f"{label}: prepared identity or provenance differs")
            require(set(data["backends"]) == set(pilot.BACKENDS), f"{label}: prepared backend set differs")
            for backend, result in data["backends"].items():
                require(result["status"] == "PREPARED" and result["file"] == pilot.ARTIFACTS[backend][0] and
                        result["artifact"] == arm["backends"][backend]["artifact"], f"{label}: backend receipt differs")
            profile = data["profile"]
            numbers = ("parameters", "macs_per_image", "conv2d_macs_per_image", "linear_macs_per_image",
                       "flops_per_image_two_per_mac")
            require(all(type(profile[k]) is int and profile[k] > 0 for k in numbers) and
                    profile["macs_per_image"] == profile["linear_macs_per_image"] + profile["conv2d_macs_per_image"] and
                    profile["flops_per_image_two_per_mac"] == 2 * profile["macs_per_image"] and
                    profile["accounting"] == ACCOUNTING, f"{label}: profiler accounting differs")
            topology = data["topology"]
            if not ratio:
                require(topology == [], f"{label}: original has an unexpected pruning topology")
                continue
            require(topology and len({layer["layer"] for layer in topology}) == len(topology),
                    f"{label}: absent or duplicate topology layers")
            for layer in topology:
                width, kept = layer["original_width"], layer["kept_indices"]
                require(layer["layer"].startswith("encoder.stages.") and type(width) is int and width > 0 and
                        layer["pruned_width"] == len(kept) == width - int(width * ratio) and kept and
                        kept == sorted(set(kept)) and all(type(i) is int and 0 <= i < width for i in kept),
                        f"{label}: invalid retained expansion indices")
            require(profile["parameters"] < original["parameters"] and profile["macs_per_image"] < original["macs_per_image"] and
                    profile["conv2d_macs_per_image"] == original["conv2d_macs_per_image"],
                    f"{label}: inconsistent expansion-only complexity")
            if previous:
                previous_profile = previous["profile"]
                require(profile["parameters"] < previous_profile["parameters"] and
                        profile["macs_per_image"] < previous_profile["macs_per_image"] and
                        all((later["layer"], later["original_width"]) == (earlier["layer"], earlier["original_width"]) and
                            set(later["kept_indices"]) <= set(earlier["kept_indices"]) for later, earlier in
                            zip(topology, previous["topology"], strict=True)), f"{label}: pruning ranks are not nested")
            previous = data
    return profiles


def read_profiles(bundle, folder):
    profiles = {}
    for dataset in pilot.DATASETS:
        record = parse((folder / dataset_file(dataset)).read_bytes())
        require(record["campaign"] == pilot.FOLDER.name and record["dataset"] == dataset and
                set(record["jobs"]) == {job for job in pilot.JOBS if job.endswith(dataset)}, "Prepared receipt dataset differs")
        profiles.update(record["jobs"])
    return validate_profiles(bundle, profiles)


def transport_profiles(bundle, files):
    expected = {digest: (job, variant) for job, data in bundle["jobs"].items()
                for variant, digest in data["validation"]["prepared_receipt_sha256"].items()}
    profiles = {job: {} for job in pilot.JOBS}
    for file in files:
        value = member(file.read_bytes())
        digest = pilot.canonical_sha(value["record"])
        require(digest in expected, "Downloaded receipt has no matching published digest")
        job, variant = expected[digest]
        require(profiles[job].get(variant, value) == value, "Conflicting copies of the same prepared receipt")
        profiles[job][variant] = value
    return validate_profiles(bundle, profiles)


def profile_bytes(profiles, dataset):
    jobs = {job: profiles[job] for job in pilot.JOBS if job.endswith(dataset)}
    # Pack only integer channel lists and other receipt content, preserving the source JSON values exactly.
    return (json.dumps({"campaign": pilot.FOLDER.name, "dataset": dataset, "jobs": jobs},
                       sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def complexity_rows(bundle, profiles, jobs=pilot.JOBS):
    for job in jobs:
        report = bundle["jobs"][job]["validation"]
        original = profiles[job]["original"]["record"]["profile"]
        for variant, ratio in pilot.VARIANTS.items():
            value = profiles[job][variant]
            profile = value["record"]["profile"]
            parameters, macs = profile["parameters"], profile["macs_per_image"]
            yield {"job": job, "dataset": report["dataset"], "variant": variant,
                   "target_pruning_percent": round(ratio * 100), "parameters": parameters,
                   "parameter_reduction_percent": 100 * (original["parameters"] - parameters) / original["parameters"],
                   "macs_per_image": macs, "mac_reduction_percent": 100 * (original["macs_per_image"] - macs) / original["macs_per_image"],
                   "flops_per_image_two_per_mac": profile["flops_per_image_two_per_mac"],
                   "parent_checkpoint_epoch": report["receipt"]["epoch"],
                   "parent_checkpoint_sha256": report["receipt"]["checkpoint_sha256"],
                   "source_receipt": f"{job}/prepared/{variant}/prepared.json",
                   "source_receipt_sha256": value["source_sha256"],
                   "canonical_receipt_sha256": report["prepared_receipt_sha256"][variant]}


def complexity_table(bundle, profiles, jobs=pilot.JOBS):
    return pilot.markdown_table(
        ["Job", "Variant", "Target", "Parameters", "Param. ↓", "MACs/image", "MACs ↓", "Parent epoch", "Receipt digest¹"],
        [[r["job"], r["variant"], f"{r['target_pruning_percent']}%", f"{r['parameters']:,}",
          f"{r['parameter_reduction_percent']:.2f}%", f"{r['macs_per_image']:,}",
          f"{r['mac_reduction_percent']:.2f}%", str(r["parent_checkpoint_epoch"]), f"`{r['canonical_receipt_sha256'][:12]}`"]
         for r in complexity_rows(bundle, profiles, jobs)])


def recovery_snapshot(protocol, sidecar):
    return {"campaign": pilot.FOLDER.name, "job": E5, "observed_date": "2026-11-21",
            "observed_via": "authenticated JupyterLab directory listing and JSON downloads",
            "source_files": {RECOVERY_PROTOCOL: member(protocol), EPOCH_ONE: member(sidecar)},
            "listing": {"recovered": ["progress", "recovery_protocol.json"],
                        "recovered/progress": ["pruned_10"],
                        "recovered/progress/pruned_10": ["epoch_01"],
                        "recovered/progress/pruned_10/epoch_01": ["state.pt", "state.pt.sha256.json"]},
            "validation_recovered_directory_present": False,
            "variant_status": {"original": "UNVERIFIED", "pruned_10": "PARTIAL",
                               "pruned_30": "UNVERIFIED", "pruned_50": "UNVERIFIED"}}


def verify_recovery(bundle, profiles, data):
    check_public(data)
    protocol = check_member(data["source_files"][RECOVERY_PROTOCOL])
    sidecar = check_member(data["source_files"][EPOCH_ONE])
    expected = recovery_snapshot(pilot.json_bytes(protocol), pilot.json_bytes(sidecar))
    report = bundle["jobs"][E5]["validation"]
    report_sha = bundle["jobs"][E5]["original_members"][f"{E5}/validation/report.json"]["sha256"]
    require(data == expected and protocol == {
        "schema": "convnext-optimization-recovery-v1", "protocol_sha256": report["protocol_sha256"],
        "framework_source_sha256": pilot.source_hashes(RECOVERY_REVISION), "validation_report_sha256": report_sha,
        "parent_receipts_sha256": {str(ratio): profiles[E5][variant]["source_sha256"]
                                   for variant, ratio in pilot.VARIANTS.items()},
        "policy": {"device": "cpu", "epochs": 3, "gpu_uuid": None, "micro_batch": 4, "threads": 1},
    }, "E5 recovery snapshot or parent lineage differs")
    require(set(sidecar) == {"bytes", "sha256"} and type(sidecar["bytes"]) is int and sidecar["bytes"] > 0,
            "E5 epoch checkpoint sidecar is invalid")
    pilot.require_hash(sidecar["sha256"])
    return data


def notebook_text(bundle, profiles, job, recovery):
    relative = f"../../{pilot.FOLDER.as_posix()}"
    interpretation = (
        "The recovered 10% arm has a saved epoch-1 checkpoint receipt and is `PARTIAL`. "
        "The 0%, 30%, and 50% arms remain `UNVERIFIED`. The audit found no aligned post-recovery validation report."
        if job == E5 and recovery["variant_status"]["pruned_10"] == "PARTIAL"
        else "The final recovery audit is scoped to E5 APTOS2019.")
    return (
        f"[Verified profiles]({relative}/complexity.csv) match the prepared digests of this pilot.\n\n"
        + complexity_table(bundle, profiles, [job])
        + "\n\n¹ The table shows the first 12 characters of the canonical prepared SHA-256. "
        "The CSV preserves full receipt and parent-checkpoint hashes. The reported MACs count executed Conv2d "
        "and Linear operations per image. E7 counts the shared weights once and the calls across all three views.\n\n"
        "![Pre-recovery Macro-F1 versus recorded MACs](../../docs/figures/pruning_pilot_quality_complexity.png)\n\n"
        + interpretation + " Shared CPU validation recorded no timing, so latency is `NOT_MEASURED`."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Read only prepared*.json and two named E5 JSON receipts")
    parser.add_argument("--write", type=Path, help="Write supplementary public evidence and CSV")
    parser.add_argument("--figures", type=Path, help="Regenerate only the three complexity figures")
    args = parser.parse_args(argv)
    folder = pilot.ROOT / pilot.FOLDER
    bundle = pilot.verify(parse((folder / "validation_evidence.json").read_bytes()))
    if args.source_dir:
        profiles = transport_profiles(bundle, sorted(args.source_dir.glob("prepared*.json")))
        recovery = recovery_snapshot((args.source_dir / "recovery_protocol.json").read_bytes(),
                                     (args.source_dir / "state.pt.sha256.json").read_bytes())
    else:
        profiles = read_profiles(bundle, folder)
        recovery = parse((folder / RECOVERY_FILE).read_bytes())
    verify_recovery(bundle, profiles, recovery)
    rows = pilot.csv_bytes(complexity_rows(bundle, profiles))
    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        for dataset in pilot.DATASETS:
            (args.write / dataset_file(dataset)).write_bytes(profile_bytes(profiles, dataset))
        (args.write / RECOVERY_FILE).write_bytes(pilot.json_bytes(recovery))
        (args.write / "complexity.csv").write_bytes(rows)
    if not args.source_dir:
        require((folder / "complexity.csv").read_bytes() == rows, "Published complexity table differs")
    if args.figures:
        pilot.plot_complexity(bundle, list(complexity_rows(bundle, profiles)), args.figures)
    print("PASS: 20 prepared profiles and E5 recovery parent binding. Snapshot records one partial arm and no validated recovery.")


if __name__ == "__main__":
    main()
