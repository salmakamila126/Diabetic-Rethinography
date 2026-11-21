"""HPC campaign orchestration. Inventory and audit never launch a workload."""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from . import backends, models, registry
from .__main__ import _parser as run_parser, _plan

GIB = 1024**3
PRIORITY = ("E5_APTOS2019", "E6_APTOS2019", "E7_APTOS2019", "E5_DDR", "E6_DDR")
STAGES = ("audit", "prepare", "validate", "measure", "recover", "measure-recovered")


def inventory():
    """Derive jobs, paths and expected digests from the published registry evidence."""
    entries = registry.registry()
    jobs = {f"{arm}_{dataset}": entry for (dataset, arm), entry in entries.items()
            if entry.status != "NOT_APPLICABLE"}
    return {key: jobs[key] for key in (*PRIORITY, *sorted(jobs.keys() - set(PRIORITY)))}


def checkpoint_relative(entry):
    if not entry.evidence:
        return None
    value = (entry.evidence.get("source_external_relative_to_Kel_13")
             or entry.evidence["provenance"]["source_location_in_hpc"])
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Evidence has an invalid relative checkpoint path")
    return (path / "best.pt").as_posix()


def receipt(entry):
    return {**entry.describe(), "checkpoint_relative_to_root": checkpoint_relative(entry),
            "scope": "frozen validation only",
            "validation_images": entry.configuration["manifest"]["counts"]["val"] if entry.evidence else None}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", *STAGES))
    parser.add_argument("--job", choices=tuple(inventory()))
    parser.add_argument("--root", type=Path, default=os.environ.get("CADVIT_ARTIFACT_ROOT"))
    parser.add_argument("--campaign-root", type=Path, default=os.environ.get("CADVIT_OPT_CAMPAIGN"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--external-source-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpus", help="Allocated Linux logical CPU IDs, e.g. 4-7")
    parser.add_argument("--allocation-id", help="Site allocation identifier or lab approval reference")
    parser.add_argument("--confirm-isolated-cpu", action="store_true")
    parser.add_argument("--shared-cpu", action="store_true", help="Untimed validate stage only, with taskset and one thread")
    parser.add_argument("--min-free-gib", type=int, default=5, help="Filesystem reserve after new outputs")
    parser.add_argument("--min-ram-gib", type=int, default=6, help="Available host/cgroup RAM floor per job")
    parser.add_argument("--sample-count", type=int, help="Omit for full frozen validation")
    parser.add_argument("--backend", nargs="+", choices=backends.BACKENDS, default=list(backends.BACKENDS))
    parser.add_argument("--recovery-epochs", type=int, choices=(1, 2, 3))
    parser.add_argument("--recovery-micro-batch", type=int, choices=(1, 4, 8, 16, 32), default=4)
    parser.add_argument("--recovery-device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--authorize-recovery", action="store_true")
    parser.add_argument("--confirm-ddr-e7-safe", action="store_true")
    parser.add_argument("--gpu-uuid", help="For approved CUDA recovery, use one explicitly allocated GPU UUID")
    return parser


def _available_ram():
    if not Path("/proc/meminfo").is_file():
        return None
    lines = Path("/proc/meminfo").read_text().splitlines()
    available = int(next(l.split()[1] for l in lines if l.startswith("MemAvailable:"))) * 1024
    for group in _cgroups():
        maximum, current = group / "memory.max", group / "memory.current"
        if maximum.is_file() and current.is_file() and maximum.read_text().strip() != "max":
            available = min(available, int(maximum.read_text()) - int(current.read_text()))
    return available


def _cgroups():
    cpu_root = Path("/sys/fs/cgroup")
    source = Path("/proc/self/cgroup")
    if not source.is_file():
        return ()
    member = next((line.split(":", 2)[-1] for line in source.read_text().splitlines()
                   if line.startswith("0::")), "").lstrip("/")
    group = cpu_root / member
    return (group, *(p for p in group.parents if p == cpu_root or cpu_root in p.parents))


def _cpu_quota():
    limits = []
    for group in _cgroups():
        cpu_max = group / "cpu.max"
        if cpu_max.is_file():
            quota, period = cpu_max.read_text().split()
            if quota != "max":
                limits.append(int(quota) / int(period))
    return min(limits) if limits else None


def _disk(path):
    real = path.resolve()
    for parent in (real, *real.parents):
        if parent.is_dir():
            return shutil.disk_usage(parent).free
    raise ValueError("No existing filesystem ancestor")


def _affinity():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def _parse_cpus(value):
    members = set()
    for item in value.split(","):
        numbers = [int(p) for p in item.split("-")]
        if not 1 <= len(numbers) <= 2 or not 0 <= numbers[0] <= numbers[-1] < 65536:
            raise ValueError("Invalid CPU list")
        members.update(range(numbers[0], numbers[-1] + 1))
    return sorted(members)


def _arguments(args, entry):
    if not args.root or not args.campaign_root or not args.job:
        raise ValueError("Provide --root, --campaign-root and --job")
    root, campaign = args.root.expanduser().resolve(), args.campaign_root.expanduser().resolve()
    workspace = root / "optimization_runs"
    if workspace not in campaign.parents:
        raise ValueError("Use a unique campaign directory under <root>/optimization_runs")
    if any(name.startswith(".") for name in campaign.relative_to(workspace).parts):
        raise ValueError("Campaign paths cannot use hidden coordinator directories")
    location = checkpoint_relative(entry)
    checkpoint = args.checkpoint or (root / location if location else None)
    rev = entry.configuration["snapshot"]["revision"] if entry.evidence else None
    snapshot = args.snapshot or (root / "hf_foundation/models--facebook--convnext-tiny-224/snapshots" / rev
                                 if rev else None)
    sidecar = root / "cad_vit_runs" / "manifests" / f"{entry.dataset}_seed42.json"
    return root, campaign, checkpoint, snapshot, sidecar


def audit(args, entry):
    root, campaign, checkpoint, snapshot, manifest = _arguments(args, entry)
    resources = {"cpu_affinity": _affinity(), "cgroup_cpu_quota": _cpu_quota(),
                 "available_ram_bytes": _available_ram(), "free_campaign_bytes": _disk(campaign)}
    capabilities = backends.capabilities()
    selected = list(dict.fromkeys(["eager", *args.backend]))
    missing = {name: capabilities[name]["missing_packages"] for name in selected
               if capabilities[name]["missing_packages"]}
    report = {"job": args.job, "experiment": receipt(entry), "resources": resources,
              "requested_backends": selected, "backend_missing_packages": missing,
              "checkpoint": str(checkpoint) if checkpoint else None, "manifest_sidecar": str(manifest),
              "snapshot": str(snapshot) if snapshot else None, "campaign": str(campaign),
              "checks": {}}
    checks = report["checks"]

    def check(name, func):
        try:
            func()
            checks[name] = {"status": "PASS"}
        except (OSError, ValueError, KeyError, PackageNotFoundError) as exc:
            checks[name] = {"status": "BLOCKED", "reason": str(exc)}

    def verify_checkpoint():
        binary = models.verify_receipt(entry, checkpoint)
        sidecar = json.loads(binary.with_name("best.pt.sha256.json").read_text(encoding="utf-8"))
        if sidecar != {"sha256": entry.checkpoint_sha256, "bytes": binary.stat().st_size}:
            raise ValueError("Checkpoint sidecar or binary length differs")

    def verify_snapshot():
        identity = entry.configuration["snapshot"]
        actual = fa.audit_snapshot("convnext_tiny", snapshot, identity["revision"], identity["weight_sha256"])
        if any(actual.get(key) != val for key, val in identity.items()):
            raise ValueError("Official snapshot metadata differs from evidence")

    def verify_environment():
        if version("transformers") != "4.57.3":
            raise ValueError("Use the pinned Transformers 4.57.3 environment")
        pillow = entry.configuration.get("pillow") if entry.evidence else None
        if pillow and version("Pillow") != pillow:
            raise ValueError(f"Use the recorded Pillow {pillow} processor environment")
        if missing:
            raise ValueError("Selected backend dependencies are absent")

    def verify_manifest():
        source = json.loads(manifest.read_text(encoding="utf-8"))["root"]
        recorded = entry.configuration["manifest"].get("manifest_sidecar_sha256")
        if recorded:
            phase.require_hash(manifest, recorded)
        models.load_rows(entry, root, source)

    if entry.evidence:
        check("sources", lambda: registry.require_sources(entry, args.external_source_dir))
        check("checkpoint_binary_and_sidecar", verify_checkpoint)
        check("official_snapshot", verify_snapshot)
        check("frozen_manifest_without_image_decode", verify_manifest)
    else:
        checks["experiment"] = {"status": "BLOCKED", "reason": entry.status}
    check("software", verify_environment)
    report["status"] = "AUDIT_PASS" if all(x["status"] == "PASS" for x in checks.values()) else "BLOCKED"
    return report


@contextmanager
def _lock(path, *, shared=False):
    import fcntl

    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Busy campaign resource: {path.name}. Wait or resume later.") from exc
        yield


def _resource_guard(args, campaign):
    if not args.allocation_id or not args.cpus or not hasattr(os, "sched_setaffinity"):
        raise ValueError("Execution requires Linux, --allocation-id and an allocated --cpus list")
    if (args.stage.startswith("measure") and not args.confirm_isolated_cpu):
        raise ValueError("Measurement requires an exclusive CPU allocation, confirmed explicitly")
    cpus = _parse_cpus(args.cpus)
    affinity = _affinity()
    if args.stage == "validate":
        if not args.shared_cpu or args.confirm_isolated_cpu:
            raise ValueError("Use --shared-cpu without --confirm-isolated-cpu for untimed validation")
        if args.threads != 1 or len(cpus) != 1 or cpus != affinity:
            raise ValueError("Start validation with taskset -c on one allowed logical CPU and use --threads 1")
    elif args.shared_cpu:
        raise ValueError("--shared-cpu applies only to untimed validation")
    if args.threads < 1 or len(cpus) < args.threads or not set(cpus) <= set(affinity):
        raise ValueError("Requested threads and CPUs exceed the process allocation")
    quota = _cpu_quota()
    if quota is not None and quota < args.threads:
        raise ValueError("CPU cgroup quota is below the requested threads")
    if args.recovery_epochs and args.stage != "recover":
        raise ValueError("Recovery settings apply only to the recover stage")
    ram = _available_ram()
    # Resource floors are conservative policy settings, not measured consumption.
    additional_disk = {"prepare": 6, "recover": 3}.get(args.stage, 1)
    if args.min_ram_gib < 1 or ram is None or ram < args.min_ram_gib * GIB:
        raise ValueError("Insufficient available RAM or unreadable RAM limit")
    if args.min_free_gib < 1 or _disk(campaign) < (args.min_free_gib + additional_disk) * GIB:
        raise ValueError("Insufficient disk above the configured reserve")
    os.sched_setaffinity(0, cpus)
    return cpus


def _gpu_guard(args):
    if args.stage != "recover" or args.recovery_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if not args.confirm_ddr_e7_safe or not args.gpu_uuid or not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("GPU recovery needs an allocated idle GPU UUID and explicit DDR E7 safety confirmation")
    available = subprocess.check_output(["nvidia-smi", "-i", args.gpu_uuid,
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    if int(available) < 8192:
        raise ValueError("GPU recovery requires at least 8 GiB free as a conservative guard")
    active = subprocess.check_output(["nvidia-smi", "-i", args.gpu_uuid,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if active:
        raise ValueError("The recovery GPU is in use")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid


def main(argv=None):
    args = _parser().parse_args(argv)
    jobs = inventory()
    if args.stage == "inventory":
        print(json.dumps({name: receipt(entry) for name, entry in jobs.items()}, indent=2))
        return
    if not args.job:
        raise SystemExit("--job is required")
    entry = jobs[args.job]
    if args.stage == "audit":
        report = audit(args, entry)
        print(json.dumps(report, indent=2))
        if report["status"] != "AUDIT_PASS":
            raise SystemExit(2)
        return
    if args.stage == "recover" and (entry.arm not in registry.INTEGRATED or not args.authorize_recovery
                                    or not args.recovery_epochs):
        raise ValueError("Recovery requires explicit authorization, epochs and an E5/E6 job")
    if args.stage != "recover" and (args.authorize_recovery or args.recovery_epochs or args.gpu_uuid
                                    or args.confirm_ddr_e7_safe or args.recovery_device != "cpu"):
        raise ValueError("Recovery flags cannot be used in another stage")
    root, campaign, checkpoint, snapshot, manifest = _arguments(args, entry)
    if not entry.evidence:
        raise ValueError(f"{args.job}: {entry.status}")
    if args.stage == "validate" and not (campaign / args.job / "protocol.json").is_file():
        raise FileNotFoundError("Validation requires an existing campaign/job protocol.json from prepare")
    cpus = _resource_guard(args, campaign)
    # Guard paths against datasets, model snapshots, checkpoint directories and Git worktrees.
    fa._safe_out(campaign / args.job, [checkpoint.parent, snapshot, root / "datasets", root / "cad_vit_runs"])
    lock_dir = root / "optimization_runs/.locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(_lock(lock_dir / "cpu-coordinator", shared=args.stage == "prepare"))
        if args.stage == "prepare":
            for slot in range(2):
                try:
                    stack.enter_context(_lock(lock_dir / f"prepare-{slot}"))
                    break
                except RuntimeError:
                    if slot == 1:
                        raise
            for cpu in cpus:
                stack.enter_context(_lock(lock_dir / f"cpu-{cpu}"))
        stack.enter_context(_lock(lock_dir / f"job-{args.job}"))
        report = audit(args, entry)
        if report["status"] != "AUDIT_PASS":
            print(json.dumps(report, indent=2))
            raise SystemExit(2)
        _gpu_guard(args)
        source_root = json.loads(manifest.read_text(encoding="utf-8"))["root"]
        cli = ["run", "--dataset", entry.dataset, "--arm", entry.arm, "--artifact-root", str(root),
               "--model-dir", str(snapshot), "--checkpoint", str(checkpoint), "--output", str(campaign / args.job),
               "--manifest-source-root", source_root, "--threads", str(args.threads), "--backend", *args.backend,
               "--trust-checkpoint", "--trust-legacy-snapshot"]
        if args.external_source_dir:
            cli += ["--external-source-dir", str(args.external_source_dir)]
        if args.sample_count is not None:
            cli += ["--sample-count", str(args.sample_count)]
        if args.stage == "recover":
            cli += ["--authorize-recovery", "--recovery-epochs", str(args.recovery_epochs),
                    "--recovery-micro-batch", str(args.recovery_micro_batch), "--recovery-device", args.recovery_device,
                    "--prune", "0", "0.1", "0.3", "0.5"]
            if args.recovery_device == "cuda:0":
                cli.append("--acknowledge-shared-gpu")
        run = run_parser().parse_args(cli)
        plan = _plan(run, entry)
        campaign.mkdir(parents=True, exist_ok=True)
        from .workflow import execute

        allocation = {"id": args.allocation_id, "cpus": cpus, "threads": args.threads,
                      "exclusive_cpu_confirmed": args.confirm_isolated_cpu}
        if args.stage == "validate":
            allocation["cpu_mode"] = "shared"
        return execute(args.stage, run, entry, plan, allocation)


if __name__ == "__main__":
    raise SystemExit(main())
