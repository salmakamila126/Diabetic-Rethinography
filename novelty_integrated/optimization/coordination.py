"""Linux process locks and resource admission for shared-host recovery."""

from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import socket
import stat
from tempfile import TemporaryDirectory
from uuid import UUID


MAX_RECOVERY_WORKERS = 6
TEMPORARY_FAILURE = 75


class LockBusy(RuntimeError):
    pass


class RecoveryDeferred(RuntimeError):
    def __init__(self, admission):
        self.admission = admission
        super().__init__("Recovery capacity is temporarily unavailable")


@contextmanager
def lock(path, *, shared=False, wait=False):
    import fcntl

    with path.open("a+b") as stream:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(stream, mode if wait else mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockBusy(f"Busy campaign resource: {path.name}. Wait or resume later.") from exc
        yield


def host_locks():
    """One lock namespace per Linux host boot and research account across artifact roots."""
    boot = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    path = Path("/var/tmp") / f"cadvit-opt-{os.getuid()}-{boot}"
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Host lock directory must be private, user-owned and locally accessible")
    return path


def parse_cpus(value):
    members = set()
    for item in value.split(","):
        ends = [int(p) for p in item.split("-")]
        if not 1 <= len(ends) <= 2 or not 0 <= ends[0] <= ends[-1] < 65536:
            raise ValueError("Invalid CPU list")
        members.update(range(ends[0], ends[-1] + 1))
    return sorted(members)


def physical_cores(cpus, sysfs=Path("/sys/devices/system/cpu")):
    groups = []
    for cpu in cpus:
        path = sysfs / f"cpu{cpu}/topology/thread_siblings_list"
        if not path.is_file():
            raise ValueError(f"Cannot verify physical core and SMT topology for logical CPU {cpu}")
        siblings = parse_cpus(path.read_text().strip())
        if cpu not in siblings or any(set(group).intersection(siblings) for group in groups):
            raise ValueError("Each requested logical CPU must belong to a different physical core")
        groups.append(siblings)
    return groups


def pressure(cgroups=(), proc=Path("/proc/pressure")):
    """Use the maximum of the visible host and cgroup PSI avg10 signals."""
    signals = {"host_pressure_known": True}
    for resource, scope in (("cpu", "some"), ("memory", "full"), ("io", "full")):
        def read(path):
            if path.is_file():
                row = next((line.split() for line in path.read_text().splitlines() if line.startswith(f"{scope} ")), [])
                value = next((float(x[6:]) for x in row if x.startswith("avg10=")), None)
                if value is not None and math.isfinite(value) and 0 <= value <= 100:
                    return value
            return None

        host = read(proc / resource)
        if host is None:
            signals["host_pressure_known"] = False
        visible = (host, *(read(path / f"{resource}.pressure") for path in cgroups))
        signals[f"{resource}_{scope}_avg10_pct"] = max((v for v in visible if v is not None), default=None)
    return signals


def _pressure_capacity(signals):
    # PSI cutoffs are conservative admission policy, not measured requirements of the models.
    thresholds = {"cpu_some_avg10_pct": (25.0, 60.0), "memory_full_avg10_pct": (0.5, 2.0),
                  "io_full_avg10_pct": (2.0, 10.0)}
    capacity = MAX_RECOVERY_WORKERS if signals["host_pressure_known"] else 1
    for key, (moderate, severe) in thresholds.items():
        value = signals.get(key)
        if value is not None and value >= severe:
            return 0
        if value is None or value >= moderate:
            capacity = 1
    return capacity


def capacity(request, active, snapshot):
    on_filesystem = [other for other in active if other["filesystem_device"] == request["filesystem_device"]]
    keep_free = max([request["disk_reserve_bytes"], *(other["disk_reserve_bytes"] for other in on_filesystem)])
    used_disk = sum(other["disk_bytes"] for other in on_filesystem)
    used_ram = sum(other["ram_bytes"] for other in active)
    used_threads = sum(other["threads"] for other in active)
    quotas = [item["cpu_quota"] for item in (snapshot, *active) if item["cpu_quota"] is not None]
    count = len(active)
    limits = {
        "configured": min([request["max_workers"], *(other["max_workers"] for other in active)]),
        "ram": count + max(0, ((snapshot["available_ram_bytes"] or 0) - used_ram) // request["ram_bytes"]),
        "disk": count + max(0, (snapshot["free_disk_bytes"] - keep_free - used_disk) // request["disk_bytes"]),
        "inodes": count + max(0, (snapshot["free_inodes"] - 100 * len(on_filesystem)) // 100),
        "quota": (count + max(0, math.floor((min(quotas) - used_threads) / request["threads"]))
                  if quotas else MAX_RECOVERY_WORKERS),
        "quota_telemetry": MAX_RECOVERY_WORKERS if snapshot["cpu_quota_known"] else 1,
        "pressure": _pressure_capacity(snapshot["pressure"]),
    }
    return {"capacity": min(MAX_RECOVERY_WORKERS, *limits.values()), "limits": limits,
            "active_count": count, "resources": snapshot}


def _slot(root, number):
    return root / f"recovery-slot-{number}"


def _active(root):
    """Call under the admission lock so a locked slot has already published its metadata."""
    active = []
    for number in range(1, MAX_RECOVERY_WORKERS + 1):
        path = _slot(root, number)
        try:
            with lock(path):
                continue
        except LockBusy:
            try:
                lease = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise LockBusy("Recovery slot is closing before metadata publication. Retry shortly.") from exc
            if lease["slot"] != number or lease["host"] != socket.gethostname():
                raise ValueError("Unexpected recovery slot metadata")
            active.append(lease)
    return active


def status(root):
    with lock(root / "recovery-admission", wait=True):
        return _active(root)


@contextmanager
def recovery_slot(root, request, read_resources):
    if not 1 <= request["max_workers"] <= MAX_RECOVERY_WORKERS:
        raise ValueError("Invalid global recovery worker limit")
    # Lock inodes are never renamed or removed. Metadata is a separate atomic publication.
    with ExitStack() as lifetime:
        with lock(root / "recovery-admission", wait=True):
            active = _active(root)
            admission = capacity(request, active, read_resources())
            if len(active) >= admission["capacity"]:
                raise RecoveryDeferred(admission)
            occupied_slots = {worker["slot"] for worker in active}
            number = next(i for i in range(1, MAX_RECOVERY_WORKERS + 1) if i not in occupied_slots)
            with ExitStack() as reservation:
                for siblings in request["physical_core_siblings"]:
                    reservation.enter_context(lock(root / f"recovery-core-{min(siblings)}"))
                path = _slot(root, number)
                reservation.enter_context(lock(path))
                lease = {**request, "pid": os.getpid(), "host": socket.gethostname(), "slot": number,
                         "cpu_quota": admission["resources"]["cpu_quota"]}
                with TemporaryDirectory(dir=root, prefix=".recovery-admission-") as scratch:
                    candidate = Path(scratch) / "lease.json"
                    candidate.write_text(json.dumps(lease), encoding="utf-8")
                    candidate.replace(path.with_suffix(".json"))
                lifetime.enter_context(reservation.pop_all())
        yield {"lease": lease, **admission}


@contextmanager
def workload_locks(host, artifact, *, stage, job, cpus, request=None, read_resources=None):
    with ExitStack() as stack:
        # The artifact-root names also coordinate with live workers on previous revisions.
        for root in (host, artifact):
            stack.enter_context(lock(root / "cpu-coordinator", shared=stage == "prepare" or request is not None))
            stack.enter_context(lock(root / f"job-{job}"))
            if stage == "prepare":
                for slot in range(2):
                    try:
                        stack.enter_context(lock(root / f"prepare-{slot}"))
                        break
                    except LockBusy:
                        if slot == 1:
                            raise
                for cpu in cpus:
                    stack.enter_context(lock(root / f"cpu-{cpu}"))
            elif request is not None:
                # Shared holds block both prepare slots, including legacy prepares on this artifact root.
                for slot in range(2):
                    stack.enter_context(lock(root / f"prepare-{slot}", shared=True))
        yield stack.enter_context(recovery_slot(host, request, read_resources)) if request is not None else None
