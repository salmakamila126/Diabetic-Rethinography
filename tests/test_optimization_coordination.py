"""CPU-only policy tests and Linux multiprocess flock regression tests."""

from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from novelty_integrated.optimization import coordination as opt


def request(job, cpu, limit=6, *, core=None, fs=1):
    return {"job": job, "cpus": [cpu], "physical_core_siblings": [core or [cpu]],
            "threads": 1, "max_workers": limit, "ram_bytes": 6, "disk_bytes": 8,
            "disk_reserve_bytes": 5, "filesystem_device": fs}


def resources(**overrides):
    return {"available_ram_bytes": 100, "free_disk_bytes": 100, "free_inodes": 2000,
            "cpu_quota": 12, "cpu_quota_known": True,
            "pressure": {"host_pressure_known": True, "cpu_some_avg10_pct": 0,
                         "memory_full_avg10_pct": 0, "io_full_avg10_pct": 0}, **overrides}


def work(host, artifact, info):
    return opt.workload_locks(host, artifact, stage="recover", job=info["job"], cpus=info["cpus"],
                              request=info, read_resources=resources)


def contender(host, artifact, info, start, finish, queue):
    if not start.wait(10):
        return
    try:
        with work(host, artifact, info) as lease:
            queue.put(("admitted", lease["lease"]["slot"]))
            finish.wait(15)
    except opt.RecoveryDeferred:
        queue.put(("deferred", None))
    except opt.LockBusy as exc:
        queue.put(("busy", str(exc)))


class AdmissionPolicyTests(unittest.TestCase):
    def test_reserves_ram_disk_and_cpu_quota_against_peer_demands(self):
        waiting = request("B", 10)
        running = {**request("A", 9), "cpu_quota": 4}
        for expected, snapshot in ((1, resources(available_ram_bytes=11)),
                                   (1, resources(free_disk_bytes=20)),
                                   (1, resources(free_inodes=199)),
                                   (1, resources(cpu_quota=1)),
                                   (1, resources(cpu_quota_known=False)),
                                   (2, resources(available_ram_bytes=12, free_disk_bytes=21, cpu_quota=2))):
            with self.subTest(snapshot=snapshot):
                self.assertEqual(opt.capacity(waiting, [running], snapshot)["capacity"], expected)
        self.assertEqual(opt.capacity(waiting, [], resources(available_ram_bytes=5))["capacity"], 0)
        self.assertEqual(opt.capacity(waiting, [], resources(free_disk_bytes=12))["capacity"], 0)
        other_disk = {**running, "filesystem_device": 2}
        self.assertEqual(opt.capacity(waiting, [other_disk], resources(free_disk_bytes=13))["capacity"], 2)
        running["disk_reserve_bytes"] = 6
        self.assertEqual(opt.capacity(waiting, [running], resources(free_disk_bytes=21))["capacity"], 1)

    def test_io_memory_and_cpu_pressure_back_off(self):
        snapshot = resources()
        waiting = request("job", 10)
        for signal, value, expected in (("cpu_some_avg10_pct", 25, 1), ("memory_full_avg10_pct", 0.5, 1),
                                        ("io_full_avg10_pct", 2, 1), ("io_full_avg10_pct", 10, 0),
                                        ("memory_full_avg10_pct", 2, 0), ("cpu_some_avg10_pct", 60, 0),
                                        ("host_pressure_known", False, 1)):
            with self.subTest(signal=signal, value=value):
                signals = {**snapshot["pressure"], signal: value}
                self.assertEqual(opt.capacity(waiting, [], {**snapshot, "pressure": signals})["capacity"], expected)

    def test_sysfs_topology_and_cgroup_pressure_are_read_conservatively(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu, siblings in ((2, "2,130"), (130, "2,130"), (4, "4,132")):
                path = root / f"cpu{cpu}/topology"
                path.mkdir(parents=True)
                (path / "thread_siblings_list").write_text(siblings)
            self.assertEqual(opt.physical_cores([130, 4], root), [[2, 130], [4, 132]])
            with self.assertRaisesRegex(ValueError, "different physical core"):
                opt.physical_cores([2, 130], root)
            with self.assertRaisesRegex(ValueError, "Cannot verify"):
                opt.physical_cores([5], root)
            group = root / "cgroup"
            group.mkdir()
            (root / "io").write_text("some avg10=1.00\nfull avg10=1.00 avg60=0.00\n")
            (group / "io.pressure").write_text("some avg10=22.00\nfull avg10=12.00 avg60=0.00\n")
            signals = opt.pressure([group], root)
            self.assertFalse(signals["host_pressure_known"])
            self.assertEqual(signals["io_full_avg10_pct"], 12)
            self.assertEqual(opt.capacity(request("job", 4), [], resources(pressure=signals))["capacity"], 0)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires kernel POSIX flock")
class ProcessLockTests(unittest.TestCase):
    @contextmanager
    def roots(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("host", "artifact1", "artifact2")]
            for path in paths:
                path.mkdir()
            yield paths

    def test_concurrent_race_admits_exactly_each_limit_from_one_to_six(self):
        ctx = multiprocessing.get_context("fork")
        for limit in range(1, 7):
            with self.subTest(limit=limit), self.roots() as (host, a, b):
                start, finish, queue = ctx.Event(), ctx.Event(), ctx.Queue()
                children = [ctx.Process(target=contender, args=(host, a if i % 2 else b, request(str(i), i, limit),
                                                                start, finish, queue)) for i in range(7)]
                try:
                    for child in children:
                        child.start()
                    start.set()
                    reports = [queue.get(timeout=10) for _ in children]
                    slots = [slot for result, slot in reports if result == "admitted"]
                    self.assertEqual(sorted(slots), list(range(1, limit + 1)))
                    self.assertEqual(reports.count(("deferred", None)), 7 - limit)
                    self.assertEqual(len(opt.status(host)), limit)
                finally:
                    finish.set()
                    for child in children:
                        if child.pid is not None:
                            child.join(timeout=5)
                            if child.is_alive():
                                child.terminate()
                                child.join(timeout=5)
                    queue.close()
                self.assertEqual(opt.status(host), [])

    def test_job_core_and_measure_locks_cover_multiple_artifact_roots(self):
        with self.roots() as (host, a, b), work(host, a, request("E5_DDR", 9, core=[9, 137])):
            with self.assertRaisesRegex(opt.LockBusy, "job-E5_DDR"):
                with work(host, b, request("E5_DDR", 10)):
                    self.fail("same job entered twice")
            for cpu in (9, 137):
                with self.assertRaisesRegex(opt.LockBusy, "recovery-core-9"):
                    with work(host, b, request("E6_DDR", cpu, core=[9, 137])):
                        self.fail("same core/sibling entered twice")
            for stage in ("measure", "measure-recovered", "prepare"):
                with self.assertRaises(opt.LockBusy):
                    with opt.workload_locks(host, b, stage=stage, job="other", cpus=[11]):
                        self.fail(f"overlapping {stage}")
            # Older programs use only the artifact-root locks. Their timing and prepare remain fenced.
            with self.assertRaises(opt.LockBusy), opt.lock(a / "cpu-coordinator"):
                self.fail("legacy measurement entered")
            for slot in range(2):
                with self.assertRaises(opt.LockBusy), opt.lock(a / f"prepare-{slot}"):
                    self.fail("legacy prepare entered")

    def test_mixed_limits_and_atomic_rollback_on_partial_admission_failure(self):
        with self.roots() as (host, a, b):
            with work(host, a, request("first", 9, limit=1)):
                with self.assertRaises(opt.RecoveryDeferred), work(host, b, request("second", 10)):
                    self.fail("increased ceiling of an active wave")
            with work(host, a, request("first", 9)):
                conflicting = {**request("second", 10), "physical_core_siblings": [[10], [9]]}
                with self.assertRaises(opt.LockBusy), work(host, b, conflicting):
                    self.fail("partially locked core set admitted")
                with patch.object(Path, "replace", side_effect=OSError("synthetic write failure")):
                    with self.assertRaisesRegex(OSError, "write failure"), work(host, b, request("second", 10)):
                        self.fail("lease without metadata admitted")
                self.assertEqual(len(opt.status(host)), 1)
                with work(host, b, request("second", 10)):
                    self.assertEqual(len(opt.status(host)), 2)

    def test_resource_deferral_does_not_reserve_a_job_core_or_slot(self):
        with self.roots() as (host, a, b), work(host, a, request("first", 9)):
            waiting = request("second", 10)
            for overrides, limit in (({"available_ram_bytes": 11}, "ram"), ({"free_disk_bytes": 20}, "disk")):
                with self.assertRaises(opt.RecoveryDeferred) as blocked:
                    with opt.workload_locks(host, b, stage="recover", job="second", cpus=[10], request=waiting,
                                            read_resources=lambda: resources(**overrides)):
                        self.fail("oversubscribed resources admitted")
                self.assertEqual(blocked.exception.admission["limits"][limit], 1)
                self.assertEqual(len(opt.status(host)), 1)
            with work(host, b, waiting) as lease:
                self.assertEqual(lease["lease"]["slot"], 2)

    def test_process_crash_releases_locks_and_leaves_completed_epoch_untouched(self):
        ctx = multiprocessing.get_context("fork")
        with self.roots() as (host, a, b):
            start, finish, queue = ctx.Event(), ctx.Event(), ctx.Queue()
            checkpoint = a / "completed-epoch"
            checkpoint.write_text("epoch state")
            info = request("E7_APTOS2019", 9, limit=1)
            child = ctx.Process(target=contender, args=(host, a, info, start, finish, queue))
            child.start()
            start.set()
            try:
                self.assertEqual(queue.get(timeout=10), ("admitted", 1))
                self.assertEqual(opt.status(host)[0]["pid"], child.pid)
            finally:
                child.kill()
                child.join(timeout=5)
                queue.close()
            self.assertFalse(child.is_alive())
            self.assertEqual(opt.status(host), [])
            with work(host, a, info) as lease:
                self.assertEqual(lease["lease"]["slot"], 1)
                self.assertEqual(checkpoint.read_text(), "epoch state")


if __name__ == "__main__":
    unittest.main()
