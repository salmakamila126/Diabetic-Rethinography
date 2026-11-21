"""Quality and latency on identical ordered validation inputs."""

import hashlib
from importlib import metadata
import math
import os
import platform
import time

import numpy as np
import torch

from novelty import foundation_adaptation as fa
from novelty.preof_phase1 import sha256


def _tensor_digest(digest, values):
    for tensor in values:
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.contiguous().numpy().tobytes())


@torch.inference_mode()
def collect_logits(infer, batches, prepare):
    result = []
    digest = hashlib.sha256()
    for batch in batches:
        inputs = prepare(batch)
        _tensor_digest(digest, inputs)
        logits = torch.as_tensor(infer(*inputs)).cpu()
        if logits.dtype != torch.float32 or logits.shape != (len(batch), 5) or not torch.isfinite(logits).all():
            raise ValueError("Backend did not produce finite FP32 categorical logits in Nx5 grade order")
        result.append(logits.clone())
    return torch.cat(result), digest.hexdigest()


def quality(logits, batches):
    labels = [row["label"] for batch in batches for row in batch]
    return fa._classification(labels, logits.softmax(dim=1).numpy())


def parity(reference, candidate, *, atol=1e-4, rtol=1e-4):
    if reference.shape != candidate.shape or not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("Parity requires aligned outputs and finite nonnegative tolerances")
    absolute = (candidate.double() - reference.double()).abs()
    mismatches = torch.where(candidate.argmax(dim=1) != reference.argmax(dim=1))[0].tolist()
    passed = bool(torch.all(absolute <= atol + rtol * reference.abs().double())) and not mismatches
    return {"passed": passed, "atol": atol, "rtol": rtol, "maximum_absolute": float(absolute.max()),
            "mean_absolute": float(absolute.mean()), "rms": float(absolute.square().mean().sqrt()),
            "maximum_relative_reference_floor_1e-6": float((absolute / reference.abs().clamp_min(1e-6)).max()),
            "prediction_agreement": 1 - len(mismatches) / len(reference), "changed_order_indices": mismatches}


def quality_delta(reference, candidate):
    fields = ("accuracy", "macro_f1", "weighted_f1", "quadratic_weighted_kappa", "grade_mae",
              "undergraded", "overgraded", "wrong")
    return {**{key: candidate[key] - reference[key] if None not in (candidate[key], reference[key]) else None
              for key in fields},
            "per_grade_recall": [new["recall"] - old["recall"]
                                 for old, new in zip(reference["per_grade"], candidate["per_grade"], strict=True)]}


def _summary(ns, batch_size):
    ms = np.asarray(ns, dtype=float) / 1_000_000
    return {"median_ms_batch": float(np.median(ms)), "mean_ms_batch": float(np.mean(ms)),
            "p95_ms_batch": float(np.percentile(ms, 95)), "median_ms_per_image": float(np.median(ms) / batch_size),
            "mean_ms_per_image": float(np.mean(ms) / batch_size),
            "p95_ms_per_image": float(np.percentile(ms, 95) / batch_size),
            "images_per_second": float(batch_size * len(ns) / (ms.sum() / 1000))}


@torch.inference_mode()
def time_backend(infer, batches, prepare, *, warmup, repeats, scope):
    if warmup < 1 or repeats < 2 or scope not in ("model_only", "end_to_end"):
        raise ValueError("Require warm-up, repeated passes and an explicit timing scope")
    batch_size = len(batches[0])
    if not batch_size or any(len(batch) != batch_size for batch in batches):
        raise ValueError("Only fixed-size batches are timed")
    example = prepare(batches[0])
    for _ in range(warmup):
        infer(*example)
    del example
    passes = []
    for _ in range(repeats):
        wall_start = time.perf_counter_ns()
        samples = []
        for batch in batches:
            inputs = prepare(batch) if scope == "model_only" else None
            start = time.perf_counter_ns()
            if inputs is None:
                inputs = prepare(batch)
            infer(*inputs)  # CPU backends use synchronous calls.
            samples.append(time.perf_counter_ns() - start)
        wall = time.perf_counter_ns() - wall_start
        passes.append({"latency_ns": samples, "pass_wall_ns": wall, **_summary(samples, batch_size)})
    return {"scope": scope, "passes": passes, **_summary([ns for p in passes for ns in p["latency_ns"]], batch_size)}


def runtime():
    versions = {}
    for name in ("torch", "transformers", "numpy", "Pillow", "onnx", "onnxruntime", "openvino", "psutil"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    cpu = platform.processor() or platform.machine()
    if not platform.processor() and os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            cpu = next((line.partition(":")[2].strip() for line in stream if line.startswith("model name")), cpu)
    return {"os": platform.system(), "os_release": platform.release(), "machine": platform.machine(), "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                  "OPENBLAS_NUM_THREADS")},
            "versions": versions, "precision": "float32", "device": "cpu"}


def memory():
    try:
        import psutil
    except ImportError:
        return {"rss_bytes": None, "process_lifetime_peak_rss_bytes": fa._process_peak_rss_bytes()}
    info = psutil.Process().memory_info()
    return {"rss_bytes": info.rss,
            "process_lifetime_peak_rss_bytes": getattr(info, "peak_wset", None)
            or fa._process_peak_rss_bytes()}


def artifact_receipt(primary):
    paths = [primary]
    if primary.suffix == ".xml":
        paths.append(primary.with_suffix(".bin"))
    return {"files": [{"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in paths],
            "bytes": sum(p.stat().st_size for p in paths)}
