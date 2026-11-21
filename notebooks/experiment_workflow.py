"""Optional notebook orchestration of the repository's own experiment CLIs."""

import os
from pathlib import Path
import subprocess
import sys


CLI = {
    **{arm: "novelty.foundation_adaptation" for arm in ("B0", "B1", "B2")},
    **{arm: f"novelty_integrated.{arm.lower()}.runner" for arm in ("E5", "E6", "E7")},
}
STEPS = {
    **{arm: ("preflight", "smoke", "train") for arm in ("B0", "B1", "B2")},
    "E5": ("prepare-cache", "verify-inputs", "smoke", "train"),
    "E6": ("estimate-cache", "prepare-cache", "verify-inputs", "smoke", "benchmark", "train"),
    "E7": ("estimate-cache", "prepare-cache", "verify-inputs", "smoke", "benchmark", "train"),
}


def _required(env, key):
    value = env.get(key)
    if not value:
        raise ValueError(f"Set {key} explicitly for this action.")
    return value


def _external(path, root):
    path = Path(path).expanduser().resolve()
    if path == root or root in path.parents or path in root.parents or any(
            (p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Use a dedicated output/cache path outside the repository.")
    return path


def command(root, arm, dataset, step, config, *, env=None):
    """Build one authentic CLI command. The caller decides whether to execute it."""
    if arm not in STEPS or step not in STEPS[arm] or dataset not in ("APTOS2019", "DDR"):
        raise ValueError("This notebook has no executable workflow for the requested run.")
    env = os.environ if env is None else env
    snapshot = config["snapshot"]
    artifact = Path(_required(env, "DR_ARTIFACT_ROOT")).expanduser().resolve()
    model = Path(_required(env, "DR_CONVNEXT_SNAPSHOT")).expanduser().resolve()
    from novelty import foundation_adaptation as foundation

    protections = [artifact / "datasets", artifact / "cad_vit_runs" / "manifests", model]
    argv = [sys.executable, "-m", CLI[arm], step,
            "--dataset", dataset, "--artifact-root", str(artifact), "--model-dir", str(model)]
    if arm != "E7":
        argv += ["--revision", snapshot["revision"], "--weight-sha256", snapshot["weight_sha256"]]
    if arm.startswith("B"):
        argv += ["--model-kind", "convnext_tiny"]
    if step == "preflight":
        return argv

    # The repository's official snapshot uses legacy weights. Trust is always explicit.
    if env.get("DR_TRUST_LEGACY_BIN") == "1":
        argv.append("--trust-legacy-bin")
    cpu = step in ("estimate-cache", "prepare-cache", "verify-inputs")
    argv += ["--device", "cpu" if cpu else env.get("DR_GPU_DEVICE", "cuda:0"),
             "--precision", "fp32" if cpu else config["precision"]]
    micro = str(config["micro_batch"])
    accumulation = str(config["accumulation"])
    if arm.startswith("B"):
        argv += ["--experiment", arm, "--mode", "full"]
        if step == "smoke":
            argv += ["--batch-size", micro]
        else:
            argv += ["--micro-batch", micro, "--accumulation", accumulation,
                     "--epochs", str(config["epochs"]), "--backbone-lr", str(config["backbone_lr"]),
                     "--head-lr", str(config["head_lr"]), "--weight-decay", str(config["weight_decay"])]
    else:
        argv += ["--micro-batch", micro, "--accumulation", accumulation]
        if env.get("DR_MANIFEST_SOURCE_ROOT"):
            argv += ["--manifest-source-root", env["DR_MANIFEST_SOURCE_ROOT"]]
        transport = config.get("input_transport", {})
        if not (arm == "E5" and step == "prepare-cache"):
            workers = transport.get("num_workers", transport.get("workers", 0))
            argv += ["--num-workers", str(workers)]
            if transport.get("persistent_workers") and workers > 0:
                argv += ["--persistent-workers"]
            if transport.get("pin_memory"):
                argv += ["--pin-memory"]
            if transport.get("prefetch_factor") is not None:
                argv += ["--prefetch-factor", str(transport["prefetch_factor"])]
        if arm == "E5":
            argv += ["--arm", "integrated"]
            if step != "prepare-cache":
                argv += ["--decoded-cache-mib", str(transport.get("decoded_ram_mib_per_worker", 0))]
        if arm in ("E5", "E7") or env.get("DR_VALIDATION_CACHE"):
            val_cache = foundation._safe_out(_external(_required(env, "DR_VALIDATION_CACHE"), root), protections)
            argv += ["--val-cache-root" if arm == "E7" else "--validation-cache-root", str(val_cache)]
        if arm in ("E6", "E7"):
            train_cache = foundation._safe_out(_external(_required(env, "DR_TRAIN_CACHE"), root), protections)
            argv += ["--train-cache-root", str(train_cache)]
        if arm == "E7":
            if env.get("DR_REUSE_E6_TRAIN_CACHE") == "1":
                argv += ["--reuse-e6-train"]
            if not cpu:
                argv += ["--acknowledge-shared-gpu", "--init", "e5",
                         "--e5-checkpoint", _required(env, "DR_E5_CHECKPOINT"),
                         "--e5-checkpoint-sha256", config["e5_checkpoint_sha256"]]
                if env.get("DR_TRUST_E5_CHECKPOINT") == "1":
                    argv += ["--trust-e5-checkpoint"]
        if step == "prepare-cache":
            if arm == "E6":
                argv += ["--cache-max-gib", _required(env, "DR_CACHE_MAX_GIB")]
            if arm == "E7":
                argv += ["--val-cache-max-gib", _required(env, "DR_VALIDATION_CACHE_MAX_GIB"),
                         "--reserve-gib", _required(env, "DR_CACHE_RESERVE_GIB")]
                if env.get("DR_REUSE_E6_TRAIN_CACHE") != "1":
                    argv += ["--train-cache-max-gib", _required(env, "DR_CACHE_MAX_GIB")]
            return argv

    if step == "estimate-cache" and arm == "E7":
        return argv
    run = foundation._safe_out(_external(_required(env, "DR_NOTEBOOK_RUN_DIR"), root), protections)
    if step == "estimate-cache":
        # E6 requires an output argument even though its estimate only prints JSON.
        return argv + ["--output-dir", str(run / "cache_estimate")]
    output = run / {"verify-inputs": "input_preflight", "smoke": "smoke",
                    "benchmark": "benchmark", "train": "train"}[step]
    if output.exists():
        raise FileExistsError(f"Choose a fresh notebook output for {step}.")
    argv += ["--output-dir", str(output)]
    if arm.startswith("E") and step in ("smoke", "benchmark", "train"):
        argv += ["--input-preflight-report", str(run / "input_preflight" / "input_equivalence.json")]
    if step == "train" or (arm == "E7" and step == "benchmark"):
        argv += ["--smoke-report", str(run / "smoke" / "smoke.json")]
    if arm in ("E6", "E7") and step == "train":
        argv += ["--benchmark-report", str(run / "benchmark" / "benchmark.json")]
    return argv


def run_step(root, arm, dataset, step, config, switches):
    if step is None:
        print("Historical evidence view. No workflow step selected.")
        return
    if not switches.get(step):
        raise ValueError(f"Explicitly enable {step} before selecting it.")
    if step in ("smoke", "benchmark", "train") and os.environ.get("DR_GPU_ALLOCATION_CONFIRMED") != "1":
        raise ValueError("Confirm the allocated device and set DR_GPU_ALLOCATION_CONFIRMED=1.")
    argv = command(root, arm, dataset, step, config)
    env = os.environ.copy()
    if "--device" in argv and argv[argv.index("--device") + 1] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(argv, cwd=root, env=env, check=True)
