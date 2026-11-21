"""Dataset-specific receipts derived from the published validation evidence."""

from dataclasses import dataclass
import json
from pathlib import Path

from novelty import preof_phase1 as phase

ROOT = Path(__file__).resolve().parents[2]
CLASS_ORDER = (0, 1, 2, 3, 4)
DATASETS = ("APTOS2019", "DDR")
INTEGRATED = {"E5": "e5_integrated", "E6": "e6_ldam_ordinal_adjacent"}
STATUSES = frozenset(("SUPPORTED", "NEEDS_ADAPTER", "MISSING_CHECKPOINT",
                      "MISSING_SOURCE", "PENDING_TRAINING", "NOT_APPLICABLE", "UNVERIFIED", "COMPLETED"))
B1_SOURCE = "2b011e422c40f52a03a3d0aa89e0d18285875da1946058c265cc07747a61f300"


@dataclass(frozen=True)
class Experiment:
    dataset: str
    arm: str
    family: str | None
    status: str
    evidence_path: str | None = None
    evidence: dict | None = None
    external_source: tuple = ()

    @property
    def configuration(self):
        return self.evidence["configuration"]

    @property
    def best(self):
        return self.evidence.get("best") or self.evidence["summary"]["best"]

    @property
    def checkpoint_sha256(self):
        if "checkpoint_sidecars" in self.evidence:
            return self.evidence["checkpoint_sidecars"]["best.pt"]["sha256"]
        return self.evidence["source_sha256"]["best.pt"]

    def describe(self):
        info = {"dataset": self.dataset, "arm": self.arm, "adapter": self.family,
                "status": self.status, "evidence": self.evidence_path,
                "external_source_requirements": dict(self.external_source),
                "recovery_implemented": self.arm in INTEGRATED}
        if self.evidence:
            info.update(checkpoint_availability="MISSING_CHECKPOINT",
                        checkpoint_sha256=self.checkpoint_sha256, best_epoch=self.best["epoch"],
                        manifest_sha256=self.configuration["manifest"]["manifest_sha256"])
        if self.arm == "E7":
            info.update(training_status="VALIDATION_COMPLETE" if self.evidence else "UNVERIFIED",
                        source_evidence_status="VERIFIED" if self.evidence else "UNVERIFIED",
                        checkpoint_digest_status="VERIFIED_ON_HPC" if self.evidence else "UNVERIFIED",
                        checkpoint_availability="MISSING_CHECKPOINT" if self.evidence else "UNVERIFIED",
                        checkpoint_load_status="NOT_RUN", original_checkpoint_export_status="NOT_RUN",
                        benchmark_status="NOT_RUN")
        return info


def _read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def registry():
    baseline = "novelty/convnext_b0_b4/validation_evidence.json"
    entries = {}
    for run in _read(baseline)["runs"]:
        dataset, arm = run["dataset"], run["arm"]
        cfg = run["configuration"]
        required = ()
        if arm == "B3":
            required = (("foundation_b3_parallel_runner.py", cfg["implementation_sha256"]),
                        ("sampler.py", cfg["sampling"]["sampler_sha256"]))
        if arm.startswith("B4-"):
            required = (("b4_runner.py", cfg["b4"]["runner_sha256"]),
                        ("b4_common.py", cfg["b4"]["common_sha256"]))
        status = "MISSING_SOURCE" if required else "SUPPORTED"
        entries[dataset, arm] = Experiment(dataset, arm, "hf_v1_linear", status, baseline, run, required)

    for dataset in DATASETS:
        for arm, directory in INTEGRATED.items():
            path = f"results/{dataset}/{directory}/validation_evidence.json"
            run = _read(path)
            if (run["configuration"]["dataset"] != dataset or
                    run["summary"]["status"] != "VALIDATION_COMPLETE"):
                raise ValueError("Integrated validation evidence does not match the registry")
            entries[dataset, arm] = Experiment(dataset, arm, "hf_v1_ordinal", "SUPPORTED", path, run)
        entries[dataset, "E7"] = Experiment(dataset, "E7", "hf_v1_global_local", "PENDING_TRAINING")
    path = "results/APTOS2019/e7_global_local/validation_evidence.json"
    if (ROOT / path).is_file():
        run = _read(path)
        if (run["experiment"] != "E7_APTOS2019" or run["configuration"]["dataset"] != "APTOS2019"
                or run["summary"]["status"] != "VALIDATION_COMPLETE" or run["summary"]["epochs_completed"] != 20
                or run["audit"]["source_files_verified"] != 18):
            raise ValueError("E7 validation evidence does not match the registry")
        entries["APTOS2019", "E7"] = Experiment("APTOS2019", "E7", "hf_v1_global_local", "COMPLETED", path, run)
    entries["DDR", "B4"] = Experiment("DDR", "B4", None, "NOT_APPLICABLE")
    return entries


def require_sources(entry, external_source_dir=None):
    """Verify the local adapter sources and audited external evidence without importing it."""
    if not entry.evidence:
        raise ValueError(f"{entry.dataset} {entry.arm}: {entry.status}; no approved checkpoint receipt")
    if entry.external_source and external_source_dir is None:
        raise ValueError("MISSING_SOURCE: supply the directory containing the exact historical source files")
    recorded = entry.configuration.get("source_sha256", {"novelty/foundation_adaptation.py": B1_SOURCE})
    if entry.arm.startswith("B4-") and entry.configuration["b4"]["b1_source_sha256"] != B1_SOURCE:
        raise ValueError("B4 foundation source does not match the reviewed adapter")
    for relative, digest in recorded.items():
        phase.require_hash(ROOT / relative, digest)
    for name, digest in entry.external_source:
        phase.require_hash(Path(external_source_dir) / name, digest)
    return {"tracked": recorded, "external": dict(entry.external_source),
            "historical_main_sha256": entry.configuration.get("implementation_sha256")}
