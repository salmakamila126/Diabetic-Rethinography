"""Generate static, no-execution result notebooks from the optimization inventory."""

from pathlib import Path
import sys

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novelty_integrated.optimization import hpc


EXECUTION = """\
from pathlib import Path
import os
import subprocess
import sys

# Run All is read-only. To dispatch one stage, set it explicitly and supply approved arguments.
RUN_STAGE = None  # audit, prepare, measure, recover, measure-recovered
STAGE_OPTIONS = []  # e.g. allocation arguments from the HPC guide

REPO = Path(os.environ.get("CADVIT_REPOSITORY", Path.cwd())).expanduser().resolve()
REPO = next((p for p in (REPO, *REPO.parents)
             if (p / "novelty_integrated/optimization/hpc.py").is_file()), None)
if REPO is None:
    raise ValueError("Set CADVIT_REPOSITORY to the checkout containing the optimization framework.")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if RUN_STAGE is not None:
    allowed = {"audit", "prepare", "measure", "recover", "measure-recovered"}
    if RUN_STAGE not in allowed:
        raise ValueError("Select a supported stage.")
    subprocess.run([sys.executable, "-m", "novelty_integrated.optimization.hpc", RUN_STAGE,
                    "--job", JOB, *STAGE_OPTIONS], cwd=REPO, check=True)
"""

DISPLAY = """\
from IPython.display import display, Markdown, SVG
from novelty_integrated.optimization.notebook import load, render

CAMPAIGN = os.environ.get("CADVIT_OPT_CAMPAIGN")
QUALITY_BACKEND = "eager"  # Detailed per-grade metrics and confusion matrices for this backend.
reports = {}
for stage in ("measurement", "post-recovery"):
    report = load(JOB, CAMPAIGN, stage) if CAMPAIGN else None
    if report is None:
        display(Markdown(f"**{stage}** — NOT_RUN"))
        continue
    if reports and any(report[k] != previous[k] for previous in reports.values()
                       for k in ("protocol_sha256", "sample")):
        raise ValueError("Measurement and post-recovery reports are from different protocols or samples.")
    reports[stage] = report
    display(Markdown(f"### {stage}\\n\\n" + render(report, QUALITY_BACKEND)))
    for figure in ("comparison.svg", "tradeoff.svg"):
        path = Path(CAMPAIGN).expanduser() / JOB / stage / figure
        display(SVG(filename=str(path)))
"""


def notebook(job, entry):
    md = nb.v4.new_markdown_cell
    code = nb.v4.new_code_cell
    provenance = entry.describe()
    if entry.evidence:
        src = hpc.checkpoint_relative(entry)
        val = entry.best["validation"]
        origin = f"""\
| Receipt | Value |
| --- | --- |
| Original relative checkpoint | `{src}` |
| Checkpoint SHA-256 | `{entry.checkpoint_sha256}` |
| Selected epoch | {entry.best['epoch']} |
| Frozen manifest SHA-256 | `{provenance['manifest_sha256']}` |
| Validation images | {entry.configuration['manifest']['counts']['val']} |
| Processor SHA-256 | `{entry.configuration['snapshot']['processor_config_sha256']}` |
| Recorded architecture | `{entry.family}` |

The published validation macro-F1 is **{val['macro_f1']:.6f}**. It documents checkpoint selection and is separate from new CPU FP32 measurements.
[Published validation evidence](../../{entry.evidence_path})"""
    else:
        origin = ("**PENDING_TRAINING.** The registry has no audited completed E7 DDR receipt. "
                  "Preparation and timing remain blocked until completed evidence is integrated. "
                  "[E7 DDR research protocol](../novelty/E7_DDR.ipynb)")
    if entry.external_source:
        origin += ("\n\nThe HPC audit requires an independently supplied external source directory with the exact receipt files: "
                   + ", ".join(f"`{name}`" for name, _ in entry.external_source) + ".")
    single = "HF ConvNeXt V1 uses the verified official 224-pixel validation processor."
    architecture = ("E7 retains its 384-pixel global view, two 224-pixel local crops, shared HF V1 backbone and original gate."
                    if entry.arm == "E7" else single)
    recovery = ("Optional recovery uses the original experiment's objective, and starts a fresh optimizer for a fixed 1–3 epochs. "
                "It prepares each recovered ratio plus the 0% recovery control. `measure-recovered` measures the original, "
                "unrecovered and recovered variants together under one CPU allocation."
                if provenance["recovery_implemented"] else
                "**Recovery not supported** for this arm. Measure original and pruned variants under the same CPU allocation.")
    cells = [
        md(f"# {entry.arm} — {entry.dataset} optimization\n\n"
           "Compare one trained checkpoint and its descendants on the same frozen validation sequence. "
           "[Optimization notebook index](README.md) · "
           "[HPC execution guide](../../novelty_integrated/optimization/README.md#staged-hpc-campaign) · "
           "[Historical backend and pruning audit](../../docs/OPTIMIZATION_COMPATIBILITY.md)"),
        md("## Original checkpoint and provenance\n\n" + origin),
        md("## Study design\n\n" + architecture + " The archived timm ConvNeXtV2 experiments use a different GRN layout and "
           "are methodological references. These notebooks exercise the receipt-matched HF V1 adapter.\n\n"
           "| Stage | Treatment |\n| --- | --- |\n"
           "| Benchmark original | FP32 eager, TorchScript, ONNX Runtime CPU, OpenVINO CPU. Every export is gated on full-sequence numerical and prediction agreement. |\n"
           "| Prune without recovery | Independent L1 expansion-channel removals of 10%, 30%, and 50%, rounded down per layer. Report the actual parameter and MAC reduction. |\n"
           "| Recovery | " + recovery + " |\n\n"
           "Batch 1, full ordered validation by default, 20 warm-ups and three timed passes per backend and scope. CPU affinity and thread count are explicit. "
           "Model-only excludes image loading and transforms. End-to-end includes them. Quality uses five grades in order 0–4. "
           "FLOPs count two operations per counted Conv2d/Linear MAC and exclude normalization, elementwise operations and I/O. "
           "E7 counts shared parameters once and computes work for all three views. The validation split selected the parent checkpoints, so trade-offs are exploratory."),
        md("## Execution controls\n\nDefault execution dispatches no process. HPC stages require separate allocation arguments. "
           "HPC jobs verify the original bytes and write exclusively under the dedicated campaign root. No stage uses final test."),
        code(f"JOB = {job!r}\n" + EXECUTION),
        md("## Results and trade-offs\n\n| Measurement | Status in this checkout |\n| --- | --- |\n"
           "| Original and pruning benchmark | `NOT_RUN` |\n"
           f"| Post-recovery benchmark | `{'NOT_RUN' if provenance['recovery_implemented'] else 'NOT_SUPPORTED'}` |\n\n"
           "Set `CADVIT_OPT_CAMPAIGN` to the external campaign root to display completed provenance-matched results. "
           "The display includes latency, throughput, model size, compute, aggregate and per-grade quality, confusion matrices, "
           "pruning/recovery deltas and publication figures."),
        code(DISPLAY),
        md("## Current conclusion\n\nNo latency or pruning quality conclusion is available before a measured run. "
           "Compare the original and derived models within the same allocation and validation sequence. "
           "Report classifier agreement for backend changes and grade-level losses for pruning or recovery. "
           "Keep this exploratory validation comparison separate from final-test assessment."),
    ]
    for i, cell in enumerate(cells):
        cell.id = f"{job.lower().replace('_', '-')}-{i:02}"
    return nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "optimization": {"job": job, "dataset": entry.dataset, "arm": entry.arm,
                         "execution_default": "NOT_RUN", "receipt_status": entry.status},
    })


def main():
    output = ROOT / "notebooks/optimization"
    output.mkdir(exist_ok=True)
    for job, entry in hpc.inventory().items():
        document = notebook(job, entry)
        nb.validate(document)
        with (output / f"{job}.ipynb").open("w", encoding="utf-8", newline="\n") as stream:
            nb.write(document, stream)


if __name__ == "__main__":
    main()
