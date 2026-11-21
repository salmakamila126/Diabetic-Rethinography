"""Verify the E7 DDR partial-run transfer and render its aggregate evidence."""

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZipFile

from docs import e5_e6_publication as prior, e7_publication as aptos

ROOT = prior.ROOT
FOLDER = Path("results/DDR/e7_global_local")
EVIDENCE = FOLDER / "validation_evidence.json"
ARCHIVE_SHA = "1b1d0b82779071552051bc3c01b478c9c4b77a50b58d2d90060b88765c9d8d85"
MEMBERS = {"STATUS.txt", "best_confusion_matrix.csv", "best_per_grade.csv", "best_summary.json",
           "history.json", "last_summary.json", "selection_record.json"}
STATUS = "PARTIAL_EARLY_STOPPED_AT_EPOCH_10"
EXPECTED = {"accuracy": 0.8338658146964856, "macro_f1": 0.6685627941443673,
            "quadratic_weighted_kappa": 0.8521508578613854, "grade_mae": 0.26118210862619806}
FIGURES = {"losses": "Train total objective and validation categorical CE by epoch",
           "epoch_scores": "Train and validation macro-F1, accuracy and QWK by epoch",
           "confusion": "Best-checkpoint counts and true-grade row percentages",
           "grade_errors": "Precision, recall, F1 and direction of errors by true grade",
           "e5_e6_e7_comparison": "Best-checkpoint comparison on the same DDR validation split"}
require, sha = prior.require, prior.sha


def comparison(run):
    records = {method: aptos.parse((ROOT / prior.run_folder(method, "DDR") / "validation_evidence.json").read_bytes())
               for method in ("E5", "E6")}
    reference = run["configuration"]["manifest"]
    for method, record in records.items():
        actual = record["configuration"]["manifest"]
        require(all(actual[key] == reference[key] for key in ("manifest_sha256", "image_order_sha256",
                    "manifest_sidecar_sha256", "counts", "grade_support")), f"{method} DDR frozen identity differs")
    parent = records["E5"]
    init = run["configuration"]["init"]
    require((init["checkpoint_epoch"], init["checkpoint_sha256"]) ==
            (parent["summary"]["best"]["epoch"], parent["checkpoint_sidecars"]["best.pt"]["sha256"]),
            "DDR E5 warm-start receipt differs")
    return {**records, "E7 (partial)": run}


def audit(run):
    cfg, summary, history = (run[k] for k in ("configuration", "summary", "history"))
    best = summary["best"]
    require(run["experiment"] == "E7_DDR" and cfg["dataset"] == "DDR" and cfg["schema"] == "convnext-e7-global-local-v1"
            and summary["status"] == STATUS and summary["final_test_status"] == cfg["final_test_status"] == "NOT_ACCESSED"
            and cfg["epochs"] == summary["epochs_planned"] == 20 and summary["epochs_completed"] == len(history) == 10
            and summary["stopped_during_epoch"] == 11, "Unexpected E7 DDR run identity or status")
    require(cfg["git_head"] == aptos.SOURCE_COMMIT and cfg["class_order"] == list(range(5))
            and cfg["checkpoint_selection"] == "max validation macro-F1; tie lower raw categorical CE; tie earliest epoch"
            and cfg["selection_weights"] == "raw" and cfg["ema_decay"] is None, "Unexpected E7 DDR selection configuration")
    require([r["epoch"] for r in history] == list(range(1, 11)), "Epochs must be exactly 1 through 10 in order")
    require(cfg["manifest"]["counts"] == {"train": 10018, "val": 1252} and
            cfg["manifest"]["grade_support"]["train"] == cfg["train_counts"] == [5014, 504, 3581, 189, 730]
            and cfg["manifest"]["grade_support"]["val"] == [626, 63, 448, 24, 91], "Unexpected frozen split support")
    require(cfg["seed"] == 42 and cfg["geometry"]["global_size"] == 384 and cfg["geometry"]["patch_size"] == 224
            and cfg["geometry"]["patch_count"] == 2 and cfg["effective_batch"] == 32,
            "Unexpected E7 DDR view or training configuration")
    incumbent = None
    for row in history:
        for split, key in (("train", "train"), ("validation", "val")):
            aptos._verify_metrics(row[split], cfg["manifest"]["grade_support"][key], f"Epoch {row['epoch']} {split}")
        val = row["validation"]
        score = val["macro_f1"], -val["validation_loss"], -row["epoch"]
        # Recorded F1 values are distinct. This is equivalent to the runner's tolerance-based comparison for this transfer.
        improved = incumbent is None or score > incumbent[0]
        if incumbent is not None:
            require(abs(score[0] - incumbent[0][0]) > 1e-12, "Transfer requires the runner's near-tie comparator")
        if improved:
            incumbent = score, row["epoch"]
        require(row["best_epoch"] == incumbent[1] and row["best_updated"] is improved
                and row["class_weight_active"] is False, "Checkpoint selection or weighting history differs")
    require(incumbent[1] == best["epoch"] == 2 and history[1]["validation"] == best["validation"]
            and all(best["validation"][key] == value for key, value in EXPECTED.items()), "Best checkpoint values differ")
    require(run["provenance"]["archive_sha256"] == ARCHIVE_SHA and set(run["provenance"]["members"]) == MEMBERS
            and all(prior.HEX.fullmatch(v["sha256"]) and type(v["bytes"]) is int and v["bytes"] > 0
                    for v in run["provenance"]["members"].values()), "Evidence archive receipt differs")
    require(set(run["checkpoint_sidecars"]) == {"best.pt", "last.pt"}, "Both checkpoint receipts are required")
    for name, epoch in (("best.pt", 2), ("last.pt", 10)):
        receipt = run["checkpoint_sidecars"][name]
        require(receipt["epoch"] == receipt["history_length"] == epoch and receipt["bytes"] > 0
                and prior.HEX.fullmatch(receipt["sha256"]), f"Invalid {name} receipt")
    prior.check_public(run)
    comparison(run)


def load_export(path):
    require(path.stat().st_size <= 1024 * 1024, "Expected a lightweight aggregate ZIP")
    original = path.read_bytes()
    require(sha(original) == ARCHIVE_SHA, "ZIP differs from the audited transfer")
    with ZipFile(io.BytesIO(original)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(MEMBERS) and {item.filename for item in entries} == MEMBERS and
                all(not item.is_dir() and 0 < item.file_size <= 1024 * 1024 for item in entries), "Unexpected ZIP members")
        raw = {item.filename: archive.read(item) for item in entries}
    require(raw["STATUS.txt"].decode().splitlines() == ["E7 DDR - partial training evidence", "Planned epochs: 20",
            "Completed epochs: 10", "Best checkpoint epoch: 2", "Stopped during epoch 11",
            "No verified E7 DDR final-test result", "Checkpoint binaries remain on HPC",
            "All metrics and figures must be derived from source evidence"], "Unexpected transfer status")
    cfg, history, best_export, last_export = (aptos.parse(raw[name]) for name in
            ("selection_record.json", "history.json", "best_summary.json", "last_summary.json"))
    require(last_export["best"] == best_export["best"], "Best and last checkpoint histories disagree about selection")
    best = best_export["best"]["validation"]

    def rows(name):
        return io.StringIO(raw[name].decode().replace("\r\r\n", "\n"), newline="")

    require([[int(x) for x in row] for row in csv.reader(rows("best_confusion_matrix.csv")) if row]
            == best["confusion_matrix"], "Original confusion CSV differs from JSON")
    grades = list(csv.DictReader(rows("best_per_grade.csv")))
    require(len(grades) == len(best["per_grade"]) == 5 and all(csv_row.keys() == json_row.keys()
            and all(type(value)(csv_row[key]) == value for key, value in json_row.items())
            for csv_row, json_row in zip(grades, best["per_grade"], strict=True)), "Original per-grade CSV differs from JSON")
    aptos.verify_sources(cfg)
    source_root = PurePosixPath(cfg["manifest"]["dataset_root"]).parents[1]
    for key in ("train_cache_root", "val_cache_root"):
        cfg[key] = PurePosixPath(cfg[key]).relative_to(source_root).as_posix()
    for key in ("dataset_root", "manifest_source_root", "mapped_dataset_root"):
        del cfg["manifest"][key]
    run = {"experiment": "E7_DDR", "configuration": cfg, "history": history,
           "summary": {"status": STATUS, "epochs_planned": 20, "epochs_completed": 10,
                       "stopped_during_epoch": 11, "final_test_status": "NOT_ACCESSED", "best": best_export["best"]},
           "checkpoint_sidecars": {name: {"epoch": export["epoch"], "history_length": export["history_length"],
                                          "sha256": export["sha256"], "bytes": export["size_bytes"]}
                                    for name, export in (("best.pt", best_export), ("last.pt", last_export))},
           "provenance": {"archive_sha256": ARCHIVE_SHA,
                          "members": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in raw.items()},
                          "checkpoint_hashes": "Reported by the HPC checkpoint summaries; binaries were not transferred",
                          "manifest_check": "Matched published DDR E5/E6 receipts; original manifest bytes not provided in this ZIP"},
           "audit": {"source_files_verified": len(cfg["source_sha256"]), "epoch_records": len(history),
                     "confusion_matrices_checked": 2 * len(history), "original_confusion_and_per_grade_csv_match": True}}
    audit(run)
    return run


def comparison_table(run):
    records = comparison(run)
    selected = [item["summary"]["best"] for item in records.values()]
    rows = [("Epochs completed", 20, 20, run["summary"]["epochs_completed"]),
            ("Selected epoch", *(x["epoch"] for x in selected))]
    rows += [(label, *(f"{x['validation'][key]:.9f}" for x in selected)) for key, label in aptos.SCORES.items()]
    rows += [(f"Grade {grade} recall", *(f"{x['validation']['per_grade'][grade]['recall']:.6f}" for x in selected))
             for grade in (1, 3)]
    return prior.table(["Frozen DDR validation", *records], rows)


def figure_data(run):
    history = run["history"]
    best = run["summary"]["best"]
    cm = best["validation"]["confusion_matrix"]
    derived = prior.derive(cm)

    def series(key, labels):
        return {"epochs": [row["epoch"] for row in history], "selected_epoch": best["epoch"],
                "series": {label: [prior.get(row, path) for row in history] for label, path in labels.items()},
                "source": f"validation_evidence.json · {key}"}

    return {"losses": series("history", {"Train total objective": "train.loss", "Validation categorical CE": "validation.validation_loss"}),
            "epoch_scores": series("history", {f"{label} · {split}": f"{split.lower()}.{key}"
                    for key, label in aptos.SCORES.items() if key in ("macro_f1", "accuracy", "quadratic_weighted_kappa")
                    for split in ("Train", "Validation")}),
            "confusion": {"counts": cm, "normalized": derived["row_normalized"], "source": "validation_evidence.json · epoch 2"},
            "grade_errors": {"grades": best["validation"]["per_grade"], "source": "validation_evidence.json · epoch 2"},
            "e5_e6_e7_comparison": {"runs": {name: {"epoch": item["summary"]["best"]["epoch"],
                    "metrics": {key: prior.best(item)[key] for key in aptos.SCORES}}
                    for name, item in comparison(run).items()}, "source": "E5, E6 and E7 DDR validation_evidence.json"}}


def plot(run, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    destination.mkdir(parents=True, exist_ok=True)
    require(not any((destination / f"ddr_e7_{name}.{ext}").exists() for name in FIGURES for ext in ("png", "svg")),
            "Figure target already exists. Use a fresh output directory.")
    data = figure_data(run)
    colors = ("#176383", "#a44918", "#247553")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "e7-ddr-partial",
                         "svg.fonttype": "none"}):
        for name, payload in data.items():
            count = {"losses": 2, "epoch_scores": 3, "confusion": 2, "grade_errors": 2, "e5_e6_e7_comparison": 5}[name]
            cols = 3 if count in (3, 5) else 2
            fig, axes = plt.subplots(2 if count == 5 else 1, cols, figsize=(12, 7.4 if count == 5 else 4.7),
                                     layout="constrained", squeeze=False)
            axes = axes.ravel()
            fig.suptitle(f"DDR · E7 partial (10/20 epochs) | {FIGURES[name]}", fontweight="bold")
            if name in ("losses", "epoch_scores"):
                panels = ((label, [label]) for label in payload["series"]) if name == "losses" else (
                    (label, [f"{label} · Train", f"{label} · Validation"]) for label in ("Macro-F1", "Accuracy", "QWK"))
                for ax, (label, keys) in zip(axes, panels):
                    for i, key in enumerate(keys):
                        ax.plot(payload["epochs"], payload["series"][key], color=colors[i], marker="o",
                                markersize=3, linestyle=("-", "--")[i], label=key)
                    ax.axvline(payload["selected_epoch"], color="#666666", linestyle=":", label="Selected epoch 2")
                    ax.set(title=label, xlabel="Completed epoch", xticks=payload["epochs"],
                           ylabel="Loss (unitless)" if name == "losses" else "Score (unitless)")
                    ax.grid(alpha=.25)
                    ax.legend(fontsize=8, frameon=False)
            elif name == "confusion":
                for ax, key, title, unit in zip(axes, ("counts", "normalized"), ("Counts", "True-grade row percentages"),
                                               ("Images", "Share of true grade")):
                    matrix = np.array(payload[key])
                    upper = 1.0 if key == "normalized" else matrix.max()
                    edges = np.arange(6) - .5
                    image = ax.pcolormesh(edges, edges, matrix, cmap="Blues", vmin=0, vmax=upper)
                    bar = fig.colorbar(image, ax=ax, shrink=.82, label=unit)
                    bar.solids.set_rasterized(False)
                    for (r, c), value in np.ndenumerate(matrix):
                        ax.text(c, r, f"{value:.1%}" if key == "normalized" else str(value), ha="center", va="center",
                                color="white" if value > .55 * upper else "#263743")
                    ax.set(title=f"Epoch 2 · {title}", xlabel="Predicted grade", ylabel="True grade", xticks=range(5), yticks=range(5))
                    ax.set_aspect("equal")
                    ax.invert_yaxis()
            elif name == "grade_errors":
                for ax, keys, ylabel in zip(axes, (("precision", "recall", "f1"), ("overgraded", "undergraded")),
                                            ("Score (unitless)", "Incorrect predictions (images)")):
                    x = np.arange(5)
                    for i, key in enumerate(keys):
                        bars = ax.bar(x + (i - (len(keys)-1)/2)*.24, [g[key] for g in payload["grades"]],
                                      width=.24, color=colors[i], label="F1" if key == "f1" else key.capitalize())
                        if len(keys) == 2:
                            ax.bar_label(bars, padding=2, fontsize=8)
                    ax.set(title="Grade metrics" if len(keys) == 3 else "Direction of error", xlabel="True grade and support", ylabel=ylabel)
                    ax.set_xticks(x, [f"{g['grade']}\n(n={g['support']})" for g in payload["grades"]])
                    ax.set_ylim(0, 1.08 if len(keys) == 3 else max(g[k] for g in payload["grades"] for k in keys)*1.24)
                    ax.legend(frameon=False, fontsize=9)
                    ax.grid(axis="y", alpha=.22)
            else:
                for ax, (key, label) in zip(axes, aptos.SCORES.items()):
                    names = [f"{arm}\nepoch {r['epoch']}" for arm, r in payload["runs"].items()]
                    bars = ax.bar(names, [item["metrics"][key] for item in payload["runs"].values()], color=colors)
                    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
                    ax.set(title=label, ylabel="Grade steps" if key == "grade_mae" else "Score (unitless)",
                           ylim=(0, .36 if key == "grade_mae" else 1.12))
                    ax.grid(axis="y", alpha=.22)
                axes[-1].axis("off")
                axes[-1].text(.02, .64, "Same 1,252 frozen validation images\nE5 and E6 completed 20 epochs\nE7 completed 10 of 20 epochs\nValidation selected each checkpoint",
                              transform=axes[-1].transAxes, va="center")
            fig.supxlabel(f"Source: {payload['source']}", fontsize=9, color="#38444b")
            metadata = {"Description": json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)}
            stem = destination / f"ddr_e7_{name}"
            fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white", metadata=metadata)
            svg = io.StringIO()
            fig.savefig(svg, format="svg", facecolor="white", metadata={**metadata, "Date": None})
            content = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
            stem.with_suffix(".svg").write_text(content, encoding="utf-8", newline="\n")
            plt.close(fig)


def report(run):
    options = run["configuration"]
    best = run["summary"]["best"]["validation"]
    grades = prior.table(["Grade", "Support", "Precision", "Recall", "F1", "Overgraded", "Undergraded"], [
        (g["grade"], g["support"], *(f"{g[k]:.6f}" for k in ("precision", "recall", "f1")),
         g["overgraded"], g["undergraded"]) for g in best["per_grade"]])
    index = "\n".join(f"- {label}. [SVG](../../../docs/figures/ddr_e7_{key}.svg) · [PNG](../../../docs/figures/ddr_e7_{key}.png)"
                      for key, label in FIGURES.items())
    hashes = "\n".join(f"- `{name}` · epoch {value['epoch']} · {value['bytes']:,} bytes · `{value['sha256']}`"
                       for name, value in run["checkpoint_sidecars"].items())
    return f"""# E7 Global/Local ConvNeXt-Tiny on DDR

**PARTIAL / EARLY STOPPED AT EPOCH 10.** The 20-epoch run stopped during epoch 11. Ten completed epochs are recorded. Validation macro-F1 selected epoch 2 from the recorded interval. Final test remains `NOT_ACCESSED`.

[Research notebook](../../../notebooks/novelty/E7_DDR.ipynb) · [Audited aggregate evidence](validation_evidence.json) · [Full epoch CSV](history.csv) · [E7 training protocol](../../../novelty_integrated/e7/README.md)

## Research question and protocol

E7 tests whether a global view and two detail views change grading performance. One shared ConvNeXt-Tiny receives a 384×384 letterboxed image and two 224×224 local FOV crops. A learned gate fuses the pooled global and mean local features. The model warm-started from DDR E5 epoch {options['init']['checkpoint_epoch']} (`{options['init']['checkpoint_sha256']}`). Seed 42, B1 training-only augmentation, raw selected weights, BF16, AdamW, effective batch 32 and the categorical plus 0.25 ordinal BCE objective are recorded in the evidence. Class reweighting was scheduled for epoch 11 and is inactive in all ten completed epochs.

Frozen support is 10,018 training and 1,252 validation images. Validation grades 0–4 contain 626, 63, 448, 24 and 91 images. Selection maximizes validation macro-F1, then minimizes validation categorical CE, then favors the earlier epoch. Training and validation used different loss definitions. See the epoch CSV for full-precision values and per-epoch resources.

## Selected validation checkpoint

{comparison_table(run)}

E7 has the highest macro-F1 and accuracy of the three selected checkpoints in this comparison. E5 retains higher QWK. E7 grade 1 recall is 11/63, below both E5 and E6. Grade 3 recall is 11/24, tied with E6 and below E5. Grade 1 has the lowest E7 F1. These small class supports limit interpretation.

{grades}

Of 1,252 validation images, {best['correct']:,} were graded correctly. {best['overgraded']} were overgraded and {best['undergraded']} were undergraded. For true grade 1, 19 predictions were grade 0 and 33 were grade 2. For true grade 3, 11 predictions were grade 2 and 2 were grade 4. These counts describe the recorded grading errors.

Training total loss falls from {run['history'][0]['train']['loss']:.4f} to {run['history'][-1]['train']['loss']:.4f}. Validation categorical CE is {best['validation_loss']:.4f} at the selected epoch and {run['history'][-1]['validation']['validation_loss']:.4f} at epoch 10. Macro-F1 at epoch 10 is {run['history'][-1]['validation']['macro_f1']:.6f}. The comparison uses one validation-selected checkpoint per method, one run per method, and unequal training budgets. It provides no causal estimate of multi-view fusion, confirmed novelty, or independent final-test performance.

## Figure index

{index}

Lines connect only the ten recorded epoch values. The vertical marker shows epoch 2. The normalized matrix divides each row by its true-grade support. The PNG and SVG versions use the same plotted data and embed the same machine-readable figure payload.

From the repository root, render all figures into a fresh directory with Matplotlib 3.11.2 installed:

```bash
python -m docs.e7_ddr_publication --figures ../e7-ddr-rendered
```

## Provenance and limitations

- Original aggregate archive SHA-256 `{ARCHIVE_SHA}`. Its seven member hashes are retained in the evidence.
- Training source revision `{options['git_head']}`. All 18 recorded source files matched both that Git revision and the local source during import.
- Frozen manifest SHA-256 `{options['manifest']['manifest_sha256']}`. Sidecar and ordered membership digests match the published DDR E5/E6 receipts. The ZIP contains aggregate evidence and no manifest bytes or checkpoint binaries.
{hashes}

The checkpoint SHA-256 values are those recorded in the exported summaries. The original binaries remain external. Private cache and dataset paths were removed from the public record. The original ten history objects were retained unchanged. No recovery, backend parity or latency result is included for E7 DDR.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Verify the original aggregate ZIP against the public record")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--figures", type=Path, help="Render the five figure pairs into a fresh directory")
    output.add_argument("--publish-to", type=Path, help="Write evidence, report and figures to a new destination tree")
    args = parser.parse_args(argv)
    run = load_export(args.archive) if args.archive else aptos.parse((ROOT / EVIDENCE).read_bytes())
    audit(run)
    if args.archive and (ROOT / EVIDENCE).is_file():
        require(run == aptos.parse((ROOT / EVIDENCE).read_bytes()), "Committed evidence differs from ZIP")
    if args.figures:
        plot(run, args.figures)
    if args.publish_to:
        folder = args.publish_to / FOLDER
        require(not folder.exists(), "Evidence destination already exists")
        plot(run, args.publish_to / "docs/figures")
        folder.mkdir(parents=True)
        for name, value in ((EVIDENCE.name, json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                            ("history.csv", aptos.history_csv(run["history"])), ("validation_report.md", report(run))):
            (folder / name).write_text(value, encoding="utf-8", newline="\n")
    print(f"{STATUS}: 10 epoch records, best epoch 2, 1,252 validation rows")


if __name__ == "__main__":
    main()
