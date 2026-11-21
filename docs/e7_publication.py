"""Audit the E7 APTOS2019 aggregate transfer and render validation-only evidence."""

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from zipfile import ZipFile

from docs import e5_e6_publication as prior
from docs.e5_e6_publication import check_public, equal, require, sha

ROOT = prior.ROOT
RUN = "foundation_e7/APTOS2019/run/train"
FOLDER = Path("results/APTOS2019/e7_global_local")
DATASET = "APTOS2019"
ARCHIVE_SHA = "f60524e43ca2dd594526332368e895fb4ab746321b2a55f57d9e0ddc3b2e431d"
SOURCE_COMMIT = "b7c42b6a3560d233960707390119e05731d635c3"
CHECKPOINTS = {
    "best.pt": "5ef9816b9c7511a2b0347f7967543f04d70a77c9d0de630fd4116f714c38d11c",
    "last.pt": "428920a006d8666e19c74850ce9ad9912b30356e120636fce3afceeb62b5f940",
}
ORIGINALS = ("run.json", "summary.json", "history.json", "best.pt.sha256.json", "last.pt.sha256.json")
MATRIX = [[175, 5, 0, 0, 0], [1, 24, 10, 0, 0], [0, 12, 80, 5, 4], [0, 0, 7, 8, 5], [0, 2, 6, 3, 19]]
EXPECTED = {"accuracy": 0.8360655737704918, "macro_f1": 0.696492253305618,
            "weighted_f1": 0.8367882547827453, "quadratic_weighted_kappa": 0.9126865167780703,
            "grade_mae": 0.20218579234972678, "validation_loss": 0.9034500641112795,
            "correct": 306, "wrong": 60, "confusion_matrix": MATRIX}
SCORES = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1",
          "quadratic_weighted_kappa": "QWK", "grade_mae": "Grade MAE"}
FIGURES = {
    "epoch_scores": "All five metrics for train and validation across 20 epochs, with epoch 19 marked.",
    "losses": "Training total, weighted or unweighted categorical term, ordinal BCE, and validation categorical CE.",
    "resources": "Epoch times, throughput, GPU allocator peaks and process RSS high-water mark.",
    "confusion": "Epoch-19 validation confusion in counts and true-grade row percentages.",
    "grade_errors": "Grade-wise precision, recall, F1, and under/overgraded counts with support.",
    "e5_e6_e7_comparison": "Selected validation checkpoints on the same APTOS2019 split, including minority-grade recall.",
}


def parse(raw):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), "Duplicate JSON keys")
        return result

    return json.loads(raw, object_pairs_hook=unique)


def _raw_archive(path):
    require(Path(path).stat().st_size <= 1024 * 1024, "Unexpected E7 aggregate archive size")
    raw = Path(path).read_bytes()
    require(sha(raw) == ARCHIVE_SHA, "E7 archive differs from the reviewed transfer")
    with ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) == len(ORIGINALS) and
                {i.filename for i in members} == {f"E7_APTOS2019/{name}" for name in ORIGINALS},
                "Unexpected or duplicated E7 archive member")
        require(all(i.file_size < 1024 * 1024 and not i.is_dir() for i in members),
                "Invalid E7 archive member type or size")
        return {name: archive.read(f"E7_APTOS2019/{name}") for name in ORIGINALS}


def comparison(run):
    records = {}
    for method, directory in prior.RUN_NAMES.items():
        record = parse((ROOT / f"results/{DATASET}/{directory}/validation_evidence.json").read_bytes())
        for key in ("manifest_sha256", "image_order_sha256", "counts", "grade_support"):
            require(record["configuration"]["manifest"][key] == run["configuration"]["manifest"][key],
                    f"{method} uses a different validation identity")
        records[method] = record
    records["E7"] = run
    return records


def verify_manifest(directory, manifest, raw_source_root):
    csv_path = Path(directory) / "APTOS2019_seed42.csv"
    meta = csv_path.with_suffix(".json").read_bytes()
    raw = csv_path.read_bytes()
    require(sha(raw) == manifest["manifest_sha256"] and sha(meta) == manifest["manifest_sidecar_sha256"],
            "External manifest or sidecar hash differs")
    sidecar = parse(meta)
    require(sidecar["root"] == raw_source_root and sidecar["dataset"] == DATASET and sidecar["seed"] == 42
            and sidecar["manifest_file_sha256"] == sha(raw), "Frozen manifest sidecar identity differs")
    grouped = {"train": [], "val": []}
    groups, digests = {}, {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        split = row["split"]
        if split == "test":
            continue
        require(split in grouped, "Unknown frozen split")
        group, digest = row["group_id"], row["sha256"]
        require(groups.setdefault(group, split) == split and digests.setdefault(digest, split) == split,
                "Group or identical-content overlap across training and validation")
        grouped[split].append(row)
    ordered = [(split, r["image_id"], r["sha256"]) for split, rows in grouped.items() for r in rows]
    require(sha(json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode()) ==
            manifest["image_order_sha256"], "Frozen train/validation order differs")
    for split, rows in grouped.items():
        support = [sum(r["label"] == str(g) for r in rows) for g in range(5)]
        require(len(rows) == manifest["counts"][split] and support == manifest["grade_support"][split]
                and len({r["group_id"] for r in rows}) == manifest["group_count"][split],
                "Frozen membership statistics differ")


def verify_sources(cfg):
    recorded = cfg["source_sha256"]
    from novelty_integrated.e7.runner import _hashes

    expected_current = _hashes()
    require(cfg["git_head"] == SOURCE_COMMIT and recorded == expected_current and len(recorded) == 18,
            "E7 source set or current source hashes differ")
    for name, digest in recorded.items():
        revision = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=ROOT)
        require(sha(revision) == digest, f"E7 source hash differs at the run commit: {name}")


def _verify_metrics(log, support, label):
    base = {k: v for k, v in log.items() if k not in ("grade1_transitions", "grade3_transitions")}
    derived = prior.verify_scores(base, support, label)
    for actual, expected in zip(log["per_grade"], derived["per_grade"], strict=True):
        require(equal(actual["mean_absolute_error"], expected["mean_absolute_error"]),
                f"{label}: per-grade MAE differs")
    if "validation_loss" in log:
        for metric in ("macro_precision", "macro_recall"):
            require(equal(log[metric], derived[metric]), f"{label}: {metric} differs")
        cm = log["confusion_matrix"]
        require(log["grade1_transitions"] == {"support": support[1], "correct": cm[1][1],
                 "to_grade0": cm[1][0], "to_grades2_4": sum(cm[1][2:])}, f"{label}: grade 1 transitions differ")
        require(log["grade3_transitions"] == {"support": support[3], "correct": cm[3][3],
                 "to_grades0_2": sum(cm[3][:3]), "to_grade4": cm[3][4]}, f"{label}: grade 3 transitions differ")
    return derived


def audit(record):
    from novelty import foundation_adaptation as fa
    from novelty_integrated.e7 import config

    cfg, summary, history = (record[k] for k in ("configuration", "summary", "history"))
    require(summary["record"] == cfg and summary["status"] == "VALIDATION_COMPLETE" and
            summary["epochs_completed"] == cfg["epochs"] == config.EPOCHS and
            summary["final_test_status"] == cfg["final_test_status"] == "NOT_ACCESSED",
            "E7 summary, run configuration or scope differs")
    require(cfg["schema"] == config.SCHEMA and cfg["dataset"] == DATASET and cfg["seed"] == config.SEED
            and cfg["class_order"] == [0, 1, 2, 3, 4] and cfg["selection_weights"] == "raw"
            and cfg["ema_decay"] is None and cfg["snapshot"] == config.OFFICIAL_SNAPSHOT
            and cfg["effective_processor_sha256"] == fa._digest_json(cfg["effective_processor"]),
            "E7 model, processor or selection configuration differs")
    geo = cfg["geometry"]
    require((geo["global_size"], geo["patch_size"], geo["patch_count"],
             geo["patch_width_and_height_fraction"], geo["fov_threshold"], geo["fov_margin_fraction"])
            == (config.GLOBAL_SIZE, config.PATCH_SIZE, config.PATCH_COUNT,
                config.PATCH_WIDTH_FRACTION, config.FOV_THRESHOLD, config.FOV_MARGIN_FRACTION),
            "E7 geometry differs from the reviewed source")
    require(cfg["effective_batch"] == cfg["micro_batch"] * cfg["accumulation"] == config.EFFECTIVE_BATCH
            and cfg["optimizer"] == "AdamW" and cfg["scheduler"] is None and cfg["precision"] == "bf16"
            and (cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"], cfg["max_grad_norm"])
            == (config.BACKBONE_LR, config.HEAD_LR, config.WEIGHT_DECAY, config.MAX_GRAD_NORM),
            "E7 optimization configuration differs")
    manifest = cfg["manifest"]
    require(manifest["dataset"] == DATASET and manifest["images_verified"] is True
            and manifest["counts"] == {"train": 2930, "val": 366}
            and manifest["manifest_sha256"] == prior.DATASETS[DATASET][2]
            and cfg["train_counts"] == manifest["grade_support"]["train"],
            "E7 frozen split identity differs")
    predecessors = comparison(record)
    warm = predecessors["E5"]
    require(cfg["init"] == {"mode": "E5_BEST_WARMSTART",
                "checkpoint_epoch": warm["summary"]["best"]["epoch"],
                "checkpoint_sha256": warm["checkpoint_sidecars"]["best.pt"]["sha256"],
                "source_git_head": warm["configuration"]["source_git_head"],
                "summary_sha256": warm["provenance"]["files"]["summary.json"]["sha256"]},
            "E7 E5 warm-start receipt differs")
    # GlobalLocalModel reuses E5's backbone and both heads, and adds Linear(1536, 1).
    parameters = warm["summary"]["parameter_counts"]["total_trainable"] + 1536 + 1
    require(summary["parameter_counts"] == {"total": parameters, "trainable": parameters},
            "E7 parameter counts do not match the transfer and fusion gate")
    for name, expected in CHECKPOINTS.items():
        sidecar = record["checkpoint_sidecars"][name]
        require(set(sidecar) == {"sha256", "bytes"} and sidecar["sha256"] == expected
                and type(sidecar["bytes"]) is int and sidecar["bytes"] > 0, "E7 sidecar receipt differs")
    require([r["epoch"] for r in history] == list(range(1, 21)), "E7 epochs missing or unordered")
    best = None
    for row in history:
        epoch, tr, val, resource = (row[k] for k in ("epoch", "train", "validation", "resource"))
        for split, values in (("train", tr), ("val", val)):
            _verify_metrics(values, manifest["grade_support"][split], f"Epoch {epoch} {split}")
        selected = fa._best_is_new(val, best["validation"] if best else None)
        if selected:
            best = {"epoch": epoch, "validation": val}
        require(row["best_updated"] is selected and row["best_epoch"] == best["epoch"],
                f"Epoch {epoch} checkpoint-selection history differs")
        require(row["class_weight_active"] is (epoch >= config.WEIGHT_START_EPOCH)
                and row["learning_rates"] == {"backbone": cfg["backbone_lr"], "head": cfg["head_lr"]},
                f"Epoch {epoch} weighting schedule or LR differs")
        require(all(math.isfinite(v) and v >= 0 for v in
                    (tr["loss"], tr["categorical_loss"], tr["ordinal_loss"], val["validation_loss"]))
                and math.isclose(tr["loss"], tr["categorical_loss"] + config.e5.ORDINAL_COEFFICIENT * tr["ordinal_loss"],
                                 rel_tol=1e-6, abs_tol=1e-8), f"Epoch {epoch} loss terms differ")
        windows = tr["accumulation_window_images"]
        batch = cfg["effective_batch"]
        quotient, remainder = divmod(manifest["counts"]["train"], batch)
        require(windows == [batch] * quotient + ([remainder] if remainder else [])
                and sum(windows) == tr["images"] and len(windows) == tr["optimizer_steps"],
                f"Epoch {epoch} accumulation counts differ")
        for split, source in (("train", "train"), ("validation", "val")):
            seconds, rate = resource[f"{split}_seconds"], resource[f"{split}_images_per_second"]
            require(seconds > 0 and equal(rate, manifest["counts"][source] / seconds),
                    f"Epoch {epoch} resource rate differs")
        require(0 < resource["gpu_peak_allocated_bytes"] <= resource["gpu_peak_reserved_bytes"]
                and resource["process_peak_rss_bytes"] > 0, f"Epoch {epoch} memory fields invalid")
    require(summary["best"] == best and best["epoch"] == 19 and all(
            best["validation"][k] == value for k, value in EXPECTED.items()), "E7 best checkpoint differs")
    return _verify_metrics(best["validation"], manifest["grade_support"]["val"], "Best E7 validation")


def load_export(archive_path, manifest_dir=None):
    originals = _raw_archive(archive_path)
    cfg, summary, history = (parse(originals[name]) for name in ORIGINALS[:3])
    require(summary["record"] == cfg, "Original summary record differs from run.json")
    verify_sources(cfg)
    if manifest_dir:
        verify_manifest(manifest_dir, cfg["manifest"], cfg["manifest"]["manifest_source_root"])
    root = PurePosixPath(cfg["manifest"]["dataset_root"]).parents[1]
    for key in ("train_cache_root", "val_cache_root"):
        cfg[key] = PurePosixPath(cfg[key]).relative_to(root).as_posix()
    cfg["manifest"] = {k: v for k, v in cfg["manifest"].items()
                       if k not in ("dataset_root", "manifest_source_root", "mapped_dataset_root")}
    summary["record"] = cfg
    record = {"experiment": "E7_APTOS2019", "configuration": cfg, "summary": summary, "history": history,
              "checkpoint_sidecars": {name: parse(originals[name + ".sha256.json"]) for name in CHECKPOINTS},
              "provenance": {"archive_sha256": ARCHIVE_SHA, "source_location_in_hpc": RUN,
                             "members": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in originals.items()},
                             "binary_hash_verification": "HPC operator independently hashed both binaries; sidecars match those digests"},
              "audit": {"source_files_verified": len(cfg["source_sha256"]), "epoch_records": len(history),
                        "confusion_matrices_checked": 2 * len(history), "local_manifest_and_sidecar_verified": bool(manifest_dir),
                        "checkpoint_binary_available_in_archive": False}}
    record["derived_best_validation"] = audit(record)
    check_public(record)
    return record


def history_csv(history):
    def flatten(row):
        return {f"{name}.{key}" if isinstance(value, dict) else name:
                json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) if isinstance(item, (list, dict)) else item
                for name, value in row.items()
                for key, item in (value.items() if isinstance(value, dict) else [(None, value)])}

    rows = [flatten(row) for row in history]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()

def comparison_table(run):
    records = tuple(comparison(run).values())
    return prior.table(["Selected validation metric", "E5", "E6", "E7"], [
        ("Best epoch", *(record["summary"]["best"]["epoch"] for record in records)),
        *[(label, *(f"{prior.best(record)[key]:.9f}" for record in records))
          for key, label in (("macro_f1", "Macro-F1"), ("accuracy", "Accuracy"),
              ("quadratic_weighted_kappa", "QWK"), ("grade_mae", "Grade MAE"), ("weighted_f1", "Weighted-F1"))],
    ])


def report(run):
    cfg, history = run["configuration"], run["history"]
    scores, derived = prior.best(run), run["derived_best_validation"]
    table = prior.table
    times = [sum(e["resource"][key] for e in history) for key in ("train_seconds", "validation_seconds")]
    peaks = [max(e["resource"][key] for e in history) / 2**30
             for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "process_peak_rss_bytes")]
    ce_min = min(history, key=lambda e: e["validation"]["validation_loss"])
    sections = [
        "# E7 Global/Local ConvNeXt-Tiny on APTOS2019",
        "**Status: VALIDATION_COMPLETE.** The original run completed 20 epochs and selected epoch 19. "
        "The 366-image validation split supplies every result on this page. The run records no final-test access.",
        "## Research question and architecture",
        "E7 asks whether combining the full retinal field with localized views changes grading performance. "
        "One shared HF ConvNeXt V1 processes a full 384×384 letterbox and two 224×224 letterboxes from overlapping left/right crops. "
        "Each crop spans 55% of estimated field width and height, centered vertically. All views use EXIF-corrected RGB and official rescale/normalization with processor resize disabled. "
        "The gate combines global and mean local embeddings with learned sigmoid weights. A categorical head supplies grade predictions and an ordered-threshold head supplies auxiliary training supervision. "
        "The learned gate and geometric crop positions do not localize verified lesions.",
        f"The recorded {run['summary']['parameter_counts']['total']:,} trainable parameters agree with the E5 trainable count plus "
        "1,537 fusion-gate parameters. Backbone parameters are shared across all three views. "
        "[Model](../../../novelty_integrated/e7/model.py) · [Geometry](../../../novelty_integrated/e7/geometry.py) · "
        "[Runner and protocol](../../../novelty_integrated/e7/README.md) · [Notebook](../../../notebooks/novelty/E7_APTOS2019.ipynb).",
        "## Training and validation protocol",
        f"E7 started from APTOS E5 epoch {cfg['init']['checkpoint_epoch']} with strict backbone/head transfer. "
        f"The E5 checkpoint SHA-256 is `{cfg['init']['checkpoint_sha256']}`. The new gate was initialized by the E7 model constructor. "
        "The run used seed 42, BF16, full fine-tuning, micro-batch 4 and accumulation 8 (effective batch 32, final window 18). "
        "AdamW used backbone LR 1e−5, head LR 1e−3, weight decay 0.01 and gradient clipping at 1.0. No scheduler or EMA was used.",
        "The training objective is categorical CE plus 0.25 × ordinal BCE. Train-only inverse-square-root weights affect "
        "categorical CE from epoch 11, with weights normalized to mean one per training example. Validation loss is ordinary "
        "categorical CE. Validation is deterministic. B1 augmentation applies only to training and precedes view generation. "
        "The recorded training cache path points to the E6 decoded-RGB cache; validation has a separate E7 decoded-RGB cache. "
        "Checkpoint selection uses raw weights, highest validation macro-F1, lower validation CE for a tie, then the earlier epoch.",
        f"The frozen APTOS manifest SHA-256 is `{cfg['manifest']['manifest_sha256']}`. "
        "The manifest contains 2,930 train and 366 validation rows, with 2,832 and 351 SHA groups respectively. "
        "The local manifest and sidecar copies matched the recorded hashes, grade support, and train/validation group separation. "
        "No images were opened in this audit. The training record reports source-image verification.",
        "## Complete epoch history",
        "The [audited JSON](validation_evidence.json) preserves all recorded fields and numerical precision, including "
        "40 train/validation confusion matrices, grade statistics and resource records. The [derived CSV](history.csv) "
        "flattens those same 20 source JSON rows. The ZIP did not contain the original runner CSV.",
    ]
    for split in ("train", "validation"):
        loss = ([("Total loss", "train.loss"), ("Categorical", "train.categorical_loss"), ("Ordinal BCE", "train.ordinal_loss")]
                if split == "train" else [("Categorical CE", "validation.validation_loss")])
        sections += [f"### {split.capitalize()} scores", prior.epoch_table(
            history, loss + [(value, f"{split}.{key}") for key, value in SCORES.items()])]
    sections += ["### Recorded resources",
        prior.epoch_table(history, [(label, f"resource.{key}") for key, label in (
            ("train_seconds", "Train s"), ("validation_seconds", "Validation s"),
            ("train_images_per_second", "Train images/s"), ("validation_images_per_second", "Val images/s"),
            ("gpu_peak_allocated_bytes", "GPU allocated B"), ("gpu_peak_reserved_bytes", "GPU reserved B"),
            ("process_peak_rss_bytes", "Peak process RSS B"))]),
        f"Summed epoch times were {times[0]:,.2f} s training and {times[1]:,.2f} s validation "
        f"({sum(times) / 3600:.2f} h combined). Peaks were {peaks[0]:.3f} GiB GPU allocated, "
        f"{peaks[1]:.3f} GiB GPU reserved and {peaks[2]:.3f} GiB process RSS. These are logged training resource "
        "records. The ZIP has no CPU or GPU utilization series or deployment latency measurement.",
        "## Selected checkpoint and grading errors",
        table(["Metric", "Epoch 19"], [(label, f"{scores[key]:.9f}") for key, label in SCORES.items()] +
              [("Validation CE", f"{scores['validation_loss']:.9f}"), ("Correct", scores["correct"]),
               ("Incorrect", scores["wrong"]), ("Undergraded", scores["undergraded"]), ("Overgraded", scores["overgraded"])]),
        "True grades are rows and predictions are columns.",
        table(["True grade", *range(5)], [(i, *row) for i, row in enumerate(scores["confusion_matrix"])]),
        "Row-normalized fractions",
        table(["True grade", *range(5)], [(i, *row) for i, row in enumerate(derived["row_normalized"])]),
        table(["Grade", "Support", "Precision", "Recall", "F1", "Under", "Over"],
              [tuple(g[k] for k in ("grade", "support", "precision", "recall", "f1", "undergraded", "overgraded"))
               for g in scores["per_grade"]]),
        f"Grade 1 recall was 24/35. Ten grade-1 images were graded 2 and one was graded 0. "
        f"Grade 3 recall was 8/20, the lowest of the five grades. Seven grade-3 images were graded 2 and five were graded 4. "
        f"Of 60 errors, {derived['adjacent']} were one grade away and {derived['non_adjacent']} were at least two grades away. "
        "The aggregate evidence does not identify clinical causes of these errors.",
        f"Validation CE reached its minimum at epoch {ce_min['epoch']} ({ce_min['validation']['validation_loss']:.6f}). "
        "Epoch 19 was selected on macro-F1 under the original rule. Train loss and validation CE measure different objectives.",
        "## APTOS2019 comparison with E5 and E6",
        comparison_table(run),
        "E5 achieved the highest macro-F1. E7 improved accuracy, weighted-F1 and QWK and reduced grade MAE relative to E5, "
        "while reducing grade-1 and grade-3 recall. E6 trailed E5 on the listed aggregate metrics. The selected checkpoints share the frozen validation set. "
        "The comparison uses a single seed and the same validation data for checkpoint selection, with no significance test. "
        "E7 warm-started from E5 and adds three-view inference, so optimization budget and computation differ. "
        "These results do not establish a gain attributable to the views or clinical superiority. "
        "[E5 validation report](../e5_integrated/validation_report.md) · [E6 validation report](../e6_ldam_ordinal_adjacent/validation_report.md).",
        "## Figure index",
        table(["Panel", "Evidence shown"], [(f"[PNG](../../../docs/figures/aptos2019_e7_{key}.png) / "
             f"[SVG](../../../docs/figures/aptos2019_e7_{key}.svg)", text) for key, text in FIGURES.items()]),
        "## Provenance and limitations",
        f"The run was archived under `{RUN}` relative to the experiment root. All 18 recorded source digests match "
        f"Git commit `{SOURCE_COMMIT}` and the local source. The transfer SHA-256 is `{ARCHIVE_SHA}`. "
        "The JSON records the five original member hashes, the processor revision and hashes, cache report digests and runtime settings. "
        "Absolute dataset roots were omitted and cache roots converted to paths relative to the experiment root.",
        table(["Checkpoint", "Bytes", "SHA-256"], [(name, receipt["bytes"], receipt["sha256"])
              for name, receipt in run["checkpoint_sidecars"].items()]),
        "The archived sidecars match the values independently hashed on HPC and supplied by the operator. "
        "The ZIP contains no binaries, source images, row-level predictions, or original smoke/benchmark reports. "
        "No E7 checkpoint was loaded or exported, and no latency benchmark was run here. DDR completion remains unverified.",
    ]
    return "\n\n".join(s.strip() for s in sections) + "\n"


def plot(run, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.4, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "svg.hashsalt": "e7-aptos-validation", "savefig.dpi": 270})
    blue, orange, green = prior.COLORS["E5"], prior.COLORS["E6"], "#247553"
    history, epochs = run["history"], list(range(1, 21))
    styles = ((blue, "-"), (orange, "--"), (green, "-."), ("#744970", ":"))

    def layout(rows, cols, title, width=11, height=6.4):
        fig, axes = plt.subplots(rows, cols, figsize=(width, height), layout="constrained", squeeze=False)
        fig.suptitle(f"APTOS2019  |  {title}", fontsize=17, fontweight="bold", x=.02, ha="left")
        return fig, axes.ravel()

    def save(fig, key):
        stem = destination / f"aptos2019_e7_{key}"
        fig.savefig(stem.with_suffix(".png"), dpi=270, facecolor="white")
        prior.save_svg(fig, stem.with_suffix(".svg"))
        plt.close(fig)

    def curve(ax, title, fields, *, unit=None, scale=1):
        for (path, label), (color, style) in zip(fields, styles):
            ax.plot(epochs, [prior.get(row, path) / scale for row in history], style, color=color,
                    label=label, linewidth=2)
        ax.axvline(19, color="#606060", linewidth=1, linestyle=":", label="Selected epoch 19")
        if unit:
            ax.set_ylabel(unit)
        ax.set(title=title, xlabel="Epoch", xticks=[1, 5, 10, 15, 19])
        ax.grid(alpha=.20)
        ax.legend(loc="best", fontsize=8.8, frameon=False)

    fig, axes = layout(2, 3, "E7 grade prediction over 20 epochs")
    for ax, (key, label) in zip(axes, SCORES.items()):
        curve(ax, label, [(f"train.{key}", "Train"), (f"validation.{key}", "Validation")])
    axes[-1].axis("off")
    axes[-1].text(.08, .65, "Frozen validation: 366 images\n\nSelected checkpoint: epoch 19\nRanked by validation macro-F1",
                  transform=axes[-1].transAxes, va="center")
    save(fig, "epoch_scores")

    fig, axes = layout(1, 2, "E7 training objective and validation CE", height=4.6)
    curve(axes[0], "Train loss components", [("train.loss", "Total"), ("train.categorical_loss", "Categorical"),
                                            ("train.ordinal_loss", "Ordinal BCE (unscaled)")])
    axes[0].axvline(11, color="#744970", linewidth=1, linestyle="--")
    axes[0].text(11.15, .03, "Class weights active", color="#744970", transform=axes[0].get_xaxis_transform())
    curve(axes[1], "Validation loss", [("validation.validation_loss", "Categorical CE")])
    save(fig, "losses")

    fig, axes = layout(2, 2, "E7 measured training resources")
    for ax, (title, unit, fields, scale) in zip(axes, (
        ("Epoch duration", "Seconds", (("train_seconds", "Train"), ("validation_seconds", "Validation")), 1),
        ("Throughput", "Images/s", (("train_images_per_second", "Train"), ("validation_images_per_second", "Validation")), 1),
        ("GPU allocator peaks", "GiB", (("gpu_peak_allocated_bytes", "Allocated"), ("gpu_peak_reserved_bytes", "Reserved")), 2**30),
        ("Process RSS high-water mark", "GiB", (("process_peak_rss_bytes", "Peak RSS"),), 2**30))):
        curve(ax, title, [(f"resource.{key}", label) for key, label in fields], unit=unit, scale=scale)
    save(fig, "resources")

    best = prior.best(run)
    fig, axes = layout(1, 2, "E7 epoch 19 validation confusion", width=10, height=5)
    for ax, (matrix, normalized) in zip(axes, ((np.array(best["confusion_matrix"]), False),
            (np.array(run["derived_best_validation"]["row_normalized"]), True))):
        max_color = 1 if normalized else matrix.max()
        edges = np.arange(6) - .5
        im = ax.pcolormesh(edges, edges, matrix, cmap="Blues", vmin=0, vmax=max_color, rasterized=False)
        bar = fig.colorbar(im, ax=ax, shrink=.80)
        bar.solids.set_rasterized(False)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.1%}" if normalized else str(value), ha="center", va="center",
                    color="white" if value > .55 * max_color else "#1b2e40")
        ax.set(title="True-grade row fractions" if normalized else "Counts",
               xlabel="Predicted grade", ylabel="True grade", xticks=range(5), yticks=range(5))
    save(fig, "confusion")

    fig, axes = layout(1, 2, "E7 epoch 19 grade-level performance", height=4.6)
    grades = np.arange(5)
    by_grade = best["per_grade"]
    for i, key in enumerate(("precision", "recall", "f1")):
        axes[0].bar(grades + (i - 1) * .24, [item[key] for item in by_grade], width=.24,
                    color=styles[i][0], label=key.upper() if key == "f1" else key.capitalize())
    for i, key in enumerate(("undergraded", "overgraded")):
        bars = axes[1].bar(grades + (i - .5) * .34, [item[key] for item in by_grade], width=.34,
                           color=styles[i][0], label=key.capitalize())
        axes[1].bar_label(bars, fontsize=8, padding=2)
    axes[0].set(title="Precision, recall and F1", ylim=(0, 1.08))
    axes[1].set(title="Direction of errors", ylabel="Images", ylim=(0, 15))
    for ax in axes:
        ax.set_xticks(grades, [f"{i}\n(n={g['support']})" for i, g in enumerate(by_grade)])
        ax.set_xlabel("True grade and support")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=.2)
    save(fig, "grade_errors")

    fig, axes = layout(2, 4, "Selected E5, E6 and E7 validation checkpoints", width=13)
    records = comparison(run)
    names = list(records)
    selected = [prior.best(r) for r in records.values()]
    values = [(label, [r[key] for r in selected]) for key, label in SCORES.items()]
    values += [(f"Grade {grade} recall", [r["per_grade"][grade]["recall"] for r in selected]) for grade in (1, 3)]
    for ax, (label, series) in zip(axes, values):
        bars = ax.bar(names, series, color=(blue, orange, green), width=.63)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_ylim(0, .33 if label == "Grade MAE" else 1.10)
        ax.set_title(label)
        ax.grid(axis="y", alpha=.2)
    axes[-1].axis("off")
    axes[-1].text(.02, .64, "Same 366 validation images\n\nSelected epochs\nE5  ·  16\nE6  ·  15\nE7  ·  19",
                  transform=axes[-1].transAxes, va="center")
    save(fig, "e5_e6_e7_comparison")


def write(run, destination):
    folder, figures = destination / FOLDER, destination / "docs/figures"
    paths = [folder / name for name in ("validation_evidence.json", "validation_report.md", "history.csv")]
    paths += [figures / f"aptos2019_e7_{key}.{ext}" for key in FIGURES for ext in ("png", "svg")]
    require(not any(path.exists() for path in paths), "E7 publication outputs already exist")
    folder.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plot(run, figures)
    report_text = report(run)
    check_public(report_text)
    for path, value in zip(paths[:3], (json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                      report_text, history_csv(run["history"]))):
        path.write_text(value, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    write(load_export(args.archive, args.manifest_dir), args.destination)


if __name__ == "__main__":
    main()
