"""Audit and plot the frozen pilot_100_v1 aggregate export, without model or dataset access."""

import argparse
import csv
from functools import lru_cache
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

from docs.e5_e6_publication import DATASETS, HEX, ROOT, check_public, equal, require, sha, verify_scores
from docs.e7_publication import parse
from novelty_integrated.optimization.reporting import validation_summary


FOLDER = Path("results/optimization/pilot_100_v1")
ARCHIVE = "PRUNING_PILOT_EVIDENCE.zip"
ARCHIVE_SHA = "bb3f0968aa7130f59763049b00ab29e38ab2f7d715477263b7988c6766ecdb4b"
ARCHIVE_BYTES = 55_642
JOBS = ("E5_APTOS2019", "E6_APTOS2019", "E7_APTOS2019", "E5_DDR", "E6_DDR")
VARIANTS = {"original": 0., "pruned_10": .1, "pruned_30": .3, "pruned_50": .5}
BACKENDS = ("eager", "torchscript", "onnxruntime", "openvino")
BACKEND_LABEL = dict(zip(BACKENDS, ("PyTorch Eager", "TorchScript", "ONNX Runtime", "OpenVINO")))
ARTIFACTS = dict(zip(BACKENDS, (["inference_state_dict.pt"], ["model.torchscript.pt"],
                                ["model.onnx"], ["model.xml", "model.bin"])))
RUNS = {"E5": "e5_integrated", "E6": "e6_ldam_ordinal_adjacent", "E7": "e7_global_local"}
PREPARE_COMMIT = "c101b1a5f72ee027bbb3adff8dd260536187c3e5"
VALIDATE_COMMITS = {job: ("c7e61a845c329f5ced544a23e2062cb377b075ba" if job == "E6_DDR"
                          else "f406ddfa78bdfad58fe6528c60c6d6c0f63ef158") for job in JOBS}
FILES = ("protocol.json", "validation/report.json", "validation/summary.csv")
METRICS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1",
           "quadratic_weighted_kappa": "QWK", "grade_mae": "Grade MAE"}
TOLERANCES = {"atol": 1e-4, "rtol": 1e-4}
STEMS = ("pruning_pilot_aptos2019_quality", "pruning_pilot_ddr_quality",
         "pruning_pilot_macro_f1_delta", "pruning_pilot_backend_parity")
# Status in the original validation export. Supplementary evidence lives in separate files.
STUDY_STATUS = {"complexity": "MISSING_PREPARED_PROFILES",
                "post_recovery": {job: "UNVERIFIED" for job in ("E5_APTOS2019", "E7_APTOS2019")},
                "latency": "NOT_MEASURED"}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def receipt(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


@lru_cache
def source_hashes(commit):
    prefix = "novelty_integrated/optimization/"
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", prefix],
                                    cwd=ROOT).decode().splitlines()
    return {path.removeprefix(prefix): sha(subprocess.check_output(["git", "show", f"{commit}:{path}"],
                                                                 cwd=ROOT))
            for path in paths if path.removeprefix(prefix).endswith(".py")}


def require_hash(value):
    require(isinstance(value, str) and HEX.fullmatch(value), "Invalid SHA-256 receipt")


def parent_receipt(recorded, arm, dataset):
    path = f"results/{dataset}/{RUNS[arm]}/validation_evidence.json"
    raw = (ROOT / path).read_bytes()
    require(recorded["evidence"] == {"path": path, "sha256": sha(raw)}, f"{arm} {dataset}: evidence receipt differs")
    evidence = parse(raw)
    best = evidence["checkpoint_sidecars"]["best.pt"]
    cfg = evidence["configuration"]
    require((recorded["checkpoint_sha256"], recorded["checkpoint_bytes"], recorded["epoch"]) ==
            (best["sha256"], best["bytes"], evidence["summary"]["best"]["epoch"]), f"{arm} {dataset}: parent checkpoint differs")
    require(recorded["manifest"]["sha256"] == cfg["manifest"]["manifest_sha256"] == DATASETS[dataset][2] and
            recorded["effective_processor_sha256"] == cfg["effective_processor_sha256"] and
            recorded["snapshot"] == cfg["snapshot"] and recorded["source"]["tracked"] == cfg["source_sha256"],
            f"{arm} {dataset}: manifest, processor, snapshot or training-source receipt differs")
    require_hash(recorded["manifest"]["sidecar_sha256"])
    return cfg["manifest"]["grade_support"]["val"]


def validate_backend(variant, name, support, label):
    result = variant["backends"][name]
    parity = result["parity"]
    require(result["status"] == "VALIDATED" and parity["passed"] is True and
            parity["changed_order_indices"] == [] and parity["prediction_agreement"] == 1.0 and
            all(parity[k] == v for k, v in TOLERANCES.items()) and
            0 <= parity["mean_absolute"] <= parity["rms"] <= parity["maximum_absolute"] <= TOLERANCES["atol"] and
            parity["maximum_relative_reference_floor_1e-6"] >= 0, f"{label}: parity record differs")
    require(name != "eager" or parity["maximum_absolute"] == 0, f"{label}: eager self-comparison differs")
    settings = result["settings"]
    require((settings.get("provider") == "CPUExecutionProvider" and
             settings.get("intra_op_threads") == settings.get("inter_op_threads") == 1)
            if name == "onnxruntime" else (settings.get("threads") == 1), f"{label}: thread/provider differs")
    if name == "openvino":
        require(settings["device"] == "CPU" and settings["precision_hint"] == "f32", f"{label}: OpenVINO settings differ")
    artifact = result["artifact"]
    require([f["file"] for f in artifact["files"]] == ARTIFACTS[name] and
            artifact["bytes"] == sum(f["bytes"] for f in artifact["files"]), f"{label}: artifact receipt differs")
    for f in artifact["files"]:
        require(type(f["bytes"]) is int and f["bytes"] > 0, f"{label}: invalid artifact size")
        require_hash(f["sha256"])
    q = result["quality"]
    derived = verify_scores(q, support, label)
    require(all(equal(a["mean_absolute_error"], b["mean_absolute_error"])
                for a, b in zip(q["per_grade"], derived["per_grade"], strict=True)),
            f"{label}: per-grade MAE differs")
    require(q["confusion_matrix"] == variant["backends"]["eager"]["quality"]["confusion_matrix"],
            f"{label}: classification counts differ despite 100% agreement")


def validate_job(job, protocol, report):
    arm, dataset = job.split("_")
    support = parent_receipt(protocol["receipt"], arm, dataset)
    expected_protocol = {"backends": list(BACKENDS), "batch_size": 1, "parity_tolerances": TOLERANCES,
                         "repeats": 3, "sample_count": 100, "sample_seed": 42, "warmup": 20}
    require(protocol["schema"] == "convnext-optimization-staged-v1" and
            protocol["protocol"] == expected_protocol and protocol["ratios"] == [.1, .3, .5] and
            protocol["arm"] == arm and protocol["dataset"] == dataset and
            protocol["framework_source_sha256"] == source_hashes(PREPARE_COMMIT), f"{job}: prepared protocol differs")
    expected_report = {"schema": "convnext-optimization-validation-v1", "stage": "validate",
                       "status": "VALIDATION_COMPLETE", "outcome": "PASS", "failures": 0, "backend_rows": 16,
                       "arm": arm, "dataset": dataset, "prepared_framework_source_sha256": protocol["framework_source_sha256"],
                       "framework_source_sha256": source_hashes(VALIDATE_COMMITS[job]),
                       "protocol_sha256": canonical_sha(protocol), "sample": protocol["sample"],
                       "parity_tolerances": TOLERANCES, "receipt": protocol["receipt"], "backends": list(BACKENDS)}
    require(all(report.get(key) == value for key, value in expected_report.items()) and
            report["timing_performed"] is False, f"{job}: validation identity/status differs")
    sample = report["sample"]
    positions = sample["frozen_validation_positions_zero_based"]
    require(sample["split"] == "val" and sample["count"] == 100 and sample["class_order"] == list(range(5)) and
            len(positions) == 100 and all(type(i) is int and 0 <= i < sum(support) for i in positions) and
            positions == sorted(set(positions)) and sum(sample["grade_support"]) == 100 and
            all(type(n) is int and 0 < n <= available for n, available in
                zip(sample["grade_support"], support, strict=True)), f"{job}: frozen sample differs")
    r, a = report["runtime"], report["allocation"]
    require(r["device"] == "cpu" and r["precision"] == "float32" and r["versions"] == protocol["versions"] and
            r["torch_threads"] == r["torch_interop_threads"] == a["threads"] == 1 and r["affinity"] == a["cpus"] and
            len(a["cpus"]) == 1 and a["cpu_mode"] == "shared" and a["exclusive_cpu_confirmed"] is False,
            f"{job}: run is not the untimed shared-CPU FP32 validation")
    require_hash(report["ordered_preprocessed_tensor_sha256"])
    require(set(report["prepared_receipt_sha256"]) == set(VARIANTS), f"{job}: missing preparation receipts")
    for digest in report["prepared_receipt_sha256"].values():
        require_hash(digest)
    require([(v["name"], v["ratio"]) for v in report["variants"]] == list(VARIANTS.items()), f"{job}: variants differ")
    for variant in report["variants"]:
        require(set(variant["backends"]) == set(BACKENDS), f"{job}: backend set differs")
        for name in BACKENDS:
            validate_backend(variant, name, sample["grade_support"], f"{job}/{variant['name']}/{name}")


def verify(bundle):
    check_public(bundle)
    require(bundle["schema"] == "convnext-pruning-pilot-publication-v1" and bundle["campaign"] == FOLDER.name and
            bundle["archive"] == {"file": ARCHIVE, "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA} and
            bundle["status"] == STUDY_STATUS and set(bundle["jobs"]) == set(JOBS), "Wrong pilot export")
    for job in JOBS:
        data = bundle["jobs"][job]
        protocol, report = data["protocol"], data["validation"]
        validate_job(job, protocol, report)
        canonical = (json_bytes(protocol), json_bytes(report), validation_summary(report).encode())
        require(data["original_members"] == {f"{job}/{name}": receipt(raw) for name, raw in
                                             zip(FILES, canonical, strict=True)}, f"{job}: original file receipts differ")
    for dataset in DATASETS:
        e5, e6 = [bundle["jobs"][f"{arm}_{dataset}"]["validation"] for arm in ("E5", "E6")]
        require(e5["sample"] == e6["sample"] and
                e5["receipt"]["manifest"] == e6["receipt"]["manifest"] and
                e5["ordered_preprocessed_tensor_sha256"] == e6["ordered_preprocessed_tensor_sha256"],
                f"{dataset}: E5/E6 sample or processor inputs differ")
    e5, e7 = [bundle["jobs"][f"{arm}_APTOS2019"]["validation"] for arm in ("E5", "E7")]
    require(e5["sample"] == e7["sample"] and e5["receipt"]["manifest"] == e7["receipt"]["manifest"] and
            e5["ordered_preprocessed_tensor_sha256"] != e7["ordered_preprocessed_tensor_sha256"],
            "APTOS2019 E7 must share the same ordered sample with distinct three-view tensors")
    return bundle


def load_archive(path):
    require(path.stat().st_size == ARCHIVE_BYTES, "Archive byte count differs from the reviewed export")
    raw = path.read_bytes()
    require(sha(raw) == ARCHIVE_SHA, "Archive SHA-256 differs from the reviewed export")
    with ZipFile(BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) == len(JOBS) * len(FILES) and
                {member.filename for member in members} == {f"{job}/{name}" for job in JOBS for name in FILES} and
                all(0 < member.file_size < 256 * 1024 for member in members) and archive.testzip() is None,
                "Unexpected ZIP inventory, size or CRC")
        jobs = {}
        for job in JOBS:
            data = {f"{job}/{name}": archive.read(f"{job}/{name}") for name in FILES}
            protocol, report, summary = data.values()
            require(summary == validation_summary(parse(report)).encode(), f"{job}: CSV differs from validation JSON")
            jobs[job] = {"protocol": parse(protocol), "validation": parse(report),
                         "original_members": {name: receipt(value) for name, value in data.items()}}
    return verify({"schema": "convnext-pruning-pilot-publication-v1", "campaign": FOLDER.name,
                   "archive": {"file": ARCHIVE, **receipt(raw)}, "status": STUDY_STATUS, "jobs": jobs})


def quality_rows(bundle, jobs=JOBS):
    for job in jobs:
        variants = bundle["jobs"][job]["validation"]["variants"]
        baseline = variants[0]["backends"]["eager"]["quality"]
        for arm in variants:
            q = arm["backends"]["eager"]["quality"]
            yield {"job": job, "target_pruning_percent": round(100 * arm["ratio"]), "samples": 100,
                   **{k: q[k] for k in METRICS}, **{f"delta_{k}": q[k] - baseline[k] for k in METRICS},
                   **{f"grade_{g}_recall": q["per_grade"][g]["recall"] for g in (1, 3)}}


def parity_rows(bundle, jobs=JOBS):
    for job in jobs:
        for arm in bundle["jobs"][job]["validation"]["variants"]:
            for backend in BACKENDS:
                result = arm["backends"][backend]
                p = result["parity"]
                yield {"job": job, "target_pruning_percent": round(100 * arm["ratio"]), "backend": backend,
                       "reference": "eager", "sample_count": 100, "device": "CPU", "precision": "FP32",
                       "status": result["status"], "parity_status": "PASS" if p["passed"] else "FAIL",
                       "argmax_agreement": p["prediction_agreement"], "maximum_absolute_logit_difference": p["maximum_absolute"],
                       "atol": p["atol"], "rtol": p["rtol"], "artifact_bytes": result["artifact"]["bytes"]}


def csv_bytes(rows):
    first, *remaining = rows
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=first, lineterminator="\n")
    writer.writeheader()
    writer.writerows([first, *remaining])
    return stream.getvalue().encode()


def markdown_table(columns, rows):
    return "\n".join("| " + " | ".join(row) + " |" for row in (columns, ["---"] * len(columns), *rows))


def quality_table(bundle, jobs=JOBS):
    return markdown_table(
        ["Job", "Target", *METRICS.values(), "Δ Macro-F1", "Grade 1 recall", "Grade 3 recall"],
        [[row["job"], f"{row['target_pruning_percent']}%", *(f"{row[k]:.4f}" for k in METRICS),
          f"{row['delta_macro_f1']:+.4f}", f"{row['grade_1_recall']:.4f}", f"{row['grade_3_recall']:.4f}"]
         for row in quality_rows(bundle, jobs)])


def parity_table(bundle, jobs=JOBS):
    rows = list(parity_rows(bundle, jobs))
    values = []
    for job in jobs:
        for name in BACKENDS[1:]:
            matched = [row for row in rows if row["job"] == job and row["backend"] == name]
            values.append([job, BACKEND_LABEL[name], f"{sum(r['parity_status'] == 'PASS' for r in matched)}/{len(matched)}",
                           f"{100 * min(r['argmax_agreement'] for r in matched):.0f}%",
                           f"{max(r['maximum_absolute_logit_difference'] for r in matched):.6g}"])
    return markdown_table(["Job", "Backend vs eager", "PASS", "Min. argmax agreement", "Max. absolute logit difference"], values)


def notebook_evidence(bundle, job, *, supplement_text=None):
    """Render static Markdown from the same records as the campaign report and figures."""
    report = bundle["jobs"][job]["validation"]
    dataset = report["dataset"]
    stem = STEMS[0] if dataset == "APTOS2019" else STEMS[1]
    support = ", ".join(map(str, report["sample"]["grade_support"]))
    rows = list(quality_rows(bundle, [job]))
    n_grade0 = sum(row[0] for row in report["variants"][-1]["backends"]["eager"]["quality"]["confusion_matrix"])
    worst = max(r["maximum_absolute_logit_difference"] for r in parity_rows(bundle, [job]))
    if supplement_text is None:
        supplement_text = ("Post-recovery evidence is `UNVERIFIED` in this export. Prepared-profile parameter counts and MACs "
                           "are unavailable. Shared CPU validation recorded no timing, so latency is `NOT_MEASURED`.")
    return (
        "## Published structured-pruning pilot — before recovery\n\n"
        f"`pilot_100_v1` used the same ordered 100 frozen validation cases across variants, "
        f"batch size 1 and untimed FP32 inference on a shared CPU. Grade support in 0–4 order is [{support}]. "
        f"Checkpoint epoch {report['receipt']['epoch']} came from the original full-validation selection. "
        f"The original row here uses the pilot subset. The reduction target refers to MLP expansion channels.\n\n"
        f"{quality_table(bundle, [job])}\n\n"
        f"The 10% arm changed Macro-F1 by {rows[1]['delta_macro_f1']:+.4f}. "
        f"At 50%, the model predicted grade 0 on {n_grade0}/100 cases. There are only "
        f"{report['sample']['grade_support'][3]} grade-3 cases, so even one case changes that recall substantially.\n\n"
        f"![Macro-F1 and QWK across pre-recovery pruning ratios for {dataset}](../../docs/figures/{stem}.png)\n\n"
        f"### Numerical parity\n\n"
        f"{parity_table(bundle, [job])}\n\n"
        f"All 12 non-eager comparisons agreed with the corresponding eager argmax. "
        f"The largest absolute logit difference was {worst:.6g} under the recorded FP32 "
        f"absolute and relative tolerances of 0.0001. Agreement checks the export against its own variant.\n\n"
        f"### Recovery, complexity and CPU timing\n\n"
        f"{supplement_text}\n\n"
        f"[Campaign report](../../docs/PRUNING_PILOT_RESULTS.md) · "
        f"[Evidence and confusion matrices](../../{(FOLDER / 'validation_evidence.json').as_posix()}) · "
        f"[Per-variant parity CSV](../../{(FOLDER / 'numerical_parity.csv').as_posix()}) · "
        f"[Python generator](../../docs/pruning_pilot_publication.py)"
    )


def save_figure(figure, output, stem, rows, evidence="validation_evidence.json"):
    data = json.dumps({"evidence": (FOLDER / evidence).as_posix(),
                       "rows": rows}, sort_keys=True, separators=(",", ":"))
    figure.savefig(output / f"{stem}.png", dpi=200, metadata={"Description": data})
    svg = StringIO()
    figure.savefig(svg, format="svg", metadata={"Date": None, "Description": data})
    (output / f"{stem}.svg").write_bytes(("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n").encode())


def plot(bundle, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        _plot(plt, bundle, output)


def _plot(plt, bundle, output):
    plt.rcParams.update({"svg.hashsalt": "pilot-100-v1", "svg.fonttype": "none", "font.family": "DejaVu Sans",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True, exist_ok=True)
    colors = {"E5": "#176383", "E6": "#a44918", "E7": "#6950a1"}
    for dataset, stem in zip(DATASETS, STEMS[:2], strict=True):
        jobs = [job for job in JOBS if job.endswith(dataset)]
        rows = list(quality_rows(bundle, jobs))
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
        for ax, key in zip(axes, ("macro_f1", "quadratic_weighted_kappa"), strict=True):
            for job in jobs:
                points = [r for r in rows if r["job"] == job]
                arm = job.split("_")[0]
                ax.plot([r["target_pruning_percent"] for r in points], [r[key] for r in points],
                        marker="o", color=colors[arm], label=arm)
            ax.set(xlabel="Target expansion pruning (%)", ylabel=METRICS[key], xticks=[0, 10, 30, 50], ylim=(-.04, 1))
            ax.grid(axis="y", alpha=.22)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{dataset} · before recovery · 100 frozen validation images")
        save_figure(fig, output, stem, rows)
        plt.close(fig)

    rows = [r for r in quality_rows(bundle) if r["target_pruning_percent"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for offset, job in enumerate(j for j in JOBS if j.endswith(dataset)):
            points = [r for r in rows if r["job"] == job]
            arm = job.split("_")[0]
            ax.bar([n + (offset - 1) * .22 for n in range(3)], [r["delta_macro_f1"] for r in points],
                   width=.21, color=colors[arm], label=arm)
        ax.set(title=dataset, xlabel="Target expansion pruning (%)", xticks=range(3), xticklabels=["10", "30", "50"],
               ylim=(-.59, .04))
        ax.axhline(0, color="#373737", linewidth=.8)
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="lower left", ncol=3, frameon=False)
    axes[0].set_ylabel("Macro-F1 difference vs same original")
    fig.suptitle("Pre-recovery quality change on the paired validation subset")
    save_figure(fig, output, STEMS[2], rows)
    plt.close(fig)

    rows = [r for r in parity_rows(bundle) if r["backend"] != "eager"]
    pairs = [(job, round(100 * ratio)) for job in JOBS for ratio in VARIANTS.values()]
    lookup = {(r["job"], r["target_pruning_percent"], r["backend"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(8.1, 9.2), layout="constrained")
    values = [[lookup[job, ratio, name]["maximum_absolute_logit_difference"] for name in BACKENDS[1:]]
              for job, ratio in pairs]
    image = ax.imshow(values, vmin=0, vmax=TOLERANCES["atol"], cmap="Blues", aspect="auto")
    for i, (job, ratio) in enumerate(pairs):
        for j, name in enumerate(BACKENDS[1:]):
            r = lookup[job, ratio, name]
            value = values[i][j]
            ax.text(j, i, f"{value:.1e}  {r['parity_status']}", ha="center", va="center", fontsize=9,
                    color="white" if value > 6.5e-5 else "#142a40")
    ax.set(xticks=range(3), xticklabels=[BACKEND_LABEL[k] for k in BACKENDS[1:]],
           yticks=range(len(pairs)), yticklabels=[f"{job.replace('_', ' · ')} · {ratio}%" for job, ratio in pairs])
    for boundary in range(4, len(pairs), 4):
        ax.axhline(boundary - .5, color="#7e96a8", lw=.8)
    colorbar = fig.colorbar(image, ax=ax, shrink=.88)
    colorbar.set_label("Maximum absolute logit difference")
    ax.set_title("Backend parity vs PyTorch Eager\n60 comparisons · 100% argmax agreement each", pad=12)
    save_figure(fig, output, STEMS[3], rows)
    plt.close(fig)


COMPLEXITY_STEMS = ("pruning_pilot_parameters", "pruning_pilot_macs", "pruning_pilot_quality_complexity")


def plot_complexity(bundle, profiles, output):
    """Plot recorded capacity and computation with the exact matching pilot quality rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    output.mkdir(parents=True, exist_ok=True)
    fp = {(row["job"], row["target_pruning_percent"]): row["macro_f1"] for row in quality_rows(bundle)}
    rows = [dict(row, macro_f1=fp[row["job"], row["target_pruning_percent"]]) for row in profiles]
    colors = {"E5": "#176383", "E6": "#a44918", "E7": "#6950a1"}
    styles = {"E5": ("-", "o", 8), "E6": ("--", "s", 4), "E7": ("-", "D", 5)}
    chart = [("parameters", 10**6, "Parameters (millions)", "Parameter count by pruning target"),
             ("macs_per_image", 10**9, "MACs per image (billions)", "Conv2d and Linear MACs by pruning target"),
             ("macro_f1", 1, "Macro-F1", "Pre-recovery Macro-F1 vs recorded MACs")]
    markers = dict(zip((0, 10, 30, 50), ("o", "s", "^", "D")))
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update({"svg.hashsalt": "pilot-100-v1-profiles", "svg.fonttype": "none",
                             "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
        for stem, (key, scale, ylabel, title) in zip(COMPLEXITY_STEMS, chart, strict=True):
            fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), layout="constrained", sharey=True)
            for ax, dataset in zip(axes, DATASETS, strict=True):
                for job in (j for j in JOBS if j.endswith(dataset)):
                    arm = job.split("_")[0]
                    data = [r for r in rows if r["job"] == job]
                    targets = [r["target_pruning_percent"] for r in data]
                    style, marker, size = styles[arm]
                    tradeoff = key == "macro_f1"
                    x = [r["macs_per_image"] / 10**9 for r in data] if tradeoff else targets
                    y = [r[key] / scale for r in data]
                    ax.plot(x, y, color=colors[arm], linestyle=style, label=arm, linewidth=1.8,
                            marker=None if tradeoff else marker, markersize=size)
                    if tradeoff:
                        for x_i, y_i, target in zip(x, y, targets, strict=True):
                            ax.plot(x_i, y_i, marker=markers[target], color=colors[arm], markersize=5)
                ax.set_title(dataset)
                ax.grid(axis="y", alpha=.22)
                ax.legend(frameon=False, loc="upper right")
                if key == "macro_f1":
                    ax.set(xlabel="MACs per image (billions, log scale)", xscale="log",
                           xticks=[3, 4, 10, 20], xlim=(2.05, 26), ylim=(.03, .76))
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                else:
                    ax.set(xlabel="Target expansion pruning (%)", xticks=[0, 10, 30, 50])
            axes[0].set_ylabel(ylabel)
            subtitle = ("100 frozen validation images per dataset · lines connect variants of the same checkpoint"
                        if key == "macro_f1" else "E5 and E6 traces overlap where recorded counts match · E7 uses three views")
            fig.suptitle(f"{title}\n{subtitle}")
            if key == "macro_f1":
                handles = [Line2D([], [], color="#343434", marker=m, linestyle="none", label=f"{r}%")
                           for r, m in markers.items()]
                axes[0].legend(handles, [h.get_label() for h in handles], loc="upper left", ncol=2,
                               frameon=False, title="Pruning target")
                axes[1].legend(
                    [Line2D([], [], color=colors[a]) for a in colors], list(colors), loc="upper right",
                    frameon=False, title="Model")
            save_figure(fig, output, stem, rows, "complexity.csv")
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Verify original HPC aggregate ZIP")
    parser.add_argument("--write-tables", type=Path, help="Write validated aggregate JSON and CSV in this directory")
    parser.add_argument("--figures", type=Path, help="Regenerate the PNG/SVG pairs in this directory")
    args = parser.parse_args(argv)
    source = ROOT / FOLDER
    bundle = load_archive(args.archive) if args.archive else verify(parse((source / "validation_evidence.json").read_bytes()))
    tables = {"quality.csv": csv_bytes(quality_rows(bundle)), "numerical_parity.csv": csv_bytes(parity_rows(bundle))}
    if args.write_tables:
        args.write_tables.mkdir(parents=True, exist_ok=True)
        (args.write_tables / "validation_evidence.json").write_bytes(json_bytes(bundle))
        for name, raw in tables.items():
            (args.write_tables / name).write_bytes(raw)
    for name, raw in tables.items():
        if not args.archive:
            require((source / name).read_bytes().replace(b"\r\n", b"\n") == raw, f"Published table differs: {name}")
    if args.figures:
        plot(bundle, args.figures)
    print("PASS: original ZIP evidence for 5 jobs, 20 variants, 80 FP32 backend rows and 60 parity comparisons.")


if __name__ == "__main__":
    main()
