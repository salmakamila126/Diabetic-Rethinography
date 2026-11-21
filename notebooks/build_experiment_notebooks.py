"""Render dataset notebooks from committed aggregate evidence without loading a model."""

import base64
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nb


ROOT = Path(__file__).resolve().parents[1]
BASE = "novelty/convnext_b0_b4"
BASE_EVIDENCE = f"{BASE}/validation_evidence.json"
FAMILIES = ("APTOS2019", "DDR")
INTEGRATED = {"E5": "e5_integrated", "E6": "e6_ldam_ordinal_adjacent"}
HYPOTHESES = {
    "B0": "Full fine-tuning with ordinary categorical CE and the deterministic official processor establishes the vanilla control.",
    "B1": "Conservative training augmentation may improve generalization while preserving retinal field of view.",
    "B2": "Train-only inverse square root class weighting, normalized to mean image weight one, may improve minority recognition. This separate arm starts from the unaugmented B0 protocol.",
    "B3": "B1 plus targeted grade 3 sampling to 10% with proportional non-grade-3 replacement tests whether more exposure improves grade 3 recognition.",
    "B4": "The B1-equivalent quarantine (Q) and removal control (R) test whether excluding train-only, exactly duplicated images with conflicting labels affects performance.",
    "E5": "A shared ordinal auxiliary head and deferred inverse square root class weighting test whether severity ordering and imbalance can be addressed together.",
    "E6": "A class-dependent margin and adjacent-grade penalties investigate whether more explicit boundary supervision changes grade 1 errors.",
    "E7": "The full 384 px field and two 224 px detail views test whether retaining more local information improves validation macro-F1. No result in this checkout tests that hypothesis.",
}
POOL = "Official Hugging Face ConvNeXt-Tiny V1, pooled width 768."
SPLIT = "Frozen seed 42 manifest, SHA group isolation, grade order 0–4. Validation uses deterministic transforms and original support. The notebook creates no split."
PROCESSOR = "EXIF transpose and RGB, then the pinned official processor at 224 px for resize/crop, rescale and normalization."
AUGMENTATION = ("B1 train-only augmentation is keyed by seed, epoch and image ID. It applies horizontal flip "
                "(p=0.50), field-of-view guarded rotation (p=0.50, ±7°), brightness (p=0.30, factor 0.90–1.10), "
                "and contrast (p=0.30, factor 0.90–1.10), in that order.")
COLS = (("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Weighted-F1", "weighted_f1"),
        ("QWK", "quadratic_weighted_kappa"), ("Grade MAE", "grade_mae"))
SETUP = """\
from pathlib import Path
import hashlib
import json
import os
import sys

from IPython.display import JSON, display

configured_root = os.environ.get("DR_RESEARCH_ROOT")
start = Path(configured_root).expanduser().resolve() if configured_root else Path.cwd().resolve()
ROOT = next((p for p in (start, *start.parents)
             if (p / "notebooks/experiment_workflow.py").is_file() and (p / "novelty_integrated").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Set DR_RESEARCH_ROOT to the reviewed repository checkout.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""
VERIFY = """\
VERIFY_LOCAL_PROVENANCE = False
if VERIFY_LOCAL_PROVENANCE:
    artifact = os.environ.get("DR_ARTIFACT_ROOT")
    checkpoint = os.environ.get("DR_CHECKPOINT")
    expected = CONFIG.get("manifest", {}).get("manifest_sha256")
    if not artifact or not expected:
        raise ValueError("Set DR_ARTIFACT_ROOT and use a notebook with a verified frozen manifest.")
    manifest = Path(artifact) / "cad_vit_runs" / "manifests" / f"{DATASET}_seed42.csv"
    with manifest.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("Frozen manifest hash differs.")
    print("Frozen manifest SHA-256 matches.")
    if checkpoint:
        run_id = os.environ.get("DR_RECORDED_RUN_ID")
        receipts = {r.get("id", r.get("experiment")):
                    r.get("source_sha256", {}).get("best.pt") or
                    r.get("checkpoint_sidecars", {}).get("best.pt", {}).get("sha256") for r in RUNS}
        expected_checkpoint = receipts.get(run_id) if run_id else CHECKPOINT_SHA256
        if len(RUNS) > 1 and not run_id:
            raise ValueError("Set DR_RECORDED_RUN_ID to the relevant B4-Q or B4-R evidence ID.")
        if not expected_checkpoint:
            raise ValueError("No repository-verified checkpoint SHA-256 exists for this run.")
        with Path(checkpoint).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_checkpoint:
                raise ValueError("Checkpoint hash differs.")
        print("Checkpoint SHA-256 matches without deserializing weights.")
"""
WORKFLOW = """\
from notebooks.experiment_workflow import run_step

ACTION = None
RUN_PREFLIGHT = False
RUN_CACHE = False
RUN_INPUT_VERIFICATION = False
RUN_GPU_SMOKE = False
RUN_BENCHMARK = False
RUN_TRAINING = False
RUN_CHECKPOINT_EXPORT = False
RUN_PRUNING = False
RUN_FINAL_TEST = False
if RUN_CHECKPOINT_EXPORT or RUN_PRUNING or RUN_FINAL_TEST:
    raise ValueError("Export, pruning and final-test inference are outside this workflow.")
run_step(ROOT, METHOD, DATASET, ACTION, CONFIG, {
    "preflight": RUN_PREFLIGHT, "estimate-cache": RUN_PREFLIGHT, "prepare-cache": RUN_CACHE,
    "verify-inputs": RUN_INPUT_VERIFICATION, "smoke": RUN_GPU_SMOKE,
    "benchmark": RUN_BENCHMARK, "train": RUN_TRAINING,
})
"""


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def md(source):
    return nb.v4.new_markdown_cell(dedent(source).strip())


def code(source, outputs=()):
    return nb.v4.new_code_cell(source.strip(), outputs=list(outputs))


def format_value(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "&#124;").replace("\n", " ")


def table(headers, rows):
    return "\n".join("| " + " | ".join(format_value(x) for x in row) + " |"
                     for row in (headers, ["---"]*len(headers), *rows))


def config_text(record):
    cfg = record["configuration"]
    fields = ("model_kind", "mode", "seed", "epochs", "micro_batch", "accumulation", "effective_batch",
              "precision", "device_type", "optimizer", "backbone_lr", "head_lr", "weight_decay",
              "max_grad_norm", "selection", "checkpoint_selection", "training_loss", "source_git_head",
              "implementation_sha256", "torch", "transformers", "pillow", "numpy")
    rows = [(f, cfg[f]) for f in fields if f in cfg]
    rows += [("manifest_sha256", cfg["manifest"]["manifest_sha256"]),
             ("train / validation counts", cfg["manifest"]["counts"]),
             ("model revision", cfg["snapshot"]["revision"]),
             ("model weights SHA-256", cfg["snapshot"]["weight_sha256"]),
             ("processor config SHA-256", cfg["snapshot"]["processor_config_sha256"])]
    extras = {key: cfg[key] for key in ("augmentation","b1_augmentation","class_weighting","objective",
              "sampling","b4","input_transport") if cfg.get(key) is not None}
    return table(["Recorded field", "Value"], rows) + "\n\n```json\n" + json.dumps(extras, indent=2) + "\n```"


def epoch_scores(history):
    sections = ["### Training metrics"]
    for split in ("train", "validation"):
        loss_cols = [("Total loss","loss"),("CE","validation_loss"),("Categorical term","categorical_loss"),
                     ("Ordinal","ordinal_loss"),("Adjacent","adjacent_loss")]
        cols = [(label,key) for label,key in loss_cols if any(key in e.get(split,{}) for e in history)]
        cols += list(COLS)
        sections.append(table(["Epoch", *(label for label,_ in cols)],
            [(e["epoch"], *(e.get(split,{}).get(key) for _,key in cols)) for e in history]))
        if split == "train":
            sections.append("### Validation metrics")
    resources = [("Train s","train_seconds"),("Validation s","validation_seconds"),
                 ("Train input s","train_input_seconds"),("Val input s","validation_input_seconds"),
                 ("Train images/s","train_images_per_second"),("Val images/s","validation_images_per_second"),
                 ("Peak VRAM B","gpu_peak_reserved_bytes"),("Process RSS B","process_peak_rss_bytes")]
    sections += ["### Timing, memory and validation errors", table(
        ["Epoch", *(name for name,_ in resources), "Under", "Over", "High-confidence wrong"],
        [(e["epoch"], *(e.get("resource",{}).get(key) for _,key in resources),
          *(e["validation"].get(key) for key in ("undergraded","overgraded","high_confidence_wrong")))
         for e in history])]
    if any("learning_rates" in row for row in history):
        sections += ["### Learning rates and checkpoint tracking", table(
            ["Epoch", "Backbone LR", "Head LR", "Class weights active", "Best epoch so far", "Best updated"],
            [(row["epoch"], row["learning_rates"]["backbone"], row["learning_rates"]["head"],
              row.get("class_weight_active"), row.get("best_epoch"), row.get("best_updated")) for row in history])]
    return "\n\n".join(sections)


def grade_report(record):
    chosen = record["best"] if "best" in record else record["summary"]["best"]
    v = chosen["validation"]
    cm = v["confusion_matrix"]
    raw = table(["True \\ Pred",0,1,2,3,4], [(i,*row) for i,row in enumerate(cm)])
    rate = table(["True \\ Pred (%)",0,1,2,3,4],
                 [(i,*(100*x/sum(row) if sum(row) else None for x in row)) for i,row in enumerate(cm)])
    scores = table(["Grade","Precision","Recall","F1","Support","True positive"],
        [(g["grade"],g["precision"],g["recall"],g["f1"],g["support"],g["correct"]) for g in v["per_grade"]])
    adjacent = sum(x for i,row in enumerate(cm) for j,x in enumerate(row) if abs(i-j)==1)
    leading = sorted(((x,i,j) for i,row in enumerate(cm) for j,x in enumerate(row) if i != j and x),
                     reverse=True)[:3]
    body = table(["Best epoch",*(name for name,_ in COLS),"Correct","Wrong","Under","Over"],
                 [(chosen["epoch"],*(v[key] for _,key in COLS),v["correct"],v["wrong"],v["undergraded"],v["overgraded"])])
    checkpoint = record.get("source_sha256", {}).get("best.pt") or record["checkpoint_sidecars"]["best.pt"]["sha256"]
    other = ", ".join(f"{i} → {j} ({n})" for n,i,j in leading)
    minority = ", ".join(f"grade {i} {cm[i][i]}/{sum(cm[i])}" for i in (1,3))
    return (f"{body}\n\nFrozen validation only. The selected checkpoint SHA-256 is `{checkpoint}`. "
            f"The recorded binary remains external.\n\n"
            f"**Raw counts**\n\n{raw}\n\n**Row-normalized percentages**\n\n{rate}\n\n"
            f"**Recorded per-grade scores**\n\n{scores}\n\n"
            f"Minority-grade predictions on the diagonal were {minority}. The largest off-diagonal "
            f"counts were {other}. The matrix gives {adjacent} adjacent errors and "
            f"{v['wrong']-adjacent} errors of at least two grades. "
            "Directional totals count every error once. Confidence thresholds depend on output-logit scale.")


def saved_figure(path):
    figure = ROOT / path
    if figure.suffix == ".svg":
        kind, mime, content = "SVG", "image/svg+xml", figure.read_text(encoding="utf-8")
    else:
        kind, mime, content = "Image", "image/png", base64.b64encode(figure.read_bytes()).decode("ascii")
    saved = nb.v4.new_output("display_data", data={mime:content,
        "text/plain":f"Original, committed figure: {path}"})
    return code(f'from IPython.display import {kind}\ndisplay({kind}(filename=str(ROOT / {path!r})))', [saved])


def archive_url(path, rev="027e586f607bde4877b395ca566082c713340e8b"):
    return f"https://github.com/salmakamila126/Diabetic-Rethinography/blob/{rev}/{path}"


def interpret(method, dataset, baseline, runs):
    by_arm = {r["arm"]:r["best"]["validation"] for r in baseline if r["dataset"] == dataset}
    if method=="B4":
        if not runs:
            return "The audited DDR training split did not contain an applicable label conflict, so there is no Q/R contrast for this dataset."
        q, r = by_arm["B4-Q"], by_arm["B4-R"]
        return (f"Quarantine Q changed macro-F1 by {q['macro_f1']-r['macro_f1']:+.6f} versus matched removal R. "
                "That one validation contrast does not establish a general effect of quarantine.")
    if method=="B0":
        return "B0 is the untreated comparison for B1 and B2 on this dataset. Training score coverage is limited by the original log."
    if method in ("B1","B2","B3"):
        parent = "B1" if method=="B3" else "B0"
        value, ref = by_arm[method],by_arm[parent]
        return (f"Compared with {parent} on this validation split, macro-F1 changed by "
                f"{value['macro_f1']-ref['macro_f1']:+.6f}, grade 1 recall by "
                f"{value['per_grade'][1]['recall']-ref['per_grade'][1]['recall']:+.4f}, and grade 3 recall by "
                f"{value['per_grade'][3]['recall']-ref['per_grade'][3]['recall']:+.4f}.")
    if method in INTEGRATED:
        a, b = [read(f"results/{dataset}/{INTEGRATED[m]}/validation_evidence.json")["summary"]["best"]["validation"]
                for m in ("E5","E6")]
        return (f"On this dataset, E6 minus E5 is {b['macro_f1']-a['macro_f1']:+.6f} macro-F1, "
                f"{b['accuracy']-a['accuracy']:+.6f} accuracy and "
                f"{b['quadratic_weighted_kappa']-a['quadratic_weighted_kappa']:+.6f} QWK. "
                "See the comparison for grade 1 false positives, grade 3 recall and confidence-scale caveats.")
    return "The 384-pixel field covers peripheral content outside the local crops. Patch usefulness remains a hypothesis until validation artifacts are verified."


def experiment(method, dataset, baseline):
    baseline_method = method.startswith("B")
    if baseline_method:
        arms = ("B4-Q","B4-R") if method == "B4" else (method,)
        runs = [r for r in baseline if r["dataset"]==dataset and r["arm"] in arms]
        evidence = BASE_EVIDENCE
        fallback = {}
        checkpoint = runs[0]["source_sha256"]["best.pt"] if runs else None
    elif method in INTEGRATED:
        folder = f"results/{dataset}/{INTEGRATED[method]}"
        evidence = f"{folder}/validation_evidence.json"
        runs = [read(evidence)]
        fallback = {}
        checkpoint = runs[0]["checkpoint_sidecars"]["best.pt"]["sha256"]
    else:
        e5 = read(f"results/{dataset}/e5_integrated/validation_evidence.json")
        runs, evidence, checkpoint = [], None, None
        fallback = {key:e5["configuration"][key] for key in ("manifest","snapshot")}
        fallback.update({"micro_batch":4, "accumulation":8, "precision":"bf16",
                         "input_transport":{"num_workers":2,"persistent_workers":True,"prefetch_factor":2,"pin_memory":True},
                         "e5_checkpoint_sha256":e5["checkpoint_sidecars"]["best.pt"]["sha256"]})

    top = [md(f"# {method} — {dataset}\n\n{HYPOTHESES[method]}\n\n"
              "[Notebook index](../README.md) · [Experiment audit](../../docs/NOTEBOOK_EXPERIMENT_AUDIT.md)")]
    if baseline_method:
        interface = "[foundation adaptation](../../novelty/foundation_adaptation.py)"
        b0_hash_commit = "de4699e7162b461195f87924d5f7ca39d2313cdb"
        b12_hash_commit = "3a563c67fef82454ce2ed34858615bd456424a13"
        if method in ("B0","B1","B2"):
            commit = b0_hash_commit if method=="B0" else b12_hash_commit
            source = f"Exact recorded main training module matched by SHA-256 to [the archived source]({archive_url('novelty/foundation_adaptation.py',commit)}). The optional CLI uses the current {interface} for a separately recorded new run."
        elif method=="B3":
            source = f"The historical runner and grade 3 sampler were external. The export records their hashes. {interface} is a component reference and exposes B0–B2 only."
        else:
            source = f"The B4 runner and selection/cache scripts were external. The B1 component matches [the archived source]({archive_url('novelty/foundation_adaptation.py',b12_hash_commit)}), but the historical B4 orchestration has no tracked entry point."
        if method=="B4" and dataset=="DDR":
            source += " No B4 DDR run exists in the evidence. The audited frozen DDR training split had no conflicting-label exact SHA group to quarantine."
    else:
        prefix = f"../../novelty_integrated/{method.lower()}"
        module_targets = ["model.py","objective.py","config.py"] if method=="E5" else (
            ["objective.py","decoded_cache.py","config.py"] if method=="E6" else ["geometry.py","model.py","input_pipeline.py","config.py"])
        source = f"[Runner]({prefix}/runner.py) · [Protocol and environment]({prefix}/README.md) · " + " · ".join(
            f"[{name}]({prefix}/{name})" for name in module_targets)
    top += [md("## Protocol and reproducibility\n\n" + source + "\n\n" + SPLIT)]
    architecture = POOL + (" A linear categorical head emits five logits." if baseline_method else
        " A categorical head emits five logits and an ordered-threshold ordinal head contributes auxiliary supervision.")
    if baseline_method:
        architecture += " Parameter counts are not published in this run evidence."
    elif method in INTEGRATED:
        counts = runs[0]["summary"].get("parameter_counts",{})
        architecture += " Recorded parameter counts are " + ", ".join(f"{k}={v:,}" for k,v in counts.items()) + "."
    if method=="E7":
        architecture += (" One shared backbone serves the full 384×384 view and two 224×224 patches. A 1,537-parameter gate fuses "
                         "the pooled global and mean local embeddings. The external checkpoint has no verified count in this checkout.")
    preprocess = PROCESSOR
    if method in ("B1","B3","B4","E5","E6","E7"):
        preprocess += " " + AUGMENTATION
    if method=="E7":
        preprocess = ("EXIF-corrected original RGB is letterboxed to 384 px. Two crops use 55% of the estimated field width and height, "
                      "are letterboxed to 224 px, and use official normalization with resize disabled. " + AUGMENTATION +
                      " Decoded train RGB can reuse compatible E6 cache read-only, and validation uses an isolated E7 cache.")
    top += [md("## Model, inputs and environment\n\n" + architecture + "\n\n" + preprocess +
               "\n\nThe evidence view requires Python, Jupyter and IPython. Optional source workflows rely on the "
               "run-specific PyTorch, Transformers, Pillow and NumPy environment, a verified local snapshot, and external "
               "datasets/manifests. The CLI and protocol links describe hardware, caching and environment details.")]
    if evidence:
        display_expr = ("[r for r in blob['runs'] if r['dataset'] == DATASET and "
                        "(r['arm'].startswith('B4-') if METHOD == 'B4' else r['arm'] == METHOD)]"
                        if baseline_method else "[blob]")
        setup = SETUP + f"\nMETHOD, DATASET = {method!r}, {dataset!r}\nEVIDENCE = {evidence!r}\n"
        setup += (f"blob = json.loads((ROOT / EVIDENCE).read_text(encoding='utf-8'))\nRUNS = {display_expr}\n"
                  "CONFIG = RUNS[0]['configuration'] if RUNS else {}\n")
    else:
        setup = SETUP + f"\nMETHOD, DATASET = {method!r}, {dataset!r}\nRUNS = []\nCONFIG = {fallback!r}\n"
    setup += f"CHECKPOINT_SHA256 = {checkpoint!r}\nprint(f'{{METHOD}} {{DATASET}}: {{len(RUNS)}} published run(s) in this notebook.')"
    top += [md("### Read committed evidence\n\nThe setup uses the dataset-specific result and performs no training or inference."), code(setup)]
    if method not in ("B3","B4") or runs:
        selection = (" For B4, set `DR_RECORDED_RUN_ID` to `APTOS2019_B4_Q` or `APTOS2019_B4_R` when supplying a checkpoint." if method=="B4" else "")
        top += [md("### Optional manifest and checkpoint byte checks\n\nSet `DR_ARTIFACT_ROOT` and optionally `DR_CHECKPOINT` before enabling this read-only gate." + selection),
                code(VERIFY)]
    for record in runs:
        title = record.get("arm",method)
        link = f"../../{evidence}"
        top += [md(f"## {title} recorded configuration and history\n\n[Authoritative aggregate JSON]({link}).\n\n"
                   + config_text(record) + "\n\nThe next tables show all completed epochs. An em dash means the metric was not recorded. "
                   "The source JSON retains full precision, resource fields and nested histories.\n\n" + epoch_scores(record["history"])),
                md(f"## {title} best-checkpoint validation\n\n" + grade_report(record))]
        if baseline_method:
            figures = record["figures"]
            path = f"{BASE}/{figures['confusion_matrix.png']['file']}"
            curve = f"{BASE}/{figures['learning_curves.svg']['file']}"
            top += [md(f"### Recorded figures\n\n[Learning curves, SVG](../../{curve}) · [Original PNG](../../{path}). "
                       "The saved cell output is the original PNG. Rows are true grades and columns are predicted grades."),
                    saved_figure(path),
                    md("The next output displays the original train and validation learning curves. Metric coverage follows the corresponding recorded history."),
                    saved_figure(curve)]
        else:
            path = f"docs/figures/{dataset.lower()}_confusion.png"
            top += [md(f"### Published comparison figures\n\n[Run report](../../{folder}/validation_report.md) · "
                       f"[Dataset figure suite](../../docs/E5_E6_VALIDATION.md#{dataset.lower()}-figure-suite) · "
                       f"[Learning curves](../../docs/figures/{dataset.lower()}_epoch_scores.png). The saved figure contains both E5 and E6, each labeled by selected epoch."),
                    saved_figure(path),
                    md("The saved score panel below places each method's selected epoch on the full recorded histories. Training and validation panels use their respective splits."),
                    saved_figure(f"docs/figures/{dataset.lower()}_epoch_scores.png")]
    if not runs:
        if method=="E7":
            top += [md("## Evidence status\n\nNo finalized E7 history, validation summary, confusion matrix, checkpoint receipt or "
                       "training curve is published here. Current HPC execution status is unverified. "
                       "The **planned** 20-epoch seed-42 pilot uses a dataset-matched E5 warm start, full fine-tuning, effective batch "
                       "32, AdamW (backbone 1e−5, head 1e−3, decay 0.01), no scheduler, and validation macro-F1 with validation CE "
                       "and earlier-epoch tie breaks. Training uses ordinary CE with deferred inverse-square-root weighting "
                       "from epoch 11 and 0.25 ordinal BCE. The warm-start SHA is from the committed E5 evidence. "
                       "It is distinct from an E7 checkpoint identity. The linked runner also requires prespecifying whether "
                       "raw or optional EMA weights generate validation predictions.")]
        else:
            top += [md("## Evidence status\n\n**Not applicable.** There are zero B4 DDR epoch records, figures, or checkpoint receipts. "
                       "See the two [APTOS2019 B4 arms](B4_APTOS2019.ipynb) and the original "
                       f"[validation evidence](../../{BASE_EVIDENCE}).")]
    if method=="E5" and dataset=="APTOS2019":
        top += [md("## Separate historical final test\n\nThe [locked final-test report](../../results/APTOS2019/e5_integrated/README.md) "
                   "documents its independent 366-image test split. The current epoch and best-checkpoint tables above use the "
                   "validation split only. The notebook exposes no test execution.")]
    top += [md("## Interpretation and limits\n\n" + interpret(method, dataset, baseline, runs) + "\n\n" + (
        "Use the [E5–E6 comparison](../../docs/E5_E6_VALIDATION.md) for the dataset-specific trade-offs, including "
        "grade 1 false positives, grade 3 recall, and the confidence-scale audit. Validation selected the best epoch, so "
        "single-run differences are descriptive." if method in INTEGRATED else
        "Do not infer a gain from geometry or multi-view fusion before validation artifacts are verified." if method=="E7" else
        "Screening comparisons share validation data and need independent confirmation. A selected checkpoint is not a final-test result. "
        "See the [baseline rationale and audit](../../docs/NOTEBOOK_EXPERIMENT_AUDIT.md#b0b4-mapping)."))]
    if method in ("B3","B4"):
        top += [md("## Execution availability\n\nThe original external orchestration and input selection are prerequisites for a historical rerun. The signed-off "
                   "reporting path reads aggregate evidence only. There is no substitute training cell for these arms.")]
    else:
        output = ("For E7, use a fresh output directory below the external `foundation_e7` root. " if method=="E7" else "")
        top += [md("## Optional execution of the authentic workflow\n\nSet the external `DR_ARTIFACT_ROOT`, `DR_CONVNEXT_SNAPSHOT` "
                   "and for writing steps `DR_NOTEBOOK_RUN_DIR`. Use the protocol's reviewed cache roots, size budgets and "
                   "paths when applicable. `DR_MANIFEST_SOURCE_ROOT` supports moved roots for E5–E7. "
                   "Authorize the local snapshot explicitly via `DR_TRUST_LEGACY_BIN=1` where required. "
                   "GPU actions also require `DR_GPU_ALLOCATION_CONFIRMED=1`. " + output +
                   "\n\nThe [shared workflow bridge](../experiment_workflow.py) lists the exact environment keys and "
                   "dispatches to the original CLI. Choose one step and enable its switch. Outputs must use a fresh external "
                   "location. Report identity and cache validity remain enforced by the runner."),
                code(WORKFLOW)]
    return nb.v4.new_notebook(cells=top, metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "research":{"family":method,"dataset":dataset,"evidence_scope":"validation",
                    "epoch_records":sum(len(r["history"]) for r in runs)},
    })


def main():
    baseline = read(BASE_EVIDENCE)["runs"]
    for group, methods in (("baselines", ("B0","B1","B2","B3","B4")),
                           ("novelty", ("E5","E6","E7"))):
        folder = ROOT / "notebooks" / group
        folder.mkdir(exist_ok=True)
        for method in methods:
            for dataset in FAMILIES:
                path = folder / f"{method}_{dataset}.ipynb"
                document = experiment(method, dataset, baseline)
                # Stable cell IDs make a second run directly comparable.
                for index, cell in enumerate(document.cells):
                    cell.id = f"{method.lower()}-{dataset.lower()}-{index:02}"
                nb.validate(document)
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    nb.write(document, stream)


if __name__ == "__main__":
    main()
