"""Inventory published figures and verify the data or archived bytes behind them."""

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from docs import e5_e6_publication as integrated, e7_ddr_publication as ddr, e7_publication as aptos
from docs import pruning_pilot_publication as pruning
from docs import pruning_pilot_supplement as supplement

ROOT = integrated.ROOT
VERIFIED = "VERIFIED"
LIMITED = "HISTORICAL_WITH_LIMITED_PROVENANCE"
MISSING = "MISSING_GENERATOR"
FIGURE_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".tif", ".tiff", ".bmp", ".pdf"}
BASE = "novelty/convnext_b0_b4"
ARCHIVE = "reproduction/archive/CAD_ViT_original_105_cells.ipynb"
REPRO = "reproduction/report_figures"
RECOVERY = "results/DDR/full_oversampling_fp32_math_mb8_80epochs"
BALANCED = "results/APTOS2019/full_oversampling_amp_fp16_80epochs"
E5_APTOS = "results/APTOS2019/e5_integrated"
SVG = "{http://www.w3.org/2000/svg}"
ARCHIVED_PNG = {"historical_v1_aptos_confusion_matrix": (89, 0), "historical_v1_ddr_confusion_matrix": (89, 1),
                "historical_v2_aptos_training_curves": (74, 2), "historical_v2_ddr_training_curves": (74, 4)}


def read(path):
    return json.loads((ROOT / path).read_bytes())


def link(path, description=None):
    return f"[{description or path}](../{path})"


def source(path, *fields):
    detail = " · ".join(fields)
    return f"{link(path, Path(path).name)} ({detail})" if detail else link(path, Path(path).name)


def source_sets(dataset):
    return " · ".join(source(f"{integrated.run_folder(method, dataset).as_posix()}/validation_evidence.json", f"{dataset} {method}")
                      for method in ("E5", "E6"))


def published_paths():
    # Exclude ignored local-only images. New report assets can be checked before staging.
    data = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    return sorted(set(data.decode("utf-8").split("\0")) - {""})


def embedded(doc):
    for index, cell in enumerate(doc["cells"], 1):
        sources = [(f"output {number}", output.get("data", {})) for number, output in enumerate(cell.get("outputs", []))]
        sources += [(f"attachment `{name}`", content) for name, content in cell.get("attachments", {}).items()]
        for origin, images in sources:
            for mime, payload in images.items():
                if mime.startswith("image/"):
                    raw = "".join(payload)
                    data = raw.encode("utf-8") if mime == "image/svg+xml" else base64.b64decode(raw)
                    yield index, origin, mime, data


def _digest(data, svg=False):
    if svg:
        data = data.replace(b"\r\n", b"\n").strip()
    return hashlib.sha256(data).hexdigest()


def _records():
    records = read(f"{BASE}/validation_evidence.json")["runs"]
    return {f"{BASE}/{figure['file']}": (r, role, figure["sha256"]) for r in records
            for role, figure in r["figures"].items()}


def describe(path, baseline, archive):
    p = Path(path)
    stem = p.stem
    if path.startswith("docs/figures/"):
        if stem in pruning.COMPLEXITY_STEMS:
            profiles = source((pruning.FOLDER / "complexity.csv").as_posix(), "20 verified prepared profiles")
            if stem.endswith("quality_complexity"):
                profiles += " · " + source((pruning.FOLDER / "quality.csv").as_posix(), "paired eager Macro-F1")
            return ("Five-job pruning pilot", profiles, source("docs/pruning_pilot_publication.py", "plot_complexity"),
                    "Prepared models · pre-recovery", VERIFIED, "Receipt-bound profilers and paired frozen-subset scores")
        if stem in pruning.STEMS:
            return ("Five-job pruning pilot", source((pruning.FOLDER / "validation_evidence.json").as_posix(), stem),
                    source("docs/pruning_pilot_publication.py", "plot"), "Frozen validation subset · pre-recovery",
                    VERIFIED, "Regenerated from verified aggregate rows with original member digests")
        if stem.startswith(("aptos2019_e7_", "ddr_e7_")):
            dataset = "DDR" if stem.startswith("ddr_") else "APTOS2019"
            module = "e7_ddr_publication" if dataset == "DDR" else "e7_publication"
            kind = stem.split("_e7_", 1)[1]
            inputs = source(f"results/{dataset}/e7_global_local/validation_evidence.json", kind)
            if kind == "e5_e6_e7_comparison":
                inputs += " · " + source_sets(dataset)
            scope = "Train + validation" if kind in ("epoch_scores", "losses", "resources") else "Validation"
            return f"{dataset} E7" + (" partial" if dataset == "DDR" else ""), inputs, source(f"docs/{module}.py", "plot"), scope + " · actual", VERIFIED, "Byte-identical regeneration from published evidence"
        if stem == "e5_e6_best_validation":
            return "E5/E6 both datasets", source_sets("APTOS2019") + " · " + source_sets("DDR"), source("docs/e5_e6_publication.py", "plot_overview"), "Validation · actual", VERIFIED, "Byte-identical regeneration"
        dataset = "DDR" if stem.startswith("ddr_") else "APTOS2019"
        kind = stem.split("_", 1)[1]
        integrated.require(kind in integrated.FIGURES, f"Unmapped publication plot: {path}")
        scope = "Train + validation" if kind in ("epoch_scores", "resources", "objective") else "Validation"
        return f"{dataset} E5/E6", source_sets(dataset), source("docs/e5_e6_publication.py", "plot_dataset"), scope + " · actual", VERIFIED, "Byte-identical regeneration"
    if path in baseline:
        r, role, receipt = baseline[path]
        integrated.require(_digest(p_on_disk(path)) == receipt, f"Baseline figure digest differs: {path}")
        inputs = source(f"{BASE}/validation_evidence.json", r["id"], "history" if p.suffix == ".svg" else "best.validation.confusion_matrix")
        if role == "learning_curves.svg":
            return r["id"], inputs, source("novelty/foundation_reporting.py", "learning_curves"), "Train + validation · actual", VERIFIED, "Regeneration gives identical SVG text after newline normalization"
        return r["id"], inputs, "Recorded Matplotlib render; original plotting function unavailable", "Validation · actual transfer", MISSING, "Figure SHA-256 matches evidence and all 25 visible counts match JSON"
    if path in {f"{RECOVERY}/{p.name}", f"reproduction/archive/{RECOVERY}/{p.name}"}:
        check = ""
        if path.startswith("reproduction/archive/"):
            integrated.require(p_on_disk(path) == p_on_disk(f"{RECOVERY}/{p.name}"), f"Archived recovery copy differs: {path}")
            check = "Exact copy of active result. "
        if stem == "confusion_matrix":
            return "CAD-ViT DDR recovery", source(f"{RECOVERY}/final_test_summary.json", "confusion_matrix"), "Original PNG render unavailable; " + source("reproduction/provenance/ddr_final_test/eval_ddr_recovery_once_executed.py", "execute writes aggregate source"), "Final test · actual archive", LIMITED, check + "Counts checked against the published test aggregate (975/1,252 correct)"
        field = {"train_loss": "train_loss", "validation_loss": "val_loss", "validation_accuracy": "val_accuracy",
                 "validation_macro_f1": "val_macro_f1"}[stem]
        return "CAD-ViT DDR recovery", source(f"{RECOVERY}/history.jsonl", field), "Original 80-epoch PNG render unavailable", ("Train" if stem == "train_loss" else "Validation") + " · actual archive", LIMITED, check + "Published 80-epoch curve; source series present, exact raster coordinates unverified"
    if path.startswith(f"{REPRO}/"):
        if stem in ARCHIVED_PNG:
            cell, output = ARCHIVED_PNG[stem]
            integrated.require(p_on_disk(path) == archive[cell, f"output {output}"], f"Archived figure differs: {path}")
            is_v1 = "_v1_" in stem
            return "CAD-ViT historical " + ("V1" if is_v1 else "V2"), source(ARCHIVE, f"cell {cell}, output {output}"), source(ARCHIVE, f"cell {cell}, inline plot from {'matrix literals' if is_v1 else 'transcribed epoch table'}"), "Final test · actual historical" if is_v1 else "Train + validation · actual historical", LIMITED, "Exact decoded notebook bytes; upstream raw logs or per-image predictions not retained here"
        if stem == "aptos_balanced_80_epoch_curves":
            return "CAD-ViT APTOS balanced", source(f"{BALANCED}/history.jsonl", "80 epochs"), "Original static SVG build script unavailable", "Train + validation · actual", MISSING, "All four SVG polylines follow source values within coordinate rounding"
        if stem.startswith("DDR_"):
            note = source(f"{REPRO}/README.md", "archive and file SHA-256")
            if stem == "DDR_gradcam_validation":
                return "CAD-ViT DDR Grad-CAM", note + " · " + source(f"{REPRO}/DDR_gradcam_metadata.json"), source("reproduction/cadvit/interpretability.py", "GradCAM; original server montage script unavailable"), "Validation · actual selected images", LIMITED, "File and transfer digests recorded; checkpoint and source pixels not rendered again"
            if stem == "DDR_error_gallery":
                return "CAD-ViT DDR retrospective errors", note + " · " + source(f"{REPRO}/DDR_error_metadata.json"), source("reproduction/cadvit/visualization.py", "plot_error_gallery; original server montage script unavailable"), "Retrospective final test · actual selected images", LIMITED, "Selected records and file digest documented; full prediction roster remains external"
            return "CAD-ViT DDR input audit", note, source("reproduction/notebooks/01_preprocessing.ipynb", "transfer display and preprocessing protocol; original montage script unavailable"), "Training · actual selected images", LIMITED, "File digest checked against transfer receipt; original image-level inputs and montage provenance incomplete"
    if path.startswith(f"{E5_APTOS}/"):
        is_test = stem == "test_confusion_matrix"
        inputs = "history.csv" if stem == "learning_curves" else "test_summary.json" if is_test else "summary.json"
        return "E5 APTOS2019 historical publication", source(f"{E5_APTOS}/{inputs}"), "Original standalone SVG build script unavailable", "Train + validation · actual" if stem == "learning_curves" else ("Final test" if is_test else "Validation") + " · actual", MISSING, "SVG coordinates or all 25 matrix labels verified against source"
    raise ValueError(f"Figure lacks an audit entry: {path}")


def p_on_disk(path):
    return (ROOT / path).read_bytes()


def _close_polyline(element, values):
    points = [tuple(map(float, value.split(","))) for value in element.attrib["points"].split()]
    integrated.require(len(points) == len(values) and all(a[0] < b[0] for a, b in zip(points, points[1:])), "SVG epoch sequence differs")
    lo, hi = values.index(min(values)), values.index(max(values))
    m = (points[hi][1] - points[lo][1]) / (values[hi] - values[lo])
    integrated.require(m < 0 and all(abs(y - points[lo][1] - (value - values[lo])*m) < .015
                       for (_, y), value in zip(points, values, strict=True)), "SVG does not follow source series")


def verify_static_data():
    import csv
    from PIL import Image

    for path in published_paths():
        suffix = Path(path).suffix.lower()
        if suffix == ".svg":
            ET.parse(ROOT / path)
        elif suffix == ".png":
            with Image.open(ROOT / path) as figure:
                figure.verify()

    report = (ROOT / REPRO / "README.md").read_text(encoding="utf-8")
    for path in (ROOT / REPRO).glob("DDR_*.png"):
        integrated.require(_digest(path.read_bytes()) in report, f"Missing DDR transfer digest: {path.name}")
    history = [json.loads(row) for row in p_on_disk(f"{BALANCED}/history.jsonl").splitlines()]
    integrated.require([r["epoch"] for r in history] == list(range(1, 81)), "Balanced history is incomplete")
    svg = ET.fromstring(p_on_disk(f"{REPRO}/aptos_balanced_80_epoch_curves.svg"))
    for line, key in zip(svg.findall(f".//{SVG}polyline"), ("train_loss", "val_loss", "val_accuracy", "val_macro_f1"), strict=True):
        _close_polyline(line, [r[key] for r in history])
    history = list(csv.DictReader((ROOT / E5_APTOS / "history.csv").read_text(encoding="utf-8").splitlines()))
    svg = ET.fromstring(p_on_disk(f"{E5_APTOS}/learning_curves.svg"))
    for line, key in zip(svg.findall(f".//{SVG}polyline"), ("train_macro_f1", "validation_macro_f1"), strict=True):
        _close_polyline(line, [float(r[key]) for r in history])
    for stem, metric in (("validation", read(f"{E5_APTOS}/summary.json")["best"]["validation"]),
                         ("test", read(f"{E5_APTOS}/test_summary.json")["metrics"])):
        svg = ET.fromstring(p_on_disk(f"{E5_APTOS}/{stem}_confusion_matrix.svg"))
        counts = [int(el.text) for el in svg.findall(f".//{SVG}text") if el.get("text-anchor") == "middle" and el.text.isdecimal()]
        integrated.require(counts == sum(metric["confusion_matrix"], []), f"E5 APTOS {stem} SVG counts differ")


def verify_regeneration():
    """Run only aggregate plotters. No notebooks, checkpoints or dataset are opened."""
    from novelty.foundation_reporting import learning_curves

    with TemporaryDirectory() as work:
        out = Path(work)
        runs = {(method, dataset): read(f"{integrated.run_folder(method, dataset).as_posix()}/validation_evidence.json")
                for method in ("E5", "E6") for dataset in ("APTOS2019", "DDR")}
        for dataset in ("APTOS2019", "DDR"):
            integrated.plot_dataset(dataset, {m: runs[m, dataset] for m in ("E5", "E6")}, out)
        integrated.plot_overview(runs, out)
        aptos.plot(read(f"{aptos.FOLDER.as_posix()}/validation_evidence.json"), out)
        ddr.plot(read(ddr.EVIDENCE.as_posix()), out)
        pilot = pruning.verify(read((pruning.FOLDER / "validation_evidence.json").as_posix()))
        pruning.plot(pilot, out)
        profiles = supplement.read_profiles(pilot, ROOT / pruning.FOLDER)
        pruning.plot_complexity(pilot, list(supplement.complexity_rows(pilot, profiles)), out)
        generated = sorted(out.iterdir())
        published = sorted((ROOT / "docs/figures").iterdir())
        integrated.require([p.name for p in generated] == [p.name for p in published], "Publication figure inventory differs")
        for new, old in zip(generated, published, strict=True):
            integrated.require(new.read_bytes() == old.read_bytes(), f"Published figure differs: {old.name}")
        for path, (run, role, _) in _records().items():
            if role == "learning_curves.svg":
                new = out / Path(path).name
                learning_curves(new, run["history"])
                integrated.require(_digest(new.read_bytes(), True) == _digest(p_on_disk(path), True), f"Baseline SVG differs: {path}")


def _embedded_origin(path, cell):
    if path.endswith("04_classification_evaluation.ipynb"):
        return ("Paper Figures 4–5", "PAPER_REFERENCE", source("reproduction/cadvit/visualization.py", "paper_binary_reference / paper_five_reference → confusion_png"),
                "Transcribed reference matrices; not a local experimental result")
    if path.endswith("06_balancing_analysis.ipynb"):
        return ("Paper Figure 6", "PAPER_REFERENCE", source("reproduction/cadvit/visualization.py", "PAPER_FIGURE_6 → balance_metrics_png"),
                "Transcribed reference values; not a local experimental result")
    if path.endswith("03_training_protocol.ipynb"):
        return ("DDR FP32 mixed protocol", "Validation · actual", source(f"{RECOVERY}/history.jsonl", "val_macro_f1, historical attachment"),
                "Exact hand-written SVG generator not retained")
    if path == ARCHIVE and cell == 92:
        return ("DDR capped pilot", "Validation · operator-reported", source(ARCHIVE, "cell 92 literal → confusion_png"), "Independent prediction artifact missing")
    if path == ARCHIVE and cell in (14, 16, 18, 20, 22, 29, 31, 33):
        return ("CAD-ViT EDA", "Raw EDA or original APTOS training input", source(ARCHIVE, f"cell {cell} plotting code"),
                "Stored actual output; exact input roster/raw images not public")
    return ("CAD-ViT historical diagrams and summaries", "EDA, architecture or mixed reference/validation/test labels",
            source(ARCHIVE, "§2, §3.1, §4.2, §5.5 or §6.8; see corresponding cell"),
            "Stored illustration or aggregate summary; exact original build chain unavailable")


def render():
    files = published_paths()
    figures = [path for path in files if Path(path).suffix.lower() in FIGURE_SUFFIXES]
    baseline = _records()
    archive = {(number, origin): raw for number, origin, _, raw in embedded(read(ARCHIVE))}
    entries = {path: describe(path, baseline, archive) for path in figures}
    hashes = {_digest(p_on_disk(path), path.endswith(".svg")): path for path in reversed(figures)}
    statuses = Counter(item[-2] for item in entries.values())
    output_rows = []
    for path in (p for p in files if p.endswith(".ipynb")):
        # An unrelated local edit must not change the published inventory.
        doc = read(path) if path != "reproduction/notebooks/preprocessing_analysis.ipynb" else json.loads(
            subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT))
        for cell, origin, mime, raw in embedded(doc):
            digest = _digest(raw, mime == "image/svg+xml")
            if digest in hashes:
                asset = hashes[digest]
                experiment, _, _, scope, status, _ = entries[asset]
                detail = "Matches the figure's encoded bytes (SVG newline-normalized)"
                lineage = link(asset)
            else:
                experiment, scope, lineage, detail = _embedded_origin(path, cell)
                status = MISSING if path.endswith("03_training_protocol.ipynb") else LIMITED
            output_rows.append((f"{link(path)} · cell {cell}, {origin}", experiment, scope, status, lineage, detail))
    table = integrated.table(["Figure", "Experiment", "Source data", "Generator / cell", "Split and result type", "Status", "Value check"],
                             [(link(path), *item) for path, item in entries.items()])
    outputs = integrated.table(["Notebook and location", "Experiment", "Split and result type", "Status", "Figure or lineage", "Check / limitation"], output_rows)
    return f"""# Figure provenance audit · `salma`

This inventory covers **{len(figures)} published figure files** and **{len(output_rows)} embedded notebook images**. Cell numbers are one-based in the JSON. Output numbers are zero-based. Notebook copies inherit the provenance of the matching figure. The same visual may appear in both tables. Raw or ignored assets and figures referenced only from other branches are outside the file totals.

- **VERIFIED ({statuses[VERIFIED]} files)** means the checked-in generator reproduced the publication bytes or the same SVG text with normalized line endings. PNG and SVG use the same scientific data. The test used Matplotlib 3.11.2, NumPy 2.5.5 and Pillow 13.0.0 for the publication panels.
- **HISTORICAL_WITH_LIMITED_PROVENANCE ({statuses[LIMITED]} files)** covers original transfers, exact archived image copies, and notebook or paper references whose upstream source or render is incomplete.
- **MISSING_GENERATOR ({statuses[MISSING]} files)** means the source aggregate or transfer receipt is available, while the script used for the existing visual is missing. Verified source values are noted individually where the format permits direct checks.

Run `python -m docs.figure_provenance --check --verify-generators` from the repository root to verify inventory, source-value checks and byte regeneration. This command reads aggregates and archived notebook images and uses a temporary output directory.

## Published figure files

{table}

## Stored notebook figures

{outputs}

## Boundaries and remaining gaps

- The [CAD-ViT historical audit](FIGURE_AUDIT.md) identifies the archived plots, transcribed paper references, three-phase V2 or mixed-protocol histories and the original loss/split definitions. The audit preserves those definitions. Transfer hashes establish byte identity. They cannot establish independent model execution, recover missing image-level predictions, or authorize redistribution of APTOS images kept locally.
- DDR preprocessing and augmentation montages, Grad-CAM and error examples are published selected panels from a real dataset. Their [transfer notes](../reproduction/report_figures/README.md) document attribution and metadata. The original server montage commands, complete input membership, checkpoint and prediction artefacts are needed to reproduce the exact figures.
- The three E5 APTOS legacy SVGs and the APTOS balanced curve were checked numerically. The ten B0–B4 confusion PNGs match the transfer receipts. Their original Matplotlib code and formatting parameters are still missing. The archived DDR mixed-protocol PNGs preserve the original outputs within the audit limits shown above.
- The [pruning pilot](PRUNING_PILOT_RESULTS.md) publishes seven reproducible PNG/SVG pairs for pre-recovery quality, backend parity, and recorded complexity. All 20 prepared profiles match published hashes. E5 APTOS2019 has a partial recovery checkpoint receipt and no aligned post-recovery validation. Latency was not measured by the untimed validation protocol. Existing [partner ConvNeXtV2/HF V1 notebook results](OPTIMIZATION_COMPATIBILITY.md) belong to separate archived runs.
- Design diagrams in Markdown or notebook text are explanatory schematics. Original notebook media and transcribed paper figures are labeled separately from local experimental output.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--verify-generators", action="store_true")
    args = parser.parse_args(argv)
    verify_static_data()
    if args.verify_generators:
        verify_regeneration()
    content = render()
    target = ROOT / "docs/FIGURE_PROVENANCE_AUDIT.md"
    if args.write:
        target.write_text(content, encoding="utf-8", newline="\n")
    else:
        integrated.require(target.read_text(encoding="utf-8") == content, "Figure audit differs from current inventory")
    print(content.splitlines()[2])


if __name__ == "__main__":
    main()
