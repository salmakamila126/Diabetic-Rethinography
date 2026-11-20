"""Audit the sanitized E5/E6 ZIP and generate aggregate validation reports and figures."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "APTOS2019": (2930, 366, "73d1546e3c2fd2bf9e3d0fd10af914cb523d28e4bc8e1f25212af454d4be8854"),
    "DDR": (10018, 1252, "7bc3e3916e012bc46fcd9cd69aa469e0daf3229e1273f1d8ecb6054899bf6eb1"),
}
RUN_NAMES = {"E5": "e5_integrated", "E6": "e6_ldam_ordinal_adjacent"}
EXPECTED = {
    ("E5", "APTOS2019"): (16, .830601093, .705791432, .899241254, .833080466, .221311475, .588535098, 304, 62, 30, 32, 23),
    ("E5", "DDR"): (18, .823482428, .653199134, .857161823, .818187567, .266773163, .885510089, 1031, 221, 86, 135, 118),
    ("E6", "APTOS2019"): (15, .789617486, .679204179, .880384339, .795890536, .273224044, .816601607, 289, 77, 35, 42, 0),
    ("E6", "DDR"): (12, .797923323, .662455052, .849553202, .812289836, .283546326, .927703809, 999, 253, 116, 137, 0),
}
BEST_KEYS = ("accuracy", "macro_f1", "quadratic_weighted_kappa", "weighted_f1",
             "grade_mae", "validation_loss", "correct", "wrong", "overgraded",
             "undergraded", "high_confidence_wrong")
PUBLIC_FILES = ("run.json", "summary.json", "history.json", "history.csv",
                "evidence_manifest.json", "best.pt.sha256.json", "last.pt.sha256.json")
PRIVATE = re.compile(r"(?i)(?:/(?:home|mnt|scratch|gpfs|lustre)/|(?<![\w])[a-z]:[\\\\/]|"
                     r"\b(?:node\d+[-\w]*|labkc[-\w]*)\.ub\.ac\.id\b|\bprobation-\d+\b|"
                     r"gh[pousr]_[a-z0-9]{20,}|hf_[a-z0-9]{20,}|"
                     r"-----BEGIN [A-Z ]*PRIVATE KEY|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})")
PRIVATE_KEYS = {"patient_id", "patient_ids", "image_path", "image_paths", "image_ids",
                "sample_ids", "predictions", "labels", "filenames", "password",
                "credentials", "token", "secret", "api_key"}
HEX = re.compile(r"[0-9a-f]{64}\Z")
COLORS = {"E5": "#176383", "E6": "#a44918"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def equal(actual, expected):
    if type(actual) in (int, float) and type(expected) in (int, float):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)
    return actual == expected


def check_public(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in PRIVATE_KEYS, "Non-aggregate or credential field in export")
            check_public(key)
            check_public(child)
    elif isinstance(value, list):
        for child in value:
            check_public(child)
    elif isinstance(value, str):
        require(not PRIVATE.search(value), "Private path, identity, or credential in export")
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite number in export")


def derive(cm):
    """Compute classification and ordinal statistics from rows=true, columns=predicted."""
    grades = range(5)
    require(isinstance(cm, list) and len(cm) == 5 and all(
        isinstance(row, list) and len(row) == 5 and all(type(x) is int and x >= 0 for x in row)
        for row in cm), "Expected 5×5 nonnegative integer cells")
    support = list(map(sum, cm))
    n = sum(support)
    require(n > 0, "Empty confusion matrix")
    predicted = [sum(cm[i][j] for i in grades) for j in grades]
    per_grade = []
    for i in grades:
        tp = cm[i][i]
        per_grade.append({
            "grade": i, "support": support[i], "correct": tp,
            "precision": tp / predicted[i] if predicted[i] else 0,
            "recall": tp / support[i] if support[i] else 0,
            "f1": 2 * tp / (support[i] + predicted[i]) if support[i] + predicted[i] else 0,
            "false_positive": predicted[i] - tp, "false_negative": support[i] - tp,
            "overgraded": sum(cm[i][i+1:]), "undergraded": sum(cm[i][:i]),
            "mean_absolute_error": sum(abs(i-j)*cm[i][j] for j in grades) / support[i] if support[i] else None,
        })
    correct = sum(cm[i][i] for i in grades)
    over = sum(cm[i][j] for i in grades for j in grades if j > i)
    under = n - correct - over
    adjacent = sum(cm[i][j] for i in grades for j in grades if abs(i-j) == 1)
    expected = sum((i-j)**2 * support[i]*predicted[j] for i in grades for j in grades)
    observed = sum((i-j)**2*cm[i][j] for i in grades for j in grades)
    return {
        "samples": n, "support": support, "correct": correct, "wrong": n-correct,
        "accuracy": correct/n, "macro_f1": sum(g["f1"] for g in per_grade)/5,
        "weighted_f1": sum(g["support"]*g["f1"] for g in per_grade)/n,
        "macro_precision": sum(g["precision"] for g in per_grade)/5,
        "macro_recall": sum(g["recall"] for g in per_grade)/5,
        "quadratic_weighted_kappa": 1 - n*observed/expected if expected else None,
        "grade_mae": sum(abs(i-j)*cm[i][j] for i in grades for j in grades)/n,
        "overgraded": over, "undergraded": under, "adjacent": adjacent, "non_adjacent": n-correct-adjacent,
        "mean_overgrade_magnitude": sum((j-i)*cm[i][j] for i in grades for j in grades if j > i)/over if over else None,
        "mean_undergrade_magnitude": sum((i-j)*cm[i][j] for i in grades for j in grades if j < i)/under if under else None,
        "per_grade": per_grade,
        "row_normalized": [[v/s if s else 0 for v in row] for row, s in zip(cm, support)],
    }


def verify_scores(recorded, support, label):
    derived = derive(recorded["confusion_matrix"])
    require(derived["support"] == support, f"{label}: frozen grade support differs")
    for key in ("correct", "wrong", "accuracy", "macro_f1", "weighted_f1",
                "quadratic_weighted_kappa", "grade_mae", "overgraded", "undergraded",
                "mean_overgrade_magnitude", "mean_undergrade_magnitude"):
        require(key in recorded and equal(recorded[key], derived[key]), f"{label}: {key} differs")
    require(len(recorded["per_grade"]) == 5, f"{label}: wrong per-grade length")
    for actual, expected in zip(recorded["per_grade"], derived["per_grade"]):
        for key in expected.keys() - {"false_positive", "false_negative", "mean_absolute_error"}:
            require(key in actual and equal(actual[key], expected[key]), f"{label}: per-grade {key} differs")
    for target, groups in ((1, ([0], [2, 3, 4])), (3, ([0, 1, 2], [4]))):
        key = f"grade{target}_transitions"
        if key in recorded:
            row = recorded["confusion_matrix"][target]
            lower, upper = (("undergraded_to_0", "overgraded_to_2_4") if target == 1
                            else ("undergraded_to_0_2", "overgraded_to_4"))
            expected = {"support": sum(row), "correct": row[target],
                        lower: sum(row[j] for j in groups[0]), upper: sum(row[j] for j in groups[1])}
            for field, value in expected.items():
                require(recorded[key].get(field) == value, f"{label}: {key}.{field} differs")
    wrong = recorded["high_confidence_wrong"]
    require(type(wrong) is int and 0 <= wrong <= derived["wrong"] and
            recorded["high_confidence_threshold"] == .9, f"{label}: confidence count/threshold invalid")
    require(equal(recorded["high_confidence_wrong_rate_all"], wrong/derived["samples"]) and
            equal(recorded["high_confidence_wrong_fraction_of_wrong"],
                  wrong/derived["wrong"] if derived["wrong"] else None),
            f"{label}: confidence rate differs")
    return derived


CSV_FIELDS = {
    "epoch": "epoch", "best_epoch": "best_epoch", "class_weight_active": "class_weight_active",
    "train_total_loss": "train.loss", "train_categorical_loss": "train.categorical_loss",
    "train_ordinal_loss": "train.ordinal_loss", "train_adjacent_loss": "train.adjacent_loss",
    "train_accuracy": "train.accuracy", "train_macro_f1": "train.macro_f1",
    "validation_categorical_ce": "validation.validation_loss",
    "validation_ordinal_loss": "validation.ordinal_loss", "validation_accuracy": "validation.accuracy",
    "validation_macro_f1": "validation.macro_f1", "validation_weighted_f1": "validation.weighted_f1",
    "validation_qwk": "validation.quadratic_weighted_kappa", "validation_grade_mae": "validation.grade_mae",
    "train_seconds": "resource.train_seconds", "validation_seconds": "resource.validation_seconds",
    **{f"{split}_{field}": f"{split}.{field}" for split in ("train", "validation")
       for field in ("confusion_matrix", "per_grade")},
    **{f"validation_grade{g}_transitions": f"validation.grade{g}_transitions" for g in (1, 3)},
}


def get(data, path):
    for key in path.split("."):
        data = data.get(key) if isinstance(data, dict) else None
    return data


def verify_csv(raw, history, label, method):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    e6_columns = {"train_adjacent_loss", "validation_grade1_transitions", "validation_grade3_transitions"}
    required = set(CSV_FIELDS) if method == "E6" else set(CSV_FIELDS) - e6_columns
    require(reader.fieldnames and set(reader.fieldnames) == required and len(reader.fieldnames) == len(required),
            f"{label}: missing, duplicate or unhandled CSV columns")
    rows = list(reader)
    require(len(rows) == len(history), f"{label}: CSV / JSON epoch count differs")
    supplements = []
    for row, expected in zip(rows, history):
        require(None not in row and None not in row.values(), f"{label}: malformed CSV row")
        supplement = {"epoch": expected["epoch"]}
        for key, cell in row.items():
            actual = None if cell == "" else {"True": True, "False": False}.get(cell, cell)
            if isinstance(actual, str):
                actual = json.loads(actual)
            if key in ("train_per_grade", "validation_per_grade"):
                split = key.removesuffix("_per_grade")
                table = derive(expected[split]["confusion_matrix"])["per_grade"]
                values = [grade["mean_absolute_error"] for grade in actual]
                require(len(values) == 5 and all(equal(mae, grade["mean_absolute_error"])
                        for mae, grade in zip(values, table)), f"{label}: CSV per-grade MAE differs")
                supplement[split] = values
                actual = [{k:v for k,v in grade.items() if k != "mean_absolute_error"} for grade in actual]
            require(equal(actual, get(expected, CSV_FIELDS[key])),
                    f"{label}: CSV {key} differs at epoch {expected['epoch']}")
        supplements.append(supplement)
    return len(rows) * len(required), supplements


def audit_run(record, method, dataset):
    label = f"{method} {dataset}"
    cfg, summary, history = (record[k] for k in ("configuration", "summary", "history"))
    require(summary["status"] == "VALIDATION_COMPLETE" and summary["epochs_completed"] == 20
            and summary["final_test_status"] == cfg["final_test_status"] == "NOT_ACCESSED"
            and summary["record"] == cfg, f"{label}: run configuration/status differs")
    require(cfg["dataset"] == dataset and cfg["seed"] == 42 and cfg["epochs"] == 20 and
            cfg["model_kind"] == "convnext_tiny" and cfg["official_pretrained_initialization"] is True
            and cfg["effective_batch"] == cfg["micro_batch"] * cfg["accumulation"] == 32,
            f"{label}: run controls differ")
    require(cfg.get("arm") == "integrated" if method == "E5" else "ldam_margins" in cfg["objective"],
            f"{label}: unexpected objective")
    manifest = cfg["manifest"]
    train_n, val_n, fingerprint = DATASETS[dataset]
    require(manifest["manifest_sha256"] == fingerprint and manifest["images_verified"] is True
            and manifest["counts"] == {"train": train_n, "val": val_n}
            and cfg["train_counts"] == manifest["grade_support"]["train"],
            f"{label}: frozen manifest identity differs")
    require([x["epoch"] for x in history] == list(range(1, 21)), f"{label}: unordered or missing epochs")
    best = None
    for item in history:
        ep = item["epoch"]
        for split, short, n in (("train", "train", train_n), ("validation", "val", val_n)):
            support = manifest["grade_support"][short]
            require(sum(support) == n, f"{label}: wrong {split} sample count")
            verify_scores(item[split], support, f"{label} {split} E{ep:02}")
        loss = item["train"]
        expected_loss = loss["categorical_loss"] + cfg["objective"]["ordinal_coefficient"]*loss["ordinal_loss"]
        if method == "E6":
            expected_loss += cfg["objective"]["adjacent_coefficient"]*loss["adjacent_loss"]
        require(math.isclose(loss["loss"], expected_loss, abs_tol=2e-6, rel_tol=1e-6),
                f"{label}: objective components differ at epoch {ep}")
        require(sum(loss["accumulation_window_images"]) == train_n and
                len(loss["accumulation_window_images"]) == loss["optimizer_steps"] and
                loss["accumulation_window_images"][-1] == loss["last_window_images"],
                f"{label}: accumulation coverage differs at epoch {ep}")
        require(item["class_weight_active"] is (ep >= cfg["objective"]["weight_start_epoch"]),
                f"{label}: weight activation differs at epoch {ep}")
        val = item["validation"]
        improved = best is None or val["macro_f1"] > best["validation"]["macro_f1"] + 1e-12 or (
            abs(val["macro_f1"] - best["validation"]["macro_f1"]) <= 1e-12
            and val["validation_loss"] < best["validation"]["validation_loss"] - 1e-12)
        if improved:
            best = {"epoch": ep, "validation": val}
        require(item["best_updated"] is improved and item["best_epoch"] == best["epoch"],
                f"{label}: best update differs at epoch {ep}")
    require(best == summary["best"], f"{label}: summary differs from best history epoch")
    reference = (best["epoch"], *(best["validation"][key] for key in BEST_KEYS))
    require(all(equal(x, y) for x, y in zip(reference, EXPECTED[method, dataset])),
            f"{label}: independently supplied comparison references differ")
    if method == "E6":
        require(all(item["validation"]["high_confidence_wrong"] == 0 for item in history),
                f"{label}: reported zero-confidence-error anomaly differs")
    return derive(best["validation"]["confusion_matrix"])


def load_export(path):
    with ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        require(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts
                    and "\\" not in n for n in names), "Unsafe ZIP member path")
        tags = {f"{m}_{d}" for m, d in EXPECTED}
        prefix = "e5_e6_publication_review"
        allowed = {f"{prefix}/{n}" for n in ("README.md", "comparison.csv")}
        allowed |= {f"{prefix}/{tag}/{n}" for tag in tags for n in (*PUBLIC_FILES, "README.md")}
        require({i.filename for i in z.infolist() if not i.is_dir()} == allowed,
                "ZIP member set differs from the aggregate-only export contract")
        require(sum(i.file_size for i in z.infolist()) < 8_000_000 and z.testzip() is None,
                "ZIP input size or CRC check failed")
        archive_sha = sha(path.read_bytes())
        result = {}
        for method, dataset in EXPECTED:
            tag = f"{method}_{dataset}"
            raw = {n: z.read(f"{prefix}/{tag}/{n}") for n in PUBLIC_FILES}
            parsed = {n: json.loads(b) for n, b in raw.items() if n.endswith(".json")}
            for value in parsed.values():
                check_public(value)
            require(not PRIVATE.search(raw["history.csv"].decode()), f"{tag}: private CSV data")
            manifest = parsed["evidence_manifest.json"]
            require(manifest["experiment"] == tag, f"{tag}: evidence manifest mismatched")
            for file in ("run.json", "summary.json", "history.json", "history.csv", "training.log",
                         "best.pt", "last.pt", "best.pt.sha256.json", "last.pt.sha256.json"):
                entry = manifest["files"][file]
                require(entry["exists"] is True and isinstance(entry["bytes"], int)
                        and entry["bytes"] > 0 and HEX.fullmatch(entry["sha256"]),
                        f"{tag}: original metadata missing for {file}")
            checkpoints = {}
            for file in ("best.pt", "last.pt"):
                sidecar = parsed[f"{file}.sha256.json"]
                expected = {key: manifest["files"][file][key] for key in ("sha256", "bytes")}
                require(sidecar == expected, f"{tag}: {file} metadata / sidecar differs")
                checkpoints[file] = sidecar
            require(manifest["files"]["config.json"]["exists"] is False,
                    f"{tag}: unexpected standalone configuration")
            record = {"experiment": tag, "configuration": parsed["run.json"],
                      "summary": parsed["summary.json"],
                      "history": parsed["history.json"], "checkpoint_sidecars": checkpoints,
                      "provenance": {**manifest, "archive_sha256": archive_sha,
                                     "transferred_member_sha256": {n: sha(b) for n, b in raw.items()}}}
            cells, supplement = verify_csv(raw["history.csv"], record["history"], tag, method)
            original_csv = manifest["files"]["history.csv"]
            crlf_csv = raw["history.csv"].replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            require((sha(raw["history.csv"]) == original_csv["sha256"] and len(raw["history.csv"]) == original_csv["bytes"])
                    or (sha(crlf_csv) == original_csv["sha256"] and len(crlf_csv) == original_csv["bytes"]),
                    f"{tag}: CSV original byte identity differs even after CRLF restoration")
            record["audit"] = {"csv_cells_checked": cells, "csv_original_digest_verified": True}
            record["csv_per_grade_mean_absolute_error"] = supplement
            record["derived_best_validation"] = audit_run(record, method, dataset)
            result[method, dataset] = record
        comparison = list(csv.DictReader(io.StringIO(z.read(f"{prefix}/comparison.csv").decode())))
        require(len(comparison) == 4 and {r["experiment"] for r in comparison} == tags,
                "Comparison CSV missing a run")
        for row in comparison:
            method, dataset = row["experiment"].split("_")
            summary = result[method, dataset]["summary"]
            for key, val in row.items():
                if key == "experiment":
                    continue
                expected = {"best_epoch": summary["best"]["epoch"], "status": summary["status"],
                            "final_test_status": summary["final_test_status"], **summary["best"]["validation"]}[key]
                require(equal(float(val) if type(expected) in (int, float) else val, expected),
                        f"{row['experiment']}: comparison CSV {key} differs")
    return result


def table(headers, rows):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.5f}"
        return str(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in
                     (headers, ["---"]*len(headers), *rows)) + "\n"


def epoch_table(history, columns):
    return table(["Epoch", *(title for title, _ in columns)],
                 [(row["epoch"], *(get(row, key) for _, key in columns)) for row in history])


def best(record):
    return record["summary"]["best"]["validation"]


def run_folder(method, dataset):
    return Path("results", dataset, RUN_NAMES[method])


def record_report(method, dataset, run):
    cfg, history, summary = (run[k] for k in ("configuration", "history", "summary"))
    b, d, m = best(run), run["derived_best_validation"], cfg["manifest"]
    files = run["provenance"]["files"]
    heading = [
        f"# {method} {dataset} frozen validation",
        f"{m['counts']['train']:,} training and {m['counts']['val']:,} validation images. "
        f"The run completed 20 epochs. The selected checkpoint was epoch **{summary['best']['epoch']}**.",
        "[E5–E6 comparison and figure index](../../../docs/E5_E6_VALIDATION.md) · "
        "[Full aggregate evidence, configuration, and hashes](validation_evidence.json)",
    ]
    if (method, dataset) == ("E5", "APTOS2019"):
        heading.append("The [separately documented historical final test](README.md) retains its original scope. "
                       "This report uses training and validation exports.")
    config = {
        "Run identity": run["provenance"]["source_location_in_hpc"],
        "Architecture": "ConvNeXt-Tiny with five-class categorical and ordered-threshold ordinal heads",
        "Foundation model / revision": f"{cfg['snapshot']['hub_id']} / {cfg['snapshot']['revision']}",
        "Seed / epochs": f"{cfg['seed']} / {cfg['epochs']}",
        "Micro-batch × accumulation → effective batch": f"{cfg['micro_batch']} × {cfg['accumulation']} → {cfg['effective_batch']}",
        "Optimizer / LR backbone / LR head / decay / clip": f"{cfg['optimizer']} / {cfg['backbone_lr']} / {cfg['head_lr']} / {cfg['weight_decay']} / {cfg['max_grad_norm']}",
        "Precision / device / scheduler": f"{cfg['precision']} / {cfg['device_type']} / none in runner",
        "Data transport, train": cfg["input_transport"]["train"],
        "Data transport, validation": cfg["input_transport"]["validation"],
        "Workers / persistent / prefetch / pin": " / ".join(str(cfg["input_transport"][k]) for k in
                    ("num_workers", "persistent_workers", "prefetch_factor", "pin_memory")),
        "Train support grades 0–4": cfg["train_counts"],
        "Validation support grades 0–4": m["grade_support"]["val"],
        "Checkpoint selection": cfg["checkpoint_selection"],
        "Parameter counts": ", ".join(f"{k}: {v:,}" for k,v in summary["parameter_counts"].items()),
        "Versions": ", ".join(f"{k} {cfg[k]}" for k in ("torch", "transformers", "pillow", "numpy")),
        "Source commit": cfg["source_git_head"],
        "Frozen manifest SHA-256": m["manifest_sha256"],
        "Manifest sidecar SHA-256": m["manifest_sidecar_sha256"],
        "Ordered train/validation identity SHA-256": m["image_order_sha256"],
        "Processor SHA-256 / effective processor SHA-256": f"{cfg['snapshot']['processor_config_sha256']} / {cfg['effective_processor_sha256']}",
        "Official weight SHA-256": cfg["snapshot"]["weight_sha256"],
    }
    config_rows = [[k, v] for k, v in config.items()]
    config_rows += [[k.replace("_", " "), v] for k,v in cfg["objective"].items()]
    config_rows += [[f"Source {key}", f"{files[key]['sha256']} ({files[key]['bytes']:,} bytes)"]
                    for key in ("best.pt", "last.pt")]
    cols = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Weighted-F1", "weighted_f1"),
            ("QWK", "quadratic_weighted_kappa"), ("Grade MAE", "grade_mae")]
    train = [("Total loss", "loss"), ("Categorical", "categorical_loss"), ("Ordinal", "ordinal_loss")]
    if method == "E6":
        train += [("Adjacent", "adjacent_loss")]
    support = [[g["grade"], g["support"], g["correct"], g["false_positive"], g["false_negative"],
                g["precision"], g["recall"], g["f1"], g["mean_absolute_error"]]
               for g in d["per_grade"]]
    off = sorted(((value, i, j) for i, row in enumerate(b["confusion_matrix"])
                  for j, value in enumerate(row) if i != j and value), reverse=True)
    top_errors = ", ".join(f"{i}→{j} ({n})" for n,i,j in off[:4])
    sections = [
        *heading, "## Run configuration and provenance", table(["Field", "Recorded value"], config_rows),
        "Processor, augmentation, optimizer, class balance, source hashes, and the complete transport configuration are in the JSON. "
        "The original run stores configuration in `run.json`. The supplied checkpoint sidecars match the recorded checkpoint digests and byte sizes.",
        "## Best checkpoint metrics",
        table(["Metric", "Value", "Source"], [
            *[(title, b[key], "Recorded") for title,key in cols],
            ("Categorical CE", b["validation_loss"], "Recorded"),
            ("Macro precision", d["macro_precision"], "Matrix-derived"),
            ("Macro recall", d["macro_recall"], "Matrix-derived"),
            *[(k.replace("_"," "), b[k], "Recorded") for k in ("correct","wrong","undergraded","overgraded",
                "mean_overgrade_magnitude","mean_undergrade_magnitude","high_confidence_wrong",
                "high_confidence_wrong_rate_all","high_confidence_wrong_fraction_of_wrong")],
            ("Adjacent mistakes (absolute grade difference 1)", d["adjacent"], "Matrix-derived"),
            ("Non-adjacent mistakes (absolute grade difference ≥2)", d["non_adjacent"], "Matrix-derived"),
        ]),
        "### Confusion matrix",
        "Rows are true grades and columns are predicted grades. Counts include the diagonal. "
        "Percentages are normalized within each true grade.",
        table(["True grade", "0", "1", "2", "3", "4", "Support"],
              [(i, *row, sum(row)) for i,row in enumerate(b["confusion_matrix"])]),
        table(["True grade", "0 (%)", "1 (%)", "2 (%)", "3 (%)", "4 (%)"],
              [(i, *(f"{v*100:.2f}" for v in row)) for i,row in enumerate(d["row_normalized"])]),
        f"Largest off-diagonal cells (true→predicted) were {top_errors}. "
        f"The grade 1 row was `{b['confusion_matrix'][1]}`; the grade 3 row was `{b['confusion_matrix'][3]}`.",
        "### Per-grade metrics",
        "Support, TP, precision, recall, and F1 are recorded and independently recomputed from the matrix. "
        "FP and FN are matrix-derived. Per-grade MAE was recorded in the CSV.",
        table(["Grade","Support","TP","FP","FN","Precision","Recall","F1","Grade MAE"], support),
        "## All 20 epochs",
        "Markdown rounds values for readability. The JSON preserves the precision and full field coverage of the export. "
        "Per-grade MAE fields removed from the JSON by the export filter are preserved separately from the CSV.",
        "### Training scores", epoch_table(history, [(title, f"train.{k}") for title,k in train+cols]),
        "### Validation scores", epoch_table(history, [(title, f"validation.{k}")
                        for title,k in [("CE","validation_loss"), ("Ordinal","ordinal_loss"), *cols]]),
        "### Selection, imbalance schedule, and errors",
        "HC wrong requires an incorrect argmax and maximum raw softmax probability ≥0.90.",
        epoch_table(history, [
            ("Best through epoch","best_epoch"), ("Updated","best_updated"), ("Weights active","class_weight_active"),
            ("Optimizer steps","train.optimizer_steps"), ("Backbone LR","learning_rates.backbone"), ("Head LR","learning_rates.head"),
            ("Train correct","train.correct"), ("Train wrong","train.wrong"), ("Train over","train.overgraded"),
            ("Train under","train.undergraded"), ("Train HC wrong","train.high_confidence_wrong"),
            ("Val correct","validation.correct"), ("Val wrong","validation.wrong"), ("Val over","validation.overgraded"),
            ("Val under","validation.undergraded"), ("Val HC wrong","validation.high_confidence_wrong")]),
        "### Resources",
        epoch_table(history, [(title, f"resource.{k}") for title,k in (
            ("Train s", "train_seconds"), ("Val s", "validation_seconds"), ("Train images/s", "train_images_per_second"),
            ("Val images/s", "validation_images_per_second"), ("Train input s", "train_input_seconds"),
            ("Val input s", "validation_input_seconds"), ("GPU allocated B", "gpu_peak_allocated_bytes"),
            ("GPU reserved B", "gpu_peak_reserved_bytes"), ("Process peak RSS B", "process_peak_rss_bytes"))]),
    ]
    resources = [h["resource"] for h in history]
    train_s = sum(r["train_seconds"] for r in resources)
    val_s = sum(r["validation_seconds"] for r in resources)
    sections.append(f"Recorded train/validation phases sum to {train_s/3600:.2f} h and {val_s/60:.2f} min. "
                    "These totals exclude any time outside the recorded phases. RSS is the parent process peak and "
                    "does not account for all worker memory. Shared compute and distinct input transport confound throughput.")
    cache_rows = [r.get("input_cache_counts") for r in resources]
    if all(cache_rows):
        keys = list(cache_rows[0])
        sections += ["### Recorded input cache counters",
                     table(["Epoch", *(k.replace("_", " ") for k in keys)],
                           [(i, *(row.get(k) for k in keys)) for i,row in enumerate(cache_rows, 1)])]
    else:
        sections.append("Per-epoch cache counters are absent in this run's resource export.")
    sections += ["## Figures", f"[{dataset} figures and captions](../../../docs/E5_E6_VALIDATION.md#{dataset.lower()}-figure-suite)",
                 "Recorded statistics are descriptive for this frozen validation split. The report contains no sample-level predictions."]
    return "\n\n".join(sections) + "\n"


FIGURES = {
    "epoch_scores": "Shared score scales by dataset. Solid E5 and dashed E6 traces use all 20 epochs. Circles mark each run’s selected epoch.",
    "objective": "Validation CE uses raw categorical logits. Training losses differ by objective. The two train categorical panels and learning-rate panel use logarithmic axes. Lower panels show the recorded weight switch and both optimizer learning rates.",
    "resources": "Durations, aggregate throughput, input wait and memory come from epoch records. Validation throughput uses a logarithmic axis to show both runs. Shared compute load, cache transport and process RSS scope limit hardware comparisons.",
    "confusion": "Raw counts and percentages within each true grade at the selected validation checkpoint. A common count scale and a common percentage scale support within-dataset comparison.",
    "grade_errors": "Precision, recall, F1 and support by grade. Direction and ordinal distance counts partition classification errors in two ways. Matrix-derived panels use the full frozen validation support.",
    "changes": "E6 minus E5 for each paired run on the same frozen validation split. Heatmap cells show changes to the row-normalized matrix in percentage points, not paired transitions for individual images.",
}


def figure_notes(records, dataset):
    e5, e6 = (records[m,dataset] for m in ("E5","E6"))
    b5, b6 = best(e5), best(e6)
    d5, d6 = (r["derived_best_validation"] for r in (e5,e6))
    tails = [r["history"][-1] for r in (e5,e6)]
    val_seconds = [sum(h["resource"]["validation_seconds"] for h in r["history"]) for r in (e5,e6)]
    ce_min = [min(r["history"], key=lambda h: h["validation"]["validation_loss"])["epoch"]
              for r in (e5,e6)]
    return {
        "epoch_scores": f"The last-epoch validation macro-F1 values were {tails[0]['validation']['macro_f1']:.3f} "
                        f"for E5 and {tails[1]['validation']['macro_f1']:.3f} for E6, below their selected peaks. "
                        "Training scores improved late in the runs, so training scores alone would favor a different stopping point.",
        "objective": f"Raw validation CE reached its minimum at epoch {ce_min[0]} for E5 and {ce_min[1]} for E6. "
                     "These CE minima and the macro-F1 selection answer different questions.",
        "resources": f"Across 20 epochs, validation phases summed to {val_seconds[0]:.1f} s (E5 cached tensors) and "
                     f"{val_seconds[1]:.1f} s (E6 uncached processing). This is an observation about these completed runs.",
        "confusion": f"E5 correctly graded {b5['correct']} images and E6 correctly graded {b6['correct']}. "
                     f"Raw grade 2→1 errors increased from {b5['confusion_matrix'][2][1]} to {b6['confusion_matrix'][2][1]}.",
        "grade_errors": f"E6 grade 1 recall rose from {d5['per_grade'][1]['recall']:.3f} to "
                        f"{d6['per_grade'][1]['recall']:.3f} alongside more grade 1 false positives. "
                        f"Non-adjacent errors moved from {d5['non_adjacent']} to {d6['non_adjacent']}.",
        "changes": f"The grade 1 diagonal increased by {(d6['per_grade'][1]['recall']-d5['per_grade'][1]['recall'])*100:.2f} "
                   f"percentage points while macro-F1 changed by {b6['macro_f1']-b5['macro_f1']:+.5f}.",
    }


def save_svg(figure, path):
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None}, facecolor="white")
    content = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def plot_dataset(dataset, pair, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.4, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "svg.hashsalt": "e5-e6-validation", "savefig.dpi": 270})
    epochs = list(range(1, 21))
    styles = {"E5": "-", "E6": "--"}
    def save(fig, name):
        stem = destination / f"{dataset.lower()}_{name}"
        fig.savefig(stem.with_suffix(".png"), dpi=270, facecolor="white")
        save_svg(fig, stem.with_suffix(".svg"))
        plt.close(fig)

    def layout(rows, columns, title, width=11, height=6.4):
        fig, axes = plt.subplots(rows, columns, figsize=(width,height), layout="constrained", squeeze=False)
        fig.suptitle(f"{dataset}  |  {title}", fontsize=17, fontweight="bold", x=.02, ha="left")
        return fig, axes.ravel()

    def metric(ax, title, paths, ylabel=None, bounds=None, skip_best=False, scale=1):
        for method, record in pair.items():
            for i, (path, suffix) in enumerate(paths):
                series = [get(epoch, path) for epoch in record["history"]]
                if not any(v is not None for v in series):
                    continue
                vals = [float(v)*scale if v is not None else np.nan for v in series]
                label = f"{method}{suffix}"
                ax.plot(epochs, vals, color=COLORS[method], linestyle=styles[method],
                        alpha=1 if not i else .53, linewidth=1.8, label=label,
                        drawstyle="steps-post" if path == "class_weight_active" else "default")
                ep = record["summary"]["best"]["epoch"]
                if not skip_best:
                    ax.scatter(ep, vals[ep-1], color=COLORS[method], edgecolors="white", s=36, zorder=5)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xlim=(1,20), xticks=[1,5,10,15,20])
        if bounds:
            ax.set_ylim(*bounds)
        ax.grid(axis="y", alpha=.23)
        ax.set_axisbelow(True)
        if ax.lines:
            ax.legend(loc="best", frameon=False, fontsize=8)
        else:
            ax.text(.5,.5,"Not recorded",transform=ax.transAxes,ha="center")

    fig, ax = layout(3, 3, "Training and validation scores", width=11.6, height=8.5)
    for panel, (title,path,bounds) in zip(ax, [
        ("Train accuracy","train.accuracy",(0,1)), ("Validation accuracy","validation.accuracy",(0,1)),
        ("Train macro-F1","train.macro_f1",(0,1)), ("Validation macro-F1","validation.macro_f1",(0,1)),
        ("Validation weighted-F1","validation.weighted_f1",(0,1)),
        ("Validation QWK","validation.quadratic_weighted_kappa",(-1,1)),
        ("Validation grade MAE","validation.grade_mae",None),
        ("Validation high-confidence errors","validation.high_confidence_wrong",None),
        ("Train high-confidence errors","train.high_confidence_wrong",None),
    ]):
        metric(panel, title, [(path,"")], "Count (p ≥ 0.90)" if path.endswith("wrong") else None, bounds)
        if bounds is None:
            panel.set_ylim(bottom=0)
    save(fig, "epoch_scores")

    fig, ax = layout(2, 3, "Objectives and optimizer schedule", width=12.3)
    metric(ax[0], "Train total and categorical loss", [("train.loss"," total"),("train.categorical_loss"," categorical")])
    ax[0].set_yscale("log")
    metric(ax[1], "Ordinal and adjacent training losses", [("train.ordinal_loss"," ordinal"),("train.adjacent_loss"," adjacent")])
    metric(ax[2], "Validation categorical CE", [("validation.validation_loss","")])
    ax[1].set_ylim(bottom=0)
    ax[2].set_ylim(bottom=0)
    metric(ax[3], "Train and validation categorical terms", [("train.categorical_loss"," train"),("validation.validation_loss"," val CE")])
    ax[3].set_yscale("log")
    metric(ax[4], "Class weighting", [("class_weight_active","")], skip_best=True, bounds=(-.15,1.15))
    ax[4].set_yticks([0,1],["Off","On"])
    metric(ax[5], "Optimizer learning rates", [("learning_rates.backbone"," backbone"),("learning_rates.head"," head")], skip_best=True)
    ax[5].set_yscale("log")
    save(fig, "objective")

    fig, ax = layout(3, 3, "Observed resource use", width=11.6, height=8.5)
    for panel, title, paths, unit, factor in [
        (0, "Train throughput", [("train_images_per_second","")], "Images / s", 1),
        (1, "Validation throughput", [("validation_images_per_second","")], "Images / s", 1),
        (2, "Train phase", [("train_seconds","")], "Seconds", 1),
        (3, "Validation phase", [("validation_seconds","")], "Seconds", 1),
        (4, "GPU memory", [("gpu_peak_allocated_bytes"," allocated"),("gpu_peak_reserved_bytes"," reserved")], "GiB", 2**-30),
        (5, "Process peak RSS", [("process_peak_rss_bytes","")], "GiB", 2**-30),
        (6, "Train input wait", [("train_input_seconds","")], "Seconds", 1),
        (7, "Validation input wait", [("validation_input_seconds","")], "Seconds", 1),
    ]:
        metric(ax[panel], title, [(f"resource.{p}", suffix) for p,suffix in paths], unit, skip_best=True, scale=factor)
        if panel == 1:
            ax[panel].set_yscale("log")
        else:
            ax[panel].set_ylim(bottom=0)
    total = ax[8]
    for method, record in pair.items():
        seconds = [r["resource"]["train_seconds"] + r["resource"]["validation_seconds"]
                   for r in record["history"]]
        total.plot(epochs, seconds, color=COLORS[method], linestyle=styles[method], label=method)
    total.set(title="Train + validation phases", xlabel="Epoch", ylabel="Seconds",
              xlim=(1,20), xticks=[1,5,10,15,20], ylim=(0,None))
    total.legend(frameon=False, fontsize=8)
    total.grid(axis="y", alpha=.23)
    save(fig, "resources")

    def heat(ax, matrix, title, scale_max, percentage=False, delta=False):
        kwargs = {"vmin": -scale_max if delta else 0, "vmax": scale_max,
                  "cmap": "PuOr_r" if delta else "Blues"}
        edges = np.arange(6) - .5
        image = ax.pcolormesh(edges, edges, matrix, rasterized=False, **kwargs)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        for i in range(5):
            for j in range(5):
                v = matrix[i,j]
                label = f"{v:+.1f}" if delta else f"{v:.1f}" if percentage else str(v)
                ax.text(j,i,label,ha="center",va="center",fontsize=9,
                        color="white" if abs(v) > scale_max*.63 else "#172a3a")
        ax.set(title=title, xlabel="Predicted grade", ylabel="True grade", xticks=range(5), yticks=range(5))
        return image

    def colorbar(fig, image, *, label, **kwargs):
        bar = fig.colorbar(image, label=label, **kwargs)
        bar.solids.set_rasterized(False)

    fig, ax = layout(2, 2, "Best-checkpoint validation confusion matrices", width=9, height=8.5)
    limit = max(max(map(max,best(record)["confusion_matrix"])) for record in pair.values())
    for col, (method, record) in enumerate(pair.items()):
        epoch = record["summary"]["best"]["epoch"]
        d = record["derived_best_validation"]
        raw = heat(ax[col],np.array(best(record)["confusion_matrix"]),f"{method} epoch {epoch}  |  counts",limit)
        normalized = heat(ax[col+2],np.array(d["row_normalized"])*100,
                          f"{method} epoch {epoch}  |  row %",100,percentage=True)
    colorbar(fig, raw, ax=list(ax[:2]), shrink=.72, label="Images")
    colorbar(fig, normalized, ax=list(ax[2:]), shrink=.72, label="Percent of true grade")
    save(fig, "confusion")

    fig, ax = layout(2, 3, "Per-grade performance and error structure", width=12.1)
    for panel, field in enumerate(("precision","recall","f1","support")):
        for offset, (method, record) in zip((-.18,.18), pair.items()):
            values = [g[field] for g in record["derived_best_validation"]["per_grade"]]
            ax[panel].bar(np.arange(5)+offset, values, .35, color=COLORS[method], label=method)
        ax[panel].set(title={"f1":"F1","support":"Validation support"}.get(field,field.title()),
                      xticks=range(5), xlabel="Grade", ylabel="Images" if field == "support" else "Score")
        if field != "support":
            ax[panel].set_ylim(0,1)
    for panel, keys, title in [(4,("undergraded","overgraded"),"Error direction"),
                               (5,("adjacent","non_adjacent"),"Ordinal distance")]:
        for offset,(method, record) in zip((-.18,.18),pair.items()):
            values = [record["derived_best_validation"][k] for k in keys]
            ax[panel].bar(np.arange(2)+offset,values,.35,color=COLORS[method],label=method)
        ax[panel].set(title=title, ylabel="Images", xticks=range(2), xticklabels=[k.replace("_"," ").title() for k in keys])
    for panel in ax:
        panel.grid(axis="y",alpha=.23)
        panel.set_axisbelow(True)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(.98,.996))
    save(fig,"grade_errors")

    fig, ax = layout(1, 2, "E6 minus E5 at the selected checkpoints", width=10.8, height=4.5)
    a,b = pair["E5"],pair["E6"]
    delta = (np.array(b["derived_best_validation"]["row_normalized"]) -
             np.array(a["derived_best_validation"]["row_normalized"]))*100
    mat = heat(ax[0], delta, "Change in validation confusion", max(float(abs(delta).max()), 1.0), delta=True)
    colorbar(fig, mat, ax=ax[0], shrink=.74, label="Percentage points")
    fields = [("accuracy","Accuracy"),("macro_f1","Macro-F1"),("weighted_f1","Weighted-F1"),
              ("quadratic_weighted_kappa","QWK"),("grade_mae","Grade MAE")]
    values = [(best(b)[field]-best(a)[field]) for field,_ in fields]
    bars = ax[1].barh(range(len(values)),values,color=["#a44918" if v>0 else "#176383" for v in values])
    ax[1].bar_label(bars,labels=[f"{v:+.4f}" for v in values],fontsize=9,padding=3)
    bound = 1.55*max(map(abs,values))
    ax[1].set(title="Absolute metric changes (original units)",xlim=(-bound,bound),
              yticks=range(len(values)),yticklabels=[k for _,k in fields])
    ax[1].axvline(0,color="#283b49",lw=.8)
    ax[1].invert_yaxis()
    save(fig,"changes")


def plot_overview(records, destination):
    import matplotlib.pyplot as plt
    import numpy as np

    fields = [
        ("Accuracy", lambda r: best(r)["accuracy"]), ("Macro-F1", lambda r: best(r)["macro_f1"]),
        ("QWK", lambda r: best(r)["quadratic_weighted_kappa"]), ("Grade MAE", lambda r: best(r)["grade_mae"]),
        *((f"Grade {g} {key.upper() if key == 'f1' else key}",
           lambda r, grade=g, field=key: best(r)["per_grade"][grade][field])
          for g in (1,3) for key in ("recall","f1")),
    ]
    fig, axes = plt.subplots(2,4, figsize=(13.5,6.7), layout="constrained")
    fig.suptitle("E5 vs E6  |  Selected frozen-validation checkpoints", fontsize=17, fontweight="bold", x=.02, ha="left")
    for ax, (name, field) in zip(axes.ravel(), fields):
        for offset, method in zip((-.18,.18), ("E5","E6")):
            values = [field(records[method,dataset]) for dataset in DATASETS]
            bars = ax.bar(np.arange(2)+offset, values, .34, color=COLORS[method], label=method)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set(title=name, xticks=[0,1], xticklabels=list(DATASETS), ylim=(0,1.10))
        ax.grid(axis="y", alpha=.23)
        ax.set_axisbelow(True)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(.98,.996))
    fig.savefig(destination / "e5_e6_best_validation.png", dpi=270, facecolor="white")
    save_svg(fig, destination / "e5_e6_best_validation.svg")
    plt.close(fig)


def compare_report(records):
    fields = ("accuracy", "macro_f1", "weighted_f1", "quadratic_weighted_kappa", "grade_mae", "validation_loss")
    names = ("Accuracy", "Macro-F1", "Weighted-F1", "QWK", "Grade MAE", "Categorical CE")
    overview = []
    changes = []
    grades = []
    lines = ["# E5–E6 ConvNeXt-Tiny validation evidence",
             "Four completed single-seed runs on frozen five-grade splits. Each row uses its own best validation macro-F1 epoch, "
             "with lower raw categorical CE and then earlier epoch as tie-breakers. APTOS2019 has 366 validation images and DDR has 1,252.",
             "[E5 protocol](../novelty_integrated/e5/README.md) · [E6 protocol](../novelty_integrated/e6/README.md) · "
             "[Audit and reproduction](#audit-and-reproduction) · [Figure index](#figure-index)"]
    for dataset in DATASETS:
        for method in ("E5","E6"):
            run = records[method,dataset]
            b = best(run)
            doc = "../" + (run_folder(method,dataset) / "validation_report.md").as_posix()
            overview.append([dataset, f"[{method}]({doc})", run["summary"]["best"]["epoch"], *(b[k] for k in fields),
                             b["correct"],b["wrong"],b["high_confidence_wrong"]])
        a,b = (best(records[method,dataset]) for method in ("E5","E6"))
        for name,k in zip(names,fields):
            changes.append([dataset,name,f"{b[k]-a[k]:+.6f}", f"{(b[k]-a[k])/a[k]*100:+.2f}%"])
        for g in (1,3):
            for method in ("E5","E6"):
                x = records[method,dataset]["derived_best_validation"]["per_grade"][g]
                grades.append([dataset,g,method,x["support"],x["correct"],x["false_positive"],
                               x["false_negative"],x["precision"],x["recall"],x["f1"]])
    lines += [
        "## Best validation checkpoints",
        table(["Dataset","Run","Epoch",*names,"Correct","Wrong","High-confidence wrong"],overview),
        "![Best validation accuracy, macro-F1, QWK, MAE and minority-grade scores](figures/e5_e6_best_validation.png)\n\n"
        "Paired bars summarize selected checkpoints on each dataset. Scores use their original units. "
        "[SVG for publication](figures/e5_e6_best_validation.svg).",
        "### Absolute and relative changes",
        "Absolute change is E6 minus E5 in metric units. Relative change divides this difference by E5. "
        "Lower Grade MAE and categorical CE are preferable. These estimates have no confidence intervals.",
        table(["Dataset","Metric","E6 − E5","Relative to E5"],changes),
        "### Grades 1 and 3", table(["Dataset","Grade","Run","Support","TP","FP","FN","Precision","Recall","F1"],grades),
        "## Interpretation",
        "**APTOS2019.** E5 retained higher accuracy, macro-F1, weighted-F1 and QWK. E6 found 28/35 grade 1 images versus "
        "26/35, yet grade 1 false positives increased from 21 to 35 and F1 fell from 0.634 to 0.571. The largest count "
        "shift was for true grade 2, with correct predictions falling from 76 to 58 and grade 2→1 errors increasing from "
        "13 to 25. Grade 3 recall and F1 rose, based on 12/20 correct versus 10/20. Adjacent and non-adjacent errors both increased.",
        "**DDR.** E6 increased macro-F1 by 0.009256 while accuracy fell by 0.025559, QWK fell by 0.007609 and grade MAE rose. "
        "Grade 1 recall increased from 12/63 to 23/63. Grade 1 predictions increased from 48 to 128, including false positives "
        "rising from 36 to 105. Precision fell from 0.250 to 0.180, and F1 increased from 0.216 to 0.241. Grade 0→1 errors "
        "rose from 10 to 51 and grade 2→1 from 26 to 52. Grade 3 recall fell from 13/24 to 11/24, while fewer false positives "
        "(12 to 3) raised F1. Adjacent errors rose from 109 to 158 and non-adjacent errors fell from 112 to 95. "
        "These shifts explain why average class F1 and ordinal agreement can move in different directions.",
        "With one seed per run, small severe-grade support, within-split duplicate groups, and checkpoint selection on the same "
        "validation sets, the comparisons are descriptive. Aggregate matrices do not identify individual prediction transitions "
        "or support paired uncertainty estimates. E6 changes several objective terms together. A matched objective ablation "
        "would be needed to attribute a gain. For E7, track grade 1 precision, grade 3 recall, QWK and MAE alongside macro-F1.",
        "## What changed between E5 and E6",
        "Both used the verified official ConvNeXt-Tiny backbone with categorical and ordered ordinal heads, 20 epochs, seed 42, "
        "effective batch 32, BF16, AdamW, backbone/head learning rates of 1e−5/1e−3, weight decay 0.01, and B1 train-only "
        "augmentation with the same official processor. The per-dataset manifest and ordered image identities match across both methods.",
        "E5 combined categorical CE with a 0.25 ordinal BCE coefficient and activated inverse-square-root training weights at epoch 11. "
        "E6 replaced the categorical training term with scaled true-class LDAM, used effective-number weighting at epoch 11, kept the "
        "0.25 ordinal term and added a 0.10 neighboring-grade penalty for pairs 0↔1 and 1↔2. Grade 3 findings therefore cannot be "
        "attributed directly to a grade 3 boundary term. Per-dataset weights and margins are in each run report.",
        "Transport also changed. E5 used bounded worker RAM for RGB and verified deterministic validation tensors. E6 used mmap training "
        "RGB and uncached official validation processing. Phase timings reflect these choices and shared load. The runs do not establish "
        "controlled hardware or cache speedups.",
        "## High-confidence error audit",
        "All four runs used the same recorded rule, an incorrect argmax with maximum categorical softmax probability at least 0.90. "
        "At the selected checkpoints E5 recorded 23 APTOS and 118 DDR high-confidence validation errors. The exported E6 series records "
        "zero on both datasets in every epoch. The matrices cannot independently reconstruct probabilities or the threshold counts.",
        "E6 [applies the target-dependent margin and scale 30 in the training objective](../novelty_integrated/e6/objective.py). "
        "Its [validation function](../novelty_integrated/e6/runner.py) delegates to the [E5 evaluator](../novelty_integrated/e5/runner.py), "
        "where CE and softmax both use raw categorical logits. At inference the categorical head supplies the argmax. The target-dependent "
        "margin cannot be applied to predictions without labels. Training on scaled logits and evaluating softmax at raw scale may compress "
        "confidence relative to E5. This is a plausible explanation for the observed zeros. The result establishes neither calibration nor "
        "clinical safety. Probability distributions and a preplanned reliability/calibration analysis would be needed to separate scale from calibration.",
        "## Figure index",
    ]
    for dataset in DATASETS:
        lines.append(f"### {dataset} figure suite")
        notes = figure_notes(records, dataset)
        for name,caption in FIGURES.items():
            basename = f"figures/{dataset.lower()}_{name}"
            title = name.replace("_"," ").capitalize()
            lines.append(f"![{dataset} {title}]({basename}.png)\n\n**{title}.** {caption} {notes[name]} [SVG]({basename}.svg).")
    archive = next(iter(records.values()))["provenance"]["archive_sha256"]
    checks = sum(r["audit"]["csv_cells_checked"] for r in records.values())
    lines += [
        "## Audit and reproduction",
        "| Status | Evidence |\n| --- | --- |\n"
        f"| PASS | Four completed summaries, 80 ordered epochs, {checks:,} CSV/JSON cell comparisons and 160 confusion matrices. "
        "Summary metrics and best-update flags agree with the recorded selection rule. |\n"
        "| PASS | Frozen manifest fingerprints, ordered sample identities and five-grade class support agree within each dataset. |\n"
        "| PASS | Eight supplied checkpoint SHA-256 values and byte sizes agree with the exported original file inventory. |\n"
        "| PASS | All exported CSVs recover their original byte digests after CRLF restoration. All 62 run-level source SHA-256 checks match the recorded Git commits. |\n"
        "| PASS | Existing E5 APTOS run, history CSV and summary bytes match the original export inventory hashes. |\n"
        "| WARNING | The archive contains sanitized copies. Original and transferred file digests are recorded separately. "
        "The eight checkpoint binaries were verified on HPC according to the supplied audit. This review verifies receipts and sidecars. |\n"
        "| WARNING | The export filter stripped per-grade MAE from JSON and omitted some fields such as the training image count. "
        "Original per-grade MAE values are preserved from CSV. Sample counts are separately derived from matrices. |\n"
        "| WARNING | E6 epoch records omit cache counters. `run.json` is the recorded configuration for all four runs. |\n"
        "| WARNING | Threshold counts require exported probabilities to audit independently. Those sample-level files are outside this package. |\n"
        "| FAIL | No failures in the aggregate evidence checks. |",
        f"Source archive SHA-256: `{archive}`. The original and transferred member hashes and source-relative run identities are in each "
        "run’s evidence JSON. The source log is represented by its digest.",
        "From the repository root, using Python with Matplotlib and Pillow, run the verification command below. "
        "`--write` generates the reports in a separate, empty output root. The plotting code uses aggregate JSON and CSV and does not import model code.",
        "```bash\npython docs/e5_e6_publication.py --export /path/to/e5_e6_publication_review.zip\n"
        "python docs/e5_e6_publication.py --export /path/to/e5_e6_publication_review.zip --write --destination /path/to/empty-output\n"
        "python -m unittest discover -s tests -p test_e5_e6_publication.py\n```",
        "## Validation and separate historical final test",
        "The four training summaries record `final_test_status=NOT_ACCESSED` at training completion. A separately documented "
        "[E5 APTOS2019 final test](../results/APTOS2019/e5_integrated/README.md) retains its original provenance and scope. "
        "These figures and comparisons use validation results.",
    ]
    return "\n\n".join(lines) + "\n"


def verify_shared(records):
    for dataset in DATASETS:
        a,b = [records[m,dataset]["configuration"] for m in ("E5","E6")]
        require(a["manifest"] == b["manifest"], f"{dataset}: split/ordered identity differs")
        for key in ("snapshot","effective_processor_sha256","b1_augmentation","seed","epochs",
                    "effective_batch","micro_batch","accumulation","optimizer","backbone_lr",
                    "head_lr","weight_decay","max_grad_norm","precision","checkpoint_selection"):
            require(a[key] == b[key], f"{dataset}: comparison control {key} differs")
    aptos = records["E5","APTOS2019"]["provenance"]["files"]
    for file in ("run.json","summary.json","history.csv"):
        require(sha((ROOT / run_folder("E5","APTOS2019") / file).read_bytes()) == aptos[file]["sha256"],
                f"Historical E5 APTOS {file} does not match the original inventory")


def verify_source(records):
    checked = {}
    for run in records.values():
        cfg = run["configuration"]
        head = cfg["source_git_head"]
        require(re.fullmatch(r"[0-9a-f]{40}", head), "Invalid source commit SHA")
        for name, expected in cfg["source_sha256"].items():
            source = PurePosixPath(name)
            require(not source.is_absolute() and ".." not in source.parts and HEX.fullmatch(expected),
                    "Invalid recorded source path or digest")
            if (head, name) not in checked:
                raw = subprocess.run(["git", "show", f"{head}:{name}"], cwd=ROOT, capture_output=True, check=True)
                checked[head, name] = sha(raw.stdout)
            require(checked[head, name] == expected, f"{run['experiment']}: {name} differs at source commit")
        run["audit"]["source_files_verified"] = len(cfg["source_sha256"])


def write_reports(records, destination):
    paths = [Path("docs/E5_E6_VALIDATION.md")]
    paths += [run_folder(m,d) / filename for m,d in records
              for filename in ("validation_report.md","validation_evidence.json")]
    paths += [Path("docs/figures") / f"{d.lower()}_{f}.{ext}"
              for d in DATASETS for f in FIGURES for ext in ("png","svg")]
    paths += [Path("docs/figures") / f"e5_e6_best_validation.{ext}" for ext in ("png","svg")]
    require(not any((destination / path).exists() for path in paths), "Generated target already exists")
    for method,dataset in records:
        folder = destination / run_folder(method,dataset)
        folder.mkdir(parents=True, exist_ok=True)
        record = records[method,dataset]
        (folder / "validation_evidence.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        (folder / "validation_report.md").write_text(record_report(method,dataset,record), encoding="utf-8", newline="\n")
    figures = destination / "docs/figures"
    figures.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        plot_dataset(dataset, {m: records[m,dataset] for m in ("E5","E6")}, figures)
    plot_overview(records, figures)
    (destination / "docs/E5_E6_VALIDATION.md").write_text(compare_report(records), encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    records = load_export(args.export)
    verify_shared(records)
    verify_source(records)
    for (m,d), record in records.items():
        print(f"PASS {m} {d}: 20 epochs, best {record['summary']['best']['epoch']}, "
              f"{record['audit']['csv_cells_checked']} CSV cells and 2 checkpoint sidecars")
    if args.write:
        write_reports(records, args.destination)
        print("Generated 4 validation reports/evidence pairs, 1 comparison and 13 PNG/SVG figure pairs.")


if __name__ == "__main__":
    main()
