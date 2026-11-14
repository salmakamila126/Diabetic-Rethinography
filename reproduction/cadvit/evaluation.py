"""Preflight and explicit final-test evaluation for the historical V2 80-epoch runs.

The preflight never opens an image. ``run`` requires a second explicit flag and
must be invoked manually only after its preflight report has been reviewed.
"""

import argparse
import ast
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath

from reproduction.cadvit.model import ARCH_CONFIG, CADViTCandidate


RUNS = {
    "APTOS2019": {
        "directory": "aptos_v2_bg10_stable_80ep_seed42",
        "best_epoch": 69,
        "best_f1": 0.6571274188819457,
        "samples": 3662,
        "logs": {"cad_vit_runs/logs/aptos_80ep.log":
                 "c81a2d3b3bfafe03ecb390bce2e2839d3d1bd7a4db0575294a7e6fd1b5b4ed08"},
    },
    "DDR": {
        "directory": "ddr_v2_fp32_weighted_fp64clip_seed42",
        "best_epoch": 72,
        "best_f1": 0.6903234255707131,
        "samples": 12522,
        "logs": {
            "cad_vit_runs/logs/ddr_80ep.log":
                "4101e32a0028211ad3f0a0126050b5c0e3c2ce2d7d01edb699b241856b5ca24b",
            "checkpoints/ddr_v2_fp32_weighted_seed42/history.jsonl":
                "379f1a53ee2871c5eacdda1f89f980094e35bfd871a5cbe6027c9a72b6eef9b6",
            "checkpoints/ddr_v2_fp32_weighted_fp64clip_seed42/history.jsonl":
                "2ec854cd4eccacd53f595e2ffc71347ebdc2a344e2bbc984fe85247fd1518932",
        },
    },
}
RESULT_NAME = "final_test_v2_80ep.json"
REQUIRED_MANIFEST_COLUMNS = {"image_id", "label", "image_path", "group_id", "sha256", "split"}


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_config_from_source():
    """Read the unchanged historical constants without importing optional OpenCV."""
    source = Path(__file__).with_name("preprocessing.py").read_text(encoding="utf-8")
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"PREPROCESS_CONFIG", "BG_V2_CONFIG"}:
                found[target.id] = ast.literal_eval(node.value)
    if set(found) != {"PREPROCESS_CONFIG", "BG_V2_CONFIG"}:
        raise ValueError("Konfigurasi preprocessing historis tidak ditemukan")
    return found


def validate_manifest(manifest_path, dataset, dataset_root, expected_sha):
    """Check split/metadata only; do not open or decode any fundus image."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != expected_sha:
        raise ValueError("Manifest SHA-256 tidak sesuai checkpoint training")
    meta_path = manifest_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dataset") != dataset or meta.get("seed") != 42:
        raise ValueError("Dataset/seed metadata manifest tidak sesuai")
    if meta.get("manifest_file_sha256") != manifest_sha:
        raise ValueError("Manifest berubah sejak metadata dibuat")
    if meta.get("manifest_sha256") and meta["manifest_sha256"] != manifest_sha:
        raise ValueError("Fingerprint manifest metadata berbeda")
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if Path(meta.get("root", "")).expanduser().resolve() != dataset_root:
        raise ValueError("Root dataset berbeda dari metadata training; audit pemindahan data dahulu")
    rows, ids, paths = [], set(), set()
    group_split, hash_split = {}, {}
    counts = {part: [0] * 5 for part in ("train", "val", "test")}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_MANIFEST_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Kolom manifest tidak lengkap")
        for row in reader:
            split, label_text = row["split"], row["label"]
            if split not in counts or label_text not in {str(i) for i in range(5)}:
                raise ValueError("Split atau label di luar protokol lima kelas")
            image_id, image_path = row["image_id"], row["image_path"]
            relative = PurePosixPath(image_path)
            if not image_id or not image_path or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("ID atau path gambar tidak aman")
            if image_id in ids or image_path in paths:
                raise ValueError("ID atau path gambar duplikat")
            ids.add(image_id)
            paths.add(image_path)
            for field, allocation in (("group_id", group_split), ("sha256", hash_split)):
                key = row[field]
                if not key or (key in allocation and allocation[key] != split):
                    raise ValueError(f"{field} kosong atau bocor antar split")
                allocation[key] = split
            if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
                raise ValueError("SHA-256 gambar di manifest tidak valid")
            counts[split][int(label_text)] += 1
            rows.append(row)
    n = len(rows)
    expected_counts = {"train": round(0.8 * n), "val": round(0.1 * n)}
    expected_counts["test"] = n - expected_counts["train"] - expected_counts["val"]
    if n != RUNS[dataset]["samples"]:
        raise ValueError(f"Jumlah gambar {n} tidak sesuai audit dataset")
    if any(sum(counts[part]) != expected_counts[part] or any(v == 0 for v in counts[part])
           for part in counts):
        raise ValueError("Rasio 8:1:1 atau representasi lima kelas tidak sesuai")
    if meta.get("counts"):
        observed = {f"{part}:{label}": count for part, values in counts.items()
                    for label, count in enumerate(values) if count}
        if meta["counts"] != observed:
            raise ValueError("Jumlah per split/kelas berbeda dari metadata")
    return {"sha256": manifest_sha, "rows": rows, "counts": counts,
            "split_sizes": {part: sum(values) for part, values in counts.items()}}


def validate_checkpoint_pair(best, last, dataset, manifest_sha):
    """Validate trusted, already-loaded checkpoint dictionaries without inference."""
    import torch

    spec = RUNS[dataset]
    if not isinstance(best, dict) or not isinstance(last, dict):
        raise ValueError("Checkpoint harus berupa dictionary")
    record = best.get("record")
    if not isinstance(record, dict) or last.get("record") != record:
        raise ValueError("Record best/last hilang atau berbeda")
    experiment = record.get("experiment")
    if record.get("pipeline_version") != "V2_BG10_STABLE" or not isinstance(experiment, dict):
        raise ValueError("Checkpoint bukan protokol V2 BG10")
    required = {"dataset": dataset, "seed": 42, "epochs": 80,
                "physical_batch_size": 16, "accumulation": 1,
                "balance": False, "augment": True, "ablation": "MFB+Attention"}
    if any(experiment.get(key) != value for key, value in required.items()):
        raise ValueError("Konfigurasi eksperimen checkpoint tidak sesuai V2 80 epoch")
    if not math.isclose(float(experiment.get("learning_rate", float("nan"))), 0.000186,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("Learning rate checkpoint berbeda")
    if record.get("manifest_sha256") != manifest_sha:
        raise ValueError("Checkpoint dan manifest memiliki SHA-256 berbeda")
    source_config = preprocessing_config_from_source()
    if record.get("preprocessing") != {"base": source_config["PREPROCESS_CONFIG"],
                                       "ben_graham": source_config["BG_V2_CONFIG"]}:
        raise ValueError("Checkpoint tidak sesuai preprocessing V2 BG10 historis")
    if record.get("model") != ARCH_CONFIG:
        raise ValueError("Konfigurasi model checkpoint berbeda dari model historis")
    model = CADViTCandidate(record["model"])
    expected_state = model.state_dict()
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 5_282_101 or record.get("parameter_count") != parameter_count:
        raise ValueError("Jumlah parameter model tidak sesuai checkpoint historis")
    if best.get("epoch") != spec["best_epoch"] - 1 or last.get("epoch") != 79:
        raise ValueError("Epoch best/last tidak sesuai log validation 80 epoch")
    for name, state in (("best", best), ("last", last)):
        if not math.isclose(float(state.get("best_f1", float("nan"))), spec["best_f1"],
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Validation macro-F1 {name} tidak sesuai log")
        weights = state.get("model")
        if not isinstance(weights, dict) or set(weights) != set(expected_state):
            raise ValueError(f"State dict model {name} tidak lengkap")
        for key, expected in expected_state.items():
            value = weights[key]
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape:
                raise ValueError(f"Shape tensor checkpoint {name}:{key} berbeda")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Tensor checkpoint {name}:{key} tidak finite")
    if dataset == "DDR":
        recovery = best.get("recovery")
        if not isinstance(recovery, dict) or last.get("recovery") != recovery:
            raise ValueError("Metadata recovery DDR best/last hilang atau berbeda")
        if (recovery.get("loss") != "inverse_sqrt_frequency_weighted_cross_entropy"
                or recovery.get("precision_during_recovery") != "FP32"
                or recovery.get("manifest_sha256") != manifest_sha):
            raise ValueError("Metadata recovery DDR tidak sesuai fase weighted FP32")
    return {"best_epoch": spec["best_epoch"], "last_epoch": 80,
            "best_validation_macro_f1": spec["best_f1"], "parameter_count": parameter_count,
            "checkpoint_format": "epoch zero-based; record/model/optimizer/scaler/best_f1/best_loss/RNG"}


def preflight(artifact_root, dataset, dataset_root=None, trust_checkpoints=False):
    """Return audit evidence; no test image is opened and no output is written."""
    import torch

    if dataset not in RUNS:
        raise ValueError("Dataset harus APTOS2019 atau DDR")
    if not trust_checkpoints:
        raise ValueError("torch.load membutuhkan checkpoint tepercaya; set --trust-checkpoints")
    root = Path(artifact_root).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if root == repo or repo in root.parents:
        raise ValueError("Artefak evaluasi harus berada di luar repository")
    spec = RUNS[dataset]
    run_dir = root / "checkpoints" / spec["directory"]
    manifest_path = root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    dataset_root = (Path(dataset_root) if dataset_root else root / "datasets" / dataset).expanduser().resolve()
    for name in (RESULT_NAME, "test_metrics.json"):
        if (run_dir / name).exists():
            raise FileExistsError(f"Hasil final test sudah ada: {run_dir / name}")
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("Checkpoint best.pt/last.pt run V2 80 epoch belum lengkap")
    # Pickle can execute code. Only use project-owned checkpoints after explicit trust.
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    record = best.get("record") if isinstance(best, dict) else None
    expected_sha = record.get("manifest_sha256") if isinstance(record, dict) else None
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("SHA-256 manifest tidak ada pada best.pt")
    manifest = validate_manifest(manifest_path, dataset, dataset_root, expected_sha)
    checkpoint = validate_checkpoint_pair(best, last, dataset, manifest["sha256"])
    config_path = run_dir / "config.json"
    if config_path.is_file():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        normalized_record = json.loads(json.dumps(record))
        if saved_config != normalized_record:
            raise ValueError("config.json run berbeda dari checkpoint")
    if dataset == "DDR":
        recovery_path = run_dir / "recovery_config.json"
        if not recovery_path.is_file():
            raise FileNotFoundError("recovery_config.json DDR tidak ditemukan")
        if json.loads(recovery_path.read_text(encoding="utf-8")) != best["recovery"]:
            raise ValueError("Konfigurasi recovery DDR berbeda dari checkpoint")
    verified_logs = {}
    for relative, expected in spec["logs"].items():
        log_path = root / relative
        actual = file_sha256(log_path)
        if actual != expected:
            raise ValueError(f"SHA-256 log berbeda dari provenance notebook: {relative}")
        verified_logs[relative] = actual
    return {"status": "PASS", "dataset": dataset, "run": spec["directory"],
            "manifest": str(manifest_path), "manifest_sha256": manifest["sha256"],
            "split_sizes": manifest["split_sizes"], "class_counts": manifest["counts"],
            "dataset_root": str(dataset_root), "run_dir": str(run_dir),
            "checkpoint_sha256": {"best.pt": file_sha256(best_path),
                                  "last.pt": file_sha256(last_path)},
            "checkpoint": checkpoint, "verified_log_sha256": verified_logs,
            "limitations": ["Per-image SHA from manifest not rehashed; no test image opened",
                            "DDR checkpoint transfer lineage is not proven by matching epoch/log metadata"]}


def classification_metrics(y_true, y_pred):
    """Notebook-compatible five-class macro one-vs-rest metrics; rows=true."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Prediksi kosong atau panjang tidak sama")
    cm = [[0] * 5 for _ in range(5)]
    for truth, pred in zip(y_true, y_pred):
        if type(truth) is not int or type(pred) is not int or not 0 <= truth < 5 or not 0 <= pred < 5:
            raise ValueError("Label/prediksi harus integer 0..4")
        cm[truth][pred] += 1
    n = len(y_true)
    report = {}
    for k in range(5):
        tp = cm[k][k]
        support = sum(cm[k])
        fp = sum(cm[i][k] for i in range(5) if i != k)
        fn = support - tp
        tn = n - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else None
        report[str(k)] = {"support": support, "precision": precision,
                          "recall": recall, "f1": f1, "specificity": specificity}
    defined_specificity = [r["specificity"] for r in report.values() if r["specificity"] is not None]
    return {"n": n, "accuracy": sum(cm[k][k] for k in range(5)) / n,
            "precision_macro": sum(r["precision"] for r in report.values()) / 5,
            "recall_macro": sum(r["recall"] for r in report.values()) / 5,
            "f1_macro": sum(r["f1"] for r in report.values()) / 5,
            "specificity_macro": (sum(defined_specificity) / len(defined_specificity)
                                  if defined_specificity else None),
            "confusion_matrix": cm, "per_class": report,
            "aggregation": "five-class macro one-vs-rest; zero_division=0 for precision/recall/F1; undefined specificity omitted"}


def run_final_test(report, batch_size=16):
    """Explicit caller-only test inference, after a successful preflight."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from reproduction.cadvit.preprocessing import (BG_V2_CONFIG, PREPROCESS_CONFIG,
                                                  preprocess_fundus_v2)

    if report.get("status") != "PASS":
        raise ValueError("Final test membutuhkan preflight PASS")
    if batch_size < 1:
        raise ValueError("Batch size harus positif")
    dataset = report["dataset"]
    root = Path(report["dataset_root"])
    with Path(report["manifest"]).open("r", encoding="utf-8-sig", newline="") as stream:
        test_rows = [r for r in csv.DictReader(stream) if r["split"] == "test"]

    class TestImages(Dataset):
        def __len__(self):
            return len(test_rows)

        def __getitem__(self, index):
            row = test_rows[index]
            image_path = (root / PurePosixPath(row["image_path"])).resolve()
            if root not in image_path.parents:
                raise ValueError("Path gambar di luar root dataset")
            with Image.open(image_path) as image:
                pixels = preprocess_fundus_v2(image, PREPROCESS_CONFIG, BG_V2_CONFIG)
            x = torch.from_numpy(np.array(pixels, copy=True, order="C")).permute(2, 0, 1).float() / 255.0
            return x, int(row["label"])

    run_dir = Path(report["run_dir"])
    result_path = run_dir / RESULT_NAME
    if result_path.exists() or (run_dir / "test_metrics.json").exists():
        raise FileExistsError("Final test telah tercatat; jangan ulang untuk tuning")
    if file_sha256(report["manifest"]) != report["manifest_sha256"]:
        raise ValueError("Manifest berubah setelah preflight")
    for name, expected in report["checkpoint_sha256"].items():
        if file_sha256(run_dir / name) != expected:
            raise ValueError(f"Checkpoint {name} berubah setelah preflight")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CADViTCandidate(best["record"]["model"]).to(device)
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    true, predicted = [], []
    loader = DataLoader(TestImages(), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            if logits.shape != (len(y), 5) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Logits test tidak valid")
            true.extend(int(v) for v in y.tolist())
            predicted.extend(int(v) for v in logits.argmax(1).cpu().tolist())
    if len(true) != report["split_sizes"]["test"]:
        raise ValueError("Jumlah prediksi test tidak cocok dengan manifest")
    metrics = classification_metrics(true, predicted)
    payload = {"dataset": dataset, "run": report["run"], "split": "test",
               "selected_epoch": report["checkpoint"]["best_epoch"],
               "selection_metric": "validation macro-F1", "manifest_sha256": report["manifest_sha256"],
               "preprocessing": "V2 BG10", "metrics": metrics,
               "provenance": "New final test; distinct from historical V1 and V2 validation"}
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return result_path, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--dataset", choices=tuple(RUNS), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Kel_13 artifact root, outside this Git repository")
    parser.add_argument("--dataset-root", type=Path,
                        help="Override dataset-specific root; must match manifest metadata")
    parser.add_argument("--trust-checkpoints", action="store_true",
                        help="Allow torch.load(weights_only=False) for project-owned checkpoints")
    parser.add_argument("--execute-final-test", action="store_true",
                        help="Required only for run; never implicit from preflight")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "run" and not args.execute_final_test:
        parser.error("run requires --execute-final-test after reviewing preflight")
    report = preflight(args.artifact_root, args.dataset, args.dataset_root, args.trust_checkpoints)
    if args.command == "preflight":
        print(json.dumps(report, indent=2))
    else:
        path, payload = run_final_test(report, args.batch_size)
        print(json.dumps({"result_path": str(path), **payload}, indent=2))


if __name__ == "__main__":
    main()