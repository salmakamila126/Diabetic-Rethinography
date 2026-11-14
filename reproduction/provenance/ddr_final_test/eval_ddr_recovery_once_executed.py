import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from reproduction.cadvit.evaluation import (
    file_sha256,
    validate_manifest,
)
from reproduction.cadvit.experiments import (
    Protocol,
    VARIANTS,
    score,
    split_loader,
)
from reproduction.cadvit.model import CADViTCandidate

ROOT = Path.home() / "Kel_13"
REPO = ROOT / "Diabetic-Rethinography-v2"
CAMPAIGN = ROOT / "experiments/cadvit_tables_1_5_balanced_fp32_math_mb8_diag"
RUN = CAMPAIGN / "ddr_full"
MANIFEST = ROOT / "cad_vit_runs/manifests/DDR_seed42.csv"
DATA_ROOT = ROOT / "datasets/DDR"
OUTPUT = RUN / "test_metrics.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit():
    require(not OUTPUT.exists(), "FINAL TEST SUDAH ADA: jangan jalankan ulang")

    cfg = load_json(RUN / "config.json")
    protocol_json = load_json(CAMPAIGN / "protocol.json")
    fork = load_json(CAMPAIGN / "FORK_PROVENANCE.json")
    summary_path = RUN / "history.jsonl"

    require(
        cfg["protocol"] == protocol_json["protocol"],
        "Protocol config dan protocol.json berbeda",
    )
    require(
        file_sha256(CAMPAIGN / "protocol.json")
        == cfg["protocol_sha256"],
        "Protocol SHA mismatch",
    )
    require(
        cfg["variant"] == {
            "name": "ddr_full",
            "dataset": "DDR",
            "task": "five_class",
            "reduction": 16,
            "use_mfb": True,
            "use_attention": True,
        },
        "Variant tidak sesuai",
    )
    require(
        cfg["model_config"] == VARIANTS["ddr_full"].model_config()
        or cfg["model_config"] == json.loads(
            json.dumps(VARIANTS["ddr_full"].model_config())
        ),
        "Konfigurasi model berbeda",
    )

    for name, expected_key in (
        ("model.py", "model_sha256"),
        ("preprocessing.py", "preprocessing_sha256"),
        ("experiments.py", "runner_sha256"),
    ):
        path = REPO / "reproduction/cadvit" / name
        require(
            file_sha256(path) == cfg[expected_key],
            f"SHA mismatch: {name}",
        )

    require(
        fork["source_epoch_completed"] == 42
        and fork["fork_micro_batch"] == 8
        and fork["fork_accumulation"] == 2
        and fork["attention_backend"] == "MATH",
        "Fork provenance berbeda",
    )

    report = validate_manifest(
        MANIFEST,
        "DDR",
        DATA_ROOT,
        cfg["manifest_sha256"],
    )
    require(
        report["split_sizes"]["test"] == 1252,
        "Jumlah test tidak sesuai audit",
    )

    history = [
        json.loads(line)
        for line in summary_path.read_text().splitlines()
        if line.strip()
    ]
    require(
        [r["epoch"] for r in history] == list(range(1, 81)),
        "History tidak lengkap",
    )
    selected = max(
        history,
        key=lambda r: (r["val_macro_f1"], -r["val_loss"]),
    )
    require(
        selected["epoch"] == 69
        and math.isclose(
            selected["val_macro_f1"],
            0.6530693590977841,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "Best validation checkpoint tidak sesuai",
    )

    # Hanya checkpoint yang dibuat dan dipercaya oleh tim sendiri.
    best = torch.load(
        RUN / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    last = torch.load(
        RUN / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    require(
        best["epoch"] == 68 and last["epoch"] == 79,
        "Epoch checkpoint tidak sesuai",
    )
    require(
        best["record"] == cfg and last["record"] == cfg,
        "Checkpoint/config berbeda",
    )
    require(
        math.isclose(
            best["best_f1"],
            selected["val_macro_f1"],
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "Best checkpoint F1 berbeda",
    )

    manifest_info = load_json(
        REPO / "results/DDR/full_oversampling_fp32_math_mb8_80epochs/checkpoint_manifest.json"
    ) if (
        REPO / "results/DDR/full_oversampling_fp32_math_mb8_80epochs/checkpoint_manifest.json"
    ).exists() else None

    if manifest_info:
        require(
            file_sha256(RUN / "best.pt")
            == manifest_info["best_checkpoint_sha256"],
            "Best checkpoint SHA mismatch",
        )
        require(
            file_sha256(RUN / "last.pt")
            == manifest_info["last_checkpoint_sha256"],
            "Last checkpoint SHA mismatch",
        )

    model = CADViTCandidate(cfg["model_config"])
    model.load_state_dict(best["model"], strict=True)
    require(
        all(torch.isfinite(t).all().item()
            for t in best["model"].values()),
        "Checkpoint memiliki tensor non-finite",
    )

    protocol = Protocol(**cfg["protocol"])
    protocol.validate()

    print("PASS: model, protocol, checkpoint, manifest, history")
    print("Best epoch:", selected["epoch"])
    print("Validation macro-F1:", selected["val_macro_f1"])
    print("Frozen test images:", report["split_sizes"]["test"])
    print("Final test:", "NOT EXECUTED")

    return cfg, report, best, protocol


def execute(cfg, report, best, protocol):
    require(not OUTPUT.exists(), "Final test sudah ada")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = CADViTCandidate(cfg["model_config"])
    model.load_state_dict(best["model"], strict=True)
    model = model.to(device).eval()

    # Batch evaluation dapat lebih kecil daripada batch training.
    # Tidak mengubah dataset, preprocessing, atau checkpoint.
    eval_protocol = replace(
        protocol,
        micro_batch_size=2,
        accumulation=8,
    )
    eval_protocol.validate()

    rows = report["rows"]
    test_rows = [r for r in rows if r["split"] == "test"]

    loader = split_loader(
        rows,
        DATA_ROOT,
        "test",
        "five_class",
        eval_protocol,
    )

    if device.type == "cuda":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        context = sdpa_kernel(SDPBackend.MATH)
    else:
        context = nullcontext()

    with context:
        metrics = score(
            model,
            loader,
            device,
            "five_class",
            collect_predictions=True,
        )

    require(metrics["n"] == len(test_rows) == 1252,
            "Jumlah prediksi tidak sesuai manifest")
    require(
        sum(map(sum, metrics["confusion_matrix"])) == 1252,
        "Jumlah confusion matrix salah",
    )

    for prediction, row in zip(metrics["predictions"], test_rows):
        require(
            prediction["true_label"] == int(row["label"]),
            "Urutan label test tidak cocok",
        )
        prediction.update({
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "image_sha256": row["sha256"],
        })

    payload = {
        "dataset": "DDR",
        "variant": "ddr_full",
        "campaign": "cadvit_tables_1_5_balanced_fp32_math_mb8_diag",
        "split": "test",
        "selected_epoch": 69,
        "selection": "validation_macro_f1",
        "protocol_type": "mixed_protocol_recovery",
        "manifest_sha256": cfg["manifest_sha256"],
        "protocol_sha256": cfg["protocol_sha256"],
        "model_sha256": cfg["model_sha256"],
        "preprocessing_sha256": cfg["preprocessing_sha256"],
        "checkpoint_sha256": file_sha256(RUN / "best.pt"),
        "metrics": metrics,
    }

    # Atomic publication without overwriting prior test results.
    temporary = OUTPUT.with_name(
        OUTPUT.name + f".{os.getpid()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    print("FINAL TEST SAVED:", OUTPUT)
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["f1_macro"])
    print("Confusion matrix:", metrics["confusion_matrix"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-final-test",
        action="store_true",
    )
    args = parser.parse_args()

    cfg, report, best, protocol = audit()

    if args.execute_final_test:
        execute(cfg, report, best, protocol)
    else:
        print("AUDIT ONLY — no test images opened")


if __name__ == "__main__":
    main()
