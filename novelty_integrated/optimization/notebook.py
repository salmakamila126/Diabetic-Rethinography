"""Read-only display of staged campaign results in Jupyter."""

import json
from pathlib import Path

from .hpc import inventory
from .reporting import _rows


def load(job, campaign, stage):
    if stage not in ("measurement", "post-recovery"):
        raise ValueError("Unknown report stage")
    entry = inventory()[job]
    path = Path(campaign).expanduser() / job / stage / "report.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_stage = "measure" if stage == "measurement" else "measure-recovered"
    if (not entry.evidence or result.get("status") != "MEASUREMENT_COMPLETE"
            or result.get("schema") != "convnext-optimization-v1" or result.get("stage") != expected_stage
            or result.get("dataset") != entry.dataset or result.get("arm") != entry.arm
            or result["receipt"]["checkpoint_sha256"] != entry.checkpoint_sha256
            or result["receipt"]["manifest"]["sha256"] != entry.configuration["manifest"]["manifest_sha256"]
            or result["sample"]["split"] != "val" or result["sample"]["class_order"] != [0, 1, 2, 3, 4]):
        raise ValueError("Report does not match the expected staged job and frozen evidence")
    return result


def table(headers, rows):
    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "&#124;").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(fmt, row)) + " |"
                     for row in (headers, ["---"] * len(headers), *rows))


def render(report, quality_backend="eager"):
    columns = (
        ("Variant", "variant"), ("Backend", "backend"), ("Status", "status"),
        ("Parameters", "parameters"), ("MACs/image", "macs_per_image"), ("FLOPs/image", "flops_per_image"),
        ("Artifact bytes", "artifact_bytes"), ("Model p50 ms", "model_p50_ms"),
        ("Model p95 ms", "model_p95_ms"), ("Model img/s", "model_images_s"),
        ("E2E p50 ms", "e2e_p50_ms"), ("E2E p95 ms", "e2e_p95_ms"), ("E2E img/s", "e2e_images_s"),
        ("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Weighted-F1", "weighted_f1"),
        ("QWK", "qwk"), ("Grade MAE", "grade_mae"),
    )
    rows = list(_rows(report))
    sections = [table([label for label, _ in columns], [[row.get(key) for _, key in columns] for row in rows])]
    sections += ["**Provenance**", table(["Field", "Value"], [
        ("Original checkpoint", report["receipt"]["checkpoint_sha256"]),
        ("Frozen manifest", report["receipt"]["manifest"]["sha256"]),
        ("Protocol digest", report["protocol_sha256"]),
        ("Selected validation images", report["sample"]["count"]),
        ("Ordered input tensor digest", report["sample"]["ordered_preprocessed_tensor_sha256"]),
        ("CPU allocation", report["allocation"]["id"]),
        ("CPU model / threads", f"{report['runtime']['cpu']} / {report['allocation']['threads']}"),
    ])]
    for variant in report["variants"]:
        backend = variant["backends"].get(quality_backend, {})
        if backend.get("status") != "VALIDATED":
            continue
        scores = backend["quality"]
        sections += [f"#### {variant['name']} on {quality_backend}",
            table(["Grade", "Precision", "Recall", "F1", "Support"],
                  [[g["grade"], g["precision"], g["recall"], g["f1"], g["support"]]
                   for g in scores["per_grade"]]),
            table(["True grade / prediction", 0, 1, 2, 3, 4],
                  [[g, *row] for g, row in enumerate(scores["confusion_matrix"])]),
            f"Undergraded **{scores['undergraded']}**, overgraded **{scores['overgraded']}**."]
        for label, values in (
            ("Compared with original, same backend", variant.get("delta_vs_original", {}).get("same_backend", {})),
            ("Compared with matched unrecovered ratio", variant.get("delta_vs_pruned_before_recovery", {})),
        ):
            delta = values.get(quality_backend)
            if delta:
                sections.append(table([label, "Delta"], [
                    ("Macro-F1", delta["quality"]["macro_f1"]),
                    ("QWK", delta["quality"]["quadratic_weighted_kappa"]),
                    ("Grade MAE", delta["quality"]["grade_mae"]),
                    ("Model-only median (ms/image)", delta["timing"]["model_only"]["median_ms_per_image"]),
                    ("End-to-end median (ms/image)", delta["timing"]["end_to_end"]["median_ms_per_image"]),
                    *[(f"Grade {grade} recall", change)
                      for grade, change in enumerate(delta["quality"]["per_grade_recall"])],
                ]))
    measured = [r for r in rows if r["status"] == "VALIDATED"]
    if measured:
        fastest = min(measured, key=lambda r: r["e2e_p50_ms"])
        best = max(measured, key=lambda r: r["macro_f1"])
        sections += ["**Descriptive conclusion**",
            f"Lowest end-to-end median: **{fastest['variant']} / {fastest['backend']}**. "
            f"Highest macro-F1: **{best['variant']} / {best['backend']}**. "
            "Decide whether the observed grade-wise changes and latency satisfy a prespecified deployment trade-off."]
    return "\n\n".join(sections)
