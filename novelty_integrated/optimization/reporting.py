"""Serialize aggregate comparisons and render a standalone academic figure."""

import csv
from html import escape

from novelty import foundation_adaptation as fa


def _rows(report):
    for variant in report["variants"]:
        for name, backend in variant["backends"].items():
            result = {"variant": variant["name"], "ratio": variant.get("ratio"),
                      "backend": name, "status": backend["status"],
                      "parameters": variant["profile"]["parameters"],
                      "macs_per_image": variant["profile"]["macs_per_image"]}
            if backend["status"] == "VALIDATED":
                result.update(artifact_bytes=backend["artifact"]["bytes"],
                              model_p50_ms=backend["timing"]["model_only"]["median_ms_per_image"],
                              model_p95_ms=backend["timing"]["model_only"]["p95_ms_per_image"],
                              e2e_p50_ms=backend["timing"]["end_to_end"]["median_ms_per_image"],
                              macro_f1=backend["quality"]["macro_f1"],
                              qwk=backend["quality"]["quadratic_weighted_kappa"],
                              grade1_recall=backend["quality"]["per_grade"][1]["recall"],
                              grade3_recall=backend["quality"]["per_grade"][3]["recall"])
            yield result


def write_report(output, report):
    rows = list(_rows(report))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_figure(output / "comparison.svg", report, rows)
    fa._atomic_json(output / "report.json", report)


def write_figure(path, report, rows):
    rows = [r for r in rows if r["status"] == "VALIDATED"]
    width, height = 960, 126 + 30 * max(1, len(rows))
    x_label, x_time, x_f1 = 24, 330, 642
    max_latency = max((r["model_p50_ms"] for r in rows), default=1) or 1
    subtitle = f"{report['dataset']} · {report['arm']} · same {report['sample']['count']} validation images · CPU FP32"
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">ConvNeXt inference and pruning comparison</title>',
           f'<desc id="desc">{escape(subtitle)}</desc>', '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="DejaVu Sans, sans-serif" fill="#163044">',
           '<text x="24" y="30" font-size="23" font-weight="bold">ConvNeXt inference and pruning</text>',
           f'<text x="24" y="55" font-size="14">{escape(subtitle)}</text>',
           f'<text x="{x_label}" y="84" font-size="14">Variant and backend</text>',
           f'<text x="{x_time}" y="84" font-size="14">Model-only p50 (ms/image)</text>',
           f'<text x="{x_f1}" y="84" font-size="14">Macro-F1 (0–1)</text>',
           '<line x1="24" x2="930" y1="92" y2="92" stroke="#b8c5cd"/>']
    for i, row in enumerate(rows):
        y = 109 + i * 30
        label = escape(f"{row['variant']} · {row['backend']}")
        latency, f1 = row["model_p50_ms"], row["macro_f1"]
        svg += [f'<text x="{x_label}" y="{y+6}" font-size="12">{label}</text>',
                f'<rect x="{x_time}" y="{y-8}" width="{205*latency/max_latency:.2f}" height="14" fill="#226e96"/>',
                f'<text x="{x_time+214}" y="{y+5}" font-size="12">{latency:.3f}</text>',
                f'<rect x="{x_f1}" y="{y-8}" width="{205*f1:.2f}" height="14" fill="#307d61"/>',
                f'<text x="{x_f1+214}" y="{y+5}" font-size="12">{f1:.4f}</text>']
    if not rows:
        svg.append('<text x="24" y="116" font-size="13">No validated backend measurements</text>')
    svg += ['</g>', '</svg>']
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(svg) + "\n")
