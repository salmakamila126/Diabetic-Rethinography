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
                      "macs_per_image": variant["profile"]["macs_per_image"],
                      "flops_per_image": variant["profile"].get("flops_per_image_two_per_mac")}
            if backend["status"] == "VALIDATED":
                result.update(artifact_bytes=backend["artifact"]["bytes"],
                              model_p50_ms=backend["timing"]["model_only"]["median_ms_per_image"],
                              model_p95_ms=backend["timing"]["model_only"]["p95_ms_per_image"],
                              model_images_s=backend["timing"]["model_only"]["images_per_second"],
                              e2e_p50_ms=backend["timing"]["end_to_end"]["median_ms_per_image"],
                              e2e_p95_ms=backend["timing"]["end_to_end"]["p95_ms_per_image"],
                              e2e_images_s=backend["timing"]["end_to_end"]["images_per_second"],
                              accuracy=backend["quality"]["accuracy"],
                              macro_f1=backend["quality"]["macro_f1"],
                              weighted_f1=backend["quality"]["weighted_f1"],
                              qwk=backend["quality"]["quadratic_weighted_kappa"],
                              grade_mae=backend["quality"]["grade_mae"],
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
    write_tradeoff(output / "tradeoff.svg", rows)
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


def write_tradeoff(path, rows):
    rows = [r for r in rows if r["status"] == "VALIDATED"]
    ceiling = 1.07 * max((r["e2e_p50_ms"] for r in rows), default=1)
    colors = {"eager": "#176383", "torchscript": "#a44918", "onnxruntime": "#247553", "openvino": "#674783"}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="510" viewBox="0 0 860 510">',
           '<title>Validation macro-F1 versus end-to-end CPU latency</title>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="DejaVu Sans,sans-serif" fill="#163044">',
           '<text x="30" y="35" font-size="23" font-weight="bold">Validation quality versus CPU latency</text>',
           '<text x="30" y="62" font-size="13">Upper left is better. Hover over a point for its treatment and exact values.</text>']
    for tick in range(6):
        x, y = 80 + 640 * tick / 5, 410 - 310 * tick / 5
        svg += [f'<path d="M80 {y}H720 M{x} 100V410" stroke="#e1e7eb"/>',
                f'<text x="70" y="{y+4}" text-anchor="end" font-size="12">{tick/5:.1f}</text>',
                f'<text x="{x}" y="431" text-anchor="middle" font-size="12">{ceiling*tick/5:.3g}</text>']
    svg += ['<path d="M80 100V410H720" fill="none" stroke="#163044"/>',
            '<text x="400" y="461" text-anchor="middle" font-size="14">End-to-end p50 (ms/image)</text>',
            '<text x="23" y="260" transform="rotate(-90 23 260)" text-anchor="middle" font-size="14">Macro-F1</text>']
    for i, (backend, color) in enumerate(colors.items()):
        y = 120 + i * 23
        svg += [f'<circle cx="746" cy="{y-4}" r="5" fill="{color}"/>',
                f'<text x="759" y="{y}" font-size="12">{backend}</text>']
    for row in rows:
        latency, score = row["e2e_p50_ms"], row["macro_f1"]
        x, y = 80 + 640 * latency / ceiling, 410 - 310 * score
        label = escape(f"{row['variant']} / {row['backend']}: {latency:.6g} ms, macro-F1 {score:.6g}")
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{colors[row["backend"]]}" fill-opacity=".80">'
                   f'<title>{label}</title></circle>')
    svg += ['</g>', '</svg>']
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(svg) + "\n")
