"""Read-only observability helpers for foundation adaptation epochs.

No model calls, optimizer steps, or test access occur in this module.
"""
import csv
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import sys
import time
import uuid


class RunLogger:
    def __init__(self, output):
        self.path = Path(output) / "training.log"
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)

    def line(self, message, *, console=True):
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} {message}"
        if console:
            print(line, flush=True)
        self.stream.write(line + chr(10))
        self.stream.flush()

    def close(self):
        self.stream.close()


class BatchProgress:
    """Tqdm in a TTY; sparse plain lines in redirected logs."""
    def __init__(self, logger, phase, epoch, total_batches, total_images):
        self.logger = logger
        self.phase = phase
        self.epoch = epoch
        self.total_batches = total_batches
        self.total_images = total_images
        self.started = time.perf_counter()
        self.last_line = self.started
        self.bar = None
        if sys.stderr.isatty():
            try:
                from tqdm import tqdm
            except ImportError:
                self.logger.line("tqdm unavailable; using sparse text progress")
            else:
                self.bar = tqdm(total=total_batches, desc=f"epoch {epoch} {phase}",
                                mininterval=2.0, dynamic_ncols=True, leave=False,
                                file=sys.stderr, unit="batch")

    def update(self, batch_done, images_done, running_loss):
        now = time.perf_counter()
        elapsed = max(now - self.started, 1e-9)
        rate = images_done / elapsed
        eta = (self.total_images - images_done) / rate if rate > 0 else 0
        if self.bar is not None:
            self.bar.set_postfix({"loss": f"{running_loss:.4f}",
                                  "img/s": f"{rate:.2f}",
                                  "ETA": f"{eta:.0f}s"}, refresh=False)
            self.bar.update(1)
        if now - self.last_line >= 60 or batch_done == self.total_batches:
            self.logger.line(f"epoch={self.epoch} phase={self.phase} "
                             f"batch={batch_done}/{self.total_batches} "
                             f"images={images_done}/{self.total_images} "
                             f"loss={running_loss:.6f} throughput={rate:.2f} img/s "
                             f"eta={eta:.0f}s", console=self.bar is None)
            self.last_line = now

    def close(self):
        if self.bar is not None:
            self.bar.close()


def epoch_lines(row, best, improved, learning_rates):
    train, val, resource = row["train"], row["validation"], row["resource"]
    qwk = val["quadratic_weighted_kappa"]
    vram = resource["gpu_peak_reserved_bytes"]
    vram_text = f"{vram / 1073741824:.2f}" if vram is not None else "NA"
    backbone_lr = learning_rates.get("backbone")
    backbone_lr_text = f"{backbone_lr:.8g}" if backbone_lr is not None else "NA"
    rss = resource.get("process_peak_rss_bytes")
    rss_text = f"{rss / 1073741824:.2f}" if rss is not None else "NA"
    lines = [
        f"epoch {row['epoch']} train loss={train['loss']:.6f} "
        f"accuracy={train['accuracy']:.6f} macro-F1={train['macro_f1']:.6f}",
        f"epoch {row['epoch']} val loss={val['validation_loss']:.6f} "
        f"accuracy={val['accuracy']:.6f} macro-F1={val['macro_f1']:.6f} "
        f"weighted-F1={val['weighted_f1']:.6f} QWK={qwk if qwk is not None else 'NA'} "
        f"grade-MAE={val['grade_mae']:.6f}",
        f"epoch {row['epoch']} lr backbone={backbone_lr_text} "
        f"head={learning_rates['head']:.8g} train_s={resource['train_seconds']:.1f} "
        f"val_s={resource['validation_seconds']:.1f} "
        f"train_img/s={resource['train_images_per_second']:.2f} "
        f"val_img/s={resource.get('validation_images_per_second', 0):.2f} "
        f"peak_reserved_GiB={vram_text} "
        f"train_input_s={resource.get('train_input_seconds', 0):.1f} "
        f"val_input_s={resource.get('validation_input_seconds', 0):.1f} "
        f"cpu_s={resource.get('process_cpu_seconds', 0):.1f} "
        f"peak_rss_GiB={rss_text}",
        f"epoch {row['epoch']} best={'UPDATED' if improved else 'retained'} "
        f"best_epoch={best['epoch']}"
    ]
    lines.append("grade  precision  recall  F1  support")
    for grade in val["per_grade"]:
        lines.append(f"{grade['grade']}  {grade['precision']:.6f}  "
                     f"{grade['recall']:.6f}  {grade['f1']:.6f}  {grade['support']}")
    return lines


def write_history_csv(path, history):
    path = Path(path)
    fields = ["epoch", "train_loss", "train_accuracy", "train_macro_f1",
              "validation_loss", "validation_accuracy", "validation_macro_f1",
              "validation_weighted_f1", "validation_qwk", "validation_grade_mae",
              "train_seconds", "validation_seconds", "train_images_per_second",
              "train_input_seconds", "validation_input_seconds",
              "process_cpu_seconds", "process_peak_rss_bytes", "validation_images_per_second",
              "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
              "backbone_lr", "head_lr", "best_epoch", "best_updated",
              "train_confusion_matrix_json", "validation_confusion_matrix_json",
              "train_per_grade_json", "validation_per_grade_json",
              "train_all_json", "validation_all_json", "resource_all_json"]
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in history:
                train, val, resource = row["train"], row["validation"], row["resource"]
                rates = row.get("learning_rates", {})
                writer.writerow({
                    "epoch": row["epoch"], "train_loss": train.get("loss"),
                    "train_accuracy": train.get("accuracy"), "train_macro_f1": train.get("macro_f1"),
                    "validation_loss": val.get("validation_loss"),
                    "validation_accuracy": val.get("accuracy"),
                    "validation_macro_f1": val.get("macro_f1"),
                    "validation_weighted_f1": val.get("weighted_f1"),
                    "validation_qwk": val.get("quadratic_weighted_kappa"),
                    "validation_grade_mae": val.get("grade_mae"),
                    "train_seconds": resource.get("train_seconds"),
                    "validation_seconds": resource.get("validation_seconds"),
                    "train_images_per_second": resource.get("train_images_per_second"),
                    "train_input_seconds": resource.get("train_input_seconds"),
                    "validation_input_seconds": resource.get("validation_input_seconds"),
                    "process_cpu_seconds": resource.get("process_cpu_seconds"),
                    "process_peak_rss_bytes": resource.get("process_peak_rss_bytes"),
                    "validation_images_per_second": resource.get("validation_images_per_second"),
                    "gpu_peak_allocated_bytes": resource.get("gpu_peak_allocated_bytes"),
                    "gpu_peak_reserved_bytes": resource.get("gpu_peak_reserved_bytes"),
                    "backbone_lr": rates.get("backbone"), "head_lr": rates.get("head"),
                    "best_epoch": row.get("best_epoch"), "best_updated": row.get("best_updated"),
                    "train_confusion_matrix_json": json.dumps(train.get("confusion_matrix")),
                    "validation_confusion_matrix_json": json.dumps(val.get("confusion_matrix")),
                    "train_per_grade_json": json.dumps(train.get("per_grade")),
                    "validation_per_grade_json": json.dumps(val.get("per_grade")),
                    "train_all_json": json.dumps(train, sort_keys=True, allow_nan=False),
                    "validation_all_json": json.dumps(val, sort_keys=True, allow_nan=False),
                    "resource_all_json": json.dumps(resource, sort_keys=True, allow_nan=False)})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def learning_curves(path, history):
    """Standalone, dependency-free SVG from stored epoch summaries."""
    panels = (("loss", "Loss", "loss", "validation_loss"),
              ("accuracy", "Accuracy", "accuracy", "accuracy"),
              ("macro_f1", "Macro-F1", "macro_f1", "macro_f1"))
    width, height, panel_width = 1110, 340, 350
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="22" font-size="16" font-family="sans-serif">'
             'Foundation adaptation learning curves</text>']
    epochs = [int(row["epoch"]) for row in history]
    for panel_index, (_key, title, train_key, val_key) in enumerate(panels):
        x0 = 40 + panel_index * 365
        y0, plot_w, plot_h = 50, 290, 225
        series = []
        for section, key, color in (("train", train_key, "#2563eb"),
                                    ("validation", val_key, "#dc2626")):
            points = [(int(row["epoch"]), float(row[section][key])) for row in history
                      if row[section].get(key) is not None]
            if points:
                series.append((section, points, color))
        all_values = [v for _, points, _ in series for _, v in points]
        if not all_values:
            continue
        lo = min(0.0, min(all_values))
        hi = max(all_values)
        if hi <= lo:
            hi = lo + 1
        max_epoch = max(epochs)
        min_epoch = min(epochs)
        span = max(max_epoch - min_epoch, 1)
        parts.extend([
            f'<text x="{x0}" y="40" font-size="14" font-family="sans-serif">{escape(title)}</text>',
            f'<path d="M{x0},{y0} V{y0+plot_h} H{x0+plot_w}" fill="none" stroke="#444"/>',
            f'<text x="{x0}" y="{y0+plot_h+20}" font-size="10">epoch {min_epoch}</text>',
            f'<text x="{x0+plot_w-45}" y="{y0+plot_h+20}" font-size="10">{max_epoch}</text>',
            f'<text x="{x0}" y="{y0-7}" font-size="10">{hi:.3f}</text>',
            f'<text x="{x0}" y="{y0+plot_h-4}" font-size="10">{lo:.3f}</text>'
        ])
        for name, points, color in series:
            coords = " ".join(f"{x0+(epoch-min_epoch)/span*plot_w:.2f},"
                              f"{y0+plot_h-(value-lo)/(hi-lo)*plot_h:.2f}"
                              for epoch, value in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                         'stroke-width="2"/>')
            legend_y = 300 if name == "train" else 318
            parts.append(f'<rect x="{x0}" y="{legend_y-9}" width="12" height="3" fill="{color}"/>'
                         f'<text x="{x0+18}" y="{legend_y}" font-size="11" '
                         f'font-family="sans-serif">{name}</text>')
    parts.append("</svg>")
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(chr(10).join(parts) + chr(10))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
