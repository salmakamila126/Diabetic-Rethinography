"""Read-only CAD-ViT paper references and artifact visualizations.

Paper matrices were transcribed from Figures 4--5 (IEEE Access 2025,
pp. 91510--91511). Rows in those figures are predictions; columns are truth.
Nothing in this module trains a model or runs final-test inference.
"""

import csv
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np

from reproduction.cadvit.evaluation import classification_metrics, file_sha256
from reproduction.cadvit.experiments import (PAPER_COMPARISONS, VARIANTS,
                                            binary_metrics, campaign_root,
                                            load_protocol, support_weighted_metrics)


PAPER_BINARY_PRED_TRUE = ((183, 0), (3, 181))  # predicted DR,No DR; true DR,No DR
PAPER_FIVE_PRED_TRUE = {
    "APTOS2019": ((184, 2, 2, 0, 0), (3, 22, 5, 0, 0), (0, 9, 86, 1, 7),
                  (0, 1, 2, 9, 1), (1, 0, 3, 5, 24)),
    "DDR": ((592, 14, 51, 0, 1), (9, 26, 7, 0, 0), (43, 14, 364, 16, 26),
            (0, 0, 1, 7, 1), (1, 0, 6, 0, 74)),
}
PAPER_TABLE_1 = ((2, 88.4, 7.9543, 1.9820), (4, 88.1, 7.5847, 1.9753),
                 (8, 88.0, 7.3999, 1.9720), (16, 88.6, 7.2629, 1.9703),
                 (32, 87.9, 7.2460, 1.9696))
PAPER_TABLE_3 = (("baseline", 83.9, 92.6), ("MFB", 86.2, 95.1),
                 ("Attention", 87.2, 96.4), ("MFB+Attention", 88.6, 96.7))
PAPER_FIGURE_6 = {
    "APTOS2019": {"imbalanced": (85.2, 85.7, 85.3, 85.3),
                  "balanced": (88.6, 88.4, 88.6, 88.4)},
    "DDR": {"imbalanced": (81.6, 82.0, 81.6, 81.5),
            "balanced": (84.8, 84.8, 84.8, 84.4)},
}  # accuracy, precision, recall, F1; PAPER_REFERENCE only


def paper_binary_reference():
    # Paper order DR,No DR -> evaluator order No DR,DR, true rows.
    cm = [[PAPER_BINARY_PRED_TRUE[1][1], PAPER_BINARY_PRED_TRUE[0][1]],
          [PAPER_BINARY_PRED_TRUE[1][0], PAPER_BINARY_PRED_TRUE[0][0]]]
    truth = [actual for actual in (0, 1) for pred in (0, 1)
             for _ in range(cm[actual][pred])]
    predicted = [pred for actual in (0, 1) for pred in (0, 1)
                 for _ in range(cm[actual][pred])]
    return {"source": "PAPER_REFERENCE Figure 4, p. 91510",
            "class_names": ["No DR", "DR"],
            "metrics": binary_metrics(truth, predicted)}


def paper_five_reference(dataset):
    if dataset not in PAPER_FIVE_PRED_TRUE:
        raise ValueError("Unknown paper dataset")
    original = PAPER_FIVE_PRED_TRUE[dataset]
    cm = [[original[pred][actual] for pred in range(5)] for actual in range(5)]
    truth = [actual for actual in range(5) for pred in range(5)
             for _ in range(cm[actual][pred])]
    predicted = [pred for actual in range(5) for pred in range(5)
                 for _ in range(cm[actual][pred])]
    metrics = classification_metrics(truth, predicted)
    metrics.update(support_weighted_metrics(metrics))
    metrics["f1_micro"] = metrics["accuracy"]  # single-label multiclass identity
    return {"source": f"PAPER_REFERENCE Figure 5, p. 91511 ({dataset})",
            "class_names": [str(i) for i in range(5)], "metrics": metrics}


def artifact_root_from_env():
    value = os.environ.get("CAD_VIT_ARTIFACT_ROOT", "").strip()
    return Path(value).expanduser().resolve() if value else None


def load_controlled_result(artifact_root, campaign, variant_name):
    """Read a completed run's JSON and config, never checkpoint/image bytes."""
    if artifact_root is None or variant_name not in VARIANTS:
        return {"status": "PENDING", "reason": "Artifact root or variant unavailable"}
    run_dir = campaign_root(artifact_root, campaign) / variant_name
    config_path, result_path = run_dir / "config.json", run_dir / "test_metrics.json"
    if not config_path.is_file() or not result_path.is_file():
        return {"status": "PENDING", "reason": "Config or frozen final-test JSON unavailable",
                "run_dir": str(run_dir)}
    try:
        _, protocol_sha = load_protocol(artifact_root, campaign)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (config.get("protocol_sha256") != protocol_sha
                or result.get("protocol_sha256") != protocol_sha
                or config.get("manifest_sha256") != result.get("manifest_sha256")
                or config.get("model_sha256") != result.get("model_sha256")
                or result.get("dataset") != VARIANTS[variant_name].dataset
                or result.get("variant") != variant_name or result.get("split") != "test"
                or config.get("variant", {}).get("name") != variant_name):
            raise ValueError("Run metadata or protocol fingerprint mismatch")
        if campaign == "balanced_v1" and (config.get("campaign") != campaign
                                          or result.get("campaign") != campaign):
            raise ValueError("Balanced campaign provenance mismatch")
    except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        return {"status": "UNRESOLVED", "reason": str(exc), "run_dir": str(run_dir)}
    return {"status": "AVAILABLE", "source": "BALANCED_V1" if campaign == "balanced_v1"
            else "IMBALANCED_CONTROL", "config": config, "result": result,
            "run_dir": str(run_dir), "result_sha256": file_sha256(result_path)}


def comparable_balancing_metrics(imbalanced, balanced):
    """Refuse Figure 6 candidate bars when anything besides balancing differs."""
    if imbalanced.get("status") != "AVAILABLE" or balanced.get("status") != "AVAILABLE":
        raise ValueError("Both final-test campaign results must be available")
    old, new = imbalanced["config"], balanced["config"]
    if (old.get("variant") != new.get("variant")
            or old.get("manifest_sha256") != new.get("manifest_sha256")
            or old.get("model_sha256") != new.get("model_sha256")
            or old.get("preprocessing_sha256") != new.get("preprocessing_sha256")):
        raise ValueError("Model, variant, manifest or preprocessing differs")
    old_protocol, new_protocol = dict(old["protocol"]), dict(new["protocol"])
    old_balance = old_protocol.pop("balancing")
    new_balance = new_protocol.pop("balancing")
    if old_balance != "none" or new_balance != "train_oversample_uniform" or old_protocol != new_protocol:
        raise ValueError("Figure 6 comparison must isolate balancing only")
    if (imbalanced["result"].get("split") != "test"
            or balanced["result"].get("split") != "test"):
        raise ValueError("Both results must be frozen final tests")

    def four(item):
        metrics = item["result"]["metrics"]
        weighted = support_weighted_metrics(metrics)
        return tuple(100 * value for value in
                     (metrics["accuracy"], weighted["precision_weighted"],
                      weighted["recall_weighted"], weighted["f1_weighted"]))

    return {"imbalanced": four(imbalanced), "balanced": four(balanced),
            "source": "CONTROLLED_FINAL_TEST; support-weighted metrics",
            "variant": old["variant"]["name"], "manifest_sha256": old["manifest_sha256"]}


def read_history(path):
    path = Path(path)
    if not path.is_file():
        return {"status": "PENDING", "reason": "Epoch history unavailable"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row.get("epoch") for row in rows] != list(range(1, len(rows) + 1)):
        return {"status": "UNRESOLVED", "reason": "Epoch sequence is incomplete/nonsequential"}
    return {"status": "AVAILABLE", "rows": rows, "sha256": file_sha256(path)}


def plot_confusion(metrics, class_names, title):
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    if cm.shape != (len(class_names), len(class_names)) or (cm < 0).any():
        raise ValueError("Confusion matrix shape/count invalid")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    image = ax.imshow(cm, cmap="Blues")
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    color="white" if cm[row, col] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(image, ax=ax, label="Images")
    fig.tight_layout()
    return fig


def confusion_png(metrics, class_names, title):
    """Portable PIL heatmap for stored notebook output, with explicit orientation."""
    from PIL import Image, ImageDraw, ImageFont

    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    n = len(class_names)
    if cm.shape != (n, n) or (cm < 0).any():
        raise ValueError("Confusion matrix shape/count invalid")
    cell = 78 if n == 5 else 115
    width, height = 180 + n * cell, 145 + n * cell
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        small = font
    x0, y0 = 95, 60
    maximum = max(1, int(cm.max()))
    draw.text((10, 10), title, fill="black", font=font)
    for row in range(n):
        draw.text((65, y0 + row * cell + cell // 2 - 8), str(class_names[row]),
                  fill="black", font=font)
        for col in range(n):
            value = int(cm[row, col])
            strength = value / maximum
            color = (int(239 - 215 * strength), int(245 - 166 * strength),
                     int(251 - 102 * strength))
            left, top = x0 + col * cell, y0 + row * cell
            draw.rectangle((left, top, left + cell - 1, top + cell - 1),
                           fill=color, outline=(220, 226, 235), width=1)
            text = str(value)
            bounds = draw.textbbox((0, 0), text, font=font)
            draw.text((left + (cell - (bounds[2] - bounds[0])) // 2,
                       top + (cell - (bounds[3] - bounds[1])) // 2 - 2),
                      text, fill="white" if strength > 0.55 else "black", font=font)
    for col, label in enumerate(class_names):
        draw.text((x0 + col * cell + cell // 2 - 8, y0 + n * cell + 7),
                  str(label), fill="black", font=font)
    draw.text((x0, y0 + n * cell + 40), "Predicted class →", fill="black", font=small)
    draw.text((7, y0 - 25), "True class ↓", fill="black", font=small)
    legend_x = x0 + n * cell + 18
    for step in range(100):
        strength = 1 - step / 99
        color = (int(239 - 215 * strength), int(245 - 166 * strength),
                 int(251 - 102 * strength))
        top = y0 + int(step * n * cell / 100)
        draw.rectangle((legend_x, top, legend_x + 18,
                        y0 + int((step + 1) * n * cell / 100)), fill=color)
    draw.text((legend_x + 23, y0), str(maximum), fill="black", font=small)
    draw.text((legend_x + 23, y0 + n * cell - 17), "0", fill="black", font=small)
    draw.text((legend_x - 3, y0 + n * cell + 7), "Images", fill="black", font=small)
    return image


def balance_metrics_png(imbalanced, balanced, title):
    """PIL bar chart for four verified percent metrics in Figure 6 order."""
    from PIL import Image, ImageDraw, ImageFont

    if len(imbalanced) != 4 or len(balanced) != 4:
        raise ValueError("Expected accuracy, precision, recall, F1")
    values = list(imbalanced) + list(balanced)
    if any(not 0 <= float(value) <= 100 for value in values):
        raise ValueError("Metrics must be percentages in [0,100]")
    image = Image.new("RGB", (850, 470), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((22, 15), title, fill="black", font=font)
    draw.rectangle((625, 70, 647, 89), fill=(76, 163, 152))
    draw.text((655, 70), "Imbalanced", fill="black", font=font)
    draw.rectangle((625, 100, 647, 119), fill=(97, 119, 193))
    draw.text((655, 100), "Balanced", fill="black", font=font)
    y0, scale = 380, 2.9
    draw.line((55, y0, 600, y0), fill="black", width=2)
    for index, label in enumerate(("Accuracy", "Precision", "Recall", "F1")):
        x = 95 + index * 130
        for offset, value, color in ((0, imbalanced[index], (76, 163, 152)),
                                     (47, balanced[index], (97, 119, 193))):
            top = y0 - int(float(value) * scale)
            draw.rectangle((x + offset, top, x + offset + 35, y0), fill=color)
            draw.text((x + offset - 4, top - 23), f"{value:.1f}", fill="black", font=font)
        draw.text((x - 4, y0 + 10), label, fill="black", font=font)
    draw.text((22, 430), "Percent; source and protocol named in title", fill="black", font=font)
    return image


def plot_training(rows, title):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError("Training history is nonsequential")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    for ax, field, label in zip(axes.flat,
                                ("train_loss", "val_loss", "val_accuracy", "val_macro_f1"),
                                ("Train loss", "Validation loss", "Validation accuracy",
                                 "Validation macro-F1")):
        ax.plot(epochs, [row[field] for row in rows], marker=".", linewidth=1)
        ax.set(xlabel="Epoch", ylabel=label, xlim=(1, max(1, len(rows))))
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_balance(before, after, title):
    import matplotlib.pyplot as plt

    labels = sorted(before, key=int)
    if labels != sorted(after, key=int):
        raise ValueError("Before/after classes differ")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(x - 0.2, [before[k] for k in labels], 0.4, label="Unique train records")
    ax.bar(x + 0.2, [after[k] for k in labels], 0.4, label="Train draws per epoch")
    ax.set(xticks=x, xticklabels=labels, xlabel="Grade", ylabel="Count", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def guarded_non_test_image(image_path, dataset_root, manifest_path, source_split):
    """Require a matching train/val manifest row before reading a real image."""
    if source_split not in {"train", "val"}:
        raise ValueError("Visual readiness permits train/validation images only")
    root = Path(dataset_root).expanduser().resolve()
    path = Path(image_path).expanduser().resolve()
    if root not in path.parents:
        raise ValueError("Image is outside dataset root")
    relative = path.relative_to(root).as_posix()
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as stream:
        matching = [row for row in csv.DictReader(stream) if row["image_path"] == relative]
    if len(matching) != 1 or matching[0]["split"] != source_split:
        raise ValueError("Image is not uniquely verified in the requested non-test split")
    if matching[0].get("sha256") and file_sha256(path) != matching[0]["sha256"]:
        raise ValueError("Image bytes differ from manifest SHA-256")
    return path, matching[0]


def preprocessing_stages(image):
    """Visualization only; final V1 output is checked pixel-for-pixel."""
    import cv2
    from PIL import ImageOps
    from reproduction.cadvit.preprocessing import (PREPROCESS_CONFIG, preprocess_fundus,
                                                    preprocess_fundus_v2)

    original = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    points = cv2.findNonZero((gray > PREPROCESS_CONFIG["crop_threshold"]).astype(np.uint8))
    if points is None:
        raise ValueError("All-dark image has no retinal crop")
    x, y, w, h = cv2.boundingRect(points)
    crop = original[y:y+h, x:x+w]
    resize = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(resize, (5, 5), sigmaX=1.0, sigmaY=1.0)
    v1 = preprocess_fundus(image)
    if not np.array_equal(blur, v1):
        raise AssertionError("Visualization stages differ from unchanged V1")
    return {"original": original, "crop": crop, "resize": resize, "gaussian_blur": blur,
            "historical_v1": v1, "historical_v2_bg10": preprocess_fundus_v2(image),
            "crop_xywh": (x, y, w, h)}


def plot_preprocessing(stages):
    import matplotlib.pyplot as plt

    names = ("original", "crop", "resize", "gaussian_blur", "historical_v1",
             "historical_v2_bg10")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    for ax, name in zip(axes.flat, names):
        ax.imshow(stages[name])
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Train/validation image only — V2 BG10 is an experimental extension")
    fig.tight_layout()
    return fig


def error_records(payload, *, binary_projection=False):
    """Classify stored predictions only; no model inference or image read."""
    if payload.get("split") != "test":
        raise ValueError("Expected a frozen final-test result")
    task = VARIANTS[payload["variant"]].task
    predictions = payload["metrics"].get("predictions")
    if predictions is None:
        return {"status": "PENDING", "reason": "Per-image predictions unavailable"}
    errors = []
    for row in predictions:
        if task == "binary_head" or binary_projection:
            actual, predicted = int(row["true_label"] > 0), int(row["predicted_label"] > 0)
            if actual == predicted:
                continue
            kind = "FP" if actual == 0 else "FN"
        else:
            if row["true_label"] == row["predicted_label"]:
                continue
            kind = "grade_error"
        errors.append({**row, "error_type": kind})
    return {"status": "AVAILABLE", "errors": errors,
            "task": "five_class_binary_projection" if task == "five_class" and binary_projection else task}


def plot_error_gallery(errors, dataset_root, *, allow_test_images=False, max_samples=6):
    """Explicit post-evaluation gallery; never called automatically by notebooks."""
    if not allow_test_images:
        raise ValueError("Explicit allow_test_images required after final-test protocol is locked")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    selected = errors[:max_samples]
    if not selected:
        raise ValueError("No error records selected")
    root = Path(dataset_root).expanduser().resolve()
    fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3.5), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, selected):
        relative = PurePosixPath(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe image path")
        path = (root / relative).resolve()
        if root not in path.parents or file_sha256(path) != row["image_sha256"]:
            raise ValueError("Image path or content differs from frozen prediction metadata")
        with Image.open(path) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(f'{row["error_type"]}: {row["true_label"]}→{row["predicted_label"]}\n'
                     f'confidence {row["confidence"]:.2f}')
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_campaign_figure(fig, artifact_root, campaign, variant_name, stem):
    """Optional user action: save a verified figure outside the Git repository."""
    if not stem.replace("_", "").isalnum() or variant_name not in VARIANTS:
        raise ValueError("Unsafe figure name or variant")
    directory = campaign_root(artifact_root, campaign) / variant_name / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stem}.png"
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    return target
