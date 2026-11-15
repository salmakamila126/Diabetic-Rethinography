"""Deterministic-per-image, training-only retinal augmentation for B1.

This module never touches validation/test images or model weights. RNG is keyed by
(seed, epoch, image ID), so a resumed epoch uses the same transformations.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from novelty import preof_phase1 as phase

PROTOCOL = {
    "name": "B1-retinal-conservative-v1",
    "order": ["horizontal_flip", "rotation", "brightness", "contrast"],
    "horizontal_flip_probability": 0.50,
    "rotation_probability": 0.50,
    "rotation_degrees": [-7.0, 7.0],
    "brightness_probability": 0.30,
    "brightness_factor": [0.90, 1.10],
    "contrast_probability": 0.30,
    "contrast_factor": [0.90, 1.10],
    "rotation_resample": "BICUBIC",
    "rotation_fill": [0, 0, 0],
    "fov_proxy_threshold_max_channel": 8,
    "minimum_fov_proxy_retention": 0.99,
    "rotation_attempts": 8,
}


def _rng(seed, epoch, image_id):
    if seed < 0 or epoch < 1 or not image_id:
        raise ValueError("Augmentation requires nonnegative seed, epoch and image ID")
    encoded = json.dumps([seed, epoch, image_id], separators=(",", ":")).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big"))


def apply(image, *, seed, epoch, image_id):
    """Return an RGB PIL image and recorded draws, with a conservative FOV proxy."""
    from PIL import Image, ImageEnhance

    if image.mode != "RGB":
        raise ValueError("Augmentation expects EXIF-corrected RGB image")
    rng = _rng(seed, epoch, image_id)
    output = image.copy()
    details = {"seed": seed, "epoch": epoch, "image_id": image_id,
               "horizontal_flip": False, "rotation_applied": False,
               "rotation_degrees": 0.0, "rotation_attempts": 0,
               "rotation_rejected_by_fov": False,
               "fov_proxy_retention": 1.0,
               "brightness_factor": 1.0, "contrast_factor": 1.0}
    if rng.random() < PROTOCOL["horizontal_flip_probability"]:
        output = output.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        details["horizontal_flip"] = True
    if rng.random() < PROTOCOL["rotation_probability"]:
        # This intensity mask is a proxy for visible retinal area, not a lesion mask.
        before = np.max(np.asarray(output), axis=2) > PROTOCOL["fov_proxy_threshold_max_channel"]
        visible = int(before.sum())
        if visible == 0:
            raise ValueError("No visible retinal area under the prespecified FOV proxy")
        mask = Image.fromarray((before * 255).astype(np.uint8), mode="L")
        for attempt in range(1, PROTOCOL["rotation_attempts"] + 1):
            angle = float(rng.uniform(*PROTOCOL["rotation_degrees"]))
            candidate_mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            retention = min(1.0, np.count_nonzero(np.asarray(candidate_mask)) / visible)
            details["rotation_attempts"] = attempt
            if retention >= PROTOCOL["minimum_fov_proxy_retention"]:
                output = output.rotate(angle, resample=Image.Resampling.BICUBIC,
                                       fillcolor=tuple(PROTOCOL["rotation_fill"]))
                details["rotation_applied"] = True
                details["rotation_degrees"] = angle
                details["fov_proxy_retention"] = float(retention)
                break
        else:
            details["rotation_rejected_by_fov"] = True
    if rng.random() < PROTOCOL["brightness_probability"]:
        factor = float(rng.uniform(*PROTOCOL["brightness_factor"]))
        output = ImageEnhance.Brightness(output).enhance(factor)
        details["brightness_factor"] = factor
    if rng.random() < PROTOCOL["contrast_probability"]:
        factor = float(rng.uniform(*PROTOCOL["contrast_factor"]))
        output = ImageEnhance.Contrast(output).enhance(factor)
        details["contrast_factor"] = factor
    return output, details


def inspect_gallery(rows, processor, output_dir, *, seed, epoch, samples_per_grade=3):
    """Write an internal-only PNG plus metadata from train rows, outside Git."""
    from PIL import Image, ImageDraw, ImageOps

    if samples_per_grade < 1 or samples_per_grade > 5:
        raise ValueError("Use 1..5 training examples per grade")
    root = phase._outside_repository(output_dir)
    if root.exists():
        raise FileExistsError("Augmentation inspection output already exists")
    selected = []
    for grade in range(5):
        ordered = sorted((row for row in rows if row["label"] == grade),
                         key=lambda row: row["image_id"])
        distinct = {}
        for row in ordered:
            distinct.setdefault(row.get("group_id", row["image_id"]), row)
        candidates = list(distinct.values())
        if len(candidates) < samples_per_grade:
            raise ValueError(f"Grade {grade} lacks inspection samples")
        indices = np.linspace(0, len(candidates) - 1, samples_per_grade, dtype=int)
        selected.extend(candidates[int(index)] for index in indices)
    tile = 224
    canvas = Image.new("RGB", (tile * 2, (tile + 26) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    records = []
    started = time.perf_counter()
    for index, row in enumerate(selected):
        if row.get("split") != "train":
            raise ValueError("Augmentation gallery accepts train rows only")
        phase.require_hash(row["path"], row["sha256"])
        with Image.open(row["path"]) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        augmented, settings = apply(original, seed=seed, epoch=epoch, image_id=row["image_id"])
        # The displayed views follow the same official 224px processor as training.
        tensors = processor(images=[original, augmented], return_tensors="pt")["pixel_values"]
        if tuple(tensors.shape) != (2, 3, 224, 224):
            raise ValueError("Official processor output differs")
        # The gallery shows raw RGB before processor; processed tensor hashes are recorded.
        top = index * (tile + 26)
        canvas.paste(ImageOps.pad(original, (tile, tile), color="black"), (0, top + 26))
        canvas.paste(ImageOps.pad(augmented, (tile, tile), color="black"), (tile, top + 26))
        draw.text((4, top + 6), f"grade {row['label']} | {row['image_id']} | original", fill="black")
        draw.text((tile + 4, top + 6), "B1 train augmentation", fill="black")
        records.append({"grade": row["label"],
                        "image_id": row["image_id"], "split": "train",
                        "source_sha256": row["sha256"], "draws": settings,
                        "original_official_tensor_sha256": hashlib.sha256(
                            tensors[0].contiguous().numpy().tobytes()).hexdigest(),
                        "augmented_official_tensor_sha256": hashlib.sha256(
                            tensors[1].contiguous().numpy().tobytes()).hexdigest()})
    root.mkdir(parents=True)
    png = root / "augmentation_gallery.png"
    temporary = root / f"augmentation_gallery.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            canvas.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, png)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"schema": "foundation-augmentation-gallery-v1", "protocol": PROTOCOL,
              "seed": seed, "epoch": epoch, "selection": "evenly spaced distinct training groups ordered by image ID",
              "samples_per_grade": samples_per_grade, "rows": records,
              "image_sha256": phase.sha256(png), "elapsed_seconds": time.perf_counter() - started,
              "redistribution": "internal inspection only; keep outside Git"}
    return report
