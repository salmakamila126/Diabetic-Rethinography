"""Read-only fundus preprocessing comparison and local audit artifacts.

The paper_described view operationalizes CAD-ViT Section III.A with the
historical V1 assumptions. It is deliberately pixel-identical to historical_v1.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


VARIANTS = ("original", "paper_described", "historical_v1", "historical_v2_bg10")
CHANNELS = ("R", "G", "B")
LUMA_WEIGHTS = np.array((0.299, 0.587, 0.114), dtype=np.float64)


def validate_source(image_path, source_split, train_manifest=None, dataset_root=None):
    """Accept only a declared train image or an independent audit image.

    A train declaration must be checked against an external manifest. For an
    independent audit image, the caller must establish that it is not test data.
    """
    path = Path(image_path).expanduser().resolve()
    if source_split not in {"train", "audit"}:
        raise ValueError("source_split must be 'train' or 'audit'; test/validation is forbidden")
    if not path.is_file():
        raise FileNotFoundError(path)
    if source_split == "train" and (train_manifest is None or dataset_root is None):
        raise ValueError("train images require train_manifest and dataset_root")
    if (train_manifest is None) != (dataset_root is None):
        raise ValueError("train_manifest and dataset_root must be supplied together")
    if train_manifest is not None:
        root = Path(dataset_root).expanduser().resolve()
        if source_split == "train" and root not in path.parents:
            raise ValueError("selected train image is outside dataset_root")
        with Path(train_manifest).expanduser().open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"image_path", "split"}.issubset(reader.fieldnames or ()):
                raise ValueError("manifest requires image_path and split columns")
            matching = []
            for row in reader:
                raw = Path(row["image_path"])
                if not raw.is_absolute() and ".." in raw.parts:
                    raise ValueError("manifest image_path must not traverse outside dataset_root")
                resolved = (raw if raw.is_absolute() else root / raw).resolve()
                if root not in resolved.parents:
                    raise ValueError("manifest image_path is outside dataset_root")
                if resolved == path:
                    matching.append(row["split"].strip().lower())
        if matching and set(matching) != {"train"}:
            raise ValueError("selected image appears outside train split")
        if source_split == "train" and not matching:
            raise ValueError("selected image is absent from train manifest")
    return path


def image_metrics(images):
    """Compute descriptive metrics without assuming pixel alignment after crop.

    Brightness = mean(0.299 R + 0.587 G + 0.114 B).
    Contrast = population standard deviation of that luminance.
    Mean-RGB shift = Euclidean distance between mean RGB vectors, in 0-255 units.
    Pixel MAE versus V1 is reported only for the aligned processed views.
    """
    original_mean = np.asarray(images["original"], dtype=np.float64).mean(axis=(0, 1))
    v1 = np.asarray(images["historical_v1"], dtype=np.float64)
    rows, histograms = [], {}
    for variant in VARIANTS:
        array = np.asarray(images[variant])
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f"{variant} must be RGB uint8")
        pixels = array.astype(np.float64)
        means = pixels.mean(axis=(0, 1))
        luminance = pixels @ LUMA_WEIGHTS
        mae = None if variant == "original" else float(np.abs(pixels - v1).mean())
        rows.append({
            "variant": variant,
            "height": array.shape[0],
            "width": array.shape[1],
            "brightness_luma_mean": float(luminance.mean()),
            "contrast_luma_std_population": float(luminance.std(ddof=0)),
            "mean_r": float(means[0]),
            "mean_g": float(means[1]),
            "mean_b": float(means[2]),
            "mean_rgb_distance_from_original": float(np.linalg.norm(means - original_mean)),
            "pixel_mae_from_v1_aligned": mae,
        })
        histograms[variant] = {
            channel: np.bincount(array[:, :, index].ravel(), minlength=256)
            for index, channel in enumerate(CHANNELS)
        }
    return rows, histograms


def audit_image(image_path, source_split, train_manifest=None, dataset_root=None,
                v1_fn=None, v2_fn=None):
    """Process one declared non-test image; never modify the input file."""
    path = validate_source(image_path, source_split, train_manifest, dataset_root)
    if v1_fn is None or v2_fn is None:
        from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2
        v1_fn = v1_fn or preprocess_fundus
        v2_fn = v2_fn or preprocess_fundus_v2
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            orientation = image.getexif().get(274, 1)
            original = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
            v1 = np.asarray(v1_fn(image)).copy()
            v2 = np.asarray(v2_fn(image)).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable or damaged image: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Preprocessing failed for {path.name}: {exc}") from exc
    for name, array in (("historical_v1", v1), ("historical_v2_bg10", v2)):
        if array.shape != (512, 512, 3) or array.dtype != np.uint8:
            raise ValueError(f"Unexpected {name} shape or dtype: {array.shape}, {array.dtype}")
    images = {
        "original": original,
        "paper_described": v1.copy(),
        "historical_v1": v1,
        "historical_v2_bg10": v2,
    }
    assert np.array_equal(images["paper_described"], images["historical_v1"])
    rows, histograms = image_metrics(images)
    return {
        "input_path": path,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_split": source_split,
        "exif_orientation": orientation,
        "paper_v1_identical": True,
        "images": images,
        "metrics": rows,
        "histograms": histograms,
    }


def _overview(images, destination):
    width, height = 1160, 1220
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 14), "One input, four documented views", fill="#15243a")
    for index, variant in enumerate(VARIANTS):
        x = 24 + (index % 2) * 570
        y = 52 + (index // 2) * 580
        draw.text((x, y), variant, fill="#15243a")
        picture = Image.fromarray(images[variant])
        picture = ImageOps.contain(picture, (530, 530))
        canvas.paste(picture, (x + (530 - picture.width) // 2, y + 26 + (530 - picture.height) // 2))
    canvas.save(destination)


def _histogram_figure(histograms, destination):
    canvas = Image.new("RGB", (1160, 920), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 12), "RGB histograms - counts per intensity bin", fill="#15243a")
    colors = {"R": "#bd3838", "G": "#28834b", "B": "#3068b0"}
    for index, variant in enumerate(VARIANTS):
        left = 34 + (index % 2) * 570
        top = 54 + (index // 2) * 440
        plot_left, plot_top = left + 48, top + 42
        plot_width, plot_height = 452, 315
        draw.text((left, top), variant, fill="#15243a")
        draw.line((plot_left, plot_top, plot_left, plot_top + plot_height,
                   plot_left + plot_width, plot_top + plot_height), fill="#333333", width=2)
        max_count = max(int(histograms[variant][channel].max()) for channel in CHANNELS) or 1
        for channel in CHANNELS:
            values = histograms[variant][channel]
            points = [
                (plot_left + round(bin_index * plot_width / 255),
                 plot_top + plot_height - round(int(count) * plot_height / max_count))
                for bin_index, count in enumerate(values)
            ]
            draw.line(points, fill=colors[channel], width=2)
        draw.text((plot_left, plot_top + plot_height + 8), "0", fill="#333333")
        draw.text((plot_left + plot_width // 2 - 10, plot_top + plot_height + 8), "128", fill="#333333")
        draw.text((plot_left + plot_width - 20, plot_top + plot_height + 8), "255", fill="#333333")
        for label_index, channel in enumerate(CHANNELS):
            draw.text((plot_left + plot_width - 75 + 25 * label_index, top),
                      channel, fill=colors[channel])
        draw.text((plot_left, plot_top - 18), f"panel max: {max_count}", fill="#333333")
    canvas.save(destination)


def save_audit(result, output_root):
    """Write only derived PNG and CSV files to an ignored output directory."""
    digest = result["input_sha256"][:12]
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_"
                        for char in result["input_path"].stem)[:60]
    root = Path(output_root).expanduser().resolve()
    input_parent = result["input_path"].parent
    if root == input_parent or input_parent in root.parents or root in input_parent.parents:
        raise ValueError("Audit output must not be written into the source image directory")
    destination = root / f"{safe_stem}_{digest}"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in VARIANTS:
        path = destination / f"{variant}.png"
        Image.fromarray(result["images"][variant]).save(path)
        paths[variant] = path
    for name, draw in (("comparison", _overview), ("rgb_histograms", _histogram_figure)):
        path = destination / f"{name}.png"
        draw(result["images"] if name == "comparison" else result["histograms"], path)
        paths[name] = path
    metrics_path = destination / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        provenance = {
            "input_sha256": result["input_sha256"],
            "source_split": result["source_split"],
            "exif_orientation": result["exif_orientation"],
            "paper_v1_identical": result["paper_v1_identical"],
        }
        writer = csv.DictWriter(stream, fieldnames=list(provenance) + list(result["metrics"][0]))
        writer.writeheader()
        writer.writerows({**provenance, **row} for row in result["metrics"])
    paths["metrics_csv"] = metrics_path
    histogram_path = destination / "rgb_histograms.csv"
    with histogram_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "channel", "intensity_0_255", "pixel_count"))
        for variant in VARIANTS:
            for channel in CHANNELS:
                for intensity, count in enumerate(result["histograms"][variant][channel]):
                    writer.writerow((variant, channel, intensity, int(count)))
    paths["histograms_csv"] = histogram_path
    return paths


def run_audit(image_path, source_split, output_root, train_manifest=None, dataset_root=None):
    """Run the historical methods on one image and save reproducible audit artifacts."""
    result = audit_image(image_path, source_split, train_manifest, dataset_root)
    return result, save_audit(result, output_root)
