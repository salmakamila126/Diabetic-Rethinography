"""Frozen B1 train/validation loader with explicit, read-only host path mapping.

The original manifest and JSON sidecar are never modified. When their recorded
dataset root belongs to Comm1, require that exact old root as an explicit CLI
argument, then verify every selected image under the new artifact root.
"""

import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase


def load_train_val(artifact_root, dataset, *, manifest_source_root=None, verify_images=True):
    if dataset not in phase.SPECS:
        raise ValueError("Unknown frozen dataset")
    root = Path(artifact_root).expanduser().resolve()
    local_root = (root / "datasets" / dataset).resolve()
    if not local_root.is_dir():
        raise FileNotFoundError(local_root)
    manifest = root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    phase.require_hash(manifest, phase.SPECS[dataset]["manifest_sha256"])
    sidecar = manifest.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (metadata.get("dataset") != dataset or metadata.get("seed") != 42
            or metadata.get("manifest_file_sha256") != phase.SPECS[dataset]["manifest_sha256"]):
        raise ValueError("Frozen manifest sidecar differs from CSV identity")
    recorded = metadata.get("root")
    if not isinstance(recorded, str):
        raise ValueError("Frozen manifest sidecar needs an absolute source dataset root")
    source = (PurePosixPath(recorded) if PurePosixPath(recorded).is_absolute()
              else PureWindowsPath(recorded))
    if not source.is_absolute():
        raise ValueError("Frozen manifest sidecar needs an absolute source dataset root")
    if source.name != dataset or source.parent.name != "datasets":
        raise ValueError("Manifest sidecar root is not this dataset directory")
    if manifest_source_root is not None:
        supplied = (PurePosixPath(manifest_source_root)
                    if PurePosixPath(manifest_source_root).is_absolute()
                    else PureWindowsPath(manifest_source_root))
        if supplied != source:
            raise ValueError("Explicit source-root mapping differs from frozen sidecar")
    native = Path(recorded).expanduser().resolve() == local_root
    if native:
        rows, info = fa.load_train_val_manifest(root, dataset, verify_images=verify_images)
        info["path_mapping"] = "native"
    else:
        if manifest_source_root is None:
            raise ValueError("Sidecar records another host: supply its exact --manifest-source-root")
        rows = {"train": [], "val": []}
        ids, paths, digest_split, group_split = set(), set(), {}, {}
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image_id", "image_path", "label", "split", "group_id", "sha256"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Frozen manifest columns incomplete")
            for raw in reader:
                split = raw["split"]
                if split == "test":
                    continue  # Never validate or open a test label/path/image.
                if split not in rows:
                    raise ValueError("Unknown frozen split")
                image_id, relative = raw["image_id"], PurePosixPath(raw["image_path"])
                digest, group = raw["sha256"], raw["group_id"]
                if (raw["label"] not in {str(i) for i in range(5)} or not image_id or not group
                        or not relative.parts or relative.is_absolute() or ".." in relative.parts
                        or "\\" in raw["image_path"] or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    raise ValueError("Invalid frozen train/validation row")
                if (image_id in ids or raw["image_path"] in paths
                        or (digest in digest_split and digest_split[digest] != split)
                        or (group in group_split and group_split[group] != split)):
                    raise ValueError("Duplicate row or cross-split group/content leakage")
                ids.add(image_id)
                paths.add(raw["image_path"])
                digest_split[digest], group_split[group] = split, split
                image_path = (local_root / Path(*relative.parts)).resolve()
                if local_root not in image_path.parents:
                    raise ValueError("Image path escapes mapped dataset root")
                if verify_images:
                    phase.require_hash(image_path, digest)
                rows[split].append({"image_id": image_id, "image_path": raw["image_path"],
                                    "path": str(image_path), "label": int(raw["label"]),
                                    "group_id": group, "sha256": digest, "split": split})
        if len(rows["val"]) != phase.SPECS[dataset]["val_count"] or not rows["train"]:
            raise ValueError("Frozen train/validation row counts differ")
        support = {split: [sum(row["label"] == grade for row in part)
                           for grade in range(5)] for split, part in rows.items()}
        if any(n == 0 for values in support.values() for n in values):
            raise ValueError("One split lacks a grade")
        info = {"dataset": dataset,
                "manifest_sha256": phase.SPECS[dataset]["manifest_sha256"],
                "dataset_root": str(local_root),
                "counts": {split: len(part) for split, part in rows.items()},
                "grade_support": support,
                "group_count": {split: len({r["group_id"] for r in part})
                                for split, part in rows.items()},
                "image_order_sha256": fa._digest_json([
                    (r["split"], r["image_id"], r["sha256"])
                    for split in ("train", "val") for r in rows[split]]),
                "images_verified": verify_images, "path_mapping": "explicit_source_to_local"}
    info["manifest_sidecar_sha256"] = phase.sha256(sidecar)
    info["manifest_source_root"] = str(source)
    info["mapped_dataset_root"] = str(local_root)
    return rows, info
