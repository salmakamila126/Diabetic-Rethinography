"""E3 exploratory, validation-only enhancement sensitivity for frozen CAD-ViT.

Consumes verified Phase 1 prediction and preprocessing caches. No work on import;
the pilot command requires explicit trust for the four historical checkpoints.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from novelty import preof_e1 as e1
from novelty import preof_phase1 as phase


SCHEMA = "preof-e3-pilot-v1"
CONDITIONS = ("v1", "mix025", "mix050", "mix075", "v2_bg10",
              "bg5", "bg20", "clahe2", "clahe4", "gamma075", "gamma125")
MODEL_NATIVE = {"v1": "v1", "v2": "v2_bg10"}
ENDPOINT_KEY = {("v1", "v1"): "p11", ("v1", "v2_bg10"): "p12",
                ("v2", "v1"): "p21", ("v2", "v2_bg10"): "p22"}


def select_pilot(rows, dataset, groups_per_grade=8, seed=42):
    """Select whole groups per grade, deterministically; retain manifest order."""
    if groups_per_grade < 1:
        raise ValueError("groups_per_grade must be positive")
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_id"], set()).add(int(row["label"]))
    selected = set()
    for grade in range(5):
        candidates = [group for group, grades in by_group.items() if grade in grades]
        if not candidates:
            raise ValueError(f"Validation grade {grade} has no groups")
        candidates.sort(key=lambda group: hashlib.sha256(
            f"{seed}|{dataset}|{grade}|{group}".encode()).hexdigest())
        selected.update(candidates[:groups_per_grade])
    subset = [row for row in rows if row["group_id"] in selected]
    support = {str(grade): sum(int(row["label"]) == grade for row in subset)
               for grade in range(5)}
    return subset, {"algorithm": "SHA-256 rank per grade; union of whole groups; manifest row order",
                    "seed": seed, "groups_per_grade": groups_per_grade,
                    "selected_groups": sorted(selected), "grade_support": support,
                    "sample_count": len(subset)}


def bg_enhance(v1, sigma):
    import cv2
    from reproduction.cadvit.preprocessing import BG_V2_CONFIG

    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("BG sigma must be finite and positive")
    pixels = v1.astype(np.float32)
    background = cv2.GaussianBlur(pixels, (0, 0), sigmaX=sigma, sigmaY=sigma)
    enhanced = (BG_V2_CONFIG["alpha"] * pixels + BG_V2_CONFIG["beta"] * background
                + BG_V2_CONFIG["offset"])
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def condition_pixels(v1, v2, condition):
    """Produce only transient uint8 RGB pixels; exact endpoints reuse cache arrays."""
    import cv2

    if v1.shape != (512, 512, 3) or v2.shape != v1.shape or v1.dtype != np.uint8 or v2.dtype != np.uint8:
        raise ValueError("E3 requires historical 512x512 uint8 RGB cache entries")
    if condition == "v1":
        return v1
    if condition == "v2_bg10":
        return v2
    if condition in {"mix025", "mix050", "mix075"}:
        alpha = {"mix025": .25, "mix050": .5, "mix075": .75}[condition]
        # Interpolation is in RGB pixel space after both historical pipelines.
        mixed = (1 - alpha) * v1.astype(np.float32) + alpha * v2.astype(np.float32)
        return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    if condition in {"bg5", "bg20"}:
        return bg_enhance(v1, 5.0 if condition == "bg5" else 20.0)
    if condition in {"clahe2", "clahe4"}:
        lab = cv2.cvtColor(v1, cv2.COLOR_RGB2LAB)
        clip = 2.0 if condition == "clahe2" else 4.0
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if condition in {"gamma075", "gamma125"}:
        gamma = .75 if condition == "gamma075" else 1.25
        corrected = 255.0 * np.power(v1.astype(np.float32) / 255.0, gamma)
        return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown E3 condition: {condition}")


def _probabilities(logits):
    values = np.array(logits, dtype=np.float64, copy=True)
    values -= values.max()
    probabilities = np.exp(values)
    return (probabilities / probabilities.sum()).tolist()


def _read_rows(path, selected, predictions):
    if not path.exists():
        return []
    saved = []
    prediction_by_id = {item["image_id"]: item for item in predictions}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n") or not line.strip() or len(saved) >= len(selected):
                raise ValueError("Truncated or extra E3 pilot row")
            item = json.loads(line)
            source = selected[len(saved)]
            cached = prediction_by_id.get(source["image_id"])
            if cached is None:
                raise ValueError("Pilot image absent from prediction cache")
            expected = {"image_id": source["image_id"], "group_id": source["group_id"],
                        "true_grade": int(source["label"]), "image_sha256": source["sha256"]}
            if any(item.get(key) != value for key, value in expected.items()) or item.get("split") != "val":
                raise ValueError("E3 row differs from frozen manifest")
            if set(item.get("models", {})) != {"v1", "v2"}:
                raise ValueError("E3 model records incomplete")
            for model in ("v1", "v2"):
                if set(item["models"][model]) != set(CONDITIONS):
                    raise ValueError("E3 condition records incomplete")
                for condition, record in item["models"][model].items():
                    logits = np.asarray(record.get("logits"), dtype=float)
                    probs = np.asarray(record.get("probabilities"), dtype=float)
                    if (logits.shape != (5,) or probs.shape != (5,)
                            or not np.isfinite(logits).all() or not np.isfinite(probs).all()
                            or not np.allclose(probs, _probabilities(logits), atol=1e-6, rtol=1e-6)):
                        raise ValueError("E3 prediction is malformed")
                    endpoint = ENDPOINT_KEY.get((model, condition))
                    if endpoint and not np.allclose(logits, cached[endpoint]["logits"], atol=0, rtol=0):
                        raise ValueError("E3 endpoint differs from verified prediction cache")
            saved.append(item)
    return saved


def _metadata(args, selection, prediction_hashes, manifest_sha256):
    import cv2
    import torch
    from reproduction.cadvit.preprocessing import BG_V2_CONFIG, PREPROCESS_CONFIG

    return {"schema": SCHEMA, "dataset": args.dataset, "split": "val",
            "selection": selection, "conditions": list(CONDITIONS),
            "manifest_sha256": manifest_sha256, "prediction_cache": prediction_hashes,
            "preprocessing_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/preprocessing.py"),
            "implementation_sha256": phase.sha256(__file__),
            "historical_base": PREPROCESS_CONFIG, "historical_bg": BG_V2_CONFIG,
            "transform": {"interpolation": "float32 RGB blend of cached V1 and V2; np.rint ties-to-even then clip 0..255 and uint8; endpoints are cached arrays",
                          "bg": "V1 float32 RGB; GaussianBlur sigmaX=sigmaY in {5,10,20}; alpha=4 beta=-4 offset=128; clip then uint8 truncation; BG10 is historical cached V2",
                          "clahe": "cached V1 RGB uint8 -> OpenCV LAB -> CLAHE L tileGrid 8x8 clipLimit 2 or 4 -> RGB uint8",
                          "gamma": "cached V1 float32 RGB /255 -> power 0.75 or 1.25 -> *255 -> np.rint -> clip -> uint8",
                          "model_input": "uint8 RGB -> copy -> CHW float32 /255; no augmentation"},
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__,
                        "torch": torch.__version__, "device": args.device},
            "endpoint_logit_tolerance": {"atol": args.atol, "rtol": args.rtol}}


def pilot(args):
    import torch

    if not args.trust_checkpoints:
        raise ValueError("Explicit --trust-checkpoints required for historical checkpoint loading")
    if not (0 <= args.atol <= 1e-3 and 0 <= args.rtol <= 1e-3):
        raise ValueError("Endpoint logit tolerances must be within 0..1e-3")
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    prediction_root = Path(args.prediction_cache_dir).expanduser().resolve()
    preprocessing_root = Path(args.preprocess_cache_root).expanduser().resolve()
    output = phase._outside_repository(args.output_dir)
    raw_root = (artifact_root / "datasets").resolve()
    for protected in (prediction_root, preprocessing_root, raw_root):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("E3 output must be separate from datasets and existing caches")
    predictions, hashes = e1.load_verified_cache(args.dataset, artifact_root, prediction_root)
    dataset_root = (artifact_root / "datasets" / args.dataset).resolve()
    manifest = artifact_root / "cad_vit_runs" / "manifests" / f"{args.dataset}_seed42.csv"
    all_rows = phase.validation_rows(manifest, args.dataset, dataset_root)
    selected, selection = select_pilot(all_rows, args.dataset, args.groups_per_grade, args.seed)
    meta = _metadata(args, selection, hashes, phase.SPECS[args.dataset]["manifest_sha256"])
    if output.exists():
        if not args.resume:
            raise FileExistsError("E3 output exists; use --resume only for identical provenance")
        if json.loads((output / "metadata.json").read_text(encoding="utf-8")) != meta:
            raise ValueError("E3 output metadata differs; refuse resume")
    else:
        output.mkdir(parents=True)
        (output / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    saved = _read_rows(output / "rows.jsonl", selected, predictions)
    if len(saved) == len(selected):
        print(f"COMPLETE: {args.dataset} E3 pilot {len(saved)} images; no inference repeated")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    cache = phase.PreprocessingCache(preprocessing_root, dataset_root)
    models, provenance = phase.checked_models(args.dataset, artifact_root, args.v2_source, True)
    if (provenance["checkpoint_sha256"] != hashes["checkpoint_sha256"]
            or provenance["model_source_sha256"] != hashes["model_source_sha256"]
            or provenance["preprocessing_source_sha256"] != hashes["preprocessing_source_sha256"]
            or provenance["v2_historical_source"]["sha256"] != hashes["historical_v2_source_sha256"]):
        raise ValueError("Model/source provenance differs from prediction cache")
    for model in models.values():
        model.to(device).eval()
    lookup = {item["image_id"]: item for item in predictions}
    with (output / "rows.jsonl").open("ab") as stream, torch.inference_mode():
        for row in selected[len(saved):]:
            cached = lookup[row["image_id"]]
            v1 = _read_pixels(cache, row["sha256"], "v1")
            v2 = _read_pixels(cache, row["sha256"], "v2_bg10")
            if not np.array_equal(bg_enhance(v1, 10.0), v2):
                raise ValueError("Historical BG10 endpoint differs from V2 preprocessing cache")
            item = {"image_id": row["image_id"], "group_id": row["group_id"],
                    "true_grade": int(row["label"]), "image_sha256": row["sha256"],
                    "split": "val", "models": {}}
            for model_name, model in models.items():
                records = {}
                for condition in CONDITIONS:
                    key = ENDPOINT_KEY.get((model_name, condition))
                    if key:
                        records[condition] = cached[key]
                        if len(saved) == 0 and row == selected[0]:
                            pixels = condition_pixels(v1, v2, condition)
                            tensor = torch.from_numpy(np.array(pixels, copy=True)).permute(2, 0, 1).float()[None].to(device) / 255
                            logits = model(tensor)[0].detach().cpu().numpy()
                            if not np.allclose(logits, cached[key]["logits"], atol=args.atol, rtol=args.rtol):
                                raise ValueError("Historical endpoint logits differ from prediction cache")
                        continue
                    pixels = condition_pixels(v1, v2, condition)
                    tensor = torch.from_numpy(np.array(pixels, copy=True)).permute(2, 0, 1).float()[None].to(device) / 255
                    logits = model(tensor)[0].detach().cpu().numpy().astype(float)
                    if logits.shape != (5,) or not np.isfinite(logits).all():
                        raise ValueError("E3 frozen forward produced malformed logits")
                    records[condition] = {"logits": logits.tolist(),
                                          "probabilities": _probabilities(logits)}
                item["models"][model_name] = records
            stream.write((json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
            saved.append(item)
            print(f"{args.dataset} E3 pilot: {len(saved)}/{len(selected)}", flush=True)
    _read_rows(output / "rows.jsonl", selected, predictions)
    print(f"COMPLETE: {args.dataset} E3 pilot {len(selected)} validation images")


def _read_pixels(cache, image_sha256, variant):
    data, meta, key, payload = cache._paths(image_sha256, variant)
    pixels, status = cache._read(data, meta, key, payload)
    if status != "hit":
        raise ValueError(f"Phase 1 preprocessing cache {variant} is {status}; E3 will not repair it")
    return pixels


def _paired_intervals(truth, groups, native, condition, repeats, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2 or repeats < 30:
        raise ValueError("Need at least two groups and 30 paired bootstrap draws")
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)

    def delta(index):
        y = truth[index]
        return [float(np.mean(condition[index] == y) - np.mean(native[index] == y)),
                float(np.mean(np.abs(condition[index] - y)) - np.mean(np.abs(native[index] - y)))]

    point = delta(np.arange(len(truth)))
    draws = np.asarray([delta(np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))]))
                        for _ in range(repeats)])
    return {name: {"point": point[i], "ci95_percentile": np.quantile(draws[:, i], [.025, .975]).tolist()}
            for i, name in enumerate(("accuracy_delta", "grade_mae_delta"))}


def analyze(args):
    if not 0 < args.high_confidence <= 1:
        raise ValueError("High-confidence threshold must be in (0,1]")
    output = phase._outside_repository(args.output_dir)
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("split") != "val":
        raise ValueError("Invalid E3 pilot metadata")
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    predictions, hashes = e1.load_verified_cache(meta["dataset"], artifact_root, args.prediction_cache_dir)
    if hashes != meta["prediction_cache"]:
        raise ValueError("Phase 1 prediction cache changed")
    dataset_root = (artifact_root / "datasets" / meta["dataset"]).resolve()
    manifest = artifact_root / "cad_vit_runs" / "manifests" / f"{meta['dataset']}_seed42.csv"
    all_rows = phase.validation_rows(manifest, meta["dataset"], dataset_root)
    selection = meta["selection"]
    selected, expected_selection = select_pilot(all_rows, meta["dataset"],
                                                selection["groups_per_grade"], selection["seed"])
    if selection != expected_selection:
        raise ValueError("E3 pilot selection differs from frozen manifest")
    saved = _read_rows(output / "rows.jsonl", selected, predictions)
    if len(saved) != len(selected):
        raise ValueError("E3 pilot is incomplete")
    truth = np.asarray([item["true_grade"] for item in saved], dtype=int)
    groups = np.asarray([item["group_id"] for item in saved])
    results = {"schema": "preof-e3-analysis-v1", "status": "EXPLORATORY_VALIDATION_PILOT",
               "dataset": meta["dataset"], "split": "val", "selection": selection,
               "provenance": {"pilot_metadata_sha256": phase.sha256(output / "metadata.json"),
                              "pilot_rows_sha256": phase.sha256(output / "rows.jsonl"),
                              "prediction_cache": hashes},
               "definitions": {"within_model": "Each condition changes pixels for one fixed checkpoint; no retraining.",
                               "between_model": "V1 and V2 differ in training protocol; cross-model contrasts cannot isolate preprocessing.",
                               "uncertainty": "Paired group bootstrap percentile intervals on selected validation groups; exploratory pilot only."},
               "models": {}}
    for model in ("v1", "v2"):
        probabilities = {condition: np.asarray([item["models"][model][condition]["probabilities"]
                                                 for item in saved], dtype=float) for condition in CONDITIONS}
        native = MODEL_NATIVE[model]
        native_pred = probabilities[native].argmax(axis=1)
        model_result = {"native_condition": native, "conditions": {}, "paired_changes": {}}
        for condition in CONDITIONS:
            probs = probabilities[condition]
            pred = probs.argmax(axis=1)
            model_result["conditions"][condition] = e1.classification(truth, probs, args.high_confidence)
            if condition != native:
                transition = e1.within_model(truth, probabilities[native], probs)
                transition["severity_direction_from_native"] = {
                    "prediction_higher": int(np.sum(pred > native_pred)),
                    "prediction_lower": int(np.sum(pred < native_pred)),
                    "unchanged": int(np.sum(pred == native_pred))}
                transition["ordinal_boundary_crossings"] = [
                    {"boundary": boundary,
                     "toward_higher": int(np.sum((native_pred < boundary) & (pred >= boundary))),
                     "toward_lower": int(np.sum((native_pred >= boundary) & (pred < boundary)))}
                    for boundary in range(1, 5)]
                transition["ordinal_error_change"] = {
                    "mean_absolute_delta": float(np.mean(np.abs(pred - truth) - np.abs(native_pred - truth))),
                    "improved": int(np.sum(np.abs(pred - truth) < np.abs(native_pred - truth))),
                    "worsened": int(np.sum(np.abs(pred - truth) > np.abs(native_pred - truth)))}
                transition["paired_group_bootstrap"] = _paired_intervals(
                    truth, groups, native_pred, pred, args.bootstrap, args.seed)
                model_result["paired_changes"][condition] = transition
        results["models"][model] = model_result
    (output / "analysis.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(output, results)
    print(f"COMPLETE: {meta['dataset']} E3 exploratory validation pilot analysis")


def _write_report(output, results):
    lines = [f"# E3 {results['dataset']} frozen-model enhancement pilot", "",
             f"Validation only; {results['selection']['sample_count']} images in "
             f"{len(results['selection']['selected_groups'])} whole groups.", "",
             "Grade support: " + ", ".join(f"{grade}: {count}" for grade, count in results["selection"]["grade_support"].items()),
             "", "| Model | Condition | Accuracy | Macro-F1 | QWK | Grade MAE | High-confidence wrong |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model, detail in results["models"].items():
        for condition in CONDITIONS:
            metric = detail["conditions"][condition]
            qwk = metric["quadratic_weighted_kappa"]
            lines.append(f"| {model} | {condition} | {metric['accuracy']:.4f} | {metric['macro_f1']:.4f} | "
                         f"{qwk:.4f} | {metric['grade_mae']:.4f} | {metric['high_confidence_wrong']} |"
                         if qwk is not None else
                         f"| {model} | {condition} | {metric['accuracy']:.4f} | {metric['macro_f1']:.4f} | NA | "
                         f"{metric['grade_mae']:.4f} | {metric['high_confidence_wrong']} |")
    lines += ["", "Inference-only sensitivity under fixed models. V1 and V2 training protocols differ.",
              "Group bootstrap intervals are descriptive for this selected pilot. No final test was accessed.",
              "No lesion or clinical mechanism is inferred from these prediction changes."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Self-contained SVG of the prespecified interpolation response, no plotting dependency.
    from html import escape
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="500" viewBox="0 0 780 500">',
             '<rect width="780" height="500" fill="white"/>',
             f'<text x="50" y="30" font-size="18">{escape(results["dataset"])}: V1–V2 pixel interpolation</text>']
    for panel, (model, detail) in enumerate(results["models"].items()):
        top = 65 + panel * 210
        parts += [f'<text x="50" y="{top + 10}" font-size="15">Frozen {model.upper()} checkpoint</text>',
                  f'<path d="M60 {top+170} H700 M60 {top+25} V{top+170}" stroke="black" fill="none"/>']
        for label, field, color, low, high in (("Accuracy", "accuracy", "#2369b0", 0, 1),
                                                ("Grade MAE", "grade_mae", "#ba4b36", 0, 4)):
            points = []
            for i, condition in enumerate(("v1", "mix025", "mix050", "mix075", "v2_bg10")):
                value = detail["conditions"][condition][field]
                points.append(f"{60+i*160},{top+170-(value-low)/(high-low)*145:.2f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{510 if label == "Accuracy" else 600}" y="{top+15}" font-size="12" fill="{color}">{label}</text>')
        for i, alpha in enumerate((0, .25, .5, .75, 1)):
            parts.append(f'<text x="{55+i*160}" y="{top+190}" font-size="12">{alpha:g}</text>')
    parts += ['<text x="320" y="490" font-size="12">Interpolation alpha</text>', '</svg>']
    (output / "interpolation_response.svg").write_text("\n".join(parts), encoding="utf-8")
    families = (("Ben Graham sigma", ("bg5", "v2_bg10", "bg20"), ("5", "10", "20")),
                ("LAB-CLAHE clip", ("clahe2", "clahe4"), ("2", "4")),
                ("Gamma exponent", ("gamma075", "gamma125"), ("0.75", "1.25")))
    strength = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="540" viewBox="0 0 900 540">',
                '<rect width="900" height="540" fill="white"/>',
                f'<text x="35" y="27" font-size="18">{escape(results["dataset"])}: condition accuracy by family</text>']
    for row_number, (model, detail) in enumerate(results["models"].items()):
        for column, (label, conditions, tick_labels) in enumerate(families):
            left, top = 45 + column * 290, 65 + row_number * 240
            strength += [f'<text x="{left}" y="{top}" font-size="14">{escape(model.upper())}: {escape(label)}</text>',
                         f'<path d="M{left+20} {top+175} H{left+250} M{left+20} {top+25} V{top+175}" stroke="black" fill="none"/>',
                         f'<text x="{left}" y="{top+30}" font-size="11">1.0</text>',
                         f'<text x="{left+4}" y="{top+177}" font-size="11">0</text>']
            values = [detail["conditions"][condition]["accuracy"] for condition in conditions]
            points = [f"{left+45+i*(175/(len(conditions)-1)):.1f},{top+175-value*150:.1f}"
                      for i, value in enumerate(values)]
            strength.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2369b0" stroke-width="2"/>')
            for i, (point, tick) in enumerate(zip(points, tick_labels)):
                x, y = point.split(",")
                strength += [f'<circle cx="{x}" cy="{y}" r="4" fill="#2369b0"/>',
                             f'<text x="{float(x)-12:.1f}" y="{top+194}" font-size="11">{tick}</text>']
    strength.append('</svg>')
    (output / "strength_response.svg").write_text("\n".join(strength), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--artifact-root", required=True)
        command.add_argument("--prediction-cache-dir", required=True)
        command.add_argument("--output-dir", required=True)
        if name == "pilot":
            command.add_argument("--dataset", choices=tuple(phase.SPECS), required=True)
            command.add_argument("--preprocess-cache-root", required=True)
            command.add_argument("--v2-source", required=True)
            command.add_argument("--trust-checkpoints", action="store_true")
            command.add_argument("--device", default="cuda")
            command.add_argument("--groups-per-grade", type=int, default=8)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--atol", type=float, default=1e-4)
            command.add_argument("--rtol", type=float, default=1e-4)
            command.add_argument("--resume", action="store_true")
        else:
            command.add_argument("--bootstrap", type=int, default=500)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--high-confidence", type=float, default=.9)
    args = parser.parse_args(argv)
    (pilot if args.command == "pilot" else analyze)(args)


if __name__ == "__main__":
    main()
