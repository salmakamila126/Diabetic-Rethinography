"""Exploratory E2: frozen validation feature extraction and drift analysis."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid

import numpy as np

from novelty import preof_e1 as e1
from novelty import preof_phase1 as phase


SCHEMA = "preof-e2-pooled-features-v1"
RESULT_SCHEMA = "preof-e2-drift-v1"
LAYERS = (("stem", 32, 2), ("blocks.3", 96, 8),
          ("blocks.8", 160, 32), ("head", 320, 32))
DIM = sum(channels for _, channels, _ in LAYERS)
CONDITIONS = (("p11", "v1", "v1"), ("p12", "v1", "v2_bg10"),
              ("p21", "v2", "v1"), ("p22", "v2", "v2_bg10"))
PAIRS = (("V1", "p11", "p12"), ("V2", "p22", "p21"))


def _extractor_fingerprint():
    """Separate feature semantics from future edits to the analysis/report."""
    names = {"_layer_modules", "pooled_forward", "_metadata", "_feature_key",
             "_atomic_npy", "extract"}
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    nodes = [ast.dump(node, include_attributes=False) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise ValueError("E2 extractor definition missing")
    payload = {"nodes": nodes, "layers": LAYERS, "conditions": CONDITIONS, "schema": SCHEMA}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _layer_modules(model):
    if (tuple(model.block_names) != ("IRB-1", "IRB+A-1", "IRB-2", "IRB+A-2", "EMB-1",
                                    "IRB+A-3", "EMB-2", "IRB+A-4", "EMB-3")
            or model.classifier.in_features != 320):
        raise ValueError("CAD-ViT architecture differs from audited feature points")
    return (model.stem, model.blocks[3], model.blocks[8], model.head)


def pooled_forward(model, inputs):
    """Capture pooled tensors only; full feature maps remain transient in forward."""
    import torch

    if inputs.ndim != 4 or inputs.shape[0] != 1 or inputs.shape[1] != 3 or inputs.shape[-2] != inputs.shape[-1]:
        raise ValueError("Expected one square RGB tensor")
    side = inputs.shape[-1]
    if side % 32:
        raise ValueError("Input side must be divisible by 32")
    observed = {}
    handles = []

    def make_hook(name, channels, stride):
        def capture(_module, _input, output):
            expected = (1, channels, side // stride, side // stride)
            if tuple(output.shape) != expected:
                raise ValueError(f"Layer {name}: expected {expected}, got {tuple(output.shape)}")
            observed[name] = model.pool(output).flatten(1).detach().to("cpu", dtype=torch.float32).numpy()[0].copy()
        return capture

    try:
        for (name, channels, stride), module in zip(LAYERS, _layer_modules(model)):
            handles.append(module.register_forward_hook(make_hook(name, channels, stride)))
        with torch.inference_mode():
            logits = model(inputs)
        if tuple(logits.shape) != (1, 5) or set(observed) != {entry[0] for entry in LAYERS}:
            raise ValueError("Incomplete feature capture or malformed logits")
        vector = np.concatenate([observed[name] for name, _, _ in LAYERS]).astype(np.float32)
        output_logits = logits[0].detach().to("cpu", dtype=torch.float32).numpy().copy()
        if vector.shape != (DIM,) or not np.isfinite(vector).all() or not np.isfinite(output_logits).all():
            raise ValueError("Non-finite or malformed frozen features/logits")
        return vector, output_logits
    finally:
        for handle in handles:
            handle.remove()


def _metadata(dataset, count, prediction_meta, prediction_hashes, device, atol, rtol):
    import torch

    spec = phase.SPECS[dataset]
    return {"schema": SCHEMA, "dataset": dataset, "split": "val",
            "validation_count": spec["val_count"], "feature_count": count,
            "manifest_sha256": spec["manifest_sha256"],
            "prediction_metadata_sha256": prediction_hashes["metadata_sha256"],
            "prediction_cache_sha256": prediction_hashes["cache_jsonl_sha256"],
            "image_order_sha256": prediction_meta["image_order_sha256"],
            "checkpoint_sha256": prediction_meta["provenance"]["checkpoint_sha256"],
            "checkpoint_record_sha256": prediction_meta["provenance"]["checkpoint_record_sha256"],
            "model_source_sha256": prediction_meta["provenance"]["model_source_sha256"],
            "preprocessing_source_sha256": prediction_meta["provenance"]["preprocessing_source_sha256"],
            "historical_v2_source_sha256": prediction_meta["provenance"]["v2_historical_source"]["sha256"],
            "layers": [{"name": name, "channels": channels, "stride": stride}
                       for name, channels, stride in LAYERS],
            "pooling": "adaptive average 1x1; float32 pooled vectors",
            "condition_order": [condition[0] for condition in CONDITIONS],
            "shape": [4, DIM], "dtype": "float32",
            "normalization": "uint8 RGB / 255, identical to Phase 1 prediction cache",
            "device": str(device), "torch": torch.__version__,
            "logit_check": {"atol": atol, "rtol": rtol},
            "feature_extractor_sha256": _extractor_fingerprint(),
            "implementation_sha256": phase.sha256(__file__)}


def _feature_key(meta, row):
    payload = {"metadata_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
               "image_sha256": row["sha256"], "image_id": row["image_id"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _feature_path(root, key):
    return root / "vectors" / key[:2] / f"{key}.npy"


def _read_features(root, meta, rows):
    """Validate the entire prefix and each npy digest before resume/analysis."""
    path = root / "rows.jsonl"
    if not path.is_file():
        return []
    entries = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n") or not line.strip() or len(entries) >= len(rows):
                raise ValueError("Truncated, blank, or extra E2 feature row")
            entry = json.loads(line)
            source = rows[len(entries)]
            expected = {"image_id": source["image_id"], "image_sha256": source["sha256"],
                        "true_grade": int(source["label"]), "group_id": source["group_id"],
                        "split": "val", "feature_key": _feature_key(meta, source)}
            if any(entry.get(key) != value for key, value in expected.items()):
                raise ValueError("E2 feature row differs from frozen validation manifest")
            vector_path = _feature_path(root, entry["feature_key"])
            if phase.sha256(vector_path) != entry.get("npy_sha256"):
                raise ValueError("E2 feature file digest differs")
            vectors = np.load(vector_path, allow_pickle=False)
            if vectors.shape != (4, DIM) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise ValueError("Corrupt E2 feature shape, dtype, or values")
            entries.append(entry)
    return entries


def _atomic_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_preprocessed(cache, image_sha256, variant):
    """E2 consumes Phase 1 pixels without creating or repairing its cache."""
    data_path, meta_path, key, payload = cache._paths(image_sha256, variant)
    array, status = cache._read(data_path, meta_path, key, payload)
    if status != "hit":
        raise ValueError(f"Preprocessing cache {variant} is {status}; preserve Phase 1 cache and inspect it")
    cache.stats["hits"] += 1
    return array


def extract(args):
    """Verify Phase 1 data; resume per-image frozen forward passes only."""
    import torch

    if (not np.isfinite(args.atol) or not np.isfinite(args.rtol)
            or not 0 <= args.atol <= 1e-3 or not 0 <= args.rtol <= 1e-3):
        raise ValueError("Logit comparison tolerances must be finite and within 0..1e-3")
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    dataset_root = (artifact_root / "datasets" / args.dataset).resolve()
    prediction_root = Path(args.prediction_cache_dir).expanduser().resolve()
    preprocessing_root = Path(args.preprocess_cache_root).expanduser().resolve()
    output = phase._outside_repository(args.output_dir)
    raw_root = (artifact_root / "datasets").resolve()
    if (output == prediction_root or prediction_root in output.parents or output == raw_root
            or raw_root in output.parents or output == preprocessing_root
            or preprocessing_root in output.parents or output in preprocessing_root.parents):
        raise ValueError("Feature output must be separate from caches and raw datasets")
    prediction, hashes = e1.load_verified_cache(args.dataset, artifact_root, prediction_root)
    manifest = artifact_root / "cad_vit_runs" / "manifests" / f"{args.dataset}_seed42.csv"
    all_rows = phase.validation_rows(manifest, args.dataset, dataset_root)
    count = len(all_rows) if args.limit is None else args.limit
    if not 1 <= count <= len(all_rows):
        raise ValueError("--limit must select validation images only")
    rows = all_rows[:count]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    prediction_meta = json.loads((prediction_root / "metadata.json").read_text(encoding="utf-8"))
    meta = _metadata(args.dataset, count, prediction_meta, hashes, device, args.atol, args.rtol)
    metadata_path = output / "metadata.json"
    if output.exists():
        if not args.resume or not metadata_path.is_file():
            raise FileExistsError("Use --resume for an existing E2 feature directory")
        saved_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if ({key: value for key, value in saved_meta.items() if key != "implementation_sha256"}
                != {key: value for key, value in meta.items() if key != "implementation_sha256"}):
            raise ValueError("E2 feature cache provenance/configuration differs")
        meta = saved_meta
        existing = _read_features(output, meta, rows)
    else:
        existing = []
    if len(existing) == count:
        print(json.dumps({"status": "COMPLETE", "dataset": args.dataset,
                          "features": count, "output_dir": str(output), "forward_passes": 0}, indent=2))
        return
    preprocessing = phase.PreprocessingCache(args.preprocess_cache_root, dataset_root)
    models, checked = phase.checked_models(args.dataset, artifact_root, args.v2_source, args.trust_checkpoints)
    for key in ("checkpoint_sha256", "checkpoint_record_sha256", "model_source_sha256",
                "preprocessing_source_sha256"):
        if checked[key] != prediction_meta["provenance"][key]:
            raise ValueError(f"Checkpoint/source provenance differs from prediction cache: {key}")
    if checked["v2_historical_source"]["sha256"] != meta["historical_v2_source_sha256"]:
        raise ValueError("Historical V2 source differs from prediction cache")
    for model in models.values():
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _layer_modules(model)
    if not output.exists():
        output.mkdir(parents=True)
        metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with (output / "rows.jsonl").open("ab") as stream:
        for index, row in enumerate(rows[len(existing):], start=len(existing)):
            rel = PurePosixPath(row["image_path"])
            image_path = (dataset_root / Path(*rel.parts)).resolve()
            if dataset_root not in image_path.parents:
                raise ValueError("Validation image path escapes dataset root")
            phase.require_hash(image_path, row["sha256"])
            pixels = {variant: _existing_preprocessed(preprocessing, row["sha256"], variant)
                      for variant in ("v1", "v2_bg10")}
            inputs = {variant: torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float()[None].to(device) / 255
                      for variant, array in pixels.items()}
            vectors = []
            max_difference = 0.0
            for condition, model_name, variant in CONDITIONS:
                vector, logits = pooled_forward(models[model_name], inputs[variant])
                expected = np.asarray(prediction[index][condition]["logits"], dtype=np.float32)
                difference = float(np.max(np.abs(logits - expected)))
                if not np.allclose(logits, expected, atol=args.atol, rtol=args.rtol):
                    raise ValueError(f"{condition} logits differ from Phase 1 cache on {row['image_id']}: max abs {difference}")
                max_difference = max(max_difference, difference)
                vectors.append(vector)
            array = np.stack(vectors).astype(np.float32)
            key = _feature_key(meta, row)
            vector_path = _feature_path(output, key)
            _atomic_npy(vector_path, array)
            entry = {"image_id": row["image_id"], "image_sha256": row["sha256"],
                     "true_grade": int(row["label"]), "group_id": row["group_id"],
                     "split": "val", "feature_key": key,
                     "npy_sha256": phase.sha256(vector_path),
                     "maximum_absolute_logit_difference": max_difference}
            stream.write((json.dumps(entry, separators=(",", ":")) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{args.dataset} E2 features: {index + 1}/{count} {row['image_id']}", flush=True)
    assert len(_read_features(output, meta, rows)) == count
    print(json.dumps({"status": "COMPLETE", "dataset": args.dataset, "features": count,
                      "output_dir": str(output), "preprocessing_cache": preprocessing.stats}, indent=2))


def load_complete_features(dataset, artifact_root, prediction_root, feature_root):
    """Read verified aggregate cache and all feature arrays; never open images."""
    prediction, prediction_hashes = e1.load_verified_cache(dataset, artifact_root, prediction_root)
    feature_root = Path(feature_root).expanduser().resolve()
    meta_path = feature_root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pred_meta = json.loads((Path(prediction_root) / "metadata.json").read_text(encoding="utf-8"))
    spec = phase.SPECS[dataset]
    expected = {"schema": SCHEMA, "dataset": dataset, "split": "val",
                "validation_count": spec["val_count"], "feature_count": spec["val_count"],
                "manifest_sha256": spec["manifest_sha256"],
                "prediction_metadata_sha256": prediction_hashes["metadata_sha256"],
                "prediction_cache_sha256": prediction_hashes["cache_jsonl_sha256"],
                "image_order_sha256": pred_meta["image_order_sha256"],
                "checkpoint_sha256": pred_meta["provenance"]["checkpoint_sha256"],
                "checkpoint_record_sha256": pred_meta["provenance"]["checkpoint_record_sha256"],
                "model_source_sha256": pred_meta["provenance"]["model_source_sha256"],
                "preprocessing_source_sha256": pred_meta["provenance"]["preprocessing_source_sha256"],
                "historical_v2_source_sha256": pred_meta["provenance"]["v2_historical_source"]["sha256"],
                "layers": [{"name": name, "channels": channels, "stride": stride}
                           for name, channels, stride in LAYERS],
                "condition_order": [item[0] for item in CONDITIONS],
                "shape": [4, DIM], "dtype": "float32"}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Incomplete or stale E2 feature cache metadata")
    fingerprint = meta.get("implementation_sha256")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or meta.get("feature_extractor_sha256") != _extractor_fingerprint()):
        raise ValueError("E2 extractor semantics/provenance changed; audit feature cache before reuse")
    tolerance = meta.get("logit_check")
    if (not isinstance(tolerance, dict) or set(tolerance) != {"atol", "rtol"}
            or any(not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1e-3
                   for value in tolerance.values())
            or not isinstance(meta.get("device"), str) or not isinstance(meta.get("torch"), str)):
        raise ValueError("E2 runtime or logit verification metadata malformed")
    if (meta.get("pooling") != "adaptive average 1x1; float32 pooled vectors"
            or meta.get("normalization") != "uint8 RGB / 255, identical to Phase 1 prediction cache"):
        raise ValueError("E2 pooling or normalization metadata differs")
    root = Path(artifact_root).expanduser().resolve()
    rows = phase.validation_rows(root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv",
                                 dataset, (root / "datasets" / dataset).resolve())
    entries = _read_features(feature_root, meta, rows)
    if len(entries) != spec["val_count"]:
        raise ValueError("E2 feature cache is partial; analyze full validation only")
    vectors = np.stack([np.load(_feature_path(feature_root, entry["feature_key"]), allow_pickle=False)
                        for entry in entries])
    return prediction, vectors, {"feature_metadata_sha256": phase.sha256(meta_path),
                                 "feature_rows_sha256": phase.sha256(feature_root / "rows.jsonl"),
                                 "prediction_metadata_sha256": prediction_hashes["metadata_sha256"],
                                 "prediction_cache_sha256": prediction_hashes["cache_jsonl_sha256"],
                                 "manifest_sha256": spec["manifest_sha256"],
                                 "checkpoint_sha256": meta["checkpoint_sha256"],
                                 "layers": meta["layers"], "logit_check": meta["logit_check"]}


def cosine_and_normalized_l2(native, cross):
    native = np.asarray(native, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(native, axis=1), np.linalg.norm(cross, axis=1)
    denominator = norm_a * norm_b
    cosine = np.divide(np.sum(native * cross, axis=1), denominator,
                       out=np.zeros(len(native)), where=denominator > 1e-12)
    cosine_distance = 1 - np.clip(cosine, -1, 1)
    both_zero = (norm_a < 1e-12) & (norm_b < 1e-12)
    cosine_distance[both_zero] = 0.0
    mean_norm = (norm_a + norm_b) / 2
    normalized_l2 = np.divide(np.linalg.norm(native - cross, axis=1), mean_norm,
                              out=np.zeros(len(native)), where=mean_norm > 1e-12)
    return cosine_distance, normalized_l2


def linear_cka(native, cross):
    """Centered linear CKA over paired validation images, not per-image CKA."""
    x, y = np.asarray(native, np.float64), np.asarray(cross, np.float64)
    if x.shape[0] < 2 or x.shape != y.shape:
        raise ValueError("CKA requires paired matrices with at least two images")
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    cross_norm = np.linalg.norm(x.T @ y, ord="fro") ** 2
    self_x = np.linalg.norm(x.T @ x, ord="fro")
    self_y = np.linalg.norm(y.T @ y, ord="fro")
    return None if self_x * self_y < 1e-20 else float(cross_norm / (self_x * self_y))


def _midranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2
        start = end
    return ranks


def spearman(a, b):
    a, b = _midranks(a), _midranks(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _group_ci(groups, measurements, repeats, seed):
    if repeats < 30:
        raise ValueError("At least 30 paired group-bootstrap draws required")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two validation groups required")
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = {key: [] for key in measurements}
    for _ in range(repeats):
        chosen = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([members[i] for i in chosen])
        for key, (drift, delta) in measurements.items():
            values[key].append((float(np.mean(drift[index])), spearman(drift[index], delta[index])))
    result = {}
    for key, (drift, delta) in measurements.items():
        correlations = [value[1] for value in values[key] if value[1] is not None]
        result[key] = {"mean_drift": {"point": float(np.mean(drift)),
                                      "ci95": [float(x) for x in np.quantile([v[0] for v in values[key]], [0.025, 0.975])]},
                       "spearman_vs_delta_grade_mae": {
                           "point": spearman(drift, delta),
                           "ci95": [float(x) for x in np.quantile(correlations, [0.025, 0.975])]
                           if len(correlations) >= max(30, repeats // 2) else None,
                           "valid_resamples": len(correlations)}}
    return result


def _oof_scores(features, labels, folds, count):
    scores = np.empty(len(labels), float)
    for fold in range(count):
        train, held = folds != fold, folds == fold
        if not np.any(held):
            raise ValueError("Empty grouped held-out fold")
        scores[held] = phase.fit_logistic(features[train], labels[train], features[held])
    return scores


def prediction_increment(cached, drift, groups, folds=5, bootstrap=500, seed=42):
    """All layers and dimension-matched controls are prespecified and reported."""
    if not 2 <= folds <= len(np.unique(groups)):
        raise ValueError("Invalid grouped fold count")
    assignments = phase.grouped_folds(groups, folds, seed)
    rng = np.random.default_rng(seed + 103)
    random_columns = rng.standard_normal((len(cached), 2 * len(LAYERS)))
    result = []
    truth = np.asarray([row["true_grade"] for row in cached], int)
    for model, native_key, _cross_key in PAIRS:
        all_drift = np.column_stack([drift[model][name][metric]
                                     for name, _, _ in LAYERS for metric in ("cosine", "normalized_l2")])
        for boundary in range(1, 5):
            q = phase.cumulative(np.asarray([row[native_key]["probabilities"] for row in cached]))[:, boundary - 1]
            target = ((q >= 0.5) != (truth >= boundary)).astype(int)
            item = {"model": model, "boundary": boundary, "target_errors": int(target.sum()),
                    "target_support": len(target), "status": "OK", "conditions": {}, "paired_bootstrap": {}}
            if len(np.unique(target)) != 2:
                item["status"] = "UNESTIMABLE: only one error class"
                result.append(item)
                continue
            base = phase.feature_sets(cached, boundary, model)
            features = {"B": base["B"], "D": base["D"],
                        "B+random2": np.column_stack([base["B"], random_columns[:, :2]]),
                        "B+random8": np.column_stack([base["B"], random_columns]),
                        "B+all_layers": np.column_stack([base["B"], all_drift]),
                        "D+all_layers": np.column_stack([base["D"], all_drift])}
            for name, _, _ in LAYERS:
                features[f"B+{name}"] = np.column_stack([base["B"], drift[model][name]["cosine"],
                                                         drift[model][name]["normalized_l2"]])
            try:
                scores = {name: _oof_scores(values, target, assignments, folds)
                          for name, values in features.items()}
                item["conditions"] = {name: {"dimension": features[name].shape[1],
                                               **phase.metric_bundle(target, value)}
                                      for name, value in scores.items()}
                for candidate, baseline in ((f"B+{name}", "B") for name, _, _ in LAYERS):
                    item["paired_bootstrap"][candidate] = {
                        "baseline": baseline,
                        "observed_delta": {metric: item["conditions"][candidate][metric] -
                                           item["conditions"][baseline][metric]
                                           for metric in ("auroc", "auprc", "brier", "log_loss")},
                        "group_intervals": phase.paired_group_bootstrap(target, scores[baseline],
                                                                        scores[candidate], groups, bootstrap, seed)}
                for candidate, baseline in (("B+random2", "B"), ("B+random8", "B"),
                                            ("B+all_layers", "B"), ("D+all_layers", "D")):
                    item["paired_bootstrap"][candidate] = {
                        "baseline": baseline,
                        "observed_delta": {metric: item["conditions"][candidate][metric] -
                                           item["conditions"][baseline][metric]
                                           for metric in ("auroc", "auprc", "brier", "log_loss")},
                        "group_intervals": phase.paired_group_bootstrap(target, scores[baseline],
                                                                        scores[candidate], groups, bootstrap, seed)}
            except (ValueError, np.linalg.LinAlgError) as exc:
                item["status"] = f"UNESTIMABLE: {exc}"
                item["conditions"] = {}
                item["paired_bootstrap"] = {}
            result.append(item)
    return result


def analyze_arrays(cached, vectors, dataset, provenance, folds=5, bootstrap=500, seed=42):
    if vectors.shape != (len(cached), 4, DIM) or not np.isfinite(vectors).all():
        raise ValueError("E2 vector shape or values invalid")
    truth = np.asarray([row["true_grade"] for row in cached], int)
    groups = np.asarray([row["group_id"] for row in cached])
    boundaries = np.cumsum([0] + [channels for _, channels, _ in LAYERS])
    drift, profiles, bootstrap_measurements = {}, {}, {}
    for model, native_key, cross_key in PAIRS:
        native_index = next(i for i, condition in enumerate(CONDITIONS) if condition[0] == native_key)
        cross_index = next(i for i, condition in enumerate(CONDITIONS) if condition[0] == cross_key)
        native_grade = np.asarray([np.argmax(row[native_key]["probabilities"]) for row in cached])
        cross_grade = np.asarray([np.argmax(row[cross_key]["probabilities"]) for row in cached])
        delta_error = np.abs(cross_grade - truth) - np.abs(native_grade - truth)
        transitions = e1.within_model(truth,
                                      np.asarray([row[native_key]["probabilities"] for row in cached]),
                                      np.asarray([row[cross_key]["probabilities"] for row in cached]))
        drift[model], profiles[model] = {}, {"transitions": transitions, "layers": {}}
        for i, (name, _channels, _stride) in enumerate(LAYERS):
            native = vectors[:, native_index, boundaries[i]:boundaries[i + 1]]
            cross = vectors[:, cross_index, boundaries[i]:boundaries[i + 1]]
            cosine, normalized_l2 = cosine_and_normalized_l2(native, cross)
            drift[model][name] = {"cosine": cosine, "normalized_l2": normalized_l2}
            profile = {"dimension": native.shape[1], "linear_cka": linear_cka(native, cross),
                       "mean_cosine_distance": float(cosine.mean()),
                       "mean_normalized_l2": float(normalized_l2.mean()),
                       "spearman_cosine_vs_delta_grade_mae": spearman(cosine, delta_error),
                       "spearman_normalized_l2_vs_delta_grade_mae": spearman(normalized_l2, delta_error),
                       "per_grade": []}
            for grade in range(5):
                mask = truth == grade
                profile["per_grade"].append({"grade": grade, "support": int(mask.sum()),
                                             "mean_cosine_distance": float(cosine[mask].mean()) if mask.any() else None,
                                             "median_cosine_distance": float(np.median(cosine[mask])) if mask.any() else None,
                                             "mean_normalized_l2": float(normalized_l2[mask].mean()) if mask.any() else None,
                                             "mean_delta_grade_mae": float(delta_error[mask].mean()) if mask.any() else None,
                                             "correct_to_wrong": int(np.sum((native_grade == truth) &
                                                                            (cross_grade != truth) & mask)),
                                             "wrong_to_correct": int(np.sum((native_grade != truth) &
                                                                            (cross_grade == truth) & mask))})
            profiles[model]["layers"][name] = profile
            for metric, values in drift[model][name].items():
                bootstrap_measurements[f"{model}:{name}:{metric}"] = (values, delta_error)
    intervals = _group_ci(groups, bootstrap_measurements, bootstrap, seed)
    increments = prediction_increment(cached, drift, groups, folds, bootstrap, seed)
    return {"schema": RESULT_SCHEMA, "status": "E2_EXPLORATORY_COMPLETE", "dataset": dataset,
            "split": "val", "image_count": len(cached), "group_count": int(len(np.unique(groups))),
            "provenance": provenance,
            "definitions": {"cosine_distance": "1-cosine(native,cross); both-zero vectors map to 0",
                            "normalized_l2": "L2(native-cross) / mean(L2(native),L2(cross)); both-zero maps to 0",
                            "linear_cka": "centered linear CKA over all paired validation images, not per-image",
                            "error_association": "Spearman drift versus cross minus native absolute grade error",
                            "error_target": "incorrect native P(Y>=k)>=0.5 boundary decision",
                            "prediction_controls": "B is E0 native confidence plus native expert disagreement; D uses all four probability distributions",
                            "dimension_controls": "B+random2 and B+random8 use one fixed, label-independent Gaussian draw",
                            "uncertainty": "paired validation-group percentile bootstrap; exploratory across all layers and boundaries"},
            "folds": folds, "bootstrap_repeats": bootstrap, "seed": seed,
            "profiles": profiles, "drift_bootstrap": intervals,
            "incremental_error_detection": increments}


def _svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="Arial, sans-serif" fill="#173042">',
            f'<text x="24" y="32" font-size="19" font-weight="bold">{title}</text>']


def profile_svg(result, metric, title):
    names = [name for name, _, _ in LAYERS]
    datasets = {model: [result["profiles"][model]["layers"][name][metric] for name in names]
                for model, _, _ in PAIRS}
    values = [value for row in datasets.values() for value in row if value is not None]
    ceiling = max(1.0 if metric == "linear_cka" else 0.1, max(values, default=0) * 1.1)
    width, height = 740, 400
    left, right, top, bottom = 92, 690, 75, 325
    parts = _svg_start(width, height, title)
    parts += [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#555"/>',
              f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#555"/>']
    for tick in range(5):
        value = ceiling * tick / 4
        y = bottom - (bottom - top) * tick / 4
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e9ed"/>',
                  f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.3f}</text>']
    for index, name in enumerate(names):
        x = left + (right - left) * (index + 0.5) / len(names)
        parts.append(f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="12">{name}</text>')
    for model, color in (("V1", "#16796f"), ("V2", "#ad503e")):
        coordinates = []
        for index, value in enumerate(datasets[model]):
            if value is None:
                continue
            x = left + (right - left) * (index + 0.5) / len(names)
            y = bottom - (bottom - top) * value / ceiling
            coordinates.append((x, y))
            parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>',
                      f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" font-size="11">{value:.3f}</text>']
        if len(coordinates) > 1:
            points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in coordinates)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts += ['<circle cx="270" cy="383" r="5" fill="#16796f"/><text x="282" y="387" font-size="13">V1 fixed model</text>',
              '<circle cx="430" cy="383" r="5" fill="#ad503e"/><text x="442" y="387" font-size="13">V2 fixed model</text>',
              '</g>', '</svg>']
    return '\n'.join(parts) + '\n'


def incremental_svg(result):
    rows = result["incremental_error_detection"]
    columns = [name for name, _, _ in LAYERS] + ["all_layers", "random2", "random8"]
    width, height = 940, 440
    parts = _svg_start(width, height, "OOF delta AUPRC versus prediction-only B (exploratory)")
    parts.append('<text x="24" y="55" font-size="12">Rows: native model and ordinal boundary; cells: observed delta. NA means unestimable.</text>')
    x0, y0, cw, ch = 205, 105, 103, 34
    for col, name in enumerate(columns):
        parts.append(f'<text x="{x0 + col*cw + cw/2:.1f}" y="89" text-anchor="middle" font-size="11">{name}</text>')
    for row_index, item in enumerate(rows):
        y = y0 + row_index * ch
        parts.append(f'<text x="24" y="{y + 22}" font-size="12">{item["model"]} / Y≥{item["boundary"]}</text>')
        for col, name in enumerate(columns):
            key = f'B+{name}'
            data = item["paired_bootstrap"].get(key)
            delta = None if data is None else data["observed_delta"]["auprc"]
            magnitude = min(abs(delta) / 0.10, 1.0) if delta is not None else 0
            if delta is None:
                fill = '#eceff1'
            elif delta >= 0:
                fill = f'rgb({int(232-110*magnitude)},235,{int(225-115*magnitude)})'
            else:
                fill = f'rgb(245,{int(232-105*magnitude)},{int(222-110*magnitude)})'
            x = x0 + col * cw
            parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="#9aa8b1"/>')
            label = 'NA' if delta is None else f'{delta:+.3f}'
            parts.append(f'<text x="{x + cw/2:.1f}" y="{y + 22}" text-anchor="middle" font-size="12">{label}</text>')
    parts += ['<text x="24" y="415" font-size="12">Color saturates at |Δ AUPRC| = 0.10; paired-group intervals are in JSON.</text>',
              '</g>', '</svg>']
    return '\n'.join(parts) + '\n'


def per_grade_svg(result):
    names = [name for name, _, _ in LAYERS]
    values = [entry["mean_normalized_l2"]
              for model, _, _ in PAIRS for name in names
              for entry in result["profiles"][model]["layers"][name]["per_grade"]
              if entry["mean_normalized_l2"] is not None]
    maximum = max(values, default=1) or 1
    width, height = 760, 450
    parts = _svg_start(width, height, "Per-grade mean normalized L2 drift")
    parts.append('<text x="24" y="54" font-size="12">Within each fixed model; support is shown beside each true grade.</text>')
    for panel, (model, _, _) in enumerate(PAIRS):
        top = 88 + panel * 175
        parts.append(f'<text x="24" y="{top}" font-size="15" font-weight="bold">{model}</text>')
        for j, name in enumerate(names):
            parts.append(f'<text x="{200 + j*130 + 65}" y="{top + 20}" text-anchor="middle" '
                         f'font-size="12">{name}</text>')
        for grade in range(5):
            y = top + 27 + grade * 25
            sample = result["profiles"][model]["layers"][names[0]]["per_grade"][grade]
            parts.append(f'<text x="25" y="{y + 17}" font-size="12">Grade {grade} (n={sample["support"]})</text>')
            for j, name in enumerate(names):
                entry = result["profiles"][model]["layers"][name]["per_grade"][grade]
                value = entry["mean_normalized_l2"]
                strength = 0 if value is None else min(value / maximum, 1)
                fill = '#edf0f1' if value is None else f'rgb({int(238-108*strength)},{int(242-48*strength)},{int(249-75*strength)})'
                x = 200 + j * 130
                parts.append(f'<rect x="{x}" y="{y}" width="130" height="25" fill="{fill}" stroke="#8b9aa3"/>')
                label = 'NA' if value is None else f'{value:.3f}'
                parts.append(f'<text x="{x + 65}" y="{y + 17}" text-anchor="middle" font-size="12">{label}</text>')
    parts += ['<text x="24" y="435" font-size="12">Color intensity uses this dataset’s largest displayed mean; compare numeric labels across panels.</text>',
              '</g>', '</svg>']
    return '\n'.join(parts) + '\n'


def report_markdown(result):
    lines = [f'# PReOF E2 — {result["dataset"]} layer-wise representation drift', '',
             '**Exploratory validation analysis. No layer is selected as independently confirmed.**', '',
             f'{result["image_count"]} validation images, {result["group_count"]} groups; '
             f'{result["folds"]}-fold grouped OOF and {result["bootstrap_repeats"]} paired group-bootstrap draws.',
             '', '## Fixed-model native versus cross-preprocessing embeddings', '',
             '| Model | Layer | Dim | Mean cosine distance | Mean normalized L2 | Linear CKA | Spearman with Δ grade MAE |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for model, _, _ in PAIRS:
        for name, _, _ in LAYERS:
            item = result["profiles"][model]["layers"][name]
            cka = 'NA' if item['linear_cka'] is None else f'{item["linear_cka"]:.4f}'
            association = item['spearman_normalized_l2_vs_delta_grade_mae']
            association_text = 'NA' if association is None else f'{association:+.4f}'
            lines.append(f'| {model} | `{name}` | {item["dimension"]} | '
                         f'{item["mean_cosine_distance"]:.4f} | {item["mean_normalized_l2"]:.4f} | '
                         f'{cka} | {association_text} |')
    lines += ['', 'Δ grade MAE is cross minus native absolute grade error. Paired group intervals '
              'and the full per-grade profiles are in the JSON.', '',
              '| Fixed model | Predictions changed | Correct → wrong | Wrong → correct | Both wrong |',
              '| --- | ---: | ---: | ---: | ---: |']
    for model, _, _ in PAIRS:
        transitions = result['profiles'][model]['transitions']
        lines.append(f'| {model} | {transitions["prediction_changed"]} | '
                     f'{transitions["correct_to_wrong"]} | {transitions["wrong_to_correct"]} | '
                     f'{transitions["both_wrong"]} |')
    lines += ['', '![Per-grade normalized L2](per_grade_drift.svg)',
              '', '![Cosine drift](cosine_profile.svg)', '',
              '![Normalized L2 drift](normalized_l2_profile.svg)', '',
              '![Linear CKA](linear_cka_profile.svg)', '',
              '## Incremental native boundary-error detection', '',
              'E0 feature set B (native confidence plus native expert disagreement) is the primary '
              'prediction-only baseline. D contains all four prediction distributions. Each layer adds '
              'two drift features (cosine and normalized L2); fixed Gaussian controls match the added '
              '2 or 8 dimensions. Every feature is standardized on training folds only.', '',
              '| Model | Y≥k | Native errors | B AUPRC | D AUPRC | Δ(B+all−B) AUPRC [95% CI] | Δ(B+random8−B) AUPRC |',
              '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for item in result["incremental_error_detection"]:
        if item["status"] != "OK":
            lines.append(f'| {item["model"]} | {item["boundary"]} | {item["target_errors"]} | '
                         f'{item["status"]} | — | — | — |')
            continue
        delta = item["paired_bootstrap"]["B+all_layers"]
        interval = delta["group_intervals"].get("auprc")
        interval_text = 'unestimable' if interval is None else \
            f'[{interval["ci95"][0]:+.4f}, {interval["ci95"][1]:+.4f}]'
        random_delta = item["paired_bootstrap"]["B+random8"]["observed_delta"]["auprc"]
        lines.append(f'| {item["model"]} | {item["boundary"]} | {item["target_errors"]} | '
                     f'{item["conditions"]["B"]["auprc"]:.4f} | {item["conditions"]["D"]["auprc"]:.4f} | '
                     f'{delta["observed_delta"]["auprc"]:+.4f} {interval_text} | {random_delta:+.4f} |')
    lines += ['', '![OOF AUPRC deltas](incremental_auprc.svg)', '',
              'All four layers, all boundaries, random controls, D baseline, Brier, AUROC, and log loss '
              'are reported in `e2_results.json`. Multiple layer/boundary comparisons are exploratory; '
              'do not pick the most favorable cell as an independently confirmed result. Stable embeddings '
              'do not imply diagnostic relevance. Prediction mismatch cannot establish lesion loss or a '
              'clinical causal mechanism.', '']
    return '\n'.join(lines)


def analyze_command(args):
    prediction_root = Path(args.prediction_cache_dir).expanduser().resolve()
    feature_root = Path(args.feature_cache_dir).expanduser().resolve()
    output = phase._outside_repository(args.output_dir)
    raw_root = (Path(args.artifact_root).expanduser().resolve() / 'datasets').resolve()
    if (output == prediction_root or prediction_root in output.parents
            or output == feature_root or feature_root in output.parents
            or output == raw_root or raw_root in output.parents):
        raise ValueError('E2 analysis output must be separate from caches and raw datasets')
    if output.exists():
        raise FileExistsError('E2 analysis directory exists; never overwrite results')
    cached, vectors, provenance = load_complete_features(args.dataset, args.artifact_root,
                                                          prediction_root, feature_root)
    result = analyze_arrays(cached, vectors, args.dataset, provenance,
                            args.folds, args.bootstrap, args.seed)
    output.mkdir(parents=True)
    (output / 'e2_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'e2_report.md').write_text(report_markdown(result), encoding='utf-8')
    for filename, content in (("cosine_profile.svg", profile_svg(result, "mean_cosine_distance", "Mean cosine distance by layer")),
                              ("normalized_l2_profile.svg", profile_svg(result, "mean_normalized_l2", "Mean normalized L2 drift by layer")),
                              ("linear_cka_profile.svg", profile_svg(result, "linear_cka", "Centered linear CKA by layer")),
                              ("per_grade_drift.svg", per_grade_svg(result)),
                              ("incremental_auprc.svg", incremental_svg(result))):
        (output / filename).write_text(content, encoding='utf-8')
    print(json.dumps({"status": result["status"], "dataset": args.dataset,
                      "validation_images": len(cached), "output_dir": str(output)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract_cmd = commands.add_parser("extract", help="Frozen validation feature cache; model inference required")
    extract_cmd.add_argument("--dataset", choices=tuple(phase.SPECS), required=True)
    extract_cmd.add_argument("--artifact-root", required=True)
    extract_cmd.add_argument("--prediction-cache-dir", required=True)
    extract_cmd.add_argument("--preprocess-cache-root", required=True)
    extract_cmd.add_argument("--v2-source", required=True)
    extract_cmd.add_argument("--trust-checkpoints", action="store_true")
    extract_cmd.add_argument("--output-dir", required=True)
    extract_cmd.add_argument("--device", default="cpu")
    extract_cmd.add_argument("--limit", type=int)
    extract_cmd.add_argument("--resume", action="store_true")
    extract_cmd.add_argument("--atol", type=float, default=1e-4)
    extract_cmd.add_argument("--rtol", type=float, default=1e-4)
    analyze_cmd = commands.add_parser("analyze", help="CPU-only, full validation feature analysis")
    analyze_cmd.add_argument("--dataset", choices=tuple(phase.SPECS), required=True)
    analyze_cmd.add_argument("--artifact-root", required=True)
    analyze_cmd.add_argument("--prediction-cache-dir", required=True)
    analyze_cmd.add_argument("--feature-cache-dir", required=True)
    analyze_cmd.add_argument("--output-dir", required=True)
    analyze_cmd.add_argument("--folds", type=int, default=5)
    analyze_cmd.add_argument("--bootstrap", type=int, default=500)
    analyze_cmd.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract(args)
    else:
        analyze_command(args)


if __name__ == "__main__":
    main()
