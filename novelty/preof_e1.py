"""PReOF E1: read-only, validation-only analysis of Phase 1 prediction caches."""

import argparse
import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np

from novelty import preof_phase1 as phase


SCHEMA = "preof-e1-aggregate-v1"
NAMES = {"p11": "V1 model / V1 preprocessing (native)",
         "p12": "V1 model / V2 BG10 preprocessing (cross)",
         "p21": "V2 model / V1 preprocessing (cross)",
         "p22": "V2 model / V2 BG10 preprocessing (native)"}
PAIRS = {"V1_within_model": ("p11", "p12"),
         "V2_within_model": ("p22", "p21")}


def load_verified_cache(dataset, artifact_root, cache_dir):
    """Check complete Phase 1 provenance and rows without opening images/models."""
    artifact_root = Path(artifact_root).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    meta_path = cache_dir / "metadata.json"
    cache_path = cache_dir / "cache.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    spec = phase.SPECS[dataset]
    provenance = meta.get("provenance")
    if (meta.get("schema") != phase.CACHE_VERSION or meta.get("dataset") != dataset
            or meta.get("split") != "val" or meta.get("validation_count") != spec["val_count"]
            or meta.get("cache_count") != spec["val_count"]
            or meta.get("manifest_sha256") != spec["manifest_sha256"]
            or not isinstance(provenance, dict)
            or provenance.get("checkpoint_sha256") !=
               {v: spec[f"{v}_sha256"] for v in ("v1", "v2")}
            or provenance.get("checkpoint_epoch") !=
               {v: spec[f"{v}_epoch"] for v in ("v1", "v2")}):
        raise ValueError("E1 requires the complete, frozen validation prediction cache")
    fingerprint = meta.get("implementation_sha256")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("Prediction-cache implementation fingerprint is malformed")
    record_hashes = provenance.get("checkpoint_record_sha256")
    if (not isinstance(record_hashes, dict) or set(record_hashes) != {"v1", "v2"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in record_hashes.values())):
        raise ValueError("Prediction-cache checkpoint record fingerprints are malformed")
    source = provenance.get("v2_historical_source")
    if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
        raise ValueError("Historical V2 source provenance is missing")
    phase.require_hash(source["path"], source["sha256"])
    if phase.verify_v2_source(source["path"])["sha256"] != source["sha256"]:
        raise ValueError("Historical V2 source differs")
    for name, key in (("model.py", "model_source_sha256"),
                      ("preprocessing.py", "preprocessing_source_sha256")):
        phase.require_hash(phase.REPO / "reproduction/cadvit" / name, provenance[key])
    dataset_root = (artifact_root / "datasets" / dataset).resolve()
    manifest = artifact_root / "cad_vit_runs" / "manifests" / f"{dataset}_seed42.csv"
    rows = phase.validation_rows(manifest, dataset, dataset_root)
    order_hash = hashlib.sha256("\n".join(row["image_id"] for row in rows).encode()).hexdigest()
    if meta.get("image_order_sha256") != order_hash:
        raise ValueError("Prediction cache image order differs from frozen manifest")
    cached = phase.read_cache(cache_dir, rows)
    if len(cached) != spec["val_count"]:
        raise ValueError("Prediction cache is incomplete")
    return cached, {"metadata_sha256": phase.sha256(meta_path),
                    "cache_jsonl_sha256": phase.sha256(cache_path),
                    "manifest_sha256": spec["manifest_sha256"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "model_source_sha256": provenance["model_source_sha256"],
                    "preprocessing_source_sha256": provenance["preprocessing_source_sha256"],
                    "historical_v2_source_sha256": source["sha256"]}


def confusion(truth, predicted):
    return np.bincount(5 * truth + predicted, minlength=25).reshape(5, 5)


def classification(truth, probabilities, high_confidence=0.9):
    predicted = probabilities.argmax(axis=1)
    matrix = confusion(truth, predicted)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted_support, out=np.zeros(5, float), where=predicted_support > 0)
    recall = np.divide(diagonal, support, out=np.zeros(5, float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(5, float), where=precision + recall > 0)
    errors = predicted - truth
    wrong = errors != 0
    high = probabilities.max(axis=1) >= high_confidence
    n = len(truth)
    observed = np.sum(matrix * (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2) / n
    expected = np.sum(np.outer(support, predicted_support) *
                      (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2) / n**2
    qwk = None if expected == 0 else float(1 - observed / expected)
    return {"confusion_matrix": matrix.tolist(), "accuracy": float(np.mean(~wrong)),
            "macro_f1": float(f1.mean()), "grade_mae": float(np.mean(np.abs(errors))),
            "quadratic_weighted_kappa": qwk,
            "overgraded": int(np.sum(errors > 0)), "undergraded": int(np.sum(errors < 0)),
            "correct": int(np.sum(~wrong)), "wrong": int(np.sum(wrong)),
            "mean_overgrade_magnitude": float(np.mean(errors[errors > 0])) if np.any(errors > 0) else None,
            "mean_undergrade_magnitude": float(np.mean(-errors[errors < 0])) if np.any(errors < 0) else None,
            "high_confidence_threshold": high_confidence,
            "high_confidence_wrong": int(np.sum(high & wrong)),
            "high_confidence_wrong_rate_all": float(np.mean(high & wrong)),
            "high_confidence_wrong_fraction_of_wrong":
                float(np.mean(high[wrong])) if np.any(wrong) else None,
            "per_grade": [{"grade": grade, "support": int(support[grade]),
                           "precision": float(precision[grade]), "recall": float(recall[grade]),
                           "f1": float(f1[grade]), "correct": int(diagonal[grade]),
                           "overgraded": int(np.sum((truth == grade) & (errors > 0))),
                           "undergraded": int(np.sum((truth == grade) & (errors < 0))),
                           "mean_absolute_error": float(np.mean(np.abs(errors[truth == grade])))
                           if support[grade] else None} for grade in range(5)]}


def within_model(truth, native_prob, cross_prob):
    """Paired preprocessing comparison for one fixed model checkpoint."""
    native, cross = native_prob.argmax(axis=1), cross_prob.argmax(axis=1)
    native_correct, cross_correct = native == truth, cross == truth
    changed = native != cross
    l1_shift = np.abs(native_prob - cross_prob).sum(axis=1)
    true_delta = cross_prob[np.arange(len(truth)), truth] - native_prob[np.arange(len(truth)), truth]
    by_grade = []
    for grade in range(5):
        subset = truth == grade
        count = int(subset.sum())
        by_grade.append({"grade": grade, "support": count,
                         "prediction_changed": int(np.sum(changed & subset)),
                         "changed_rate": float(np.mean(changed[subset])) if count else None,
                         "correct_to_wrong": int(np.sum(native_correct & ~cross_correct & subset)),
                         "wrong_to_correct": int(np.sum(~native_correct & cross_correct & subset)),
                         "native_accuracy": float(np.mean(native_correct[subset])) if count else None,
                         "cross_accuracy": float(np.mean(cross_correct[subset])) if count else None,
                         "accuracy_delta_cross_minus_native":
                             float(np.mean(cross_correct[subset]) - np.mean(native_correct[subset])) if count else None,
                         "mean_probability_l1_shift": float(np.mean(l1_shift[subset])) if count else None,
                         "mean_true_class_probability_delta": float(np.mean(true_delta[subset])) if count else None})
    return {"prediction_transition_matrix_native_rows_cross_columns": confusion(native, cross).tolist(),
            "prediction_changed": int(changed.sum()), "prediction_changed_rate": float(changed.mean()),
            "both_correct": int(np.sum(native_correct & cross_correct)),
            "correct_to_wrong": int(np.sum(native_correct & ~cross_correct)),
            "wrong_to_correct": int(np.sum(~native_correct & cross_correct)),
            "both_wrong": int(np.sum(~native_correct & ~cross_correct)),
            "mean_probability_l1_shift": float(l1_shift.mean()),
            "mean_true_class_probability_delta": float(true_delta.mean()),
            "per_grade": by_grade}


def complementarity(truth, predicted):
    correct = {key: pred == truth for key, pred in predicted.items()}
    native_a, native_b = correct["p11"], correct["p22"]
    native_error = np.minimum(np.abs(predicted["p11"] - truth), np.abs(predicted["p22"] - truth))
    all_error = np.minimum.reduce([np.abs(pred - truth) for pred in predicted.values()])
    return {"native_both_correct": int(np.sum(native_a & native_b)),
            "native_V1_only_correct": int(np.sum(native_a & ~native_b)),
            "native_V2_only_correct": int(np.sum(~native_a & native_b)),
            "native_both_wrong": int(np.sum(~native_a & ~native_b)),
            "native_oracle_accuracy": float(np.mean(native_a | native_b)),
            "native_oracle_mae": float(np.mean(native_error)),
            "four_distribution_oracle_accuracy": float(np.mean(np.logical_or.reduce(list(correct.values())))),
            "four_distribution_oracle_mae": float(np.mean(all_error)),
            "oracle_definition": "Uses true validation labels to select a correct prediction if available, otherwise minimum absolute grade error; upper bound only."}


def group_bootstrap(truth, predicted, groups, repeats=500, seed=42):
    """Paired resampling of image groups, with every condition using the same draw."""
    if repeats < 30:
        raise ValueError("At least 30 bootstrap draws are required")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two validation groups are required")
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)

    def measures(index):
        y = truth[index]
        values = {}
        for name, (native_key, cross_key) in PAIRS.items():
            native, cross = predicted[native_key][index], predicted[cross_key][index]
            values[f"{name}_accuracy_delta"] = float(np.mean(cross == y) - np.mean(native == y))
            values[f"{name}_mae_delta"] = float(np.mean(np.abs(cross - y)) - np.mean(np.abs(native - y)))
        a, b = predicted["p11"][index], predicted["p22"][index]
        values["native_oracle_accuracy_gain_vs_better_expert"] = float(
            np.mean((a == y) | (b == y)) - max(np.mean(a == y), np.mean(b == y)))
        return values

    point = measures(np.arange(len(truth)))
    draws = {key: [] for key in point}
    for _ in range(repeats):
        selected = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([members[i] for i in selected])
        for key, value in measures(index).items():
            draws[key].append(value)
    return {key: {"point": value, "ci95_percentile":
                  [float(np.quantile(draws[key], 0.025)), float(np.quantile(draws[key], 0.975))]}
            for key, value in point.items()}


def analyze(cached, dataset, provenance, high_confidence=0.9, bootstrap=500, seed=42):
    if not 0 < high_confidence <= 1:
        raise ValueError("High-confidence threshold must be in (0, 1]")
    truth = np.asarray([row["true_grade"] for row in cached], dtype=int)
    groups = np.asarray([row["group_id"] for row in cached])
    probabilities = {key: np.asarray([row[key]["probabilities"] for row in cached], dtype=float)
                     for key in phase.KEYS}
    predicted = {key: value.argmax(axis=1) for key, value in probabilities.items()}
    results = {"schema": SCHEMA, "status": "E1_COMPLETE", "dataset": dataset, "split": "val",
               "sample_count": len(cached), "group_count": int(len(np.unique(groups))),
               "provenance": provenance,
               "definitions": {"grade_transition": "Rows are native predicted grades; columns are cross-preprocessing predicted grades.",
                               "high_confidence": "Maximum five-class probability >= threshold and prediction incorrect.",
                               "within_model": "One fixed checkpoint with V1 versus V2 BG10 preprocessing; descriptive paired change, not randomized intervention.",
                               "between_model": "Native V1 versus native V2 differs in training protocol and preprocessing; no causal attribution.",
                               "uncertainty": "Paired percentile 95% intervals from validation group bootstrap; descriptive, not a confirmatory test."},
               "conditions": {key: {"label": NAMES[key], **classification(truth, probabilities[key], high_confidence)}
                              for key in phase.KEYS},
               "within_model": {name: within_model(truth, probabilities[native], probabilities[cross])
                                for name, (native, cross) in PAIRS.items()},
               "expert_complementarity": complementarity(truth, predicted),
               "paired_group_bootstrap": {"repeats": bootstrap, "seed": seed,
                                          "intervals": group_bootstrap(truth, predicted, groups, bootstrap, seed)}}
    return results


def _heatmap_panel(matrix, title, left, top):
    """Labelled count heatmap; row totals remain explicit in the JSON report."""
    maximum = max(max(row) for row in matrix) or 1
    parts = [f'<text x="{left + 145}" y="{top + 20}" text-anchor="middle" '
             f'font-size="16" font-weight="bold">{escape(title)}</text>']
    cell = 43
    x0, y0 = left + 45, top + 48
    for grade in range(5):
        parts.append(f'<text x="{x0 + grade * cell + 21}" y="{y0 - 8}" text-anchor="middle">{grade}</text>')
        parts.append(f'<text x="{x0 - 12}" y="{y0 + grade * cell + 27}" text-anchor="middle">{grade}</text>')
        for prediction in range(5):
            count = matrix[grade][prediction]
            intensity = count / maximum
            blue = int(245 - 145 * intensity)
            fill = f'rgb({blue},{blue + 3},{255})'
            x, y = x0 + prediction * cell, y0 + grade * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="{fill}" stroke="#607080"/>')
            parts.append(f'<text x="{x + 21}" y="{y + 27}" text-anchor="middle" '
                         f'font-size="13">{count}</text>')
    return "".join(parts)


def heatmap_svg(results, transitions=False):
    if transitions:
        panels = [(name, results["within_model"][name]["prediction_transition_matrix_native_rows_cross_columns"])
                  for name in PAIRS]
        heading = "Within-model grade transitions: native rows, cross-preprocessing columns"
        width, height = 720, 370
        positions = [(20, 75), (370, 75)]
    else:
        panels = [(key, results["conditions"][key]["confusion_matrix"]) for key in phase.KEYS]
        heading = "Validation confusion matrices: true-grade rows, predicted-grade columns"
        width, height = 720, 690
        positions = [(20, 75), (370, 75), (20, 390), (370, 390)]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#172b3a">',
             f'<text x="20" y="35" font-size="18" font-weight="bold">{escape(heading)}</text>']
    for (title, matrix), (left, top) in zip(panels, positions):
        parts.append(_heatmap_panel(matrix, title, left, top))
    parts.extend(['</g>', '</svg>'])
    return "\n".join(parts) + "\n"


def grade_delta_svg(results):
    width, height = 760, 430
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#172b3a">',
             '<text x="20" y="32" font-size="18" font-weight="bold">'
             'Per-grade accuracy change: cross minus native preprocessing</text>',
             '<text x="20" y="56" font-size="13">Paired validation images; bar labels show support (n).</text>']
    scale, center = 225, 440
    for panel_index, name in enumerate(PAIRS):
        top = 86 + panel_index * 168
        parts.append(f'<text x="20" y="{top}" font-size="15" font-weight="bold">{name}</text>')
        parts.append(f'<line x1="{center}" y1="{top + 12}" x2="{center}" y2="{top + 135}" stroke="#333"/>')
        for entry in results["within_model"][name]["per_grade"]:
            grade = entry["grade"]
            delta = entry["accuracy_delta_cross_minus_native"]
            y = top + 15 + grade * 23
            parts.append(f'<text x="35" y="{y + 14}" font-size="13">Grade {grade} (n={entry["support"]})</text>')
            if delta is None:
                parts.append(f'<text x="{center + 8}" y="{y + 14}" font-size="13">NA</text>')
                continue
            extent = abs(delta) * scale
            x = center if delta >= 0 else center - extent
            color = '#18746e' if delta >= 0 else '#aa4a45'
            parts.append(f'<rect x="{x:.2f}" y="{y}" width="{extent:.2f}" height="17" fill="{color}"/>')
            label_x = center + extent + 8 if delta >= 0 else center - extent - 8
            anchor = 'start' if delta >= 0 else 'end'
            parts.append(f'<text x="{label_x:.2f}" y="{y + 14}" text-anchor="{anchor}" '
                         f'font-size="12">{delta:+.3f}</text>')
    parts.extend(['<text x="195" y="417" font-size="12">−1.0</text>',
                  '<text x="438" y="417" font-size="12">0</text>',
                  '<text x="655" y="417" font-size="12">+1.0</text>', '</g>', '</svg>'])
    return "\n".join(parts) + "\n"


def markdown_report(results):
    lines = [f'# PReOF E1 — {results["dataset"]} validation prediction failures', '',
             f'Status: **{results["status"]}**. {results["sample_count"]} validation images '
             f'across {results["group_count"]} groups. All numbers come from the verified Phase 1 '
             'prediction cache; no images, model inference, or final-test data were accessed.', '',
             '## Four fixed model/preprocessing conditions', '',
             '| Condition | Accuracy | Macro-F1 | Grade MAE | Over / under | High-confidence wrong |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in phase.KEYS:
        entry = results['conditions'][key]
        lines.append(f'| {key}: {NAMES[key]} | {entry["accuracy"]:.4f} | '
                     f'{entry["macro_f1"]:.4f} | {entry["grade_mae"]:.4f} | '
                     f'{entry["overgraded"]} / {entry["undergraded"]} | '
                     f'{entry["high_confidence_wrong"]} |')
    lines += ['', 'High-confidence wrong means maximum five-class probability '
              f'≥ {results["conditions"]["p11"]["high_confidence_threshold"]:.2f} and an incorrect grade. '
              'The threshold is descriptive, not calibrated.', '',
              '![Four confusion matrices](confusion_matrices.svg)', '',
              '## Within-model preprocessing transitions', '',
              '| Fixed model | Predictions changed | Correct → wrong | Wrong → correct | '
              'Mean probability L1 shift | Bootstrap Δ accuracy [95% CI] | Bootstrap Δ MAE [95% CI] |',
              '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    intervals = results['paired_group_bootstrap']['intervals']
    for name in PAIRS:
        entry = results['within_model'][name]
        accuracy = intervals[f'{name}_accuracy_delta']
        mae = intervals[f'{name}_mae_delta']
        lines.append(f'| {name} | {entry["prediction_changed"]} | '
                     f'{entry["correct_to_wrong"]} | {entry["wrong_to_correct"]} | '
                     f'{entry["mean_probability_l1_shift"]:.4f} | '
                     f'{accuracy["point"]:+.4f} [{accuracy["ci95_percentile"][0]:+.4f}, '
                     f'{accuracy["ci95_percentile"][1]:+.4f}] | '
                     f'{mae["point"]:+.4f} [{mae["ci95_percentile"][0]:+.4f}, '
                     f'{mae["ci95_percentile"][1]:+.4f}] |')
    lines += ['', 'The native and cross condition in each row use the **same checkpoint**. '
              'These paired changes describe sensitivity to preprocessing and do not identify a clinical cause.',
              '', '![Predicted-grade transitions](grade_transitions.svg)', '',
              '![Per-grade accuracy change](per_grade_accuracy_delta.svg)', '',
              '## Native expert complementarity and oracle upper bound', '']
    comp = results['expert_complementarity']
    lines += [f'Both correct: {comp["native_both_correct"]}; V1 only correct: '
              f'{comp["native_V1_only_correct"]}; V2 only correct: '
              f'{comp["native_V2_only_correct"]}; both wrong: {comp["native_both_wrong"]}.',
              f'Label-aware native oracle accuracy: {comp["native_oracle_accuracy"]:.4f}; '
              f'oracle MAE: {comp["native_oracle_mae"]:.4f}. '
              '**This is an unattainable upper bound without a valid selector.**',
              '', 'Native V1 versus native V2 also differs in training protocol. Expert complementarity '
              'cannot be attributed to preprocessing alone. The JSON contains all four-condition '
              'oracle bounds, five-grade support, transition counts, ordinal error, and paired '
              'group-bootstrap intervals. E1 does not overturn the previously reported E0 NO-GO '
              'decision for M2–M4.', '']
    return '\n'.join(lines)


def run(args):
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output = phase._outside_repository(args.output_dir)
    raw_root = (Path(args.artifact_root).expanduser().resolve() / 'datasets').resolve()
    if (output == cache_dir or cache_dir in output.parents or output == raw_root
            or raw_root in output.parents):
        raise ValueError('E1 output must be separate from caches and raw datasets')
    if output.exists():
        raise FileExistsError('E1 output already exists; results are never overwritten')
    cached, provenance = load_verified_cache(args.dataset, args.artifact_root, cache_dir)
    result = analyze(cached, args.dataset, provenance, args.high_confidence,
                     args.bootstrap, args.seed)
    output.mkdir(parents=True)
    (output / 'e1_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'e1_report.md').write_text(markdown_report(result), encoding='utf-8')
    (output / 'confusion_matrices.svg').write_text(heatmap_svg(result), encoding='utf-8')
    (output / 'grade_transitions.svg').write_text(heatmap_svg(result, transitions=True), encoding='utf-8')
    (output / 'per_grade_accuracy_delta.svg').write_text(grade_delta_svg(result), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'dataset': args.dataset,
                      'validation_images': len(cached), 'output_dir': str(output)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(phase.SPECS), required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--high-confidence', type=float, default=0.9)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
