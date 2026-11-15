"""Synthetic-only checks for E3 transforms, selection, and paired analysis."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from novelty import preof_e3 as e3
from novelty import preof_phase1 as phase
from reproduction.cadvit.preprocessing import preprocess_fundus, preprocess_fundus_v2


class E3Tests(unittest.TestCase):
    def test_historical_endpoints_and_condition_types(self):
        rng = np.random.default_rng(12)
        raw = rng.integers(20, 255, size=(600, 640, 3), dtype=np.uint8)
        image = Image.fromarray(raw, "RGB")
        v1, v2 = preprocess_fundus(image), preprocess_fundus_v2(image)
        self.assertIs(e3.condition_pixels(v1, v2, "v1"), v1)
        self.assertIs(e3.condition_pixels(v1, v2, "v2_bg10"), v2)
        np.testing.assert_array_equal(e3.bg_enhance(v1, 10), v2)
        self.assertEqual(len(e3.CONDITIONS), 11)
        for condition in e3.CONDITIONS:
            result = e3.condition_pixels(v1, v2, condition)
            self.assertEqual(result.shape, (512, 512, 3))
            self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(e3.condition_pixels(v1, v2, "mix050"),
                                      np.rint((v1.astype(np.float32) + v2.astype(np.float32)) / 2).astype(np.uint8))

    def test_grouped_grade_stratified_pilot_is_deterministic(self):
        rows = [{"image_id": str(i), "group_id": f"g{i // 2}", "label": str((i // 2) % 5)}
                for i in range(40)]
        selected, meta = e3.select_pilot(rows, "DDR", groups_per_grade=2, seed=42)
        selected_again, meta_again = e3.select_pilot(rows, "DDR", groups_per_grade=2, seed=42)
        self.assertEqual((selected, meta), (selected_again, meta_again))
        self.assertEqual(sum(meta["grade_support"].values()), len(selected))
        self.assertTrue(all(count == 4 for count in meta["grade_support"].values()))
        self.assertEqual({r["group_id"] for r in selected}, set(meta["selected_groups"]))
        self.assertTrue(all(sum(r["group_id"] == group for r in selected) == 2
                            for group in meta["selected_groups"]))

    def test_paired_uncertainty_and_report_are_valid(self):
        truth = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        native = truth.copy()
        changed = truth.copy()
        changed[::2] = (changed[::2] + 1) % 5
        groups = np.array([f"g{i // 2}" for i in range(10)])
        intervals = e3._paired_intervals(truth, groups, native, changed, 30, 42)
        self.assertAlmostEqual(intervals["accuracy_delta"]["point"], -.5)
        self.assertEqual(intervals, e3._paired_intervals(truth, groups, native, changed, 30, 42))
        metrics = {name: {"accuracy": .8, "macro_f1": .7, "quadratic_weighted_kappa": .75,
                          "grade_mae": .3, "high_confidence_wrong": 1} for name in e3.CONDITIONS}
        result = {"dataset": "DDR", "selection": {"sample_count": 10,
                  "selected_groups": sorted(set(groups)), "grade_support": {str(i): 2 for i in range(5)}},
                  "models": {"v1": {"conditions": metrics}, "v2": {"conditions": metrics}}}
        with TemporaryDirectory() as temp:
            e3._write_report(Path(temp), result)
            ElementTree.parse(Path(temp) / "interpolation_response.svg")
            ElementTree.parse(Path(temp) / "strength_response.svg")
            self.assertIn("Validation only", (Path(temp) / "report.md").read_text())

    def test_synthetic_pilot_resume_and_cpu_analysis(self):
        import torch

        class TinyFrozen:
            def __init__(self, bias):
                self.bias = bias

            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor):
                mean = tensor.mean() + self.bias
                return torch.stack([mean * (i + 1) for i in range(5)])[None]

        v1 = np.full((512, 512, 3), 160, np.uint8)
        v2 = e3.bg_enhance(v1, 10)
        models = {"v1": TinyFrozen(.01), "v2": TinyFrozen(.02)}

        def prediction(model, pixels):
            tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()[None] / 255
            logits = model(tensor)[0].tolist()
            return {"logits": logits, "probabilities": e3._probabilities(logits)}

        rows = [{"image_id": f"i{i}", "group_id": f"g{i}", "label": str(i),
                 "sha256": f"{i:064x}"} for i in range(5)]
        cached = []
        for row in rows:
            item = {"image_id": row["image_id"]}
            for key, model_name, pixels in (("p11", "v1", v1), ("p12", "v1", v2),
                                            ("p21", "v2", v1), ("p22", "v2", v2)):
                item[key] = prediction(models[model_name], pixels)
            cached.append(item)
        provenance = {"checkpoint_sha256": {v: phase.SPECS["APTOS2019"][f"{v}_sha256"]
                                                for v in ("v1", "v2")},
                      "model_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/model.py"),
                      "preprocessing_source_sha256": phase.sha256(phase.REPO / "reproduction/cadvit/preprocessing.py"),
                      "v2_historical_source": {"sha256": "a" * 64}}
        hashes = {"checkpoint_sha256": provenance["checkpoint_sha256"],
                  "model_source_sha256": provenance["model_source_sha256"],
                  "preprocessing_source_sha256": provenance["preprocessing_source_sha256"],
                  "historical_v2_source_sha256": "a" * 64,
                  "metadata_sha256": "b" * 64, "cache_jsonl_sha256": "c" * 64,
                  "manifest_sha256": phase.SPECS["APTOS2019"]["manifest_sha256"]}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(dataset="APTOS2019", artifact_root=str(root),
                                   prediction_cache_dir=str(root / "predictions"),
                                   preprocess_cache_root=str(root / "preprocessing"),
                                   output_dir=str(root / "e3"), v2_source=str(root / "v2.ipynb"),
                                   trust_checkpoints=True, device="cpu", groups_per_grade=1,
                                   seed=42, atol=1e-4, rtol=1e-4, resume=False)
            with (patch.object(e3.e1, "load_verified_cache", return_value=(cached, hashes)),
                  patch.object(e3.phase, "validation_rows", return_value=rows),
                  patch.object(e3.phase, "PreprocessingCache", return_value=object()),
                  patch.object(e3.phase, "checked_models", return_value=(models, provenance)),
                  patch.object(e3, "_read_pixels", side_effect=lambda _cache, _sha, variant:
                               v1 if variant == "v1" else v2)):
                e3.pilot(args)
                original = (root / "e3" / "rows.jsonl").read_bytes()
                args.resume = True
                e3.pilot(args)
                self.assertEqual((root / "e3" / "rows.jsonl").read_bytes(), original)
                e3.analyze(SimpleNamespace(output_dir=args.output_dir, artifact_root=args.artifact_root,
                                           prediction_cache_dir=args.prediction_cache_dir,
                                           high_confidence=.9, bootstrap=30, seed=42))
            result = json.loads((root / "e3" / "analysis.json").read_text())
            self.assertEqual(result["selection"]["sample_count"], 5)
            self.assertEqual(set(result["models"]), {"v1", "v2"})
            self.assertEqual(result["status"], "EXPLORATORY_VALIDATION_PILOT")


if __name__ == "__main__":
    unittest.main()
