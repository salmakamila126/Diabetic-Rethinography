"""Synthetic and paper-reference checks; no dataset or final-test inference."""

import unittest
from dataclasses import asdict
from importlib.util import find_spec

import numpy as np
import torch
from PIL import Image

from reproduction.cadvit.interpretability import gradcam
from reproduction.cadvit.experiments import BALANCED_TRAINING, Protocol, VARIANTS
from reproduction.cadvit.model import CADViTCandidate
from reproduction.cadvit.visualization import (comparable_balancing_metrics, confusion_png,
                                               error_records, guarded_non_test_image,
                                               paper_binary_reference, paper_five_reference,
                                               preprocessing_stages)


class VisualizationTests(unittest.TestCase):
    def test_paper_matrices_and_weighted_f1(self):
        binary = paper_binary_reference()["metrics"]
        self.assertEqual(binary["confusion_matrix"], [[181, 0], [3, 183]])
        self.assertAlmostEqual(binary["accuracy"], 364 / 367)
        aptos = paper_five_reference("APTOS2019")["metrics"]
        ddr = paper_five_reference("DDR")["metrics"]
        self.assertEqual((aptos["n"], ddr["n"]), (367, 1253))
        self.assertAlmostEqual(aptos["f1_weighted"], 0.8842215883, places=8)
        self.assertAlmostEqual(ddr["f1_weighted"], 0.8443922537, places=8)
        self.assertEqual(aptos["f1_micro"], aptos["accuracy"])
        image = confusion_png(aptos, [str(i) for i in range(5)],
                              "PAPER_REFERENCE Figure 5 APTOS2019")
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(image.size[0], 500)

    def test_error_type_uses_binary_semantics(self):
        payload = {"split": "test", "variant": "aptos_binary_head",
                   "metrics": {"predictions": [
                       {"true_label": 0, "predicted_label": 1, "confidence": 0.7},
                       {"true_label": 1, "predicted_label": 0, "confidence": 0.8},
                       {"true_label": 1, "predicted_label": 1, "confidence": 0.9}]}}
        result = error_records(payload)
        self.assertEqual([row["error_type"] for row in result["errors"]], ["FP", "FN"])
        projected = {**payload, "variant": "ddr_full",
                     "metrics": {"predictions": [
                         {"true_label": 0, "predicted_label": 3, "confidence": 0.7},
                         {"true_label": 2, "predicted_label": 0, "confidence": 0.8},
                         {"true_label": 2, "predicted_label": 3, "confidence": 0.9}]}}
        self.assertEqual([x["error_type"] for x in error_records(projected,
                          binary_projection=True)["errors"]], ["FP", "FN"])
        self.assertEqual(len(error_records(projected)["errors"]), 3)

    def test_visual_guard_rejects_test_before_opening_image(self):
        with self.assertRaises(ValueError):
            guarded_non_test_image("missing.png", ".", "missing.csv", "test")

    @unittest.skipUnless(find_spec("cv2"), "OpenCV unavailable locally")
    def test_figure_1_stages_preserve_historical_v1_pixels(self):
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[5:44, 7:58] = [90, 140, 180]
        pixels[20:30, 25:40] = [160, 100, 70]
        stages = preprocessing_stages(Image.fromarray(pixels))
        self.assertEqual(stages["crop_xywh"], (7, 5, 51, 39))
        self.assertTrue(np.array_equal(stages["gaussian_blur"], stages["historical_v1"]))
        self.assertEqual(stages["historical_v1"].shape, (512, 512, 3))
        self.assertEqual(stages["historical_v2_bg10"].dtype, np.uint8)

    def test_gradcam_synthetic_cpu(self):
        torch.set_num_threads(1)
        model = CADViTCandidate().train()
        result = gradcam(model, torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(result["heatmap"].shape), (64, 64))
        self.assertTrue(bool(torch.isfinite(result["heatmap"]).all()))
        self.assertGreaterEqual(float(result["heatmap"].min()), 0.0)
        self.assertLessEqual(float(result["heatmap"].max()), 1.0)
        self.assertTrue(model.training)
        self.assertFalse(model.blocks[5].attention._forward_hooks)

    def test_figure_6_pair_rejects_other_protocol_changes(self):
        metric = paper_five_reference("APTOS2019")["metrics"]
        base_config = {"variant": asdict(VARIANTS["aptos_r16"]),
                       "manifest_sha256": "a" * 64, "model_sha256": "b" * 64,
                       "preprocessing_sha256": "c" * 64}
        before = {"status": "AVAILABLE", "config": {**base_config,
                  "protocol": asdict(Protocol())},
                  "result": {"split": "test", "metrics": metric}}
        after = {"status": "AVAILABLE", "config": {**base_config,
                 "protocol": asdict(Protocol(balancing=BALANCED_TRAINING))},
                 "result": {"split": "test", "metrics": metric}}
        self.assertEqual(comparable_balancing_metrics(before, after)["balanced"][0],
                         100 * metric["accuracy"])
        after["config"]["protocol"]["learning_rate"] = 0.001
        with self.assertRaisesRegex(ValueError, "balancing only"):
            comparable_balancing_metrics(before, after)


if __name__ == "__main__":
    unittest.main()
