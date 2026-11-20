"""CPU-only E6 objective and provenance gates; no image datasets or checkpoints."""
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from novelty_integrated.e5.model import E5Heads
from novelty_integrated.e6 import config, objective, runner

COUNTS = (100, 5, 40, 3, 10)


class TestE6Objective(unittest.TestCase):
    def test_ldam_margins_and_order(self):
        margins = config.ldam_margins(COUNTS)
        self.assertAlmostEqual(max(margins), 0.5)
        self.assertGreater(margins[3], margins[0])
        self.assertEqual(config.WEIGHT_START_EPOCH, 11)
        self.assertIsNone(config.weights_for_epoch(COUNTS, 10))
        weights = config.weights_for_epoch(COUNTS, 11)
        self.assertAlmostEqual(sum(n*w for n,w in zip(COUNTS,weights))/sum(COUNTS), 1.0)
        self.assertGreater(weights[3], weights[0])

    def test_true_class_margin_and_deferred_weight(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], requires_grad=True)
        ordinal = torch.zeros((1,4), requires_grad=True)
        labels = torch.tensor([1])
        result = objective.loss_components(logits, ordinal, labels, train_counts=COUNTS, epoch=1)
        adjusted = logits.detach().clone()
        adjusted[0,1] -= config.ldam_margins(COUNTS)[1]
        self.assertTrue(torch.allclose(result["categorical"],
                         F.cross_entropy(30 * adjusted, labels)))
        weighted = objective.loss_components(logits, ordinal, labels, train_counts=COUNTS, epoch=11)
        self.assertAlmostEqual((weighted["categorical"] / result["categorical"]).item(),
                               config.effective_number_weights(COUNTS)[1], places=5)
        result["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(ordinal.grad).all())

    def test_adjacent_boundary_only_012(self):
        logits = torch.zeros((5,5), requires_grad=True)
        ordinal = torch.zeros((5,4), requires_grad=True)
        result = objective.loss_components(logits, ordinal, torch.arange(5),
                                           train_counts=COUNTS, epoch=1)
        self.assertAlmostEqual(result["adjacent"].item(), 3 * math.log(2) / 30 / 5, places=6)
        result["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bfloat16_inputs_have_finite_fp32_loss(self):
        logits = torch.randn(4, 5, dtype=torch.bfloat16, requires_grad=True)
        ordinal = torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True)
        result = objective.loss_components(logits, ordinal, torch.tensor([0,1,2,3]),
                                           train_counts=COUNTS, epoch=11)
        self.assertEqual(result["total"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["total"]))
        result["total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(ordinal.grad).all())

    def test_monotone_head_and_categorical_predictions(self):
        heads = E5Heads(768, COUNTS, ordinal=True)
        features = torch.randn(4,768)
        categorical, ordinal = heads(features)
        self.assertEqual(tuple(categorical.shape), (4,5))
        self.assertEqual(tuple(ordinal.shape), (4,4))
        self.assertTrue(torch.all(heads.ordinal.thresholds()[1:] > heads.ordinal.thresholds()[:-1]))
        self.assertTrue(torch.equal(categorical, heads.predict_logits(features)))

    def test_incomplete_accumulation_window(self):
        backbone = torch.nn.Linear(2, 768)
        heads = E5Heads(768, COUNTS, ordinal=True)
        optimizer = torch.optim.SGD(list(backbone.parameters()) + list(heads.parameters()), lr=1e-4)
        pixels = torch.randn(5, 2)
        labels = torch.arange(5)
        batches = [(pixels[i:i+2], labels[i:i+2]) for i in (0,2,4)]
        with patch.object(runner.fa, "pooled", side_effect=lambda model, x, kind: model(x)):
            report = runner._train_batches(backbone, heads, optimizer, batches,
                device=torch.device("cpu"), precision="fp32", total_images=5,
                micro_batch=2, accumulation=2, train_counts=COUNTS, epoch=1)
        self.assertEqual(report["accumulation_window_images"], [4, 1])
        self.assertEqual(report["optimizer_steps"], 2)
        self.assertTrue(math.isfinite(report["loss"]))

    def test_train_gate_rejects_before_model_or_data(self):
        class Args:
            smoke_report = None
            benchmark_report = None
        with patch.object(runner, "_prepare", side_effect=AssertionError("must not load")):
            with self.assertRaises(ValueError):
                runner.train(Args())

    def test_no_test_command(self):
        with self.assertRaises(SystemExit):
            runner.main(["test"])


if __name__ == "__main__":
    unittest.main()
