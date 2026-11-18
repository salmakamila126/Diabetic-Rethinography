"""CPU-only checks for checkpoint-compatible CAD-ViT research ablations."""
import hashlib
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novelty.cadvit_model import (  # noqa: E402
    AdaptiveMultiScaleFeatureBlock, BoundaryCoordinateAttention,
    NovelCADViT, NoveltyFlags, PRESETS, parameter_counts,
)
from novelty.cadvit_training import (  # noqa: E402
    ordinal_targets, research_loss, restricted_checkpoint_state,
    transfer_baseline_state,
)
from reproduction.cadvit.model import CADViTCandidate  # noqa: E402


ARCHIVE = ROOT.parent / "experiment_archive_80ep_20261105"
CHECKPOINTS = {
    "APTOS": (ARCHIVE / "APTOS" / "best.pt",
              "e72573f0f1c26baac53e18d950076845952da503261e212021ff6320c9ae72d5"),
    "DDR": (ARCHIVE / "DDR" / "best.pt",
            "844665f0c597142915286b727a21f1d2223f43e0f057d083b6c1dd035c180ac1"),
}
BASELINE_SOURCE_SHA = "6aff75039dbd2d3e545e6ff1998674221b3671df7d58362792e0b2ca249a0c25"


class CADViTNoveltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reproduction_source_unchanged(self):
        path = ROOT / "reproduction" / "cadvit" / "model.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), BASELINE_SOURCE_SHA)

    def test_presets_and_parameter_counts(self):
        expected = {
            "baseline": 5282101, "p1_only": 5282965,
            "p2_only": 5283649, "ordinary_gate_control": 5283649,
            "p3_probe_only": 5282761, "p3_only": 5285961,
            "p4_only": 5283385, "p2_p3_probe": 5284309,
            "p2_p3_full": 5287509, "p123": 5288373,
            "p1234": 5289657,
        }
        self.assertEqual(set(PRESETS), set(expected))
        for name, flags in PRESETS.items():
            with self.subTest(name=name):
                model = NovelCADViT(flags)
                self.assertEqual(parameter_counts(model)["total"], expected[name])
                self.assertEqual(parameter_counts(model)["trainable"], expected[name])
                self.assertEqual(model.configuration(), asdict(flags))
                self.assertEqual(len([m for m in model.modules()
                                      if isinstance(m, AdaptiveMultiScaleFeatureBlock)]),
                                 3 if flags.p2_adaptive_mfb else 0)
                self.assertEqual(len([m for m in model.modules()
                                      if isinstance(m, BoundaryCoordinateAttention)]),
                                 4 if flags.p3_boundary_probe else 0)
                self.assertEqual(hasattr(model, "detail_path"), flags.p1_detail)
                self.assertEqual(hasattr(model, "ordinal_head"), flags.p4_ordinal_head)
        self.assertEqual(PRESETS["p2_only"], PRESETS["ordinary_gate_control"])
        with self.assertRaises(ValueError):
            NoveltyFlags(p3_modulation=True)
        with self.assertRaises(TypeError):
            NoveltyFlags(p1_detail=1)

    def test_baseline_equivalence_at_initialization(self):
        torch.manual_seed(42)
        baseline = CADViTCandidate().eval()
        state = baseline.state_dict()
        x = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            reference = baseline(x)
        for name, flags in PRESETS.items():
            with self.subTest(name=name):
                model = NovelCADViT(flags).eval()
                report = transfer_baseline_state(model, state)
                self.assertFalse(report["missing_original_tensors"])
                self.assertFalse(report["unexpected_checkpoint_tensors"])
                with torch.inference_mode():
                    observed = model(x)
                torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    def test_shapes_finiteness_and_no_label_input(self):
        model = NovelCADViT(PRESETS["p1234"]).eval()
        for batch in (1, 2):
            with self.subTest(batch=batch), torch.inference_mode():
                x = torch.randn(batch, 3, 64, 64)
                out = model.forward_with_aux(x)
                self.assertEqual(out.logits.shape, (batch, 5))
                self.assertEqual(out.ordinal_logits.shape, (batch, 4))
                self.assertEqual(len(out.boundary_logits), 4)
                self.assertTrue(torch.isfinite(out.logits).all())
                self.assertTrue(torch.isfinite(out.ordinal_logits).all())
                self.assertTrue(all(z.shape == (batch, 4) and torch.isfinite(z).all()
                                    for z in out.boundary_logits))
                torch.testing.assert_close(model(x), out.logits, rtol=0, atol=0)
                _, shapes = model.forward_features(x, record_shapes=True)
                self.assertEqual(shapes["stem"][0], batch)
                self.assertEqual(shapes["head"], (batch, 320, 2, 2))

    def test_gates_thresholds_and_gradients(self):
        model = NovelCADViT(PRESETS["p1234"]).train()
        self.assertEqual(
            [tuple(m.rates) for m in model.modules()
             if isinstance(m, AdaptiveMultiScaleFeatureBlock)],
            [(3, 5, 7, 9)] * 3,
        )
        for module in model.modules():
            if isinstance(module, AdaptiveMultiScaleFeatureBlock):
                torch.testing.assert_close(
                    module.branch_coefficients(torch.randn(2, module.gate.in_features, 4, 4)),
                    torch.ones(2, 4), atol=0, rtol=0,
                )
            if isinstance(module, BoundaryCoordinateAttention):
                threshold = module.ordered_thresholds()
                self.assertTrue(bool(torch.all(threshold[1:] > threshold[:-1])))
                self.assertTrue(bool(torch.all(module.boundary_modulation.weight == 0)))
        labels = torch.tensor([0, 4])
        out = model.forward_with_aux(torch.randn(2, 3, 64, 64))
        losses = research_loss(out, labels, model.flags)
        self.assertTrue(all(bool(torch.isfinite(v)) for v in losses.values()))
        losses["total"].backward()
        names = (
            "detail_path.projection.weight",
            "blocks.4.mfb.gate.weight",
            "blocks.1.attention.severity_score.weight",
            "blocks.1.attention.threshold_start",
            "blocks.1.attention.boundary_modulation.weight",
            "ordinal_head.weight",
        )
        parameters = dict(model.named_parameters())
        for name in names:
            with self.subTest(gradient=name):
                self.assertIsNotNone(parameters[name].grad)
                self.assertTrue(torch.isfinite(parameters[name].grad).all())
                self.assertGreater(parameters[name].grad.abs().sum().item(), 0)

    def test_ordinal_encoding_and_loss_ablation(self):
        expected = torch.tensor([
            [0., 0., 0., 0.], [1., 0., 0., 0.],
            [1., 1., 0., 0.], [1., 1., 1., 0.],
            [1., 1., 1., 1.],
        ])
        torch.testing.assert_close(ordinal_targets(torch.arange(5)), expected)
        with self.assertRaises(ValueError):
            ordinal_targets(torch.tensor([5]))
        with self.assertRaises(TypeError):
            ordinal_targets(torch.tensor([1.0]))
        labels = torch.tensor([0, 4])
        x = torch.randn(2, 3, 64, 64)
        for name in ("baseline", "p3_probe_only", "p3_only", "p4_only", "p1234"):
            with self.subTest(name=name):
                model = NovelCADViT(PRESETS[name]).eval()
                out = model.forward_with_aux(x)
                losses = research_loss(out, labels, model.flags)
                expected_loss = losses["cross_entropy"]
                if model.flags.p3_boundary_probe:
                    expected_loss = expected_loss + 0.2 * losses["boundary_bce"]
                else:
                    self.assertEqual(losses["boundary_bce"].item(), 0)
                if model.flags.p4_ordinal_head:
                    expected_loss = expected_loss + 0.2 * losses["ordinal_bce"]
                else:
                    self.assertEqual(losses["ordinal_bce"].item(), 0)
                torch.testing.assert_close(losses["total"], expected_loss, atol=0, rtol=0)

    def test_probe_only_and_train_mode_batchnorm_equivalence(self):
        self.assertEqual(PRESETS["p3_probe_only"].p3_modulation, False)
        probe = NovelCADViT(PRESETS["p3_probe_only"])
        self.assertFalse(any("boundary_modulation" in name for name, _ in probe.named_parameters()))
        self.assertEqual(len([m for m in probe.modules()
                              if isinstance(m, BoundaryCoordinateAttention)]), 4)
        torch.manual_seed(23)
        baseline = CADViTCandidate()
        extended = NovelCADViT(PRESETS["p1234"])
        transfer_baseline_state(extended, baseline.state_dict())
        x = torch.randn(2, 3, 64, 64)
        baseline.train()
        extended.train()
        with torch.no_grad():
            expected = baseline(x)
            actual = extended(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        original_bn = {k: v for k, v in baseline.state_dict().items()
                       if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        extended_bn = extended.state_dict()
        for name, value in original_bn.items():
            with self.subTest(batchnorm=name):
                torch.testing.assert_close(extended_bn[name], value, rtol=0, atol=0)

    def test_synthetic_optimizer_step_updates_new_paths(self):
        torch.manual_seed(29)
        model = NovelCADViT(PRESETS["p1234"]).train()
        names = (
            "detail_path.projection.weight",
            "blocks.4.mfb.gate.weight",
            "blocks.1.attention.severity_score.weight",
            "blocks.1.attention.threshold_start",
            "blocks.1.attention.boundary_modulation.weight",
            "ordinal_head.weight",
        )
        before = {name: dict(model.named_parameters())[name].detach().clone() for name in names}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_with_aux(torch.randn(2, 3, 64, 64))
        loss = research_loss(out, torch.tensor([0, 4]), model.flags)["total"]
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        optimizer.step()
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        for name in names:
            with self.subTest(updated=name):
                parameter = dict(model.named_parameters())[name]
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertFalse(torch.equal(parameter.detach(), before[name]))

    def test_transfer_rejects_missing_extra_and_shape_mismatch(self):
        state = CADViTCandidate().state_dict()
        key = next(iter(state))
        model = NovelCADViT(PRESETS["p1234"])
        for mutation in ("missing", "extra", "shape"):
            altered = dict(state)
            if mutation == "missing":
                del altered[key]
            elif mutation == "extra":
                altered["invented.weight"] = torch.zeros(1)
            else:
                altered[key] = torch.zeros(1)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                transfer_baseline_state(model, altered)

    def test_verified_archived_checkpoints(self):
        for dataset, (path, expected_hash) in CHECKPOINTS.items():
            with self.subTest(dataset=dataset):
                if not path.is_file():
                    self.skipTest(f"KC checkpoint unavailable: {path}")
                state, provenance = restricted_checkpoint_state(path, expected_hash)
                self.assertEqual(provenance["sha256"], expected_hash)
                base = CADViTCandidate().eval()
                base.load_state_dict(state, strict=True)
                extended = NovelCADViT(PRESETS["p1234"]).eval()
                report = transfer_baseline_state(extended, state)
                self.assertEqual(report["original_tensors_loaded"], len(state))
                self.assertTrue(report["newly_initialized_tensors"])
                x = torch.randn(1, 3, 64, 64)
                with torch.inference_mode():
                    expected = base(x)
                    actual = extended(x)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                base.train()
                extended.train()
                train_input = torch.randn(2, 3, 64, 64)
                with torch.no_grad():
                    train_expected = base(train_input)
                    train_actual = extended(train_input)
                torch.testing.assert_close(train_actual, train_expected, atol=0, rtol=0)
                for name, value in base.state_dict().items():
                    if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
                        torch.testing.assert_close(extended.state_dict()[name], value,
                                                   atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
