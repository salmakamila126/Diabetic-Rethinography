import math
from types import SimpleNamespace
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn

from novelty_integrated.e7 import geometry, inference, model, runner
from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from novelty_integrated.e5 import config as e5config


class DummyProcessor:
    def __call__(self, *, images, do_resize, return_tensors):
        assert not do_resize and return_tensors == "pt"
        return {"pixel_values": torch.stack([torch.from_numpy(
            np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255
            for image in images])}


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, *, pixel_values):
        batch, _, height, width = pixel_values.shape
        vector = pixel_values.mean((1, 2, 3), keepdim=False)[:, None] * self.scale
        vector = vector.expand(batch, 768)
        hidden = vector[:, :, None, None].expand(batch, 768, height // 32, width // 32)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=vector)


class TestE7Integrated(unittest.TestCase):
  def test_tta_combines_whole_image_views(self):
    from PIL import ImageOps
    net = model.GlobalLocalModel(DummyBackbone(), [20, 5, 10, 3, 7])
    rgb = Image.new("RGB", (120, 64))
    rgb.paste((80, 130, 200), (5, 3, 60, 58))
    processor = DummyProcessor()
    expected = []
    with torch.inference_mode():
        for source in (rgb, ImageOps.mirror(rgb)):
            g, p, _ = geometry.model_views(source, processor)
            expected.append(net(g[None], p[None])[0].softmax(1)[0])
    actual = inference.tta_probabilities(net, processor, rgb, torch.device("cpu"))
    torch.testing.assert_close(actual, torch.stack(expected).mean(0))

  def test_strict_e5_initialization(self):
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "best.pt"
        manifest = {"manifest_sha256": "a" * 64}
        counts = [20, 5, 10, 3, 7]
        net = model.GlobalLocalModel(DummyBackbone(), counts)
        source = model.GlobalLocalModel(DummyBackbone(), counts)
        source.backbone.scale.data.fill_(2.0)
        record = {"schema": e5config.SCHEMA, "dataset": "DDR", "arm": "integrated",
                  "manifest": manifest, "train_counts": counts, "snapshot": e5config.OFFICIAL_SNAPSHOT}
        best = {"epoch": 8}
        state = {"record": record, "best": best, "epoch": 8,
                 "backbone": source.backbone.state_dict(), "heads": source.heads.state_dict()}
        fa._atomic_json(checkpoint.parent / "summary.json",
                        {"status": "VALIDATION_COMPLETE", "epochs_completed": 20, "best": best})
        fa._save_checkpoint(checkpoint, state)
        args = SimpleNamespace(init="e5", dataset="DDR", e5_checkpoint=checkpoint,
                               e5_checkpoint_sha256=phase.sha256(checkpoint), trust_e5_checkpoint=True)
        with patch.object(runner, "_safe_path", return_value=checkpoint):
            self.assertEqual(runner._load_e5_initialization(args, net, manifest, counts)["checkpoint_epoch"], 8)
            self.assertEqual(net.backbone.scale.item(), 2.0)
            args.dataset = "APTOS2019"
            with self.assertRaisesRegex(ValueError, "lineage differs"):
                runner._load_e5_initialization(args, net, manifest, counts)
            args.dataset = "DDR"
            state["heads"] = {}
            fa._save_checkpoint(checkpoint, state)
            args.e5_checkpoint_sha256 = phase.sha256(checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                runner._load_e5_initialization(args, net, manifest, counts)

  def test_geometry_and_input_equivalence(self):
    rgb = Image.new("RGB", (400, 200), (0, 0, 0))
    rgb.paste(Image.new("RGB", (300, 160), (50, 100, 150)), (50, 20))
    original, patches, boxes = geometry.pil_views(rgb)
    assert original.size == (384, 384)
    assert [image.size for image in patches] == [(224, 224)] * 2
    assert boxes[0][0] <= boxes[1][0] < boxes[0][2] <= boxes[1][2]
    fov = geometry.fov_box(rgb)
    crop_scale = 224 / max(boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1])
    global_scale = 384 / max(rgb.size)
    assert crop_scale > global_scale
    assert boxes[0][1] > fov[1] and boxes[0][3] < fov[3]
    global_pixels, patch_pixels, again = geometry.model_views(rgb, DummyProcessor())
    assert global_pixels.shape == (3, 384, 384)
    assert patch_pixels.shape == (2, 3, 224, 224)
    assert again == boxes
    with self.assertRaises(ValueError):
        geometry.pil_views(rgb, training=True)


  def test_model_gradients_and_ema(self):
    torch.manual_seed(1)
    net = model.GlobalLocalModel(DummyBackbone(), [20, 5, 10, 3, 7])
    x = torch.rand(2, 3, 384, 384)
    p = torch.rand(2, 2, 3, 224, 224)
    logits, ordinal, gate = net(x, p)
    assert logits.shape == (2, 5) and ordinal.shape == (2, 4) and gate.shape == (2, 1)
    assert 0 < gate.min() and gate.max() < 1
    loss = logits.square().mean() + ordinal.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in net.parameters())
    ema = model.ModelEMA(net, 0.9)
    ema.update(net)
    assert ema.updates == 1
    with self.assertRaises(ValueError):
        model.pooled_convnext(net.backbone, x, 7)


  def test_ensemble_calibration_and_split_guard(self):
    sample = {"dataset": "APTOS2019", "manifest_sha256": "a" * 64,
        "image_ids": ["a", "b"], "class_order": [0, 1, 2, 3, 4],
        "split": "val", "checkpoint_sha256": "b" * 64,
        "probabilities": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]}
    result = inference.ensemble([sample, sample], [0.25, 0.75])
    assert np.array_equal(result, np.asarray(sample["probabilities"]))
    with self.assertRaises(ValueError):
        inference.ensemble([sample, {**sample, "image_ids": ["b", "a"]}], [0.5, 0.5])
    with self.assertRaises(ValueError):
        inference.select_temperature([[1, 0, 0, 0, 0]], [0], split="test",
                                     sample_ids=["cal"], forbidden_ids=["val"])
    with self.assertRaises(ValueError):
        inference.select_temperature([[1, 0, 0, 0, 0]], [0], split="calibration",
                                     sample_ids=["val"], forbidden_ids=["val"])
    selected = inference.select_temperature([[1, 0, 0, 0, 0]], [0], split="calibration",
                                            sample_ids=["cal"], forbidden_ids=["val"])
    assert selected in (0.75, 1.0, 1.25)
    assert np.allclose(inference.temperature_probabilities([[1, 0, 0, 0, 0]], 1).sum(1), 1)


class TinyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.categorical = nn.Parameter(torch.zeros(5))
        self.ordinal = nn.Parameter(torch.zeros(4))

    def forward(self, global_pixels, patches):
        n = len(global_pixels)
        return self.categorical.expand(n, 5), self.ordinal.expand(n, 4), torch.zeros(n, 1)


class TestE7Accumulation(unittest.TestCase):
  def test_incomplete_accumulation_window(self):
    net = TinyTrainModel()
    optim = torch.optim.SGD(net.parameters(), lr=0.1)
    batches = []
    for labels in ([0, 1], [2, 3], [4]):
        n = len(labels)
        batches.append((torch.zeros(n, 1), torch.zeros(n, 1),
                        torch.tensor(labels, dtype=torch.long), list(range(n))))
    result = runner._train_batches(net, optim, None, batches, device=torch.device("cpu"),
        precision="fp32", counts=[10] * 5, epoch=1, total_images=5,
        micro_batch=2, accumulation=2)
    assert result["accumulation_window_images"] == [4, 1]
    assert result["optimizer_steps"] == 2 and math.isfinite(result["loss"])

  def test_resume_requires_exact_record_and_checkpoint(self):
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        net = TinyTrainModel()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        record = {"schema": "synthetic-e7", "dataset": "APTOS2019"}
        metric = {"macro_f1": 0.25, "validation_loss": 1.0}
        best = {"epoch": 1, "validation": metric}
        history = [{"epoch": 1, "validation": metric}]
        state = {"record": record, "epoch": 1, "model": net.state_dict(),
                 "optimizer": optimizer.state_dict(), "ema": None,
                 "rng": fa._rng_state(), "history": history, "best": best}
        fa._atomic_json(out / "run.json", record)
        fa._atomic_json(out / "history.json", history)
        fa._save_checkpoint(out / "last.pt", state)
        fa._save_checkpoint(out / "best.pt", state)
        resumed = runner._resume(out, record, net, optimizer, None, trust=True)
        assert resumed[2] == 2 and resumed[1] == best
        with self.assertRaises(ValueError):
            runner._resume(out, {"schema": "different"}, net, optimizer, None, trust=True)
