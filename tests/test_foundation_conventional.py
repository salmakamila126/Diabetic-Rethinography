import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch

from novelty import foundation_adaptation as fa
from novelty import foundation_augmentation as aug
from novelty import preof_phase1 as phase


class ConventionalBaselineTests(unittest.TestCase):
    def test_training_only_weight_formula(self):
        train = [{"label": grade} for grade, count in enumerate((20, 5, 3, 2, 1))
                 for _ in range(count)]
        counts, weights = fa._class_weights(train)
        self.assertEqual(counts, [20, 5, 3, 2, 1])
        self.assertAlmostEqual(np.dot(counts, weights) / sum(counts), 1.0, places=12)
        self.assertGreater(weights[4], weights[0])
        with self.assertRaises(ValueError):
            fa._class_weights(train[:-1])

    def test_weighted_accumulation_matches_reference_incomplete_window(self):
        torch.manual_seed(31)
        features = torch.randn(7, 384)
        labels = torch.tensor([0, 0, 1, 2, 3, 4, 4])
        actual, reference = torch.nn.Linear(384, 5), torch.nn.Linear(384, 5)
        reference.load_state_dict(actual.state_dict())
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.03)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        weights = [0.4, 0.8, 1.0, 1.2, 1.6]
        batches = [(features[i:i + 2], labels[i:i + 2]) for i in range(0, 7, 2)]
        info = fa._train_epoch(None, actual, "frozen", "dinov3_vits16", batches,
                               optimizer, None, torch.device("cpu"), "fp32", 2,
                               1e9, 4, 7, 2, class_weights=weights)
        for start, end in ((0, 4), (4, 7)):
            ref_optimizer.zero_grad(set_to_none=True)
            per_image = torch.nn.functional.cross_entropy(
                reference(features[start:end]), labels[start:end], reduction="none")
            loss = (per_image * torch.tensor(weights)[labels[start:end]]).sum() / (end - start)
            loss.backward()
            ref_optimizer.step()
        self.assertEqual(info["accumulation_window_images"], [4, 3])
        for got, wanted in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(got, wanted, atol=1e-6, rtol=1e-6)

    def test_augmentation_reproducible_by_epoch_and_preserves_fov_proxy(self):
        image = Image.new("RGB", (96, 96), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 84, 84), fill=(120, 60, 30))
        results = []
        for epoch in range(1, 7):
            first, a = aug.apply(image, seed=42, epoch=epoch, image_id="grade2")
            second, b = aug.apply(image, seed=42, epoch=epoch, image_id="grade2")
            self.assertEqual(a, b)
            self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
            self.assertGreaterEqual(a["fov_proxy_retention"], 0.99)
            results.append(first.tobytes())
        self.assertGreater(len(set(results)), 1)
        self.assertEqual(aug.PROTOCOL["rotation_degrees"], [-7.0, 7.0])

    def test_b1_bypasses_train_cache_but_can_cache_validation(self):
        rows = {"train": [{"label": i, "image_id": f"train{i}", "split": "train"}
                          for i in range(5)],
                "val": [{"label": 0, "split": "val"}]}
        class Cache:
            def batch(self, selected):
                if any(row["split"] != "val" for row in selected):
                    raise AssertionError("B1 must not use final-tensor train cache")
                return torch.zeros(len(selected), 3, 224, 224), torch.tensor([0])
        calls = []
        def image_batch(selected, processor, **kwargs):
            calls.append(kwargs)
            return torch.zeros(len(selected), 3, 224, 224), torch.tensor(
                [row["label"] for row in selected])
        with patch.object(fa, "_image_batch", side_effect=image_batch):
            list(fa._epochs_batches("full", rows, object(), None, 2, 42, 2,
                                    input_cache=Cache(), experiment="B1"))
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call == {"augment_seed": 42, "augment_epoch": 2}
                                for call in calls))
            list(fa._validation_batches("full", rows, object(), None, 2, input_cache=Cache()))

    def test_gallery_uses_only_training_grades_outside_git(self):
        class Processor:
            def __call__(self, *, images, return_tensors):
                arrays = [np.asarray(image.resize((224, 224)), dtype=np.float32)
                          for image in images]
                return {"pixel_values": torch.from_numpy(
                    np.stack(arrays).transpose(0, 3, 1, 2).copy())}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for grade in range(5):
                path = root / f"grade{grade}.png"
                image = Image.new("RGB", (48, 48), "black")
                ImageDraw.Draw(image).ellipse((5, 5, 43, 43),
                                              fill=(80 + grade * 10, 40, 20))
                image.save(path)
                rows.append({"label": grade, "split": "train", "image_id": f"g{grade}",
                             "path": str(path), "sha256": phase.sha256(path)})
            report = aug.inspect_gallery(rows, Processor(), root / "gallery", seed=42,
                                         epoch=1, samples_per_grade=1)
            self.assertEqual([r["grade"] for r in report["rows"]], list(range(5)))
            self.assertEqual(len(report["rows"]), 5)
            self.assertEqual(phase.sha256(root / "gallery" / "augmentation_gallery.png"),
                             report["image_sha256"])
            rows[0]["split"] = "val"
            with self.assertRaises(ValueError):
                aug.inspect_gallery(rows, Processor(), root / "invalid", seed=42,
                                    epoch=1, samples_per_grade=1)

    def test_b1_b2_one_epoch_records_provenance_without_extra_inference(self):
        from types import SimpleNamespace

        class TinyConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(768))
                self.config = SimpleNamespace(model_type="convnext")
                self.forward_calls = 0

            def forward(self, pixel_values):
                self.forward_calls += 1
                pooled = pixel_values.mean(dim=(1, 2, 3))[:, None] * self.weight[None]
                return SimpleNamespace(pooler_output=pooled,
                                       last_hidden_state=pooled[:, :, None, None].expand(
                                           -1, -1, 7, 7))

        with TemporaryDirectory() as temp:
            root = Path(temp)
            train_labels = [0, 0, 0, 1, 1, 2, 3, 4]
            rows = {"train": [{"label": label, "image_id": f"t{i}", "split": "train"}
                              for i, label in enumerate(train_labels)],
                    "val": [{"label": i, "image_id": f"v{i}", "split": "val"}
                            for i in range(5)]}
            manifest = {"manifest_sha256": "a" * 64, "dataset": "DDR",
                        "grade_support": {"train": [3, 2, 1, 1, 1]}}
            snapshot = {"hub_id": fa.MODELS["convnext_tiny"]["hub_id"],
                        "revision": "b" * 40, "config_sha256": "c" * 64,
                        "processor_config_sha256": "d" * 64,
                        "weight_file": "pytorch_model.bin", "weight_sha256": "e" * 64}
            for arm in ("B1", "B2"):
                with self.subTest(arm=arm):
                    smoke = {"schema": fa.SCHEMA, "status": "BATCH_SMOKE_PASS",
                             "dataset": "DDR", "model_kind": "convnext_tiny", "mode": "full",
                             "snapshot_weight_sha256": "e" * 64, "snapshot_revision": "b" * 40,
                             "processor_config_sha256": "d" * 64, "manifest_sha256": "a" * 64,
                             "device_type": "cpu", "batch_size": 1, "precision": "fp32",
                             "optimizer_steps": 2, "weights_saved": False,
                             "implementation_sha256": phase.sha256(fa.__file__),
                             "experiment": arm}
                    if arm == "B1":
                        smoke["augmentation"] = fa._augmentation_metadata("B1")
                    else:
                        counts, weights = fa._class_weights(rows["train"])
                        smoke["class_weighting"] = {
                            "formula": "w_c=(n_c^-0.5)/(sum_j n_j*n_j^-0.5 / N_train)",
                            "train_counts": counts, "weights": weights}
                    smoke_path = root / f"{arm}_smoke.json"
                    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
                    args = SimpleNamespace(mode="full", experiment=arm, epochs=1,
                        micro_batch=1, accumulation=32, seed=42, head_lr=.001,
                        backbone_lr=.00001, weight_decay=.01, max_grad_norm=1.,
                        dataset="DDR", artifact_root=str(root), model_kind="convnext_tiny",
                        model_dir=str(root / "snapshot"), revision="b" * 40,
                        weight_sha256="e" * 64, feature_cache_dir=None, input_cache_dir=None,
                        output_dir=str(root / f"{arm}_run"), device="cpu", precision="fp32",
                        smoke_report=str(smoke_path), trust_legacy_bin=True, resume=False,
                        trust_run_checkpoint=False, extend_from_run=None)
                    backbone = TinyConv()
                    calls = []
                    def image_batch(selected, processor, **kwargs):
                        calls.append((selected[0]["split"], kwargs))
                        return (torch.ones(len(selected), 3, 224, 224),
                                torch.tensor([row["label"] for row in selected]))
                    with (patch.object(fa, "load_train_val_manifest",
                                       return_value=(rows, manifest)),
                          patch.object(fa, "audit_snapshot", return_value=snapshot),
                          patch.object(fa, "load_local_backbone",
                                       return_value=(backbone, object())),
                          patch.object(fa, "_image_batch", side_effect=image_batch)):
                        fa.train(args)
                    record = json.loads((root / f"{arm}_run" / "run.json").read_text())
                    history = json.loads((root / f"{arm}_run" / "history.json").read_text())
                    self.assertEqual(record["experiment"], arm)
                    self.assertEqual(len(history), 1)
                    self.assertEqual(backbone.forward_calls, len(rows["train"]) + len(rows["val"]))
                    self.assertTrue(all(kwargs.get("augment_epoch") == 1
                                        for split, kwargs in calls if split == "train")
                                    if arm == "B1" else
                                    all(not kwargs for split, kwargs in calls))
                    self.assertTrue(all(not kwargs for split, kwargs in calls if split == "val"))
                    self.assertEqual(record["class_weighting"] is not None, arm == "B2")
                    self.assertIn("train_input_seconds", history[0]["resource"])

    def test_current_b1_extension_rejects_any_scientific_change(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "runner.py"
            source.write_bytes(Path(fa.__file__).read_bytes())
            old_smoke_path = root / "smoke.json"
            smoke = {"effective_processor": {"size": 224}, "batch_size": 4,
                     "precision": "bf16", "snapshot_weight_sha256": "a" * 64,
                     "experiment": "B1", "augmentation": aug.PROTOCOL,
                     "class_weighting": None}
            old_smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
            old = root / "completed20"
            old.mkdir()
            old_record = {"epochs": 20, "mode": "full", "experiment": "B1",
                          "augmentation": fa._augmentation_metadata("B1"), "class_weighting": None,
                          "implementation_sha256": phase.sha256(source),
                          "smoke_report_sha256": phase.sha256(old_smoke_path)}
            (old / "run.json").write_text(json.dumps(old_record), encoding="utf-8")
            history = [{"epoch": i, "validation": {"macro_f1": i / 100,
                       "validation_loss": 1.0}, "train": {}, "resource": {}}
                       for i in range(1, 21)]
            best = {"epoch": 20, "validation": history[-1]["validation"]}
            last = {"record": old_record, "epoch": 20, "history": history,
                    "best": best, "optimizer": {"step": 1}, "rng": {"seed": 42},
                    "backbone": {"x": 1}, "head": {"x": 1}}
            for name in ("last.pt", "best.pt"):
                (old / name).write_bytes(b"synthetic")
            (old / "history.json").write_text(json.dumps(history), encoding="utf-8")
            (old / "summary.json").write_text(json.dumps({
                "status": "VALIDATION_COMPLETE", "epochs_completed": 20,
                "record": old_record, "best": best}), encoding="utf-8")
            args = type("Args", (), {"trust_run_checkpoint": True, "mode": "full",
                "epochs": 80, "extend_from_run": str(old),
                "source_file": str(source), "source_smoke_report": str(old_smoke_path),
                "legacy_source_file": None, "legacy_smoke_report": None})()
            new_record = {**old_record, "epochs": 80}
            with patch.object(fa, "_load_checkpoint", side_effect=[last, last]):
                info, _, _ = fa._audit_extension(args, new_record, smoke, root / "extension")
            self.assertEqual(info["source_runner_sha256"], old_record["implementation_sha256"])
            changed = {**new_record, "experiment": "B2"}
            with self.assertRaisesRegex(ValueError, "scientific"):
                fa._audit_extension(args, changed, smoke, root / "extension")
