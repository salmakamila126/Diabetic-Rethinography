"""Synthetic CPU-only E5 tests; never open KC images or weights."""

import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from novelty import foundation_adaptation as fa
from novelty import foundation_augmentation as augment
from novelty import preof_phase1 as phase
from novelty_integrated.e5 import config
from novelty_integrated.e5 import data
from novelty_integrated.e5.model import E5Heads, OrdinalHead
from novelty_integrated.e5.objective import loss_components, ordinal_targets
from novelty_integrated.e5 import runner


COUNTS = [40, 10, 20, 5, 5]


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 768)

    def forward(self, *, pixel_values):
        source = (pixel_values.float().mean(dim=(2, 3)) if pixel_values.ndim == 4
                  else pixel_values.float())
        pooled = self.linear(source)
        return SimpleNamespace(pooler_output=pooled,
                               last_hidden_state=pooled[:, :, None, None].expand(-1, -1, 7, 7))


class E5Tests(unittest.TestCase):
    def test_ordered_thresholds_survive_updates(self):
        head = OrdinalHead(768, COUNTS)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 773)
        optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        for _ in range(5):
            logits = head(torch.randn(7, 768))
            self.assertEqual(tuple(logits.shape), (7, 4))
            thresholds = head.thresholds().detach()
            self.assertTrue(bool(torch.all(thresholds[1:] > thresholds[:-1])))
            self.assertTrue(bool(torch.all(logits[:, :-1] > logits[:, 1:])))
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(logits, ordinal_targets(
                torch.tensor([0, 1, 2, 3, 4, 1, 3]))).backward()
            optimizer.step()

    def test_objective_fp32_and_separable_components(self):
        logits = torch.tensor([[2., 0., -1., -2., -3.], [0., 1., 2., -1., -2.]],
                              requires_grad=True)
        ordinal = torch.tensor([[1., -1., -2., -3.], [2., 1., -.5, -1.]],
                               requires_grad=True)
        labels = torch.tensor([0, 2])
        weights = config.train_class_weights(COUNTS)
        result = loss_components(logits, ordinal, labels, class_weights=weights)
        expected_ce = (F.cross_entropy(logits, labels, reduction="none") *
                       torch.tensor(weights)[labels]).mean()
        expected_ord = F.binary_cross_entropy_with_logits(ordinal, ordinal_targets(labels))
        self.assertTrue(torch.allclose(result["categorical"], expected_ce))
        self.assertTrue(torch.allclose(result["ordinal"], expected_ord))
        self.assertTrue(torch.allclose(result["total"], expected_ce + .25 * expected_ord))
        result["total"].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(ordinal.grad).all()))
        control = loss_components(logits.detach(), None, labels)
        self.assertEqual(float(control["ordinal"]), 0.)
        self.assertTrue(torch.allclose(control["total"], F.cross_entropy(logits.detach(), labels)))

    def test_objective_fp32_under_bfloat16_autocast(self):
        labels = torch.tensor([0, 1, 3, 4])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            categorical = torch.nn.Linear(8, 5)(torch.randn(4, 8))
            ordinal = torch.nn.Linear(8, 4)(torch.randn(4, 8))
            losses = loss_components(categorical, ordinal, labels,
                                     class_weights=config.train_class_weights(COUNTS))
        self.assertEqual(categorical.dtype, torch.bfloat16)
        self.assertEqual(losses["total"].dtype, torch.float32)
        self.assertEqual(losses["categorical"].dtype, torch.float32)
        self.assertEqual(losses["ordinal"].dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(losses["total"])))

    def test_class_weight_schedule_and_training_only_counts(self):
        weights = config.train_class_weights(COUNTS)
        self.assertAlmostEqual(sum(n * w for n, w in zip(COUNTS, weights)) / sum(COUNTS), 1.)
        for arm in ("control", "ordinal", "deferred", "integrated"):
            self.assertIsNone(config.weights_for_epoch(COUNTS, arm, 10))
            if arm in ("deferred", "integrated"):
                self.assertEqual(config.weights_for_epoch(COUNTS, arm, 11), weights)
                self.assertEqual(config.weights_for_epoch(COUNTS, arm, 20), weights)
            else:
                self.assertIsNone(config.weights_for_epoch(COUNTS, arm, 11))
        with self.assertRaises(ValueError):
            config.weights_for_epoch(COUNTS, "integrated", 21)
        with self.assertRaises(ValueError):
            config.validate_batch(16, 1)
        config.validate_batch(16, 2)
        config.require_official_snapshot(config.OFFICIAL_SNAPSHOT)
        with self.assertRaises(ValueError):
            config.require_official_snapshot({**config.OFFICIAL_SNAPSHOT,
                                              "processor_config_sha256": "0" * 64})

    def test_frozen_loader_excludes_test_rows_and_detects_train_tampering(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "datasets" / "APTOS2019"
            data.mkdir(parents=True)
            manifest = root / "cad_vit_runs" / "manifests" / "APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            records = []
            for split in ("train", "val"):
                for grade in range(5):
                    name = f"{split}_{grade}.dat"
                    raw = f"{split}:{grade}".encode()
                    (data / name).write_bytes(raw)
                    records.append({"image_id": name, "image_path": name,
                                    "label": str(grade), "split": split,
                                    "group_id": name, "sha256": hashlib.sha256(raw).hexdigest()})
            # A malformed test path must never be resolved or opened.
            records.append({"image_id": "test", "image_path": "../forbidden",
                            "label": "bad", "split": "test", "group_id": "test",
                            "sha256": "bad"})
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            digest = phase.sha256(manifest)
            manifest.with_suffix(".json").write_text(json.dumps({
                "dataset": "APTOS2019", "seed": 42, "root": str(data.resolve()),
                "manifest_file_sha256": digest}), encoding="utf-8")
            with patch.dict(phase.SPECS, {"APTOS2019": {"manifest_sha256": digest,
                                                        "val_count": 5}}):
                rows, info = fa.load_train_val_manifest(root, "APTOS2019")
                self.assertEqual([len(rows[key]) for key in ("train", "val")], [5, 5])
                self.assertEqual(info["grade_support"]["train"], [1] * 5)
                (data / "train_0.dat").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    fa.load_train_val_manifest(root, "APTOS2019")

    def test_explicit_host_mapping_keeps_sidecar_and_test_isolated(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            local = root / "datasets" / "APTOS2019"
            local.mkdir(parents=True)
            manifest = root / "cad_vit_runs" / "manifests" / "APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            records = []
            for split in ("train", "val"):
                for grade in range(5):
                    name = f"{split}_{grade}.dat"
                    contents = f"{split}:{grade}".encode()
                    (local / name).write_bytes(contents)
                    records.append({"image_id": name, "image_path": name,
                                    "label": str(grade), "split": split, "group_id": name,
                                    "sha256": hashlib.sha256(contents).hexdigest()})
            records.append({"image_id": "test", "image_path": "../test/forbidden",
                            "label": "bad", "split": "test", "group_id": "test", "sha256": "bad"})
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            source_root = "/mnt/home/probation-03/Kel_13/datasets/APTOS2019"
            digest = phase.sha256(manifest)
            sidecar = manifest.with_suffix(".json")
            sidecar.write_text(json.dumps({"dataset": "APTOS2019", "seed": 42,
                                           "root": source_root,
                                           "manifest_file_sha256": digest}), encoding="utf-8")
            sidecar_hash = phase.sha256(sidecar)
            with patch.dict(phase.SPECS, {"APTOS2019": {"manifest_sha256": digest,
                                                        "val_count": 5}}):
                with self.assertRaises(ValueError):
                    data.load_train_val(root, "APTOS2019")
                with self.assertRaises(ValueError):
                    data.load_train_val(root, "APTOS2019",
                                        manifest_source_root="/wrong/datasets/APTOS2019")
                rows, info = data.load_train_val(root, "APTOS2019",
                                                 manifest_source_root=source_root)
                self.assertEqual([len(rows[s]) for s in ("train", "val")], [5, 5])
                self.assertEqual(info["manifest_sidecar_sha256"], sidecar_hash)
                self.assertEqual(info["path_mapping"], "explicit_source_to_local")
                self.assertEqual(phase.sha256(sidecar), sidecar_hash)
                (local / "val_3.dat").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    data.load_train_val(root, "APTOS2019",
                                        manifest_source_root=source_root)

    def test_augmentation_and_head_initialization_are_deterministic(self):
        image = Image.fromarray(np.full((40, 40, 3), 130, dtype=np.uint8))
        first, draws_a = augment.apply(image, seed=42, epoch=3, image_id="train-1")
        second, draws_b = augment.apply(image, seed=42, epoch=3, image_id="train-1")
        self.assertEqual(draws_a, draws_b)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        torch.manual_seed(42)
        control = E5Heads(768, COUNTS, ordinal=False)
        torch.manual_seed(42)
        integrated = E5Heads(768, COUNTS, ordinal=True)
        self.assertTrue(torch.equal(control.categorical.weight, integrated.categorical.weight))
        self.assertTrue(torch.equal(control.categorical.bias, integrated.categorical.bias))

    def test_only_categorical_head_determines_predictions(self):
        heads = E5Heads(768, COUNTS, ordinal=True)
        features = torch.randn(3, 768)
        before = heads.predict_logits(features).detach().clone()
        with torch.no_grad():
            heads.ordinal.severity.weight.add_(10)
            heads.ordinal.base_threshold.add_(10)
        after = heads.predict_logits(features)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(after, heads(features)[0]))

    def test_all_four_arms_take_a_synthetic_optimizer_step(self):
        for arm in config.ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(11)
                backbone = TinyBackbone()
                heads = E5Heads(768, COUNTS, ordinal=config.arm_components(arm)[0])
                optimizer = torch.optim.AdamW([
                    {"params": list(backbone.parameters()), "lr": config.BACKBONE_LR},
                    {"params": list(heads.parameters()), "lr": config.HEAD_LR}],
                    weight_decay=config.WEIGHT_DECAY)
                pixels = torch.randn(5, 3)
                labels = torch.arange(5, dtype=torch.long)
                result = runner._train_batches(
                    backbone, heads, optimizer, [(pixels, labels)],
                    device=torch.device("cpu"), precision="fp32",
                    total_images=5, micro_batch=5, accumulation=1,
                    class_weights=config.weights_for_epoch(COUNTS, arm, 11))
                self.assertEqual(result["optimizer_steps"], 1)
                self.assertEqual(result["images"], 5)
                self.assertEqual(result["ordinal_loss"] > 0, config.arm_components(arm)[0])

    def test_incomplete_accumulation_window_matches_full_batch(self):
        torch.manual_seed(7)
        backbone_a, heads_a = TinyBackbone(), E5Heads(768, COUNTS, ordinal=True)
        torch.manual_seed(7)
        backbone_b, heads_b = TinyBackbone(), E5Heads(768, COUNTS, ordinal=True)
        opt_a = torch.optim.SGD(list(backbone_a.parameters()) + list(heads_a.parameters()), lr=.001)
        opt_b = torch.optim.SGD(list(backbone_b.parameters()) + list(heads_b.parameters()), lr=.001)
        pixels = torch.randn(35, 3)
        labels = torch.tensor([i % 5 for i in range(35)])
        windows = [(pixels[:16], labels[:16]), (pixels[16:32], labels[16:32]),
                   (pixels[32:], labels[32:])]
        result = runner._train_batches(backbone_a, heads_a, opt_a, windows,
                                       device=torch.device("cpu"), precision="fp32",
                                       total_images=35, micro_batch=16, accumulation=2,
                                       class_weights=config.train_class_weights(COUNTS))
        self.assertEqual(result["accumulation_window_images"], [32, 3])
        for start, end in ((0, 32), (32, 35)):
            opt_b.zero_grad()
            pooled = fa.pooled(backbone_b, pixels[start:end], "convnext_tiny")
            categorical, ordinal = heads_b(pooled)
            loss_components(categorical, ordinal, labels[start:end],
                            class_weights=config.train_class_weights(COUNTS))["total"].backward()
            runner._check_gradients(backbone_b, heads_b, opt_b)
            opt_b.step()
        for expected, observed in zip((*backbone_b.parameters(), *heads_b.parameters()),
                                      (*backbone_a.parameters(), *heads_a.parameters())):
            self.assertTrue(torch.allclose(expected, observed, atol=2e-6, rtol=1e-6))

    def test_control_optimizer_step_matches_existing_b1_loop(self):
        torch.manual_seed(27)
        backbone_a, heads_a = TinyBackbone(), E5Heads(768, COUNTS, ordinal=False)
        torch.manual_seed(27)
        backbone_b, head_b = TinyBackbone(), torch.nn.Linear(768, 5)
        opt_a = torch.optim.AdamW([
            {"params": list(backbone_a.parameters()), "lr": config.BACKBONE_LR},
            {"params": list(heads_a.parameters()), "lr": config.HEAD_LR}],
            weight_decay=config.WEIGHT_DECAY)
        opt_b = torch.optim.AdamW([
            {"params": list(backbone_b.parameters()), "lr": config.BACKBONE_LR},
            {"params": list(head_b.parameters()), "lr": config.HEAD_LR}],
            weight_decay=config.WEIGHT_DECAY)
        pixels = torch.randn(35, 3)
        labels = torch.tensor([i % 5 for i in range(35)])
        batches = [(pixels[:16], labels[:16]), (pixels[16:32], labels[16:32]),
                   (pixels[32:], labels[32:])]
        a = runner._train_batches(backbone_a, heads_a, opt_a, batches,
                                  device=torch.device("cpu"), precision="fp32",
                                  total_images=35, micro_batch=16, accumulation=2,
                                  class_weights=None)
        b = fa._train_epoch(backbone_b, head_b, "full", "convnext_tiny", batches,
                            opt_b, None, torch.device("cpu"), "fp32", 2,
                            config.MAX_GRAD_NORM, 3, 35, 16)
        self.assertEqual(a["accumulation_window_images"], b["accumulation_window_images"])
        self.assertAlmostEqual(a["loss"], b["loss"], places=6)
        for expected, observed in zip((*backbone_b.parameters(), *head_b.parameters()),
                                      (*backbone_a.parameters(), *heads_a.parameters())):
            self.assertTrue(torch.allclose(expected, observed, atol=2e-6, rtol=1e-6))

    def test_guarded_checkpoint_resume_restores_optimizer_and_rng(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            torch.manual_seed(9)
            backbone, heads = TinyBackbone(), E5Heads(768, COUNTS, ordinal=True)
            optimizer = torch.optim.AdamW([
                {"params": list(backbone.parameters()), "lr": config.BACKBONE_LR},
                {"params": list(heads.parameters()), "lr": config.HEAD_LR}],
                weight_decay=config.WEIGHT_DECAY)
            pixels, labels = torch.randn(5, 3), torch.tensor([0, 1, 2, 3, 4])
            runner._train_batches(backbone, heads, optimizer, [(pixels, labels)],
                                  device=torch.device("cpu"), precision="fp32",
                                  total_images=5, micro_batch=5, accumulation=1,
                                  class_weights=None)
            record = {"schema": config.SCHEMA, "arm": "integrated"}
            best = {"epoch": 1, "validation": {"macro_f1": .5, "validation_loss": .9}}
            history = [{"epoch": 1, "validation": best["validation"]}]
            state = {"record": record, "epoch": 1, "backbone": backbone.state_dict(),
                     "heads": heads.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng": fa._rng_state(), "history": history, "best": best}
            fa._atomic_json(output / "run.json", record)
            fa._atomic_json(output / "history.json", history)
            fa._save_checkpoint(output / "best.pt", state)
            fa._save_checkpoint(output / "last.pt", state)
            expected = torch.rand(3)
            saved_head = heads.categorical.weight.detach().clone()
            with torch.no_grad():
                heads.categorical.weight.add_(5)
            resumed_history, resumed_best, next_epoch = runner._resume(
                output, record, backbone, heads, optimizer, trust=True)
            self.assertEqual((len(resumed_history), resumed_best["epoch"], next_epoch), (1, 1, 2))
            self.assertTrue(torch.equal(saved_head, heads.categorical.weight))
            self.assertTrue(torch.equal(expected, torch.rand(3)))
            self.assertGreater(len(optimizer.state), 0)
            with (output / "last.pt").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                runner._resume(output, record, backbone, heads, optimizer, trust=True)

    def test_synthetic_runner_smoke_interruption_and_resume(self):
        class Processor:
            def to_dict(self):
                return {"synthetic": True, "size": 224}

        def fake_batch(selected, _processor, *, augment_seed=None, augment_epoch=None):
            pixels = torch.stack([torch.full((3, 224, 224),
                                             .1 + row["label"] / 10 +
                                             (augment_epoch or 0) / 1000)
                                  for row in selected])
            labels = torch.tensor([row["label"] for row in selected], dtype=torch.long)
            return pixels, labels

        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = {
                "train": [{"label": i % 5, "image_id": f"t{i}", "group_id": f"g{i}",
                           "split": "train"} for i in range(35)],
                "val": [{"label": i, "image_id": f"v{i}", "group_id": f"vg{i}",
                         "split": "val"} for i in range(5)],
            }
            manifest = {"dataset": "APTOS2019", "manifest_sha256": "a" * 64,
                        "counts": {"train": 35, "val": 5}, "images_verified": True}
            snapshot = config.OFFICIAL_SNAPSHOT.copy()
            args = SimpleNamespace(dataset="APTOS2019", arm="integrated",
                                   artifact_root=str(root), model_dir=str(root / "official"),
                                   revision=config.OFFICIAL_SNAPSHOT["revision"],
                                   weight_sha256=config.OFFICIAL_SNAPSHOT["weight_sha256"],
                                   trust_legacy_bin=True, device="cpu", precision="fp32",
                                   seed=42, micro_batch=32, accumulation=1,
                                   output_dir=str(root / "smoke"), smoke_report=None,
                                   resume=False, trust_run_checkpoint=False)
            with (patch.object(data, "load_train_val", return_value=(rows, manifest)),
                  patch.object(fa, "audit_snapshot", return_value=snapshot),
                  patch.object(fa, "load_local_backbone",
                               side_effect=lambda *_a, **_kw: (TinyBackbone(), Processor())),
                  patch.object(fa, "configure_trainable", return_value=["entire_backbone"]),
                  patch.object(fa, "_image_batch", side_effect=fake_batch),
                  patch.dict(sys.modules, {"transformers": SimpleNamespace(__version__="4.57.3")}),
                  patch.object(config, "PILOT_EPOCHS", 2),
                  patch.object(config, "WEIGHT_START_EPOCH", 2)):
                runner.smoke(args)
                args.smoke_report = str(root / "smoke" / "smoke.json")
                args.output_dir = str(root / "resumed")
                runner.train(args, stop_after_epoch=1)
                self.assertFalse((root / "resumed" / "summary.json").exists())
                args.resume = True
                args.trust_run_checkpoint = True
                runner.train(args)
                summary = json.loads((root / "resumed" / "summary.json").read_text())
                self.assertEqual(summary["epochs_completed"], 2)
                self.assertEqual(summary["final_test_status"], "NOT_ACCESSED")
                history = json.loads((root / "resumed" / "history.json").read_text())
                self.assertEqual([row["class_weight_active"] for row in history], [False, True])
                self.assertEqual(len(history[0]["validation"]["per_grade"]), 5)
                args.resume = False
                args.output_dir = str(root / "uninterrupted")
                runner.train(args)
                a = fa._load_checkpoint(root / "resumed" / "last.pt", trust=True)
                b = fa._load_checkpoint(root / "uninterrupted" / "last.pt", trust=True)
                for key in a["backbone"]:
                    self.assertTrue(torch.equal(a["backbone"][key], b["backbone"][key]))
                for key in a["heads"]:
                    self.assertTrue(torch.equal(a["heads"][key], b["heads"][key]))


if __name__ == "__main__":
    unittest.main()
