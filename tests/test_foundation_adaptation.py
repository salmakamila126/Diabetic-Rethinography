"""Synthetic-only integrity and training-path tests; no Hub or KC access."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from novelty import foundation_adaptation as fa
from novelty import preof_e3 as e3
from novelty import preof_phase1 as phase


class FoundationAdaptationTests(unittest.TestCase):
    def test_train_val_manifest_hash_group_and_image_integrity(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "datasets" / "APTOS2019"
            data.mkdir(parents=True)
            manifest = root / "cad_vit_runs" / "manifests" / "APTOS2019_seed42.csv"
            manifest.parent.mkdir(parents=True)
            records = []
            for split, count in (("train", 10), ("val", 5)):
                for index in range(count):
                    name = f"{split}_{index}.dat"
                    content = f"{split}:{index}".encode()
                    (data / name).write_bytes(content)
                    records.append({"image_id": name, "image_path": name, "label": str(index % 5),
                                    "split": split, "group_id": name,
                                    "sha256": hashlib.sha256(content).hexdigest()})
            # The test row's label/path are deliberately invalid; E4/E5 must not inspect them.
            records.append({"image_id": "test", "image_path": "../test", "label": "invalid",
                            "split": "test", "group_id": "test", "sha256": "invalid"})
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            digest = phase.sha256(manifest)
            manifest.with_suffix(".json").write_text(json.dumps(
                {"dataset": "APTOS2019", "seed": 42, "root": str(data.resolve()),
                 "manifest_file_sha256": digest}), encoding="utf-8")
            with patch.dict(phase.SPECS, {"APTOS2019": {"manifest_sha256": digest, "val_count": 5}}):
                rows, info = fa.load_train_val_manifest(root, "APTOS2019")
                self.assertEqual((len(rows["train"]), len(rows["val"])), (10, 5))
                self.assertEqual(info["grade_support"]["val"], [1] * 5)
                (data / "train_0.dat").write_bytes(b"altered")
                with self.assertRaises(ValueError):
                    fa.load_train_val_manifest(root, "APTOS2019")

    def test_snapshot_guard_and_partial_freeze(self):
        with TemporaryDirectory() as temp:
            revision = "a" * 40
            directory = Path(temp) / revision
            directory.mkdir()
            (directory / "config.json").write_text('{"model_type":"dinov3_vit"}', encoding="utf-8")
            (directory / "preprocessor_config.json").write_text('{"size":224}', encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"synthetic")
            digest = phase.sha256(directory / "model.safetensors")
            audited = fa.audit_snapshot("dinov3_vits16", directory, revision, digest)
            self.assertEqual(audited["weight_sha256"], digest)
            with self.assertRaises(ValueError):
                fa.audit_snapshot("dinov3_vits16", directory, revision, "0" * 64)

        class TinyViT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(12)])
                self.norm = torch.nn.LayerNorm(1)
                self.config = SimpleNamespace(hidden_size=384)

        backbone = TinyViT()
        self.assertEqual(fa.configure_trainable(backbone, "dinov3_vits16", "frozen"), [])
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
        selected = fa.configure_trainable(backbone, "dinov3_vits16", "partial")
        self.assertEqual(selected, ["layer.10", "layer.11", "norm"])
        self.assertFalse(any(p.requires_grad for p in backbone.layer[9].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.layer[10].parameters()))
        self.assertEqual(fa.configure_trainable(backbone, "dinov3_vits16", "full"), ["entire_backbone"])
        self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
        with self.assertRaises(NotImplementedError):
            fa.configure_trainable(backbone, "dinov3_vits16", "peft")

        class TinyConvNext(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.stages = torch.nn.ModuleList()
                for _ in range(4):
                    stage = torch.nn.Module()
                    stage.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(3)])
                    self.encoder.stages.append(stage)
                self.layernorm = torch.nn.LayerNorm(1)
                self.config = SimpleNamespace(hidden_sizes=[96, 192, 384, 768])

        convnext = TinyConvNext()
        self.assertEqual(fa.configure_trainable(convnext, "convnext_tiny", "partial"),
                         ["encoder.stages.3.layers.1", "encoder.stages.3.layers.2", "layernorm"])
        self.assertFalse(any(p.requires_grad for p in convnext.encoder.stages[3].layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in convnext.encoder.stages[3].layers[2].parameters()))
        self.assertEqual(fa.configure_trainable(convnext, "convnext_tiny", "full"), ["entire_backbone"])
        self.assertTrue(all(p.requires_grad for p in convnext.parameters()))

    def test_feature_index_detects_corruption_and_selection_rule(self):
        rows = [{"image_id": f"i{i}", "split": "train", "label": i,
                 "group_id": f"g{i}", "sha256": f"{i:064x}"} for i in range(2)]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            vectors = np.lib.format.open_memmap(root / "features.npy", mode="w+",
                                                 shape=(2, 384), dtype=np.float32)
            vectors[0] = 1
            vectors[1] = 2
            vectors.flush()
            with (root / "rows.jsonl").open("w", encoding="utf-8") as stream:
                for i, row in enumerate(rows):
                    stream.write(json.dumps({**row, "vector_sha256": hashlib.sha256(
                        np.asarray(vectors[i]).tobytes()).hexdigest()}) + "\n")
            loaded, entries = fa._read_feature_index(root, rows, {"vector_dimension": 384})
            self.assertEqual(len(entries), 2)
            del loaded
            vectors[1, 0] = 7
            vectors.flush()
            with self.assertRaises(ValueError):
                fa._read_feature_index(root, rows, {"vector_dimension": 384})
            del vectors
            import gc
            gc.collect()
        self.assertTrue(fa._best_is_new({"macro_f1": .7, "validation_loss": .5},
                                        {"macro_f1": .7, "validation_loss": .6}))
        self.assertFalse(fa._best_is_new({"macro_f1": .7, "validation_loss": .5},
                                         {"macro_f1": .7, "validation_loss": .5}))

    def test_frozen_probe_epoch_checkpoint_and_resume_without_model_load(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            train_rows = [{"label": i % 5, "image_id": f"t{i}", "group_id": f"gt{i}",
                           "sha256": f"{i:064x}", "split": "train"} for i in range(15)]
            val_rows = [{"label": i, "image_id": f"v{i}", "group_id": f"gv{i}",
                         "sha256": f"{i+100:064x}", "split": "val"} for i in range(5)]
            rows = {"train": train_rows, "val": val_rows}
            manifest = {"dataset": "APTOS2019", "manifest_sha256": "a" * 64,
                        "counts": {"train": 15, "val": 5}, "images_verified": True}
            snapshot = {"hub_id": fa.MODELS["dinov3_vits16"]["hub_id"],
                        "revision": "b" * 40, "config_sha256": "c" * 64,
                        "processor_config_sha256": "d" * 64,
                        "weight_file": "model.safetensors", "weight_sha256": "e" * 64}
            feature_root = root / "features"
            feature_root.mkdir()
            (feature_root / "metadata.json").write_text("{}")
            (feature_root / "rows.jsonl").write_text("{}\n")
            vectors = np.random.default_rng(42).normal(size=(20, 384)).astype(np.float32)
            feature_meta = {"backbone_parameter_count": 21_000_000}
            smoke_path = root / "smoke.json"
            smoke_path.write_text(json.dumps({"schema": fa.SCHEMA, "dataset": "APTOS2019",
                "model_kind": "dinov3_vits16", "mode": "frozen", "snapshot_weight_sha256": "e" * 64,
                "snapshot_revision": "b" * 40, "processor_config_sha256": "d" * 64,
                "manifest_sha256": "a" * 64, "device_type": "cpu", "batch_size": 2,
                "status": "BATCH_SMOKE_PASS", "optimizer_steps": 2, "weights_saved": False,
                "precision": "fp32", "implementation_sha256": phase.sha256(fa.__file__)}), encoding="utf-8")
            args = SimpleNamespace(mode="frozen", epochs=2, micro_batch=2, accumulation=3,
                seed=42, head_lr=1e-3, backbone_lr=1e-5, weight_decay=.01,
                max_grad_norm=1.0, dataset="APTOS2019", artifact_root=str(root),
                model_kind="dinov3_vits16", model_dir=str(root / "snapshot"),
                revision="b" * 40, weight_sha256="e" * 64,
                feature_cache_dir=str(feature_root), output_dir=str(root / "run"),
                device="cpu", precision="fp32", smoke_report=str(smoke_path),
                trust_legacy_bin=False, resume=False, trust_run_checkpoint=False)
            with (patch.object(fa, "load_train_val_manifest", return_value=(rows, manifest)),
                  patch.object(fa, "audit_snapshot", return_value=snapshot),
                  patch.object(fa, "_feature_cache", return_value=(vectors, feature_meta))):
                fa.train(args)
                result = json.loads((root / "run" / "summary.json").read_text())
                self.assertEqual(result["epochs_completed"], 2)
                self.assertEqual(result["parameter_counts"]["backbone_total"], 21_000_000)
                self.assertEqual(result["final_test_status"], "NOT_ACCESSED")
                args.resume, args.trust_run_checkpoint = True, True
                fa.train(args)
                self.assertEqual(len(json.loads((root / "run" / "history.json").read_text())), 2)
                args.mode = "partial"
                with self.assertRaises(ValueError):
                    fa.train(args)

    def test_partial_tuning_uses_raw_batch_path_and_selected_blocks(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(12)])
                self.norm = torch.nn.LayerNorm(384)
                self.config = SimpleNamespace(hidden_size=384, model_type="dinov3_vit")
                self.training_modes = []

            def forward(self, pixel_values):
                self.training_modes.append(self.training)
                value = pixel_values.mean(dim=(1, 2, 3))[:, None]
                for layer in self.layer:
                    value = layer(value)
                vector = self.norm(value.expand(-1, 384))
                return SimpleNamespace(pooler_output=vector,
                                       last_hidden_state=vector[:, None, :].expand(-1, 201, -1))

        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = {split: [{"label": i, "image_id": f"{split}{i}", "group_id": f"{split}{i}"}
                            for i in range(5)] for split in ("train", "val")}
            manifest = {"manifest_sha256": "a" * 64, "images_verified": True}
            snapshot = {"hub_id": fa.MODELS["dinov3_vits16"]["hub_id"],
                        "revision": "b" * 40, "config_sha256": "c" * 64,
                        "processor_config_sha256": "d" * 64,
                        "weight_file": "model.safetensors", "weight_sha256": "e" * 64}
            smoke_path = root / "smoke.json"
            smoke_path.write_text(json.dumps({"schema": fa.SCHEMA, "dataset": "DDR",
                "model_kind": "dinov3_vits16", "mode": "partial", "snapshot_weight_sha256": "e" * 64,
                "snapshot_revision": "b" * 40, "processor_config_sha256": "d" * 64,
                "manifest_sha256": "a" * 64, "device_type": "cpu", "batch_size": 1,
                "status": "BATCH_SMOKE_PASS", "optimizer_steps": 2, "weights_saved": False,
                "precision": "fp32", "implementation_sha256": phase.sha256(fa.__file__)}), encoding="utf-8")
            args = SimpleNamespace(mode="partial", epochs=1, micro_batch=1, accumulation=2,
                seed=42, head_lr=1e-3, backbone_lr=1e-5, weight_decay=.01,
                max_grad_norm=1.0, dataset="DDR", artifact_root=str(root),
                model_kind="dinov3_vits16", model_dir=str(root / "snapshot"),
                revision="b" * 40, weight_sha256="e" * 64, feature_cache_dir=None,
                output_dir=str(root / "run"), device="cpu", precision="fp32",
                smoke_report=str(smoke_path), trust_legacy_bin=False,
                resume=False, trust_run_checkpoint=False)
            backbone = TinyBackbone()
            def fake_batch(batch, _processor):
                return (torch.stack([torch.full((3, 224, 224), float(r["label"] + 1) / 10)
                                     for r in batch]), torch.tensor([r["label"] for r in batch]))

            with (patch.object(fa, "load_train_val_manifest", return_value=(rows, manifest)),
                  patch.object(fa, "audit_snapshot", return_value=snapshot),
                  patch.object(fa, "load_local_backbone", return_value=(backbone, object())),
                  patch.object(fa, "_image_batch", side_effect=fake_batch),
                  patch.object(fa, "_feature_cache", side_effect=AssertionError("Never use frozen cache"))):
                fa.train(args)
            result = json.loads((root / "run" / "summary.json").read_text())
            self.assertGreater(result["parameter_counts"]["backbone_trainable"], 0)
            self.assertEqual(result["epochs_completed"], 1)
            self.assertNotIn(True, backbone.training_modes)

            full_backbone = TinyBackbone()
            args.mode, args.accumulation = "full", 32
            args.output_dir = str(root / "full_run")
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            smoke["mode"] = "full"
            smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
            with (patch.object(fa, "load_train_val_manifest", return_value=(rows, manifest)),
                  patch.object(fa, "audit_snapshot", return_value=snapshot),
                  patch.object(fa, "load_local_backbone", return_value=(full_backbone, object())),
                  patch.object(fa, "_image_batch", side_effect=fake_batch),
                  patch.object(fa, "_feature_cache", side_effect=AssertionError("Never use frozen cache"))):
                fa.train(args)
                args.resume, args.trust_run_checkpoint = True, True
                fa.train(args)
            summary = json.loads((root / "full_run" / "summary.json").read_text())
            history = json.loads((root / "full_run" / "history.json").read_text())
            self.assertEqual(summary["parameter_counts"]["backbone_trainable"],
                             summary["parameter_counts"]["backbone_total"])
            self.assertEqual(history[0]["train"]["accumulation_window_images"], [5])
            self.assertIn(True, full_backbone.training_modes)
            self.assertIn(False, full_backbone.training_modes)
            self.assertEqual(summary["final_test_status"], "NOT_ACCESSED")

    def test_accumulation_matches_image_weighted_large_batches(self):
        for n, micro, accumulation, expected in ((7, 2, 2, [4, 3]), (5, 1, 4, [4, 1])):
            with self.subTest(n=n, micro=micro):
                torch.manual_seed(17)
                features = torch.randn(n, 384)
                labels = torch.arange(n) % 5
                actual = torch.nn.Linear(384, 5)
                reference = torch.nn.Linear(384, 5)
                reference.load_state_dict(actual.state_dict())
                optimizer = torch.optim.SGD(actual.parameters(), lr=.1)
                ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
                batches = [(features[i:i + micro], labels[i:i + micro])
                           for i in range(0, n, micro)]
                info = fa._train_epoch(None, actual, "frozen", "dinov3_vits16",
                    batches, optimizer, None, torch.device("cpu"), "fp32",
                    accumulation, 1e9, len(batches), n, micro)
                offset = 0
                for window in expected:
                    ref_optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(
                        reference(features[offset:offset + window]), labels[offset:offset + window])
                    loss.backward()
                    ref_optimizer.step()
                    offset += window
                self.assertEqual(info["accumulation_window_images"], expected)
                self.assertEqual(info["optimizer_steps"], len(expected))
                for value, wanted in zip(actual.parameters(), reference.parameters()):
                    torch.testing.assert_close(value, wanted, atol=1e-6, rtol=1e-6)

    def test_full_smoke_benchmarks_all_prespecified_micro_batches(self):
        class MiniConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(768))
                self.config = SimpleNamespace(model_type="convnext")

            def forward(self, pixel_values):
                values = pixel_values.mean(dim=(1, 2, 3))[:, None] * self.weight[None]
                feature_map = values[:, :, None, None].expand(-1, -1, 7, 7)
                return SimpleNamespace(pooler_output=values, last_hidden_state=feature_map)

        class Processor:
            def to_dict(self):
                return {"size": 224}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = {"train": [{"label": i % 5} for i in range(32)], "val": []}
            snapshot = {"weight_sha256": "a" * 64, "revision": "b" * 40,
                        "processor_config_sha256": "c" * 64}
            for batch_size in (1, 4, 8, 16, 32):
                with self.subTest(batch_size=batch_size):
                    args = SimpleNamespace(mode="full", batch_size=batch_size,
                        artifact_root=str(root), dataset="DDR", model_kind="convnext_tiny",
                        model_dir=str(root / "snapshot"), revision="b" * 40,
                        weight_sha256="a" * 64, output_dir=str(root / f"smoke{batch_size}"),
                        seed=42, device="cpu", precision="fp32", trust_legacy_bin=True)
                    def fake_batch(selected, _processor):
                        pixels = torch.ones(len(selected), 3, 224, 224)
                        return pixels, torch.tensor([row["label"] for row in selected])
                    with (patch.object(fa, "load_train_val_manifest",
                                       return_value=(rows, {"manifest_sha256": "d" * 64})),
                          patch.object(fa, "audit_snapshot", return_value=snapshot),
                          patch.object(fa, "load_local_backbone",
                                       return_value=(MiniConv(), Processor())),
                          patch.object(fa, "_image_batch", side_effect=fake_batch)):
                        fa.smoke(args)
                    result = json.loads((root / f"smoke{batch_size}" / "smoke.json").read_text())
                    self.assertEqual(result["batch_size"], batch_size)
                    self.assertEqual(result["optimizer_steps"], 2)
                    self.assertEqual(result["parameter_counts"]["backbone_trainable"],
                                     result["parameter_counts"]["backbone_total"])
                    self.assertFalse(result["weights_saved"])

    def test_full_mode_rejects_wrong_effective_batch_and_frozen_cache(self):
        args = SimpleNamespace(mode="full", epochs=20, micro_batch=4, accumulation=4,
            seed=42, head_lr=1e-3, backbone_lr=1e-5, weight_decay=.01,
            max_grad_norm=1., feature_cache_dir=None)
        with self.assertRaisesRegex(ValueError, "effective batch 32"):
            fa.train(args)
        args.accumulation, args.feature_cache_dir = 8, "forbidden"
        with self.assertRaisesRegex(ValueError, "frozen embeddings"):
            fa.train(args)

    def test_e3_sensitivity_uses_exact_validation_pilot_only(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "datasets" / "APTOS2019"
            image_dir.mkdir(parents=True)
            val = []
            for grade in range(5):
                path = image_dir / f"v{grade}.png"
                Image.fromarray(np.full((8, 8, 3), 30 + grade, np.uint8)).save(path)
                val.append({"image_id": f"v{grade}", "group_id": f"g{grade}",
                            "label": grade, "sha256": phase.sha256(path),
                            "path": str(path), "split": "val"})
            rows = {"train": [], "val": val}
            manifest = {"manifest_sha256": "a" * 64, "dataset_root": str(image_dir)}
            snapshot = {"hub_id": fa.MODELS["dinov3_vits16"]["hub_id"],
                        "revision": "b" * 40, "config_sha256": "c" * 64,
                        "processor_config_sha256": "d" * 64,
                        "weight_file": "model.safetensors", "weight_sha256": "e" * 64}
            run_dir, pilot_dir = root / "run", root / "e3"
            run_dir.mkdir()
            pilot_dir.mkdir()
            record = {"dataset": "APTOS2019", "model_kind": "dinov3_vits16",
                      "mode": "frozen", "manifest": manifest,
                      "snapshot": {key: snapshot[key] for key in
                          ("hub_id", "revision", "config_sha256", "processor_config_sha256",
                           "weight_file", "weight_sha256")},
                      "precision": "fp32", "seed": 42}
            (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
            head = torch.nn.Linear(384, 5)
            fa._save_checkpoint(run_dir / "best.pt", {"record": record, "epoch": 1,
                "best": {"epoch": 1}, "head": head.state_dict(), "backbone": None})
            selected, selection = e3.select_pilot(val, "APTOS2019", 1, 42)
            (pilot_dir / "metadata.json").write_text(json.dumps({"schema": e3.SCHEMA,
                "dataset": "APTOS2019", "manifest_sha256": "a" * 64,
                "selection": selection}), encoding="utf-8")
            (pilot_dir / "rows.jsonl").write_text("".join(json.dumps({
                "image_id": r["image_id"], "image_sha256": r["sha256"],
                "true_grade": r["label"], "group_id": r["group_id"], "split": "val"}) + "\n"
                for r in selected), encoding="utf-8")

            class FakeProcessor:
                def __call__(self, images, return_tensors):
                    return {"pixel_values": torch.zeros(1, 3, 224, 224)}

            args = SimpleNamespace(dataset="APTOS2019", artifact_root=str(root),
                model_kind="dinov3_vits16", model_dir=str(root / "snapshot"),
                revision="b" * 40, weight_sha256="e" * 64,
                run_dir=str(run_dir), e3_pilot_dir=str(pilot_dir),
                preprocess_cache_root=str(root / "preprocessing"),
                output_dir=str(root / "sensitivity"), trust_run_checkpoint=True,
                trust_legacy_bin=False, device="cpu")
            v1 = np.full((512, 512, 3), 160, np.uint8)
            v2 = e3.bg_enhance(v1, 10)
            with (patch.object(fa, "load_train_val_manifest", return_value=(rows, manifest)),
                  patch.object(fa, "audit_snapshot", return_value=snapshot),
                  patch.object(fa, "load_local_backbone", return_value=(torch.nn.Identity(), FakeProcessor())),
                  patch.object(fa, "pooled", return_value=torch.ones(1, 384)),
                  patch.object(fa.phase, "PreprocessingCache", return_value=object()),
                  patch.object(e3, "_read_pixels", side_effect=lambda _cache, _sha, variant:
                               v1 if variant == "v1" else v2)):
                fa.sensitivity(args)
            result = json.loads((root / "sensitivity" / "sensitivity.json").read_text())
            self.assertEqual(result["selection"]["sample_count"], 5)
            self.assertEqual(len(result["conditions"]), 12)
            self.assertEqual(result["split"], "val")


if __name__ == "__main__":
    unittest.main()
