"""CPU-only framework tests with synthetic inputs and temporary artifacts."""

import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
from PIL import Image
import torch
from torch import nn

from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from novelty_integrated.e5.model import E5Heads
from novelty_integrated.e7.model import GlobalLocalModel
from novelty_integrated.optimization import __main__ as cli
from novelty_integrated.optimization import backends, benchmark, models, pruning, recovery, registry


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(4, 4, 3, padding=1, groups=4)
        self.norm = nn.LayerNorm(4)
        self.pwconv1 = nn.Linear(4, 16)
        self.pwconv2 = nn.Linear(16, 4)

    def forward(self, pixels):
        features = self.norm(self.dwconv(pixels).permute(0, 2, 3, 1))
        output = self.pwconv2(torch.nn.functional.gelu(self.pwconv1(features)))
        return pixels + output.permute(0, 3, 1, 2)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="convnext")
        self.stem = nn.Conv2d(3, 4, 2, stride=32)
        stage = nn.Module()
        stage.layers = nn.Sequential(Block())
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList([stage])
        self.output = nn.Conv2d(4, 768, 1)

    def forward(self, pixel_values):
        hidden = self.output(self.encoder.stages[0].layers(self.stem(pixel_values)))
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=hidden.mean((2, 3)))


class FakeProcessor:
    def __call__(self, *, images, return_tensors, do_resize=True):
        parts = [torch.from_numpy(np.array(image.resize((224, 224)) if do_resize else image))
                 .permute(2, 0, 1).float() / 255 for image in images]
        return {"pixel_values": torch.stack(parts)}

    def to_dict(self):
        return {"synthetic": True}


def val_rows():
    return [{"label": grade, "split": "val", "image_id": str(grade), "sha256": f"{grade:064x}"}
            for grade in range(5)]


def args_for_test():
    return ["run", "--dataset", "APTOS2019", "--arm", "E5", "--artifact-root", "unused",
            "--model-dir", "unused", "--checkpoint", "unused", "--output", "unused"]


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.entries = registry.registry()

    def test_research_registry(self):
        self.assertEqual(len(self.entries), 17)
        self.assertEqual(registry.STATUSES, {"SUPPORTED", "NEEDS_ADAPTER", "MISSING_CHECKPOINT",
                         "MISSING_SOURCE", "PENDING_TRAINING", "NOT_APPLICABLE", "UNVERIFIED", "COMPLETED"})
        self.assertEqual(self.entries["DDR", "B4"].status, "NOT_APPLICABLE")
        self.assertEqual(self.entries["DDR", "E7"].status, "PENDING_TRAINING")
        e7 = self.entries["APTOS2019", "E7"]
        self.assertEqual(e7.status, "COMPLETED")
        self.assertEqual(e7.describe()["checkpoint_digest_status"], "VERIFIED_ON_HPC")
        self.assertEqual(e7.describe()["checkpoint_availability"], "MISSING_CHECKPOINT")
        self.assertEqual(e7.describe()["benchmark_status"], "NOT_RUN")
        registry.require_sources(e7)
        for dataset in registry.DATASETS:
            expected = phase.SPECS[dataset]["manifest_sha256"]
            for arm in ("B0", "B1", "B2", "B3", "E5", "E6"):
                entry = self.entries[dataset, arm]
                self.assertEqual(entry.configuration["manifest"]["manifest_sha256"], expected)
                self.assertEqual(len(entry.checkpoint_sha256), 64)
                if arm != "B3":
                    registry.require_sources(entry)
        for entry in self.entries.values():
            if entry.external_source:
                with self.assertRaisesRegex(ValueError, "MISSING_SOURCE"):
                    registry.require_sources(entry)

    def test_cli_defaults_and_recovery_gate(self):
        with patch.object(cli, "execute") as execute, contextlib.redirect_stdout(io.StringIO()) as stdout:
            cli.main(args_for_test())
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "PLAN_ONLY")
            self.assertEqual(result["split"], "val")
            execute.assert_not_called()
        for extra in (["--recovery-epochs", "2"], ["--prune", "0"], ["--authorize-recovery"]):
            with self.assertRaises(ValueError):
                cli.main(args_for_test() + extra)
        args = cli._parser().parse_args(args_for_test() + ["--recovery-epochs", "1", "--authorize-recovery",
                                                          "--prune", "0", "0.3"])
        self.assertEqual(cli._plan(args, self.entries["APTOS2019", "E5"])["prune"], [0.0, 0.3])
        with patch.object(models, "verify_receipt") as verify:
            with self.assertRaisesRegex(ValueError, "optimization.hpc recover"):
                cli.main(args_for_test() + ["--execute", "--recovery-epochs", "3", "--authorize-recovery", "--prune", "0.3"])
            verify.assert_not_called()

    def test_missing_checkpoint_and_pending_never_deserialize(self):
        entry = self.entries["APTOS2019", "E5"]
        with tempfile.TemporaryDirectory() as work, patch.object(fa, "_load_checkpoint") as load:
            with self.assertRaisesRegex(FileNotFoundError, "MISSING_CHECKPOINT"):
                models.load_original(entry, Path(work) / "best.pt", work)
            (Path(work) / "best.pt").write_bytes(b"wrong receipt")
            with self.assertRaises(ValueError):
                models.load_original(entry, Path(work) / "best.pt", work, trust_checkpoint=True)
            with self.assertRaisesRegex(ValueError, "PENDING_TRAINING"):
                models.load_original(self.entries["DDR", "E7"], Path(work) / "best.pt", work)
            load.assert_not_called()

    def test_e7_record_requires_completed_run_lineage_and_raw_weights(self):
        entry = self.entries["APTOS2019", "E7"]
        state = {"record": entry.configuration, "epoch": 19, "best": entry.best, "selection_weights": "raw"}
        models.verify_record(entry, state)
        for changed in ({**state, "selection_weights": "ema"}, {**state, "epoch": 20},
                        {**state, "record": {**state["record"], "geometry": {}}}):
            with self.assertRaises(ValueError):
                models.verify_record(entry, changed)
        with tempfile.TemporaryDirectory() as work, patch.object(fa, "_load_checkpoint") as load:
            with self.assertRaisesRegex(FileNotFoundError, "MISSING_CHECKPOINT"):
                models.load_original(entry, Path(work) / "best.pt", work)
            load.assert_not_called()

    def test_strict_original_loader_checkpoint_identity_and_immutability(self):
        entry = self.entries["APTOS2019", "E5"]
        cfg = copy.deepcopy(entry.configuration)
        processor = FakeProcessor()
        cfg["effective_processor_sha256"] = fa._digest_json(processor.to_dict())
        head = E5Heads(768, cfg["train_counts"], ordinal=True)
        backbone = Backbone()
        state = {"record": cfg, "epoch": entry.best["epoch"], "best": entry.best,
                 "backbone": backbone.state_dict(), "heads": head.state_dict()}
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "best.pt"
            fa._save_checkpoint(path, state)
            files = [path, path.with_name("best.pt.sha256.json")]
            before = [p.read_bytes() for p in files]
            evidence = {**entry.evidence, "configuration": cfg, "checkpoint_sidecars": {
                "best.pt": {"sha256": phase.sha256(path), "bytes": path.stat().st_size}}}
            synthetic = registry.Experiment(entry.dataset, entry.arm, entry.family, entry.status,
                                            entry.evidence_path, evidence)
            with patch.object(fa, "audit_snapshot", return_value=cfg["snapshot"]), patch.object(
                    fa, "load_local_backbone", return_value=(Backbone(), processor)) as construct:
                with self.assertRaisesRegex(ValueError, "requires"):
                    models.load_original(synthetic, path, work)
                loaded, _, receipt = models.load_original(synthetic, path, work,
                                                            trust_checkpoint=True, trust_snapshot=True)
                construct.assert_called_once()
                self.assertEqual(receipt["epoch"], entry.best["epoch"])
                pixels = torch.zeros(1, 3, 224, 224)
                torch.testing.assert_close(loaded(pixels), head.predict_logits(backbone(pixels).pooler_output))
                pruned, _ = pruning.prune_hf(loaded, 0.5)
                pruning.profile(pruned, (pixels,))
            self.assertEqual([p.read_bytes() for p in files], before)
        invalid = {**state, "record": {**cfg, "dataset": "DDR"}}
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            models.verify_record(entry, invalid)
        with self.assertRaises(RuntimeError):
            models.reconstruct(entry.family, Backbone(), {**state, "heads": {}}, cfg["train_counts"])

    def test_preprocessing_rejects_test_before_image_access(self):
        rows = val_rows()
        self.assertEqual(models.validation_batches(rows, batch_size=1),
                         models.validation_batches(rows, batch_size=1, sample_count=5))
        with patch.object(models.Image, "open") as opened:
            bad = [{**rows[0], "split": "test"}]
            for function in (models.prepare_inputs, models.verify_inputs):
                with self.assertRaisesRegex(ValueError, "validation"):
                    function(bad, FakeProcessor(), "hf_v1_linear")
            with self.assertRaisesRegex(ValueError, "validation"):
                models.validation_batches(bad, batch_size=1)
            opened.assert_not_called()
        with self.assertRaisesRegex(ValueError, "divisible"):
            models.validation_batches(rows, batch_size=2)

    def test_direct_preprocessing_matches_source_including_e7_views(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "image.png"
            Image.new("RGB", (63, 49), (100, 40, 24)).save(path)
            row = {**val_rows()[0], "path": str(path), "sha256": phase.sha256(path)}
            for family, dimensions in (("hf_v1_linear", [(1, 3, 224, 224)]),
                                       ("hf_v1_ordinal", [(1, 3, 224, 224)]),
                                       ("hf_v1_global_local", [(1, 3, 384, 384), (1, 2, 3, 224, 224)])):
                self.assertEqual([tuple(t.shape) for t in models.verify_inputs(
                    [row], FakeProcessor(), family)], dimensions)
            path.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                models.verified_images([[row]])

    def test_pruning_counts_deterministically_and_reconstructs(self):
        torch.manual_seed(42)
        original = models.CategoricalModel(Backbone(), nn.Linear(768, 5)).eval()
        saved = {k: v.clone() for k, v in original.state_dict().items()}
        inputs = (torch.zeros(2, 3, 224, 224),)
        parent = pruning.profile(original, inputs)
        for ratio in pruning.RATIOS:
            with self.subTest(ratio=ratio):
                pruned, topology = pruning.prune_hf(original, ratio)
                removed = 16 - topology[0]["pruned_width"]
                result = pruning.profile(pruned, inputs)
                self.assertEqual(parent["parameters"] - result["parameters"], (2 * 4 + 1) * removed)
                self.assertEqual(parent["macs_per_image"] - result["macs_per_image"], 2 * 4 * removed * 49)
                rebuilt, _ = pruning.prune_hf(original, ratio, topology=topology)
                rebuilt.load_state_dict(pruned.state_dict(), strict=True)
                torch.testing.assert_close(pruned(*inputs), rebuilt(*inputs))
        self.assertTrue(all(torch.equal(saved[k], v) for k, v in original.state_dict().items()))
        bad = models.CategoricalModel(Backbone(), nn.Linear(768, 5))
        bad.backbone.encoder.stages[0].layers[0].grn = nn.Identity()
        with self.assertRaisesRegex(ValueError, "without GRN"):
            pruning.prune_hf(bad, 0.3)

    def test_e7_selected_ema_and_shared_three_view_accounting(self):
        counts = [2] * 5
        raw = GlobalLocalModel(Backbone(), counts).eval()
        ema = copy.deepcopy(raw)
        with torch.no_grad():
            ema.heads.categorical.bias.copy_(torch.arange(5, dtype=torch.float32))
        state = {"selection_weights": "ema", "model": raw.state_dict(), "ema": {"shadow": ema.state_dict()}}
        loaded = models.reconstruct("hf_v1_global_local", Backbone(), state, counts)
        sample = (torch.zeros(1, 3, 384, 384), torch.zeros(1, 2, 3, 224, 224))
        torch.testing.assert_close(loaded(*sample), ema(*sample)[0])
        pruned, topology = pruning.prune_hf(loaded, 0.5)
        before, after = pruning.profile(loaded, sample), pruning.profile(pruned, sample)
        self.assertEqual(len(topology), 1)
        self.assertEqual(before["macs_per_image"] - after["macs_per_image"], 2 * 4 * 8 * (144 + 2 * 49))
        torch.testing.assert_close(loaded.model.gate.weight, pruned.model.gate.weight)
        with self.assertRaisesRegex(ValueError, "requires"):
            loaded(sample[0], torch.zeros(1, 1, 3, 224, 224))

    def test_e6_recovery_dispatch_preserves_objective_and_train_only_schedule(self):
        entry = self.entries["APTOS2019", "E6"]
        cfg = entry.configuration
        model = models.CategoricalModel(Backbone(), E5Heads(768, cfg["train_counts"], ordinal=True))
        rows = {"train": [{"label": grade, "split": "train"} for grade, count in enumerate(cfg["train_counts"])
                          for _ in range(count)], "val": val_rows()}
        with patch.object(fa, "_epochs_batches", return_value=()) as batches, patch.object(
                recovery.e6runner, "_train_batches", return_value={"adjacent_loss": 0.0}) as e6, patch.object(
                recovery.e5runner, "_train_batches") as e5:
            info = recovery.recover(model, entry, rows, FakeProcessor(), epochs=1, micro_batch=4)
            e5.assert_not_called()
        self.assertEqual(e6.call_args.kwargs["epoch"], entry.best["epoch"])
        self.assertEqual(batches.call_args.args[-1], entry.best["epoch"] + 1)
        self.assertEqual(info["history"][0]["train"]["adjacent_loss"], 0.0)

    def test_e7_recovery_keeps_multiview_collation_native_loss_and_gate(self):
        entry = self.entries["APTOS2019", "E7"]
        configuration = {**entry.configuration, "train_counts": [1] * 5}
        entry = registry.Experiment(entry.dataset, entry.arm, entry.family, entry.status, entry.evidence_path,
                                    {**entry.evidence, "configuration": configuration})
        model = models.GlobalLocalCategorical(models.GlobalLocalModel(Backbone(), [1] * 5))
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "sample.png"
            Image.new("RGB", (31, 35), (100, 50, 25)).save(image)
            rows = [{**row, "split": "train", "path": str(image), "sha256": phase.sha256(image)} for row in val_rows()]

            def native(multiview, optimizer, ema, batches, **protocol):
                self.assertIs(multiview, model.model)
                self.assertIn(id(multiview.gate.weight), [id(p) for p in optimizer.param_groups[1]["params"]])
                self.assertIsNone(ema)
                self.assertEqual(protocol["epoch"], entry.best["epoch"])
                self.assertEqual(protocol["accumulation"], 8)
                prepared = list(batches)
                self.assertEqual(tuple(prepared[0][0].shape), (4, 3, 384, 384))
                self.assertEqual(tuple(prepared[0][1].shape), (4, 2, 3, 224, 224))
                self.assertEqual(sum(len(batch[2]) for batch in prepared), 5)
                return {"loss": 1.0}

            with patch.object(recovery.e7runner, "_train_batches", side_effect=native) as runner, patch.object(
                    recovery.e5runner, "_train_batches") as e5, patch.object(recovery.e6runner, "_train_batches") as e6:
                result = recovery.recover(model, entry, {"train": rows, "val": [{"split": "test"}]},
                                          FakeProcessor(), epochs=3, micro_batch=4)
                self.assertEqual(runner.call_count, 3)
                e5.assert_not_called()
                e6.assert_not_called()
            self.assertEqual(result["export_weights"], "raw")
            self.assertEqual(result["history"][-1]["augmentation_epoch"], entry.best["epoch"] + 3)
            with self.assertRaisesRegex(ValueError, "original frozen training"):
                recovery.recover(model, entry, {"train": [{**rows[0], "split": "test"}, *rows[1:]]},
                                 FakeProcessor(), epochs=3, micro_batch=4)

    def test_recovery_epoch_resume_restores_optimizer_and_rng(self):
        entry = self.entries["APTOS2019", "E5"]
        counts = entry.configuration["train_counts"]
        model = models.CategoricalModel(Backbone(), E5Heads(768, counts, ordinal=True))
        train = [{"label": grade, "split": "train"} for grade, count in enumerate(counts) for _ in range(count)]

        def one_step(backbone, heads, optimizer, batches, **opts):
            param = next(backbone.parameters())
            signal = float(torch.rand(())) + float(np.random.random()) + random.random()
            optimizer.zero_grad(set_to_none=True)
            param.grad = torch.full_like(param, signal)
            optimizer.step()
            return {"loss": signal}

        with tempfile.TemporaryDirectory() as tmp, patch.object(fa, "_epochs_batches", return_value=()), patch.object(
                recovery.e5runner, "_train_batches", side_effect=one_step), contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "state.pt"

            def interrupt(state):
                fa._save_checkpoint(path, state)
                raise InterruptedError

            processor = FakeProcessor()
            opts = {"epochs": 3, "micro_batch": 4}
            reference, resumed = copy.deepcopy(model), copy.deepcopy(model)
            full = recovery.recover(reference, entry, {"train": train}, processor, **opts)
            with self.assertRaises(InterruptedError):
                recovery.recover(resumed, entry, {"train": train}, processor, save_epoch=interrupt, **opts)
            boundary = torch.load(path, map_location="cpu", weights_only=True)
            self.assertTrue(boundary["optimizer"]["state"])
            resumed_history = recovery.recover(resumed, entry, {"train": train}, processor, resume=boundary, **opts)
        for key, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
        self.assertEqual([r["train"] for r in full["history"]], [r["train"] for r in resumed_history["history"]])

    def test_cpu_smoke_synthetic_eager_torchscript_timing_serialization(self):
        fake = nn.Linear(6, 5).eval()
        batches = [[row] for row in val_rows()]
        def prepare(batch):
            return (torch.nn.functional.one_hot(torch.tensor([r["label"] for r in batch]), 6).float(),)
        truth, digest = benchmark.collect_logits(fake, batches, prepare)
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            plan = {"backends": ["eager", "torchscript"], "backend_candidates": backends.capabilities(),
                    "threads": 1, "warmup": 1, "repeats": 2, "parity_tolerances": {"atol": 1e-4, "rtol": 1e-4}}
            variant = cli._measure_variant("original", fake, batches, prepare, plan, root / "original",
                                           digest, checked_logits=truth)
            for result in variant["backends"].values():
                self.assertEqual(result["status"], "VALIDATED")
                self.assertEqual(result["parity"]["prediction_agreement"], 1.0)
                self.assertGreater(result["artifact"]["bytes"], 0)
                for scope in ("model_only", "end_to_end"):
                    self.assertEqual([len(p["latency_ns"]) for p in result["timing"][scope]["passes"]], [5, 5])
            shifted = truth.roll(1, dims=1)
            self.assertFalse(benchmark.parity(truth, shifted)["passed"])
            cli.reporting.write_report(root, {"dataset": "synthetic", "arm": "test", "sample": {"count": 5},
                                               "variants": [variant]})
            self.assertEqual(json.loads((root / "report.json").read_text())["sample"]["count"], 5)
            ElementTree.parse(root / "comparison.svg")
            with (root / "summary.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
        self.assertEqual(backends.capabilities()["eager"]["status"], "CANDIDATE")

    def test_complete_synthetic_run_compares_same_samples_and_creates_external_outputs(self):
        entry = self.entries["APTOS2019", "E5"]
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            artifact = root / "artifacts"
            manifests = artifact / "cad_vit_runs" / "manifests"
            manifests.mkdir(parents=True)
            manifest = manifests / "APTOS2019_seed42.csv"
            manifest.write_text("synthetic\n", encoding="utf-8")
            manifest.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            rows = val_rows()
            for row in rows:
                path = root / f"{row['label']}.png"
                Image.new("RGB", (21, 23), (40 * row["label"], 20, 60)).save(path)
                row.update(path=str(path), sha256=phase.sha256(path))
            cfg = copy.deepcopy(entry.configuration)
            cfg["manifest"]["manifest_sha256"] = phase.sha256(manifest)
            cfg["effective_processor_sha256"] = fa._digest_json(processor.to_dict())
            checkpoint = root / "run" / "best.pt"
            checkpoint.parent.mkdir()
            fa._save_checkpoint(checkpoint, {"record": cfg, "epoch": entry.best["epoch"], "best": entry.best,
                "backbone": Backbone().state_dict(), "heads": E5Heads(768, cfg["train_counts"], ordinal=True).state_dict()})
            evidence = {**entry.evidence, "configuration": cfg, "checkpoint_sidecars": {
                "best.pt": {"sha256": phase.sha256(checkpoint), "bytes": checkpoint.stat().st_size}}}
            synthetic = registry.Experiment(entry.dataset, entry.arm, entry.family, entry.status,
                                            entry.evidence_path, evidence)
            output = root / "output"
            args = cli._parser().parse_args(["run", "--dataset", "APTOS2019", "--arm", "E5",
                "--artifact-root", str(artifact), "--model-dir", str(root / "snapshot"), "--checkpoint", str(checkpoint),
                "--output", str(output), "--trust-checkpoint", "--trust-legacy-snapshot",
                "--backend", "eager", "--warmup", "1", "--repeats", "2", "--prune", "0.3", "--execute"])
            plan = cli._plan(args, synthetic)
            with patch.object(models, "load_rows", return_value={"val": rows}), patch.object(fa, "audit_snapshot",
                    return_value=cfg["snapshot"]), patch.object(fa, "load_local_backbone",
                    return_value=(Backbone(), processor)), patch.object(torch, "set_num_interop_threads"), patch.object(
                    recovery, "recover") as train, contextlib.redirect_stdout(io.StringIO()):
                cli.execute(args, synthetic, plan)
                train.assert_not_called()
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["sample"]["frozen_validation_positions_zero_based"], list(range(5)))
            native, pruned = report["variants"]
            self.assertEqual(pruned["treatment"], "pruned_without_recovery")
            self.assertLess(pruned["profile"]["parameters"], native["profile"]["parameters"])
            self.assertIn("per_grade_recall", pruned["delta_vs_original"]["same_backend"]["eager"]["quality"])
            self.assertNotIn(str(root), json.dumps(report))
            self.assertEqual(phase.sha256(checkpoint), synthetic.checkpoint_sha256)
            with self.assertRaisesRegex(ValueError, "fresh external"):
                cli.execute(args, synthetic, plan)

    def _optional_export(self, name):
        model = nn.Linear(6, 5).eval()
        sample = (torch.randn(2, 6),)
        with tempfile.TemporaryDirectory() as work:
            candidate = backends.build(name, model, sample, Path(work), threads=1)
            reference = model(*sample).detach()
            self.assertTrue(benchmark.parity(reference, torch.as_tensor(candidate.infer(*sample)))["passed"])

    def test_optional_onnxruntime_export(self):
        if backends.capabilities()["onnxruntime"]["missing_packages"]:
            self.skipTest("onnx and/or onnxruntime unavailable")
        self._optional_export("onnxruntime")

    def test_optional_openvino_export(self):
        if backends.capabilities()["openvino"]["missing_packages"]:
            self.skipTest("onnx and/or openvino unavailable")
        self._optional_export("openvino")

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Transformers unavailable")
    def test_actual_hf_class_with_random_weights(self):
        from transformers import ConvNextConfig, ConvNextModel

        config = ConvNextConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 768])
        model = models.CategoricalModel(ConvNextModel(config), nn.Linear(768, 5)).eval()
        pruned, topology = pruning.prune_hf(model, 0.3)
        sample = (torch.randn(1, 3, 224, 224),)
        self.assertEqual(len(topology), 4)
        self.assertLess(pruning.profile(pruned, sample)["macs_per_image"], pruning.profile(model, sample)["macs_per_image"])

    @unittest.skipUnless(importlib.util.find_spec("timm"), "timm unavailable")
    def test_actual_timm_grn_class_with_random_weights(self):
        from timm.models.convnext import ConvNeXt
        model = ConvNeXt(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64), use_grn=True, num_classes=5).eval()
        pruned, topology = pruning.prune_timm_v2(model, 0.3)
        sample = (torch.randn(1, 3, 224, 224),)
        self.assertEqual(len(topology), 4)
        self.assertLess(pruning.profile(pruned, sample)["macs_per_image"], pruning.profile(model, sample)["macs_per_image"])


if __name__ == "__main__":
    unittest.main()
