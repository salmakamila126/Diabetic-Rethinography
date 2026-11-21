"""Synthetic CPU checks of staged optimization and the notebook suite."""

import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import nbformat
from PIL import Image
import torch

from notebooks import build_optimization_notebooks as builder
from novelty import foundation_adaptation as fa
from novelty import preof_phase1 as phase
from novelty_integrated.e5.model import E5Heads
from novelty_integrated.optimization import __main__ as cli
from novelty_integrated.optimization import backends, hpc, models, notebook, recovery, registry, workflow
from tests.test_convnext_optimization import Backbone, FakeProcessor, val_rows


class CampaignTests(unittest.TestCase):
    def test_inventory_and_notebooks_are_static(self):
        jobs = hpc.inventory()
        self.assertEqual(tuple(jobs)[:5], hpc.PRIORITY)
        self.assertEqual(len(jobs), 16)
        self.assertNotIn("B4_DDR", jobs)
        self.assertEqual(jobs["E7_DDR"].status, "PENDING_TRAINING")
        self.assertEqual(hpc.checkpoint_relative(jobs["E7_APTOS2019"]),
                         "foundation_e7/APTOS2019/run/train/best.pt")
        self.assertEqual(hpc.receipt(jobs["E7_APTOS2019"])["validation_images"], 366)
        self.assertIsNone(hpc.checkpoint_relative(jobs["E7_DDR"]))
        directory = builder.ROOT / "notebooks/optimization"
        self.assertEqual({p.stem for p in directory.glob("*.ipynb")}, set(jobs))
        for key, entry in jobs.items():
            document = nbformat.read(directory / f"{key}.ipynb", as_version=4)
            nbformat.validate(document)
            self.assertEqual(document, builder.notebook(key, entry))
            self.assertEqual(document.metadata.optimization.execution_default, "NOT_RUN")
            source = "\n".join(c.source for c in document.cells if c.cell_type == "code")
            ast.parse(source)
            self.assertIn("RUN_STAGE = None", source)
            self.assertTrue(all(not c.outputs for c in document.cells if c.cell_type == "code"))
            for cell in document.cells:
                if cell.cell_type == "markdown":
                    for path in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", cell.source):
                        self.assertTrue((directory / path).exists(), path)

    def test_resource_guards_refuse_unallocated_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = hpc._parser().parse_args(["measure", "--job", "E5_APTOS2019"])
            with patch.object(hpc.os, "sched_setaffinity", create=True) as set_affinity:
                args.allocation_id, args.cpus = "synthetic", "2-5"
                with self.assertRaisesRegex(ValueError, "exclusive CPU allocation"):
                    hpc._resource_guard(args, Path(tmp))
                args.confirm_isolated_cpu = True
                with patch.object(hpc, "_affinity", return_value=[2, 3, 4, 5]), patch.object(
                        hpc, "_cpu_quota", return_value=2):
                    with self.assertRaisesRegex(ValueError, "cgroup quota"):
                        hpc._resource_guard(args, Path(tmp))
                set_affinity.assert_not_called()
        self.assertEqual(hpc._parse_cpus("2-4,6,8-9"), [2, 3, 4, 6, 8, 9])

    def test_notebook_default_dispatches_no_process(self):
        from IPython import display

        env = {"CADVIT_REPOSITORY": str(builder.ROOT), "CADVIT_OPT_CAMPAIGN": ""}
        with patch.dict(os.environ, env), patch.object(subprocess, "run") as run, patch.object(display, "display"):
            for key, entry in hpc.inventory().items():
                variables = {}
                for cell in builder.notebook(key, entry).cells:
                    if cell.cell_type == "code":
                        exec(compile(cell.source, key, "exec"), variables)
            run.assert_not_called()

    def test_audit_missing_checkpoint_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = hpc._parser().parse_args(["audit", "--job", "E7_APTOS2019",
                "--root", str(root), "--campaign-root", str(root / "optimization_runs/x"), "--backend", "eager"])
            with patch.object(models, "load_original") as deserialize:
                report = hpc.audit(args, hpc.inventory()[args.job])
            deserialize.assert_not_called()
            self.assertEqual(report["status"], "BLOCKED")
            self.assertIn("MISSING_CHECKPOINT", report["checks"]["checkpoint_binary_and_sidecar"]["reason"])
            self.assertFalse(any(root.iterdir()))

    def test_atomic_stage_restarts_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "prepared"
            def fail(scratch):
                (scratch / "partial").touch()
                raise InterruptedError
            with self.assertRaises(InterruptedError):
                workflow._publish(target, fail)
            self.assertFalse(target.exists())
            workflow._publish(target, lambda scratch: (scratch / "complete").touch())
            self.assertTrue((target / "complete").exists())
            with self.assertRaises(FileExistsError):
                workflow._publish(target, lambda scratch: (scratch / "discard").touch())
            self.assertFalse((target / "discard").exists())

    def test_stage_reuses_artifacts_and_reports_paired_recovery(self):
        entry = hpc.inventory()["E5_APTOS2019"]
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact"
            manifests = artifact / "cad_vit_runs/manifests"
            manifests.mkdir(parents=True)
            manifest = manifests / "APTOS2019_seed42.csv"
            manifest.write_text("synthetic\n", encoding="utf-8")
            manifest.with_suffix(".json").write_text("{}\n", encoding="utf-8")
            rows = val_rows()
            for row in rows:
                image = root / f"{row['label']}.png"
                Image.new("RGB", (21, 23), (40 * row["label"], 20, 60)).save(image)
                row.update(path=str(image), sha256=phase.sha256(image))
            cfg = copy.deepcopy(entry.configuration)
            cfg["manifest"]["manifest_sha256"] = phase.sha256(manifest)
            cfg["effective_processor_sha256"] = fa._digest_json(processor.to_dict())
            checkpoint = root / "original/best.pt"
            checkpoint.parent.mkdir()
            fa._save_checkpoint(checkpoint, {"record": cfg, "epoch": entry.best["epoch"], "best": entry.best,
                "backbone": Backbone().state_dict(), "heads": E5Heads(768, cfg["train_counts"], ordinal=True).state_dict()})
            synthetic = registry.Experiment(entry.dataset, entry.arm, entry.family, entry.status, entry.evidence_path,
                {**entry.evidence, "configuration": cfg, "checkpoint_sidecars": {
                    "best.pt": {"sha256": phase.sha256(checkpoint), "bytes": checkpoint.stat().st_size}}})
            output = root / "campaign"
            arguments = ["run", "--dataset", "APTOS2019", "--arm", "E5", "--artifact-root", str(artifact),
                "--model-dir", str(root / "snapshot"), "--checkpoint", str(checkpoint), "--output", str(output),
                "--trust-checkpoint", "--trust-legacy-snapshot", "--backend", "eager", "torchscript",
                "--warmup", "1", "--repeats", "2"]
            base = cli._parser().parse_args(arguments)
            recovered = cli._parser().parse_args(arguments + ["--authorize-recovery", "--recovery-epochs", "1", "--prune", "0.3"])
            allocation = {"id": "synthetic", "cpus": [0], "threads": 1, "exclusive_cpu_confirmed": True}
            with patch.object(models, "load_rows", return_value={"val": rows, "train": []}), patch.object(
                    fa, "audit_snapshot", return_value=cfg["snapshot"]), patch.object(
                    fa, "load_local_backbone", side_effect=lambda *a, **k: (Backbone(), processor)), patch.object(
                    torch, "set_num_interop_threads"), contextlib.redirect_stdout(io.StringIO()):
                workflow.execute("prepare", base, synthetic, cli._plan(base, synthetic), allocation)
                with self.assertRaisesRegex(ValueError, "unrecovered measurement"):
                    workflow.execute("recover", recovered, synthetic, cli._plan(recovered, synthetic), allocation)
                with patch.object(backends, "build", side_effect=AssertionError("unexpected export")):
                    workflow.execute("prepare", base, synthetic, cli._plan(base, synthetic), allocation)
                    workflow.execute("measure", base, synthetic, cli._plan(base, synthetic), allocation)
                with patch.object(recovery, "recover", return_value={"objective": "synthetic"}) as trainer:
                    workflow.execute("recover", recovered, synthetic, cli._plan(recovered, synthetic), allocation)
                    self.assertEqual(trainer.call_count, 4)
                with patch.object(backends, "build", side_effect=AssertionError("unexpected export")):
                    workflow.execute("measure-recovered", base, synthetic, cli._plan(base, synthetic), allocation)
            raw = workflow.read(output / "measurement/report.json")
            post = workflow.read(output / "post-recovery/report.json")
            self.assertEqual(raw["allocation"], allocation)
            self.assertEqual(raw["sample"], post["sample"])
            self.assertEqual(raw["protocol_sha256"], post["protocol_sha256"])
            self.assertEqual(len(raw["variants"]), 4)
            self.assertEqual(len(post["variants"]), 8)
            self.assertEqual(raw["plan"]["prune"], [0.1, 0.3, 0.5])
            self.assertEqual(post["recovery_policy"]["epochs"], 1)
            self.assertTrue(all(backend["status"] == "VALIDATED" for variant in post["variants"]
                                for backend in variant["backends"].values()))
            self.assertIn("delta_vs_pruned_before_recovery", post["variants"][-1])
            self.assertIn("Grade MAE", notebook.render(post))
            ElementTree.parse(output / "post-recovery/tradeoff.svg")
            self.assertEqual(phase.sha256(checkpoint), synthetic.checkpoint_sha256)
            path = output / "prepared/pruned_30"
            (path / "inference_state_dict.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Invalid prepared artifact"):
                workflow._prepared(path, raw["protocol_sha256"], .3)


if __name__ == "__main__":
    unittest.main()
