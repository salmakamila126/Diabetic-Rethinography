"""Synthetic CPU checks of staged optimization and the notebook suite."""

import ast
import contextlib
import copy
import csv
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
from novelty_integrated.optimization import backends, benchmark, hpc, models, notebook, recovery, registry, workflow
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

    def test_shared_validation_requires_single_cpu_affinity_and_no_isolation_claim(self):
        options = ["--job", "E5_APTOS2019", "--allocation-id", "synthetic", "--cpus", "2", "--threads", "1"]
        parser = hpc._parser()
        directory = Path("synthetic")
        with patch.object(hpc.os, "sched_setaffinity", create=True) as pin, patch.object(
                hpc, "_cpu_quota", return_value=1), patch.object(hpc, "_affinity", return_value=[2]) as affinity, patch.object(
                hpc, "_available_ram", return_value=6 * hpc.GIB), patch.object(hpc, "_disk", return_value=6 * hpc.GIB):
            with self.assertRaisesRegex(ValueError, "Use --shared-cpu"):
                hpc._resource_guard(parser.parse_args(["validate", *options]), directory)
            with self.assertRaisesRegex(ValueError, "without --confirm-isolated-cpu"):
                hpc._resource_guard(parser.parse_args(["validate", *options, "--shared-cpu", "--confirm-isolated-cpu"]), directory)
            args = parser.parse_args(["validate", *options, "--shared-cpu"])
            affinity.return_value = [2, 3]
            with self.assertRaisesRegex(ValueError, "taskset"):
                hpc._resource_guard(args, directory)
            affinity.return_value = [2]
            with self.assertRaisesRegex(ValueError, "--threads 1"):
                hpc._resource_guard(parser.parse_args(["validate", *options, "--shared-cpu", "--threads", "2"]), directory)
            self.assertEqual(hpc._resource_guard(args, directory), [2])
            post = parser.parse_args(["validate-recovered", *options, "--shared-cpu"])
            self.assertEqual(hpc._resource_guard(post, directory), [2])
            for stage in ("measure", "measure-recovered"):
                with self.assertRaisesRegex(ValueError, "exclusive CPU allocation"):
                    hpc._resource_guard(parser.parse_args([stage, *options, "--shared-cpu"]), directory)
                with self.assertRaisesRegex(ValueError, "applies only to untimed validation"):
                    hpc._resource_guard(parser.parse_args([stage, *options, "--shared-cpu", "--confirm-isolated-cpu"]), directory)
            self.assertEqual(pin.call_count, 2)

    def test_gpu_recovery_refuses_occupied_gpu_and_cpu_hides_cuda(self):
        cpu = hpc._parser().parse_args(["recover", "--job", "E7_APTOS2019", "--recovery-epochs", "3"])
        gpu = hpc._parser().parse_args(["recover", "--job", "E7_APTOS2019", "--recovery-epochs", "3",
                                       "--recovery-device", "cuda:0", "--gpu-uuid", "GPU-synthetic",
                                       "--confirm-ddr-e7-safe"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch.object(
                subprocess, "check_output", side_effect=["16000", "1234"]) as smi:
            with self.assertRaisesRegex(ValueError, "GPU is in use"):
                hpc._gpu_guard(gpu)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            hpc._gpu_guard(cpu)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(smi.call_count, 2)

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

    @contextlib.contextmanager
    def _fixture(self, backend_names=("eager", "torchscript"), extra_rows=0):
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
            rows += [{**rows[i % 5], "image_id": f"fixture-{i}"} for i in range(extra_rows)]
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
                "--trust-checkpoint", "--trust-legacy-snapshot", "--backend", *backend_names,
                "--warmup", "1", "--repeats", "2"]
            with patch.object(models, "load_rows", return_value={"val": rows, "train": []}), patch.object(
                    fa, "audit_snapshot", return_value=cfg["snapshot"]), patch.object(
                    fa, "load_local_backbone", side_effect=lambda *a, **k: (Backbone(), processor)), patch.object(
                    torch, "set_num_interop_threads"), contextlib.redirect_stdout(io.StringIO()):
                yield synthetic, arguments, output, checkpoint

    def test_stage_reuses_artifacts_and_reports_paired_recovery(self):
        with self._fixture() as (synthetic, arguments, output, checkpoint):
            base = cli._parser().parse_args(arguments)
            recovered = cli._parser().parse_args(arguments + ["--authorize-recovery", "--recovery-epochs", "3", "--prune", "0.3"])
            allocation = {"id": "synthetic", "cpus": [0], "threads": 1, "exclusive_cpu_confirmed": True}
            plan, recover_plan = cli._plan(base, synthetic), cli._plan(recovered, synthetic)
            source = {k: v for k, v in workflow._source().items() if k != "coordination.py"}
            with patch.object(workflow, "_source", return_value={**source, **workflow._PRE_VALIDATION_SOURCE}):
                workflow.execute("prepare", base, synthetic, plan, allocation)
            with self.assertRaisesRegex(ValueError, "passing unrecovered validate"):
                workflow.execute("recover", recovered, synthetic, recover_plan, allocation)
            shared = {**allocation, "exclusive_cpu_confirmed": False, "cpu_mode": "shared"}
            with patch.object(benchmark, "time_backend") as timer, patch.object(
                    workflow, "_source", return_value={**source, **workflow._PRE_RECOVERY_SOURCE}):
                self.assertEqual(workflow.execute("validate", base, synthetic, plan, shared), 0)
                timer.assert_not_called()
            native = output / "validation/report.json"
            old_hashes = {p: phase.sha256(p) for p in [checkpoint, native, native.with_name("summary.csv"),
                         output / "protocol.json", *(output / "prepared").glob("*/*")]}
            self.assertEqual(workflow.read(native)["outcome"], "PASS")
            self.assertFalse((output / "measurement").exists())

            def train(model, entry, rows, processor, *, epochs, micro_batch, device, resume, save_epoch):
                self.assertEqual(epochs, 3)
                self.assertEqual(set(rows), {"train"})
                self.assertEqual(device, "cpu")
                history = [] if resume is None else resume["history"]
                for epoch in range(len(history) + 1, epochs + 1):
                    history.append({"recovery_epoch": epoch})
                    save_epoch({"model": model.state_dict(), "history": history})
                    if trainer.call_count == 1:
                        raise InterruptedError("synthetic interruption")
                return {"objective": "synthetic", "history": history}

            recovered.recovery_ratio = 0.3
            with patch.object(recovery, "recover", side_effect=train) as trainer, patch.object(
                    benchmark, "time_backend") as timer:
                with patch.object(workflow, "_source", return_value={**source, **workflow._SERIAL_RECOVERY_SOURCE}):
                    with self.assertRaises(InterruptedError):
                        workflow.execute("recover", recovered, synthetic, recover_plan, allocation)
                progress_protocol = output / "recovered/recovery_protocol.json"
                old_recovery_sha = phase.sha256(progress_protocol)
                self.assertFalse((output / "recovered/pruned_30/prepared.json").exists())
                workflow.execute("recover", recovered, synthetic, recover_plan, allocation)
                self.assertEqual(phase.sha256(progress_protocol), old_recovery_sha)
                self.assertIsNotNone(trainer.call_args.kwargs["resume"])
                with self.assertRaisesRegex(ValueError, "arm incomplete"):
                    workflow.execute("validate-recovered", base, synthetic, plan, shared)
                recovered.recovery_ratio = None
                workflow.execute("recover", recovered, synthetic, recover_plan, allocation)
                self.assertEqual(trainer.call_count, 5)
                self.assertEqual(workflow.execute("validate-recovered", base, synthetic, plan, shared), 0)
                self.assertEqual(workflow.execute("validate", base, synthetic, plan, shared), 0)
                timer.assert_not_called()
            self.assertFalse((output / "measurement").exists())
            report = workflow.read(output / "validation-recovered/report.json")
            self.assertFalse(report["timing_performed"])
            self.assertEqual(report["unrecovered_validation_report_sha256"], old_hashes[native])
            self.assertEqual(report["ordered_preprocessed_tensor_sha256"],
                             workflow.read(native)["ordered_preprocessed_tensor_sha256"])
            expected_order = sorted((v for v in report["variants"] if v["ratio"]), key=lambda v: (
                -v["backends"]["eager"]["quality"]["macro_f1"], v["backends"]["eager"]["quality"]["grade_mae"], v["ratio"]))
            self.assertEqual(report["comparison"]["pruned_ratio_ranking"], [v["ratio"] for v in expected_order])
            self.assertEqual(report["comparison"]["selected_pruned_ratio"], expected_order[0]["ratio"])
            self.assertTrue(all(phase.sha256(path) == digest for path, digest in old_hashes.items()))

            with patch.object(backends, "build", side_effect=AssertionError("unexpected export")):
                workflow.execute("prepare", base, synthetic, plan, allocation)
                workflow.execute("measure", base, synthetic, plan, allocation)
                workflow.execute("measure-recovered", base, synthetic, plan, allocation)
            raw = workflow.read(output / "measurement/report.json")
            post = workflow.read(output / "post-recovery/report.json")
            self.assertEqual(raw["allocation"], allocation)
            self.assertEqual(raw["sample"], post["sample"])
            self.assertEqual(raw["protocol_sha256"], post["protocol_sha256"])
            self.assertEqual(len(raw["variants"]), 4)
            self.assertEqual(len(post["variants"]), 8)
            self.assertEqual(raw["plan"]["prune"], [0.1, 0.3, 0.5])
            self.assertEqual(post["recovery_policy"]["epochs"], 3)
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

    def test_validation_of_predecessor_preserves_artifacts_and_reports_all_backends(self):
        # Eager and TorchScript really execute. The two optional runtimes use fixtures here.
        build, load = backends.build, backends.load

        def export_fixture(name, model, inputs, output, **opts):
            if name in ("eager", "torchscript"):
                return build(name, model, inputs, output, **opts)
            path = output / ("model.onnx" if name == "onnxruntime" else "model.xml")
            path.write_text(name, encoding="utf-8")
            if name == "openvino":
                path.with_suffix(".bin").write_bytes(b"fixture")
            return backends.Backend(model, path, {})

        def load_fixture(name, model, path, names, **opts):
            if name in ("eager", "torchscript"):
                return load(name, model, path, names, **opts)
            self.assertEqual(names, ["pixels"])
            variant = path.parent.name
            if variant == "pruned_50" and name == "openvino":
                raise RuntimeError("synthetic load failure")

            def infer(*inputs):
                logits = model(*inputs)
                if variant == "pruned_30" and name == "onnxruntime":
                    return torch.nn.functional.one_hot((logits.argmax(dim=1) + 1) % 5, 5).float()
                return logits

            return backends.Backend(infer, path, {})

        with self._fixture(backends.BACKENDS, extra_rows=100) as (entry, argv, output, checkpoint):
            args = cli._parser().parse_args(argv + ["--sample-count", "100"])
            capabilities = {name: {"status": "CANDIDATE", "missing_packages": []} for name in backends.BACKENDS}
            with patch.object(backends, "capabilities", return_value=capabilities):
                plan = cli._plan(args, entry)
            allocation = {"cpu_mode": "shared", "exclusive_cpu_confirmed": False, "threads": 1, "cpus": [0]}
            predecessor = {**{k: v for k, v in workflow._source().items() if k != "coordination.py"},
                           **workflow._PRE_VALIDATION_SOURCE}
            with patch.object(workflow, "_source", return_value=predecessor), patch.object(
                    backends, "build", side_effect=export_fixture):
                workflow.execute("prepare", args, entry, plan, allocation)
            protocol = output / "protocol.json"
            inventory = {path: phase.sha256(path) for path in [checkpoint, protocol, *(output / "prepared").glob("*/*")]}
            with self.assertRaisesRegex(ValueError, "Run protocol differs"):
                workflow.execute("validate", args, entry, {**plan, "sample_count": 99}, allocation)
            with patch.object(workflow, "_source", return_value={**workflow._source(), "models.py": "unreviewed"}):
                with self.assertRaisesRegex(ValueError, "Run protocol differs"):
                    workflow.execute("validate", args, entry, plan, allocation)
            with patch.object(backends, "load", side_effect=load_fixture) as loader, patch.object(
                    backends, "build") as exporter, patch.object(benchmark, "time_backend") as timer, patch.object(
                    workflow.pruning, "profile") as profiler, patch.object(recovery, "recover") as trainer:
                self.assertEqual(workflow.execute("validate", args, entry, plan, allocation), 2)
                self.assertEqual(loader.call_count, 16)
                self.assertEqual(workflow.execute("validate", args, entry, plan, allocation), 2)
                self.assertEqual(loader.call_count, 16, "completed validation was rerun")
                for forbidden in (exporter, timer, profiler, trainer):
                    forbidden.assert_not_called()
            report = workflow.read(output / "validation/report.json")
            self.assertEqual(report["outcome"], "BACKEND_FAILURES")
            self.assertEqual(report["backend_rows"], 16)
            self.assertEqual(report["failures"], 2)
            self.assertEqual(report["sample"], workflow.read(protocol)["sample"])
            self.assertEqual(report["sample"]["count"], 100)
            self.assertEqual(report["sample"]["split"], "val")
            self.assertEqual(report["prepared_framework_source_sha256"], predecessor)
            self.assertEqual(report["framework_source_sha256"], workflow._source())
            self.assertEqual(report["allocation"], allocation)
            self.assertFalse(report["timing_performed"])
            self.assertRegex(report["ordered_preprocessed_tensor_sha256"], r"^[0-9a-f]{64}$")
            for variant in report["variants"]:
                eager = variant["backends"]["eager"]
                self.assertEqual(variant["backends"]["torchscript"]["quality"], eager["quality"])
                self.assertEqual(eager["parity"]["prediction_agreement"], 1.0)
                self.assertTrue(all("timing" not in row for row in variant["backends"].values()))
            self.assertEqual(report["variants"][2]["backends"]["onnxruntime"]["status"], "PARITY_FAILED")
            self.assertEqual(report["variants"][2]["backends"]["onnxruntime"]["parity"]["prediction_agreement"], 0)
            self.assertEqual(report["variants"][3]["backends"]["openvino"]["status"], "BACKEND_FAILED")
            summary = output / "validation/summary.csv"
            with summary.open(newline="", encoding="utf-8") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 16)
            self.assertTrue(all(table[0][metric] for metric in ("macro_f1", "accuracy", "qwk", "grade_mae")))
            self.assertListEqual(sorted(path.name for path in output.iterdir()), ["prepared", "protocol.json", "validation"])
            self.assertTrue(all(phase.sha256(path) == digest for path, digest in inventory.items()))
            summary.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Existing validation report differs"):
                workflow.execute("validate", args, entry, plan, allocation)
            (output / "prepared/pruned_50/model.bin").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "Invalid prepared artifact"):
                workflow.execute("validate", args, entry, plan, allocation)


if __name__ == "__main__":
    unittest.main()
