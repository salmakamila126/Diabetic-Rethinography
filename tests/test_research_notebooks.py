"""CPU-only integrity checks for the dataset-specific research notebooks."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = "reproduction/CAD_ViT_Reproduction.ipynb"
DATASETS = ("APTOS2019", "DDR")
NOTEBOOKS = {
    f"{folder}/{arm}_{dataset}.ipynb": (arm, dataset)
    for folder, arms in (("baselines", ("B0","B1","B2","B3","B4")), ("novelty", ("E5","E6","E7")))
    for arm in arms for dataset in DATASETS
}
PRE_MIGRATION = "027e586f607bde4877b395ca566082c713340e8b"


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class ResearchNotebookIntegrity(unittest.TestCase):
    def test_notebook_schema_syntax_and_guarded_actions(self):
        self.assertEqual({p.relative_to(ROOT / "notebooks").as_posix()
                          for p in (ROOT / "notebooks").rglob("*.ipynb")},
                         {REPRODUCTION, *NOTEBOOKS})
        self.assertFalse((ROOT / "CAD_ViT_Novelty.ipynb").exists())
        for name in (REPRODUCTION, *NOTEBOOKS):
            path = ROOT / "notebooks" / name
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            if name in NOTEBOOKS:
                arm, dataset = NOTEBOOKS[name]
                self.assertTrue(notebook.cells[0].source.startswith(f"# {arm} — {dataset}"), name)
                self.assertEqual(notebook.metadata.research.dataset, dataset)
                self.assertEqual(notebook.metadata.research.evidence_scope, "validation")
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            self.assertGreaterEqual(len(code_cells), 1, name)
            for position, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                tree = ast.parse(cell.source, filename=f"{name}:{position}")
                self.assertEqual(notebook.cells[position - 1].cell_type, "markdown")
                self.assertIsNone(cell.execution_count)
                for output in cell.outputs:
                    self.assertEqual(output.output_type, "display_data", name)
                    origin = output.data["text/plain"].removeprefix("Original, committed figure: ")
                    artifact = (ROOT / origin).resolve()
                    self.assertTrue(artifact.is_relative_to(ROOT), origin)
                    if "image/png" in output.data:
                        self.assertEqual(base64.b64decode(output.data["image/png"]), artifact.read_bytes())
                    else:
                        self.assertEqual(output.data["image/svg+xml"], artifact.read_text(encoding="utf-8"))
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.startswith("RUN_"):
                                self.assertIs(ast.literal_eval(node.value), False, (name, target.id))
                self.assertNotIn("--execute-final-test", cell.source)
                self.assertNotIn("pip install", cell.source)
            source = "\n".join(cell.source for cell in code_cells)
            if name == REPRODUCTION or NOTEBOOKS[name][0] in ("B0","B1","B2","E5","E6","E7"):
                self.assertIn("RUN_TRAINING = False", source)
            if name != REPRODUCTION and NOTEBOOKS[name][0] in ("B3","B4"):
                self.assertNotIn("run_step", source)

    def test_local_markdown_links_and_figures(self):
        paths = [ROOT / "README.md", ROOT / "notebooks" / "README.md",
                 *(ROOT / "docs" / p for p in ("RESEARCH_CONSOLIDATION_AUDIT.md",
                     "NOTEBOOK_EXPERIMENT_AUDIT.md", "OPTIMIZATION_COMPATIBILITY.md",
                     "CADVIT_NOVELTY_PROTOCOL.md", "COLLABORATOR_GUIDE.md", "FOUNDATION_TRAINING_READINESS.md")),
                 ROOT / "novelty_integrated/e5/README.md", ROOT / "results/APTOS2019/e5_integrated/README.md"]
        for name in (REPRODUCTION, *NOTEBOOKS):
            paths.append(ROOT / "notebooks" / name)
        pattern = re.compile(r"\]\(([^)]+)\)")
        for path in paths:
            if path.suffix == ".ipynb":
                text = "\n".join(cell.source for cell in nbformat.read(path, as_version=4).cells
                                 if cell.cell_type == "markdown")
            else:
                text = path.read_text(encoding="utf-8")
            for target in pattern.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if target:
                    self.assertTrue((path.parent / target).exists(), f"{path}: {target}")

    def test_b0_b4_evidence_is_validation_only_and_complete(self):
        evidence = read_json("novelty/convnext_b0_b4/validation_evidence.json")
        runs = evidence["runs"]
        self.assertEqual(len(runs), 10)
        self.assertEqual(sum(len(run["history"]) for run in runs), 200)
        self.assertEqual({run["dataset"] for run in runs if run["arm"].startswith("B4")},
                         {"APTOS2019"})
        for run in runs:
            self.assertEqual([row["epoch"] for row in run["history"]], list(range(1, 21)))
            matrix = run["best"]["validation"]["confusion_matrix"]
            support = [sum(row) for row in matrix]
            self.assertEqual(support, [grade["support"] for grade in
                                       run["best"]["validation"]["per_grade"]])
            self.assertEqual(sum(matrix[i][i] for i in range(5)),
                             run["best"]["validation"]["correct"])
            for metadata in run["figures"].values():
                figure = ROOT / "novelty" / "convnext_b0_b4" / metadata["file"]
                self.assertEqual(hashlib.sha256(figure.read_bytes()).hexdigest(),
                                 metadata["sha256"])

    def test_static_tables_are_traceable_to_source_rows(self):
        first = nbformat.read(ROOT / "notebooks" / REPRODUCTION, as_version=4)
        raw = subprocess.check_output(
            ["git","show",f"{PRE_MIGRATION}:notebooks/01_CAD_ViT_Reproduction.ipynb"], cwd=ROOT)
        self.assertEqual((ROOT / "notebooks" / REPRODUCTION).read_bytes(),
                         raw.replace(b"](../", b"](../../"))
        markdown = [cell.source for cell in first.cells if cell.cell_type == "markdown"]
        for label, folder in (("APTOS2019 balanced", "results/APTOS2019/full_oversampling_amp_fp16_80epochs"),
                              ("DDR recovery", "results/DDR/full_oversampling_fp32_math_mb8_80epochs")):
            cell = next(text for text in markdown if text.startswith(f"**{label}: complete original 80-epoch"))
            rows = [json.loads(line) for line in (ROOT / folder / "history.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            lines = [line for line in cell.splitlines() if line.startswith("| ")][2:]
            self.assertEqual(len(lines), len(rows))
            for line, source in zip(lines, rows):
                values = [value.strip() for value in line.strip("| ").split("|")]
                self.assertEqual(int(values[0]), source["epoch"])
                for displayed, key in zip(values[1:], ("train_loss", "val_loss", "val_accuracy", "val_macro_f1")):
                    self.assertAlmostEqual(float(displayed), source[key], delta=0.0000005)
        from notebooks import build_experiment_notebooks as builder

        base = read_json("novelty/convnext_b0_b4/validation_evidence.json")["runs"]
        totals = {"B":0,"E":0}
        for path, (arm,dataset) in NOTEBOOKS.items():
            actual = nbformat.read(ROOT / "notebooks" / path, as_version=4)
            expected = builder.experiment(arm,dataset,base)
            for i, cell in enumerate(expected.cells):
                cell.id = f"{arm.lower()}-{dataset.lower()}-{i:02}"
            self.assertEqual(actual, expected, path)
            totals[arm[0]] += actual.metadata.research.epoch_records
        self.assertEqual(totals, {"B":200,"E":80})

    def test_e5_validation_and_locked_test_are_distinct(self):
        base = "results/APTOS2019/e5_integrated/"
        summary = read_json(base + "summary.json")
        test = read_json(base + "test_summary.json")
        with (ROOT / base / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in history], list(range(1, 21)))
        self.assertEqual(summary["best"]["epoch"], 16)
        self.assertEqual(summary["final_test_status"], "NOT_ACCESSED")
        self.assertEqual(test["status"], "FINAL_TEST_COMPLETE")
        self.assertEqual(test["checkpoint_epoch"], 16)
        for metrics, total, correct in ((summary["best"]["validation"], 366, 304),
                                        (test["metrics"], 366, 302)):
            matrix = metrics["confusion_matrix"]
            self.assertEqual(sum(map(sum, matrix)), total)
            self.assertEqual(sum(matrix[i][i] for i in range(5)), correct)
            self.assertEqual([sum(row) for row in matrix],
                             [grade["support"] for grade in metrics["per_grade"]])
        self.assertEqual(test["metrics"]["per_grade"][3]["correct"], 4)
        self.assertEqual(test["metrics"]["per_grade"][3]["support"], 19)

    def test_default_dataset_cells_execute_without_external_actions(self):
        with patch("IPython.display.display"), patch("subprocess.run") as child, patch.dict(
                "os.environ", {"DR_RESEARCH_ROOT": str(ROOT)}, clear=False):
            for path in NOTEBOOKS:
                notebook = nbformat.read(ROOT / "notebooks" / path, as_version=4)
                env = {}
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        exec(compile(cell.source, path, "exec"), env)
            child.assert_not_called()

    def test_optional_workflows_parse_through_original_clis_without_execution(self):
        from importlib import import_module
        from notebooks import build_experiment_notebooks as builder, experiment_workflow as workflow

        base = read_json(builder.BASE_EVIDENCE)["runs"]
        with tempfile.TemporaryDirectory() as root:
            artifact = Path(root)
            env = {
                "DR_ARTIFACT_ROOT": root, "DR_CONVNEXT_SNAPSHOT": str(artifact / "model"),
                "DR_NOTEBOOK_RUN_DIR": str(artifact / "foundation_e7/new_sample_run"),
                "DR_VALIDATION_CACHE": str(artifact / "foundation_e7/val"),
                "DR_TRAIN_CACHE": str(artifact / "foundation_e7/train_cache"),
                "DR_E5_CHECKPOINT": str(artifact / "e5/best.pt"), "DR_CACHE_MAX_GIB": "20",
                "DR_VALIDATION_CACHE_MAX_GIB": "5", "DR_CACHE_RESERVE_GIB": "10",
                "DR_TRUST_LEGACY_BIN": "1", "DR_TRUST_E5_CHECKPOINT": "1",
            }
            for arm, module in workflow.CLI.items():
                config = (next(r["configuration"] for r in base if r["arm"] == arm and
                               r["dataset"] == "APTOS2019") if arm.startswith("B") else
                          builder.read(f"results/APTOS2019/{builder.INTEGRATED.get(arm,'e5_integrated')}/"
                                       "validation_evidence.json")["configuration"])
                if arm == "E7":
                    config = {**config, "micro_batch": 4, "accumulation": 8,
                              "e5_checkpoint_sha256": "a" * 64}
                runner = import_module(module)
                for step in workflow.STEPS[arm]:
                    fn_name = step.replace("-", "_")
                    argv = workflow.command(ROOT, arm, "APTOS2019", step, config, env=env)
                    with self.subTest(arm=arm, step=step), patch.object(runner, fn_name) as action:
                        runner.main(argv[3:])
                        action.assert_called_once()

    def test_cadvit_recovery_epoch_and_final_test_counts(self):
        for dataset, folder, total, correct in (
            ("APTOS2019", "results/APTOS2019/full_oversampling_amp_fp16_80epochs", 366, 268),
            ("DDR", "results/DDR/full_oversampling_fp32_math_mb8_80epochs", 1252, 975),
        ):
            path = ROOT / folder
            rows = [json.loads(line) for line in (path / "history.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["epoch"] for row in rows], list(range(1, 81)), dataset)
            final = read_json(str(Path(folder) / "final_test_summary.json"))
            metrics = final.get("metrics", final)
            matrix = metrics["confusion_matrix"]
            self.assertEqual(sum(map(sum, matrix)), total)
            self.assertEqual(sum(matrix[i][i] for i in range(5)), correct)


if __name__ == "__main__":
    unittest.main()
