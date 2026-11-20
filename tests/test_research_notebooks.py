"""CPU-only integrity checks for the three primary research notebooks."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_CAD_ViT_Reproduction.ipynb",
    "02_ConvNeXt_B0_B4_Validation.ipynb",
    "03_ConvNeXt_E5_APTOS2019.ipynb",
)


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class ResearchNotebookIntegrity(unittest.TestCase):
    def test_notebook_schema_syntax_and_guarded_actions(self):
        for index, name in enumerate(NOTEBOOKS, start=1):
            path = ROOT / "notebooks" / name
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            self.assertTrue(notebook.cells[0].source.startswith(f"# {index:02d} —"), name)
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            self.assertGreaterEqual(len(code_cells), 10, name)
            for position, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                ast.parse(cell.source, filename=f"{name}:{position}")
                self.assertEqual(notebook.cells[position - 1].cell_type, "markdown")
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                self.assertNotIn("--execute-final-test", cell.source)
                self.assertNotIn("pip install", cell.source)
            source = "\n".join(cell.source for cell in code_cells)
            self.assertIn("RUN_TRAINING = False", source)
            self.assertIn("SKIPPED_MISSING_DATA", source)

    def test_local_markdown_links_and_figures(self):
        paths = [ROOT / "README.md", ROOT / "notebooks" / "README.md",
                 ROOT / "docs" / "RESEARCH_CONSOLIDATION_AUDIT.md"]
        for name in NOTEBOOKS:
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
        first = nbformat.read(ROOT / "notebooks" / NOTEBOOKS[0], as_version=4)
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
        second = nbformat.read(ROOT / "notebooks" / NOTEBOOKS[1], as_version=4)
        markdown2 = [cell.source for cell in second.cells if cell.cell_type == "markdown"]
        comparison = next(text for text in markdown2 if text.startswith("**Committed validation comparison"))
        grades = next(text for text in markdown2 if text.startswith("**Per-grade best-validation evidence"))
        self.assertEqual(sum(line.startswith("| ") for line in comparison.splitlines()) - 2, 10)
        self.assertEqual(sum(line.startswith("| ") for line in grades.splitlines()) - 2, 50)
        third = nbformat.read(ROOT / "notebooks" / NOTEBOOKS[2], as_version=4)
        history = next(cell.source for cell in third.cells if cell.cell_type == "markdown"
                       and cell.source.startswith("**All 20 original E5 APTOS"))
        self.assertEqual(sum(line.startswith("| ") for line in history.splitlines()) - 2, 20)

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
