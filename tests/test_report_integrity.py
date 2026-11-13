"""Static integrity checks for the read-only research report and its archive."""

import ast
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproduction/archive/CAD_ViT_original_105_cells.ipynb"
ORIGINAL_SHA256 = "78ba1d4807d5cc4581d3725c53510173f014ab0b92f58043e8bc733d65fb0d1b"
V2_SHA256 = {
    "APTOS2019": "5db8e70f6c8ef78d0f9047f08f6ba1bcdae4c97c6d6b39c33ba58e8792b28e3c",
    "DDR": "95b0a47ef36626e23e6721ebb1bfeacd6265a39939457ff37ae5115ca58381d6",
}
SOURCE_SHA256 = {
    "reproduction/cadvit/model.py": "6aff75039dbd2d3e545e6ff1998674221b3671df7d58362792e0b2ca249a0c25",
    "reproduction/cadvit/preprocessing.py": "34d4e39bda7a3a996cc1662bfc7144c09667071a52435050491af531e1fbb7b5",
}


def markdown(cell):
    return "".join(cell["source"])


def old_history(archive, name):
    source = markdown(archive["cells"][73])
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


class ReportIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
        cls.main = json.loads((ROOT / "CAD_ViT.ipynb").read_text(encoding="utf-8"))
        cls.experiments = {
            name: json.loads((ROOT / "reproduction/experiments" / name).read_text(encoding="utf-8"))
            for name in (
                "01_experiment_initial_cadvit_v1.ipynb",
                "02_experiment_bg10_cadvit_v2.ipynb",
                "03_experiment_balanced_training_and_ddr_recovery.ipynb",
            )
        }
        cls.report = {cell["id"]: markdown(cell) for notebook in cls.experiments.values()
                      for cell in notebook["cells"]}

    def test_archive_bytes_cells_outputs_and_attachments(self):
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), ORIGINAL_SHA256)
        self.assertEqual(len(self.archive["cells"]), 105)
        self.assertEqual(sum(len(c.get("outputs", [])) for c in self.archive["cells"]), 32)
        self.assertEqual(sum(bool(c.get("attachments")) for c in self.archive["cells"]), 6)
        mirrors = list((ROOT / "reproduction/archive/results").rglob("*"))
        mirrors = [path for path in mirrors if path.is_file()]
        self.assertEqual(len(mirrors), 8)
        for path in mirrors:
            with self.subTest(mirror=path):
                original = ROOT / path.relative_to(ROOT / "reproduction/archive")
                self.assertEqual(path.read_bytes(), original.read_bytes())

    def test_main_has_only_static_markdown(self):
        self.assertLessEqual(len(self.main["cells"]), 12)
        for name, notebook in (("main", self.main), *self.experiments.items()):
            with self.subTest(notebook=name):
                self.assertEqual(notebook["nbformat"], 4)
                self.assertTrue(all(c["cell_type"] == "markdown" for c in notebook["cells"]))
                ids = [c["id"] for c in notebook["cells"]]
                self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("log mentah tidak tersedia", self.report["v1-train"].lower())

    def test_baseline_source_hashes_unchanged(self):
        for relative, expected in SOURCE_SHA256.items():
            with self.subTest(source=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_all_four_complete_training_tables_match_sources(self):
        cases = (
            ("balanced-train-aptos", ROOT / "results/APTOS2019/full_oversampling_amp_fp16_80epochs/history.jsonl", 24),
            ("balanced-train-ddr", ROOT / "results/DDR/full_oversampling_fp32_math_mb8_80epochs/history.jsonl", 69),
        )
        for cell_id, path, best in cases:
            with self.subTest(cell=cell_id):
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual([row["epoch"] for row in rows], list(range(1, 81)))
                text = self.report[cell_id]
                displayed = re.findall(r"^\| (\*\*)?(\d+)(?:\*\*)? \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$", text, re.M)
                self.assertEqual(len(displayed), 80)
                for actual, row in zip(displayed, rows):
                    self.assertEqual(actual[1:], (str(row["epoch"]), *(f"{row[k]:.9f}" for k in
                        ("train_loss", "val_loss", "val_accuracy", "val_macro_f1"))))
                    self.assertEqual(bool(actual[0]), row["epoch"] == best)
        for cell_id, key, best in (("v2-train-aptos", "APTOS_80_ROWS", 69),
                                   ("v2-train-ddr", "DDR_80_ROWS", 72)):
            with self.subTest(cell=cell_id):
                rows = old_history(self.archive, key)
                self.assertEqual([row[0] for row in rows], list(range(1, 81)))
                displayed = re.findall(r"^\| (\*\*)?(\d+)(?:\*\*)? \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$", self.report[cell_id], re.M)
                self.assertEqual(len(displayed), 80)
                for actual, row in zip(displayed, rows):
                    self.assertEqual(actual[1:], (str(row[0]), *(f"{float(value):.9f}" for value in row[1:5])))
                    self.assertEqual(bool(actual[0]), row[0] == best)

    def test_v2_result_hashes_and_matrix_counts(self):
        for dataset, expected in V2_SHA256.items():
            with self.subTest(dataset=dataset):
                path = ROOT / f"results/{dataset}/v2_bg10_historical_80epochs/final_test_v2_80ep.json"
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
                data = json.loads(path.read_text(encoding="utf-8"))
                cm = data["metrics"]["confusion_matrix"]
                n = sum(map(sum, cm))
                self.assertEqual(n, data["metrics"]["n"])
                self.assertAlmostEqual(sum(cm[i][i] for i in range(5)) / n,
                                       data["metrics"]["accuracy"], places=12)

    def test_six_displayed_final_test_matrices_match_recorded_sources(self):
        historical = None
        for node in ast.parse(markdown(self.archive["cells"][88])).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "final_test_confusion_matrices"):
                historical = ast.literal_eval(node.value)
        self.assertIsNotNone(historical)
        cases = {
            "v1-aptos": historical["APTOS2019"],
            "v1-ddr": historical["DDR"],
            "v2-aptos": json.loads((ROOT / "results/APTOS2019/v2_bg10_historical_80epochs/final_test_v2_80ep.json").read_text(encoding="utf-8"))["metrics"]["confusion_matrix"],
            "v2-ddr": json.loads((ROOT / "results/DDR/v2_bg10_historical_80epochs/final_test_v2_80ep.json").read_text(encoding="utf-8"))["metrics"]["confusion_matrix"],
            "balanced-aptos": json.loads((ROOT / "results/APTOS2019/full_oversampling_amp_fp16_80epochs/final_test_summary.json").read_text(encoding="utf-8"))["metrics"]["confusion_matrix"],
            "balanced-ddr": json.loads((ROOT / "results/DDR/full_oversampling_fp32_math_mb8_80epochs/final_test_summary.json").read_text(encoding="utf-8"))["confusion_matrix"],
        }
        pattern = re.compile(r"^\| ([0-4]) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", re.M)
        for cell_id, expected in cases.items():
            with self.subTest(cell=cell_id):
                actual = [tuple(map(int, match.groups())) for match in pattern.finditer(self.report[cell_id])]
                self.assertEqual(len(actual), 5)
                self.assertEqual([row[0] for row in actual], list(range(5)))
                self.assertEqual([list(row[1:6]) for row in actual], expected)
                self.assertEqual([row[6] for row in actual], list(map(sum, expected)))

    def test_local_report_and_archive_links(self):
        cases = [(ROOT / "CAD_ViT.ipynb", self.main), (ARCHIVE, self.archive)]
        cases.extend((ROOT / "reproduction/experiments" / name, notebook)
                     for name, notebook in self.experiments.items())
        for path, notebook in cases:
            for cell in notebook["cells"]:
                if cell["cell_type"] != "markdown":
                    continue
                text = re.sub(r"```[\s\S]*?```", "", markdown(cell))
                for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                    if target.startswith(("http:", "https:", "#")):
                        continue
                    if target.startswith("attachment:"):
                        self.assertIn(target.removeprefix("attachment:"), cell.get("attachments", {}))
                    else:
                        self.assertTrue((path.parent / target).exists(), f"{path}: {target}")


if __name__ == "__main__":
    unittest.main()
