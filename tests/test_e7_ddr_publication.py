"""Read-only validation of the E7 DDR checkpoint export and published figure data."""

import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from PIL import Image

from docs import e7_ddr_publication as ddr, e7_publication as aptos, figure_provenance as provenance


class E7DDRPublicationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = provenance.read(ddr.EVIDENCE.as_posix())

    def test_run_status_selection_and_history_csv(self):
        ddr.audit(self.evidence)
        summary = self.evidence["summary"]
        self.assertEqual(summary["status"], ddr.STATUS)
        self.assertEqual(summary["final_test_status"], "NOT_ACCESSED")
        self.assertEqual(summary["stopped_during_epoch"], 11)
        self.assertEqual(summary["epochs_completed"], 10)
        self.assertEqual(summary["best"]["epoch"], 2)
        results = ddr.ROOT / ddr.FOLDER
        self.assertEqual((results / "validation_report.md").read_text(encoding="utf-8"), ddr.report(self.evidence))
        published = (results / "history.csv").read_text(encoding="utf-8")
        self.assertEqual(published, aptos.history_csv(self.evidence["history"]))
        rows = list(csv.DictReader(published.splitlines()))
        self.assertEqual([int(row["epoch"]) for row in rows], list(range(1, 11)))
        self.assertEqual(sum(map(sum, summary["best"]["validation"]["confusion_matrix"])), 1252)

    def test_modified_history_fails_audit(self):
        tampered = copy.deepcopy(self.evidence)
        tampered["history"][2]["validation"]["confusion_matrix"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "grade support"):
            ddr.audit(tampered)
        tampered = copy.deepcopy(self.evidence)
        tampered["summary"]["epochs_completed"] = 20
        with self.assertRaisesRegex(ValueError, "status"):
            ddr.audit(tampered)

    def test_original_archive_agrees_with_public_evidence(self):
        original = os.environ.get("E7_DDR_PUBLICATION_EXPORT")
        if not original:
            self.skipTest("Set E7_DDR_PUBLICATION_EXPORT to verify the uncommitted original ZIP")
        self.assertEqual(ddr.load_export(Path(original)), self.evidence)

    def test_png_svg_and_source_data_agree(self):
        data = ddr.figure_data(self.evidence)
        figures = ddr.ROOT / "docs/figures"
        self.assertEqual(data["epoch_scores"]["epochs"], list(range(1, 11)))
        self.assertEqual(data["losses"]["selected_epoch"], 2)
        self.assertEqual(data["epoch_scores"]["series"]["QWK · Validation"], [
            row["validation"]["quadratic_weighted_kappa"] for row in self.evidence["history"]])
        self.assertEqual(data["grade_errors"]["grades"][1]["overgraded"], 33)
        for name, payload in data.items():
            with self.subTest(name=name):
                stem = figures / f"ddr_e7_{name}"
                with Image.open(stem.with_suffix(".png")) as im:
                    self.assertEqual(json.loads(im.info["Description"]), payload)
                    im.verify()
                svg = ET.parse(stem.with_suffix(".svg"))
                self.assertEqual(json.loads(svg.find(".//{http://purl.org/dc/elements/1.1/}description").text), payload)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is required for regeneration")
    def test_generator_reproduces_files_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            ddr.plot(self.evidence, destination)
            self.assertEqual(len(list(destination.iterdir())), 10)
            for path in destination.iterdir():
                self.assertEqual(path.read_bytes(), (ddr.ROOT / "docs/figures" / path.name).read_bytes(), path.name)
            with self.assertRaisesRegex(ValueError, "already exists"):
                ddr.plot(self.evidence, destination)

    def test_figure_audit_inventory_and_source_numbers(self):
        provenance.verify_static_data()
        path = ddr.ROOT / "docs/FIGURE_PROVENANCE_AUDIT.md"
        self.assertEqual(path.read_text(encoding="utf-8"), provenance.render())


if __name__ == "__main__":
    unittest.main()
