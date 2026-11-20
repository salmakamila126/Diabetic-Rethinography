"""CPU-only checks of staged E5/E6 aggregate evidence and rendered artifacts."""

import copy
import json
import os
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
from xml.etree import ElementTree

from PIL import Image

from docs import e5_e6_publication as publication


ROOT = publication.ROOT


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {}
        for method, dataset in publication.EXPECTED:
            path = ROOT / publication.run_folder(method, dataset) / "validation_evidence.json"
            cls.records[method, dataset] = json.loads(path.read_text(encoding="utf-8"))

    def test_all_recorded_epochs_and_selection(self):
        self.assertEqual(sum(len(r["history"]) for r in self.records.values()), 80)
        self.assertEqual(sum(r["audit"]["csv_cells_checked"] for r in self.records.values()), 1800)
        publication.verify_shared(self.records)
        publication.verify_source(self.records)
        self.assertEqual(sum(r["audit"]["source_files_verified"] for r in self.records.values()), 62)
        for (method, dataset), record in self.records.items():
            with self.subTest(method=method, dataset=dataset):
                publication.check_public(record)
                self.assertEqual(publication.audit_run(record, method, dataset),
                                 record["derived_best_validation"])
                self.assertTrue(record["audit"]["csv_original_digest_verified"])
                for key in ("best.pt", "last.pt"):
                    sidecar = record["checkpoint_sidecars"][key]
                    source = record["provenance"]["files"][key]
                    self.assertEqual(sidecar, {"sha256": source["sha256"], "bytes": source["bytes"]})
                for epoch, supplemental in zip(record["history"], record["csv_per_grade_mean_absolute_error"], strict=True):
                    self.assertEqual(epoch["epoch"], supplemental["epoch"])
                    for split in ("train", "validation"):
                        actual = publication.derive(epoch[split]["confusion_matrix"])
                        self.assertEqual(len(supplemental[split]), 5)
                        for item, grade in zip(supplemental[split], actual["per_grade"]):
                            self.assertAlmostEqual(item, grade["mean_absolute_error"])

    def test_saved_evidence_reproduces_all_reports_and_links(self):
        target = ROOT / "docs/E5_E6_VALIDATION.md"
        reports = [(target, publication.compare_report(self.records))]
        for (method, dataset), record in self.records.items():
            path = ROOT / publication.run_folder(method, dataset) / "validation_report.md"
            reports.append((path, publication.record_report(method, dataset, record)))
        for path, expected in reports:
            with self.subTest(path=path.name):
                actual = path.read_text(encoding="utf-8")
                self.assertEqual(actual, expected)
                self.assertFalse(publication.PRIVATE.search(actual))
                for href in re.findall(r"\]\(([^)]+)\)", actual):
                    file, _, anchor = unquote(href).partition("#")
                    link = (path.parent / file).resolve() if file else path
                    self.assertTrue(link.is_relative_to(ROOT) and link.is_file(), href)
                    if anchor:
                        headings = re.findall(r"^#+ +(.*)$", link.read_text(encoding="utf-8"), re.M)
                        slugs = [re.sub(r"[^\w -]", "", item.lower()).replace(" ", "-") for item in headings]
                        self.assertIn(anchor, slugs, href)

    def test_figure_pairs_are_readable_and_labeled(self):
        directory = ROOT / "docs/figures"
        names = ["e5_e6_best_validation", *(f"{dataset.lower()}_{figure}" for dataset in publication.DATASETS
                                            for figure in publication.FIGURES)]
        self.assertEqual(len(names), 13)
        for name in names:
            with self.subTest(name=name):
                with Image.open(directory / f"{name}.png") as rendered:
                    self.assertEqual(rendered.format, "PNG")
                    self.assertGreaterEqual(min(rendered.size), 1100)
                    rendered.verify()
                svg_path = directory / f"{name}.svg"
                svg = svg_path.read_text(encoding="utf-8")
                self.assertTrue(ElementTree.fromstring(svg).tag.endswith("}svg"))
                self.assertFalse(publication.PRIVATE.search(svg))
                self.assertNotIn("data:image/", svg)
                self.assertNotIn(b"\r\n", svg_path.read_bytes())
                self.assertIn("E5", svg)
                self.assertIn("E6", svg)
                if name.endswith("_confusion"):
                    dataset = name.split("_")[0].upper()
                    for method in ("E5", "E6"):
                        result = self.records[method, dataset]["summary"]["best"]
                        self.assertIn(f"{method} epoch {result['epoch']}", svg)
                        for true_grade, row in enumerate(result["validation"]["confusion_matrix"]):
                            self.assertIn(f"<!-- {row[true_grade]} -->", svg)

    def test_audit_rejects_corrupt_matrix_history_and_private_material(self):
        sample = self.records["E6", "DDR"]
        bad = copy.deepcopy(sample)
        bad["history"][0]["validation"]["confusion_matrix"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "frozen grade support"):
            publication.audit_run(bad, "E6", "DDR")
        bad = copy.deepcopy(sample)
        bad["history"].pop()
        with self.assertRaisesRegex(ValueError, "unordered or missing epochs"):
            publication.audit_run(bad, "E6", "DDR")
        with self.assertRaisesRegex(ValueError, "Private path"):
            publication.check_public({"config": "/home/example/private"})
        with self.assertRaisesRegex(ValueError, "Non-aggregate"):
            publication.check_public({"patient_ids": ["synthetic"]})

    def test_optional_archive_matches_public_evidence(self):
        supplied = os.environ.get("E5_E6_PUBLICATION_EXPORT")
        if not supplied:
            self.skipTest("Set E5_E6_PUBLICATION_EXPORT to compare the exact transferred archive")
        expected = publication.load_export(Path(supplied))
        publication.verify_shared(expected)
        publication.verify_source(expected)
        self.assertEqual(self.records, expected)


if __name__ == "__main__":
    unittest.main()
