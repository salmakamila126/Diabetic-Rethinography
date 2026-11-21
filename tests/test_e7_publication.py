"""Read-only regression checks for the published E7 APTOS2019 transfer."""

import copy
import csv
import json
import os
from pathlib import Path
import re
import unittest
from xml.etree import ElementTree

from PIL import Image

from docs import e7_publication as publication
from docs.e5_e6_publication import check_public


class E7PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = publication.ROOT / publication.FOLDER
        cls.evidence = publication.parse((cls.folder / "validation_evidence.json").read_bytes())

    def test_all_epochs_source_receipts_and_privacy(self):
        evidence = self.evidence
        self.assertEqual(publication.audit(evidence), evidence["derived_best_validation"])
        self.assertEqual(evidence["audit"]["confusion_matrices_checked"], 40)
        self.assertTrue(evidence["audit"]["local_manifest_and_sidecar_verified"])
        publication.verify_sources(evidence["configuration"])
        check_public(evidence)
        self.assertEqual(evidence["provenance"]["archive_sha256"], publication.ARCHIVE_SHA)
        self.assertEqual(set(evidence["provenance"]["members"]), set(publication.ORIGINALS))

    def test_corrupted_history_and_checkpoint_are_rejected(self):
        evidence = copy.deepcopy(self.evidence)
        evidence["history"][0]["train"]["confusion_matrix"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "support"):
            publication.audit(evidence)
        evidence = copy.deepcopy(self.evidence)
        evidence["history"][18]["best_updated"] = False
        with self.assertRaisesRegex(ValueError, "checkpoint-selection"):
            publication.audit(evidence)
        evidence = copy.deepcopy(self.evidence)
        evidence["checkpoint_sidecars"]["best.pt"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sidecar"):
            publication.audit(evidence)

    def test_full_precision_csv_and_markdown_links(self):
        history = self.evidence["history"]
        text = (self.folder / "history.csv").read_text(encoding="utf-8")
        self.assertEqual(text, publication.history_csv(history))
        rows = csv.DictReader(text.splitlines())
        for row, expected in zip(rows, history, strict=True):
            self.assertEqual(int(row["epoch"]), expected["epoch"])
            for key in ("train", "validation"):
                self.assertEqual(json.loads(row[f"{key}.confusion_matrix"]), expected[key]["confusion_matrix"])
                self.assertEqual(json.loads(row[f"{key}.per_grade"]), expected[key]["per_grade"])
                self.assertEqual(float(row[f"{key}.macro_f1"]), expected[key]["macro_f1"])
        report = (self.folder / "validation_report.md").read_text(encoding="utf-8")
        self.assertEqual(report, publication.report(self.evidence))
        check_public(report)
        for target in re.findall(r"\]\(([^)]+)\)", report):
            self.assertTrue((self.folder / target.split("#")[0]).is_file(), target)

    def test_figure_pairs_and_selected_checkpoint_labels(self):
        for key in publication.FIGURES:
            stem = publication.ROOT / f"docs/figures/aptos2019_e7_{key}"
            with Image.open(stem.with_suffix(".png")) as png:
                self.assertEqual(png.format, "PNG")
                self.assertGreaterEqual(min(png.size), 1100)
                png.verify()
            data = stem.with_suffix(".svg").read_text(encoding="utf-8")
            self.assertTrue(ElementTree.fromstring(data).tag.endswith("}svg"))
            check_public(data)
            self.assertNotIn("data:image/", data)
            if key == "confusion":
                self.assertIn("epoch 19", data)
                for grade in range(5):
                    self.assertIn(f"<!-- {publication.MATRIX[grade][grade]} -->", data)
            if key == "e5_e6_e7_comparison":
                self.assertIn("0.696", data)
                self.assertIn("0.913", data)

    def test_original_transfer_and_local_manifest_match_public_evidence(self):
        archive, manifest = os.environ.get("E7_PUBLICATION_EXPORT"), os.environ.get("E7_PUBLICATION_MANIFEST_DIR")
        if not archive or not manifest:
            self.skipTest("Set E7_PUBLICATION_EXPORT and E7_PUBLICATION_MANIFEST_DIR for original-file verification")
        self.assertEqual(publication.load_export(Path(archive), Path(manifest)), self.evidence)


if __name__ == "__main__":
    unittest.main()
