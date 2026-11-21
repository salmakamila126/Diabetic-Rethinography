"""Verify the review export and publication without requiring checkpoints or dataset images."""

import copy
import csv
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

import nbformat
from PIL import Image

from docs import pruning_pilot_publication as pilot, pruning_pilot_supplement as supplement


class PruningPilotPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = pilot.ROOT / pilot.FOLDER
        cls.evidence = pilot.verify(pilot.parse((cls.directory / "validation_evidence.json").read_bytes()))
        cls.profiles = supplement.read_profiles(cls.evidence, cls.directory)
        cls.recovery = supplement.verify_recovery(
            cls.evidence, cls.profiles, pilot.parse((cls.directory / supplement.RECOVERY_FILE).read_bytes()))

    def test_exact_quality_and_parity_tables(self):
        for name, rows, count in (("quality.csv", pilot.quality_rows(self.evidence), 20),
                                  ("numerical_parity.csv", pilot.parity_rows(self.evidence), 80)):
            raw = (self.directory / name).read_text(encoding="utf-8")
            self.assertEqual(raw.encode(), pilot.csv_bytes(rows))
            values = list(csv.DictReader(StringIO(raw)))
            self.assertEqual(len(values), count)
        parity = [r for r in values if r["backend"] != "eager"]
        self.assertEqual(len(parity), 60)
        self.assertTrue(all(r["parity_status"] == "PASS" and float(r["argmax_agreement"]) == 1.0 for r in parity))

    def test_reject_metric_drift_even_with_recomputed_member_sha(self):
        evidence = copy.deepcopy(self.evidence)
        job = evidence["jobs"]["E7_APTOS2019"]
        report = job["validation"]
        report["variants"][0]["backends"]["eager"]["quality"]["macro_f1"] = .95
        job["original_members"]["E7_APTOS2019/validation/report.json"] = pilot.receipt(pilot.json_bytes(report))
        job["original_members"]["E7_APTOS2019/validation/summary.csv"] = pilot.receipt(pilot.validation_summary(report).encode())
        with self.assertRaisesRegex(ValueError, "macro_f1 differs"):
            pilot.verify(evidence)

    def test_reject_tampered_provenance_or_member(self):
        evidence = copy.deepcopy(self.evidence)
        record = evidence["jobs"]["E6_DDR"]
        record["original_members"]["E6_DDR/validation/summary.csv"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original file receipts differ"):
            pilot.verify(evidence)
        record["validation"]["framework_source_sha256"] = pilot.source_hashes(pilot.VALIDATE_COMMITS["E5_DDR"])
        with self.assertRaisesRegex(ValueError, "validation identity/status differs"):
            pilot.verify(evidence)

    def test_reject_mismatched_ordered_input_and_private_path(self):
        evidence = copy.deepcopy(self.evidence)
        job = evidence["jobs"]["E6_APTOS2019"]
        job["validation"]["ordered_preprocessed_tensor_sha256"] = "0" * 64
        job["original_members"]["E6_APTOS2019/validation/report.json"] = pilot.receipt(pilot.json_bytes(job["validation"]))
        with self.assertRaisesRegex(ValueError, "sample or processor inputs differ"):
            pilot.verify(evidence)
        job["validation"]["allocation"]["id"] = "/" + "/".join(("home", "private-account", "work"))
        with self.assertRaisesRegex(ValueError, "Private path"):
            pilot.verify(evidence)

    def test_report_and_existing_notebooks_use_same_aggregate(self):
        report = (pilot.ROOT / "docs/PRUNING_PILOT_RESULTS.md").read_text(encoding="utf-8")
        self.assertIn(pilot.quality_table(self.evidence), report)
        self.assertIn(pilot.parity_table(self.evidence), report)
        self.assertIn("20/20 prepared receipts verified", report)
        self.assertIn(supplement.complexity_table(self.evidence, self.profiles), report)
        self.assertEqual(self.evidence["status"], pilot.STUDY_STATUS)
        for job in pilot.JOBS:
            notebook = nbformat.read(pilot.ROOT / "notebooks/optimization" / f"{job}.ipynb", as_version=4)
            nbformat.validate(notebook)
            results = [cell.source for cell in notebook.cells if cell.source.startswith("## Published structured-pruning")]
            text = supplement.notebook_text(self.evidence, self.profiles, job, self.recovery)
            self.assertEqual(results, [pilot.notebook_evidence(self.evidence, job, supplement_text=text)])
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertEqual(cell.outputs, [])
                    self.assertIsNone(cell.execution_count)

    def test_figures_embed_the_exact_visualized_rows(self):
        rows = list(pilot.quality_rows(self.evidence))
        expected = [[r for r in rows if r["job"].endswith(name)] for name in pilot.DATASETS]
        expected.append([r for r in rows if r["target_pruning_percent"]])
        expected.append([r for r in pilot.parity_rows(self.evidence) if r["backend"] != "eager"])
        for stem, inputs in zip(pilot.STEMS, expected, strict=True):
            base = pilot.ROOT / "docs/figures" / stem
            with Image.open(base.with_suffix(".png")) as png:
                raster = json.loads(png.info["Description"])
                self.assertGreaterEqual(png.width, 1500)
                self.assertGreaterEqual(png.height, 750)
                png.verify()
            svg = ET.parse(base.with_suffix(".svg"))
            description = svg.find(".//{http://purl.org/dc/elements/1.1/}description")
            self.assertEqual(json.loads(description.text), raster)
            self.assertEqual(raster["rows"], inputs)

    def test_reviewed_archive_digest_guard(self):
        with TemporaryDirectory() as temporary:
            invalid = Path(temporary) / pilot.ARCHIVE
            invalid.write_bytes(b"0" * pilot.ARCHIVE_BYTES)
            with self.assertRaisesRegex(ValueError, "Archive SHA-256 differs"):
                pilot.load_archive(invalid)

    def test_recorded_complexity_and_exact_figures(self):
        rows = list(supplement.complexity_rows(self.evidence, self.profiles))
        self.assertEqual(len(rows), 20)
        self.assertEqual((self.directory / "complexity.csv").read_bytes(), pilot.csv_bytes(rows))
        expected = {(row["job"], row["target_pruning_percent"]): row["macro_f1"]
                    for row in pilot.quality_rows(self.evidence)}
        joined = [dict(row, macro_f1=expected[row["job"], row["target_pruning_percent"]]) for row in rows]
        for stem in pilot.COMPLEXITY_STEMS:
            base = pilot.ROOT / "docs/figures" / stem
            with Image.open(base.with_suffix(".png")) as png:
                raster = json.loads(png.info["Description"])
                png.verify()
            svg = ET.parse(base.with_suffix(".svg"))
            description = svg.find(".//{http://purl.org/dc/elements/1.1/}description")
            self.assertEqual(json.loads(description.text), raster)
            self.assertEqual(raster["rows"], joined)

    def test_reject_changes_to_complexity_even_with_recomputed_file_sha(self):
        profiles = copy.deepcopy(self.profiles)
        value = profiles["E7_APTOS2019"]["pruned_30"]["record"]
        value["profile"]["parameters"] += 1
        profiles["E7_APTOS2019"]["pruned_30"] = supplement.member(pilot.json_bytes(value))
        with self.assertRaisesRegex(ValueError, "prepared identity or provenance differs"):
            supplement.validate_profiles(self.evidence, profiles)

    def test_recovery_evidence_does_not_promote_missing_evaluation(self):
        record = copy.deepcopy(self.recovery)
        self.assertEqual(record["variant_status"], {"original": "UNVERIFIED", "pruned_10": "PARTIAL",
                                                   "pruned_30": "UNVERIFIED", "pruned_50": "UNVERIFIED"})
        record["validation_recovered_directory_present"] = True
        with self.assertRaisesRegex(ValueError, "recovery snapshot or parent lineage differs"):
            supplement.verify_recovery(self.evidence, self.profiles, record)
        record = copy.deepcopy(self.recovery)
        value = record["source_files"][supplement.RECOVERY_PROTOCOL]["record"]
        value["parent_receipts_sha256"]["0.1"] = "0" * 64
        record["source_files"][supplement.RECOVERY_PROTOCOL] = supplement.member(pilot.json_bytes(value))
        with self.assertRaisesRegex(ValueError, "recovery snapshot or parent lineage differs"):
            supplement.verify_recovery(self.evidence, self.profiles, record)


if __name__ == "__main__":
    unittest.main()
