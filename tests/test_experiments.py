"""CPU-only, synthetic tests; no dataset, training epoch, or final test."""

import unittest
from pathlib import Path

from reproduction.cadvit.experiments import (Protocol, VARIANTS, artifact_paths, binary_metrics,
                                             core_macs, validate_variant_matrix)
from reproduction.cadvit.model import CADViTCandidate


class ExperimentTests(unittest.TestCase):
    def test_table_1_and_3_change_only_intended_fields(self):
        self.assertTrue(validate_variant_matrix())
        base = VARIANTS["aptos_r16"].model_config()
        for r in (2, 4, 8, 16, 32):
            current = VARIANTS[f"aptos_r{r}"].model_config()
            self.assertEqual({k for k in base if current[k] != base[k]},
                             set() if r == 16 else {"reduction"})

    def test_parameter_counts_and_flops_convention(self):
        expected = {2: 5_443_941, 4: 5_351_461, 8: 5_305_221,
                    16: 5_282_101, 32: 5_270_541}
        macs = []
        for r, count in expected.items():
            config = VARIANTS[f"aptos_r{r}"].model_config()
            self.assertEqual(sum(p.numel() for p in CADViTCandidate(config).parameters()), count)
            flop = core_macs(config, 512)
            self.assertEqual(flop["input_shape"], [1, 3, 512, 512])
            self.assertAlmostEqual(flop["arithmetic_gflops_2_per_mac"], 2 * flop["gmac"])
            macs.append(flop["core_macs"])
        self.assertEqual(macs, sorted(macs, reverse=True))

    def test_binary_metric_orientation_and_per_class_definitions(self):
        metric = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metric["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metric["accuracy"], 0.75)
        self.assertAlmostEqual(metric["per_class"]["DR"]["precision"], 2 / 3)
        self.assertEqual(metric["per_class"]["DR"]["recall"], 1.0)
        self.assertEqual(metric["per_class"]["No DR"]["specificity"], 1.0)
        self.assertEqual(metric["per_class"]["DR"]["one_vs_rest_accuracy"], 0.75)

    def test_protocol_cannot_mix_effective_batch_or_preprocessing(self):
        Protocol(micro_batch_size=4, accumulation=4).validate()
        with self.assertRaises(ValueError):
            Protocol(micro_batch_size=4, accumulation=1).validate()
        with self.assertRaises(ValueError):
            Protocol(preprocessing="v2_bg10").validate()

    def test_explicit_ddr_root_is_not_appended_twice(self):
        root = Path("/tmp/cadvit_artifacts")
        ddr = Path("/tmp/cadvit_artifacts/datasets/DDR/DDR-dataset/DR_grading")
        paths = artifact_paths(root, VARIANTS["ddr_full"], ddr)
        self.assertEqual(paths["dataset_root"], ddr.resolve())


if __name__ == "__main__":
    unittest.main()
