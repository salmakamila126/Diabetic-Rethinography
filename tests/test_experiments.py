"""CPU-only, synthetic tests; no dataset, training epoch, or final test."""

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset

from reproduction.cadvit.experiments import (BALANCED_TRAINING, CAMPAIGNS, Protocol, VARIANTS,
                                             _atomic_save,
                                             artifact_paths, balanced_train_indices,
                                             binary_metrics, core_macs, create_protocol,
                                             load_protocol, main, support_weighted_metrics,
                                             score, split_loader, training_class_counts,
                                             validate_variant_matrix)
from reproduction.cadvit.evaluation import classification_metrics
from reproduction.cadvit.model import CADViTCandidate


class ExperimentTests(unittest.TestCase):
    def test_checkpoint_serialization_is_atomic_for_synthetic_payload(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            checkpoint = Path(directory) / "dummy.pt"
            payload = {"epoch": 0, "model": {"synthetic_weight": torch.arange(3)}}
            _atomic_save(checkpoint, payload)
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(checkpoint.with_name("dummy.pt.tmp").exists())
            restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(restored["epoch"], 0)
            self.assertTrue(torch.equal(restored["model"]["synthetic_weight"],
                                        payload["model"]["synthetic_weight"]))

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

    def test_balancing_is_deterministic_and_train_only(self):
        rows = ([{"split": "train", "label": str(label)}
                 for label in (0, 0, 0, 1, 2, 3, 4)]
                + [{"split": "val", "label": "1"},
                   {"split": "test", "label": "4"}])
        before, after = training_class_counts(rows, "five_class", BALANCED_TRAINING)
        self.assertEqual(before, {"0": 3, "1": 1, "2": 1, "3": 1, "4": 1})
        self.assertEqual(after, {str(i): 3 for i in range(5)})
        indices = balanced_train_indices(rows, "five_class", 42, 0)
        self.assertEqual(indices, balanced_train_indices(rows, "five_class", 42, 0))
        self.assertEqual(len(indices), 15)
        self.assertTrue(all(0 <= index < 7 for index in indices))
        self.assertEqual([sum(int(rows[index]["label"]) == label for index in indices)
                          for label in range(5)], [3] * 5)
        binary_before, binary_after = training_class_counts(rows, "binary_head", BALANCED_TRAINING)
        self.assertEqual(binary_before, {"0": 3, "1": 4})
        self.assertEqual(binary_after, {"0": 4, "1": 4})

    def test_balanced_campaign_is_separate_and_fail_closed(self):
        # The Windows sandbox temp directory is not writable; confine this synthetic
        # test to a temporary repo directory while mocking the production root guard.
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as root:
            with patch("reproduction.cadvit.experiments.campaign_root",
                       side_effect=lambda artifact_root, campaign="paper_v1":
                       Path(artifact_root) / "experiments" / CAMPAIGNS[campaign]):
                old = create_protocol(root, campaign="paper_v1")
                new = create_protocol(root, campaign="balanced_v1")
                self.assertNotEqual(old, new)
                self.assertEqual(load_protocol(root, "paper_v1")[0].balancing, "none")
                self.assertEqual(load_protocol(root, "balanced_v1")[0].balancing, BALANCED_TRAINING)
                self.assertNotEqual(load_protocol(root, "paper_v1")[1],
                                    load_protocol(root, "balanced_v1")[1])
                with self.assertRaises(FileExistsError):
                    create_protocol(root, campaign="balanced_v1")

    def test_figure_5_support_weighted_metric_alignment(self):
        # Figure 5 uses predicted rows / true columns; evaluator uses the reverse.
        paper_matrices = {
            "APTOS2019": ([[184, 2, 2, 0, 0], [3, 22, 5, 0, 0],
                           [0, 9, 86, 1, 7], [0, 1, 2, 9, 1], [1, 0, 3, 5, 24]],
                          367, 0.8842215883),
            "DDR": ([[592, 14, 51, 0, 1], [9, 26, 7, 0, 0],
                     [43, 14, 364, 16, 26], [0, 0, 1, 7, 1], [1, 0, 6, 0, 74]],
                    1253, 0.8443922537),
        }
        for name, (matrix, count, expected_f1) in paper_matrices.items():
            with self.subTest(dataset=name):
                truth = [actual for predicted in range(5) for actual in range(5)
                         for _ in range(matrix[predicted][actual])]
                predicted = [label for label in range(5) for actual in range(5)
                             for _ in range(matrix[label][actual])]
                metrics = classification_metrics(truth, predicted)
                self.assertEqual(metrics["n"], count)
                weighted = support_weighted_metrics(metrics)
                self.assertAlmostEqual(weighted["f1_weighted"], expected_f1, places=8)
                self.assertNotAlmostEqual(weighted["f1_weighted"], metrics["f1_macro"], places=2)

    def test_balanced_loader_never_samples_validation_or_test(self):
        rows = ([{"split": "train", "label": str(label)}
                 for label in (0, 0, 0, 1, 2, 3, 4)]
                + [{"split": "val", "label": "1"}, {"split": "test", "label": "4"}])

        def synthetic_split(_rows, _root, split, _task, _augment):
            selected = [row for row in _rows if row["split"] == split]
            labels = torch.tensor([int(row["label"]) for row in selected])
            return TensorDataset(torch.zeros(len(selected), 1), labels)

        protocol = Protocol(micro_batch_size=4, accumulation=4, balancing=BALANCED_TRAINING)
        with patch("reproduction.cadvit.experiments.dataset_for", side_effect=synthetic_split):
            train = split_loader(rows, None, "train", "five_class", protocol)
            val = split_loader(rows, None, "val", "five_class", protocol)
            test = split_loader(rows, None, "test", "five_class", protocol)
            self.assertEqual(sum(len(y) for _, y in train), 15)
            self.assertEqual(sum(len(y) for _, y in val), 1)
            self.assertEqual(sum(len(y) for _, y in test), 1)
            self.assertEqual([int(y.item()) for _, batch in val for y in batch], [1])

    def test_gpu_memory_check_requires_explicit_flag(self):
        with patch("reproduction.cadvit.experiments.gpu_memory_check") as check:
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    main(["memory-check", "--variant", "aptos_r16"])
            check.assert_not_called()

    def test_synthetic_evaluation_can_store_prediction_metadata(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 5)).eval()
        loader = torch.utils.data.DataLoader(
            TensorDataset(torch.arange(5, dtype=torch.float32).reshape(5, 1),
                          torch.arange(5, dtype=torch.long)), batch_size=2)
        metrics = score(model, loader, torch.device("cpu"), "five_class",
                        collect_predictions=True)
        self.assertEqual(metrics["n"], 5)
        self.assertEqual(len(metrics["predictions"]), 5)
        self.assertEqual([row["true_label"] for row in metrics["predictions"]], list(range(5)))
        self.assertTrue(all(0 <= row["confidence"] <= 1 for row in metrics["predictions"]))
        self.assertIn("f1_weighted", metrics)


if __name__ == "__main__":
    unittest.main()
