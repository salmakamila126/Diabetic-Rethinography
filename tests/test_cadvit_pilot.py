"""Synthetic and metadata-only checks for the train/validation pilot CLI."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novelty import cadvit_pilot as pilot  # noqa: E402
from novelty.cadvit_model import NovelCADViT, NoveltyFlags, NoveltyOutput  # noqa: E402
from reproduction.cadvit.model import CADViTCandidate  # noqa: E402
from novelty.preof_phase1 import SPECS  # noqa: E402
from reproduction.cadvit.preprocessing import preprocess_fundus_v2  # noqa: E402


class TinyFiveClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 5)

    def forward_with_aux(self, x):
        return NoveltyOutput(self.layer(x), (), None)


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preflight_environment_does_not_initialize_cuda(self):
        with patch.object(pilot.torch.cuda, "get_device_name",
                          side_effect=AssertionError("CUDA context in preflight")):
            with patch.object(pilot.subprocess, "run",
                              return_value=SimpleNamespace(returncode=0,
                                                           stdout="RTX 3060\n")):
                environment = pilot._environment()
        self.assertEqual(environment["gpu_inventory"], ["RTX 3060"])
        self.assertIn("cuda_visible_devices", environment)

    def test_baseline_arm_is_original_class(self):
        self.assertIs(type(pilot._make_model("baseline")), CADViTCandidate)
        self.assertIs(type(pilot._make_model("p1_only")), NovelCADViT)

    def test_run_paths_are_independent_and_guarded(self):
        with TemporaryDirectory() as root:
            artifact = Path(root) / "artifacts"
            output = Path(root) / "novelty_outputs"
            a = pilot._run_dir(artifact, output, "APTOS2019", "p2_p3_probe", 1, 2)
            d = pilot._run_dir(artifact, output, "DDR", "p2_p3_probe", 1, 2)
            self.assertNotEqual(a, d)
            self.assertTrue(str(a).startswith(str(output)))
            with self.assertRaises(ValueError):
                pilot._run_dir(artifact, ROOT, "DDR", "baseline", 1, 2)
            with self.assertRaises(ValueError):
                pilot._run_dir(artifact, output, "DDR", "baseline", 3, 2)
            with self.assertRaises(ValueError):
                pilot._run_dir(artifact, output, "DDR", "baseline", 1, 80)

    def test_cached_protocol_rejects_legacy_output_and_overwrite(self):
        artifact = ROOT.parent
        old_output = artifact / "cadvit_novelty_pilots"
        rows = {"train": [{}], "val": [{}]}
        with patch.object(pilot, "load_train_val_manifest",
                          return_value=(rows, {"manifest_sha256": "a"*64})):
            with patch.object(pilot, "_checkpoint_preflight",
                              return_value=({}, {})):
                with self.assertRaisesRegex(ValueError, "new output root"):
                    pilot.preflight(artifact, old_output, "APTOS2019",
                                    "p2_p3_probe", 1, 2, "on",
                                    artifact / "cadvit_v2_cache")
        with TemporaryDirectory() as existing:
            with self.assertRaises(FileExistsError):
                pilot.pilot({"ablation": "p2_p3_probe"}, {}, {},
                            Path(existing), device_name="cpu")

    def test_only_train_and_validation_dataset(self):
        with self.assertRaises(ValueError):
            pilot.V2FundusSplit({"test": []}, "test")
        with self.assertRaises(ValueError):
            pilot.make_loader({"test": []}, "test", 1)
        with TemporaryDirectory() as root:
            path = Path(root) / "synthetic.png"
            image = np.full((40, 40, 3), 60, dtype=np.uint8)
            image[8:20, 11:23] = [220, 90, 40]
            Image.fromarray(image).save(path)
            rows = {
                "train": [{"split": "train", "path": str(path), "label": 2}],
                "val": [{"split": "val", "path": str(path), "label": 2}],
            }
            train = pilot.V2FundusSplit(rows, "train")
            val = pilot.V2FundusSplit(rows, "val")
            self.assertIsNotNone(train.augment)
            self.assertIsNone(val.augment)
            observed, label = val[0]
            with Image.open(path) as original:
                expected_pixels = preprocess_fundus_v2(original)
            expected = torch.from_numpy(
                np.array(expected_pixels, copy=True, order="C")
            ).permute(2, 0, 1).float() / 255
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            self.assertEqual(observed.shape, (3, 512, 512))
            self.assertEqual(label, 2)

    def test_checkpoint_metadata_matches_both_pinned_datasets(self):
        artifact = ROOT.parent
        for dataset in ("APTOS2019", "DDR"):
            with self.subTest(dataset=dataset):
                _, checked = pilot._checkpoint_preflight(
                    dataset, artifact, SPECS[dataset]["manifest_sha256"])
                self.assertEqual(checked["sha256"], pilot.CHECKPOINTS[dataset]["sha256"])
                self.assertEqual(checked["original_weight_transfer"]["original_tensors_loaded"], 285)
                self.assertEqual(checked["mixed_protocol_recovery"], dataset == "DDR")
                with self.assertRaises(ValueError):
                    pilot._checkpoint_preflight(dataset, artifact, "0" * 64)

    def test_ordinal_metrics_from_confusion_matrix(self):
        metrics = {"n": 4, "confusion_matrix": [
            [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
        ]}
        result = pilot._ordinal_metrics(metrics)
        self.assertEqual(result["grade_mae"], 0.25)
        self.assertGreater(result["qwk"], 0.8)

    def test_incomplete_accumulation_window_matches_manual_steps(self):
        torch.manual_seed(4)
        model = TinyFiveClass()
        reference = TinyFiveClass()
        reference.load_state_dict(model.state_dict())
        x = torch.randn(5, 3)
        y = torch.tensor([0, 1, 2, 3, 4])
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = pilot._train_epoch(
            model, loader, optimizer, torch.device("cpu"), NoveltyFlags(), accumulation=2)
        manual = torch.optim.SGD(reference.parameters(), lr=0.01)
        for start, stop in ((0, 4), (4, 5)):
            manual.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(reference.layer(x[start:stop]), y[start:stop]).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), pilot.GRAD_CLIP)
            manual.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        self.assertEqual(metrics["n"], 5)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 5)

    def test_resume_rejects_history_mismatch(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = TinyFiveClass()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            record = {"schema": pilot.SCHEMA, "synthetic": True,
                      "cache": {"enabled": True, "identity": "a"*64}}
            (run_dir / "config.json").write_text(json.dumps(record), encoding="utf-8")
            entry = {"epoch": 1, "val": {"f1_macro": 0.5, "loss": 0.8}}
            (run_dir / "history.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps({"last_epoch": 1}), encoding="utf-8")
            state = {
                "record": record, "epoch": 0, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": pilot._rng_state(),
                "best_epoch": 1, "best_f1": 0.5, "best_loss": 0.8,
            }
            torch.save(state, run_dir / "last.pt")
            torch.save(state, run_dir / "best.pt")
            self.assertEqual(pilot._restore_run(
                run_dir, record, model, optimizer, torch.device("cpu"))[0], 1)
            changed = dict(record)
            changed["cache"] = {"enabled": True, "identity": "b"*64}
            with self.assertRaisesRegex(ValueError, "config changed"):
                pilot._restore_run(run_dir, changed, model, optimizer,
                                   torch.device("cpu"))
            (run_dir / "history.jsonl").write_text(
                json.dumps({"epoch": 2, "val": entry["val"]}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                pilot._restore_run(run_dir, record, model, optimizer, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
