"""Regression tests for the notebook extraction of CADViTCandidate.

Only the two historical architecture cells are executed; no dataset, training,
checkpoint, or evaluation cell is loaded.
"""

import ast
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reproduction.cadvit import model  # noqa: E402


COMPONENTS = (
    "ConvBNAct",
    "MultiScaleFeatureBlock",
    "CoordinateAttention",
    "InvertedResidualBlock",
    "EnhancedMobileViTBlock",
    "CADViTCandidate",
    "ARCH_CONFIG",
)


def architecture_sources():
    notebook = json.loads((ROOT / "CAD_ViT.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    first = [source for source in code if "class ConvBNAct(nn.Sequential):" in source]
    second = [source for source in code if "class CADViTCandidate(nn.Module):" in source]
    if len(first) != 1 or len(second) != 1:
        raise AssertionError("Expected exactly two historical architecture cells")
    return first[0], second[0]


def named_nodes(source):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                found[target.id] = node
    return found


class ModelExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        first, second = architecture_sources()
        cls.historical_source = first + "\n" + second
        cls.historical = {"__name__": "historical_cadvit_architecture"}
        exec(compile(first, "CAD_ViT.ipynb:architecture_1", "exec"), cls.historical)
        exec(compile(second, "CAD_ViT.ipynb:architecture_2", "exec"), cls.historical)

    def test_architecture_definitions_match_notebook(self):
        historical = named_nodes(self.historical_source)
        extracted = named_nodes((ROOT / "reproduction/cadvit/model.py").read_text(encoding="utf-8"))
        for name in COMPONENTS:
            with self.subTest(component=name):
                self.assertIn(name, historical)
                self.assertIn(name, extracted)
                self.assertEqual(
                    ast.dump(historical[name], include_attributes=False),
                    ast.dump(extracted[name], include_attributes=False),
                )

    def test_full_model_parameter_count_and_classification_shape(self):
        candidate = model.CADViTCandidate().eval()
        self.assertEqual(sum(parameter.numel() for parameter in candidate.parameters()), 5_282_101)
        with torch.inference_mode():
            output = candidate(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 5))

    def test_module_output_shapes(self):
        modules = (
            (model.ConvBNAct(3, 8, 3, stride=2), (1, 3, 16, 16), (1, 8, 8, 8)),
            (model.MultiScaleFeatureBlock(8), (1, 8, 8, 8), (1, 8, 8, 8)),
            (model.CoordinateAttention(16, reduction=16), (1, 16, 8, 10), (1, 16, 8, 10)),
            (model.InvertedResidualBlock(8, 8), (1, 8, 8, 8), (1, 8, 8, 8)),
            (model.InvertedResidualBlock(8, 16, stride=2, with_attention=True),
             (1, 8, 8, 8), (1, 16, 4, 4)),
            (model.EnhancedMobileViTBlock(16), (1, 16, 8, 8), (1, 16, 8, 8)),
        )
        with torch.inference_mode():
            for layer, input_shape, expected in modules:
                with self.subTest(module=type(layer).__name__, input_shape=input_shape):
                    output = layer.eval()(torch.randn(input_shape))
                    self.assertEqual(tuple(output.shape), expected)

    def test_unfold_and_fold_are_inverses(self):
        block = model.EnhancedMobileViTBlock(16, patch_size=2).eval()
        features = torch.randn(2, 16, 8, 12)
        tokens = block.unfold_tokens(features)
        self.assertEqual(tuple(tokens.shape), (8, 24, 16))
        restored = block.fold_tokens(tokens, (8, 12), batch=2)
        torch.testing.assert_close(restored, features, rtol=0, atol=0)

    def test_extracted_and_historical_forward_match_with_shared_weights(self):
        torch.manual_seed(42)
        extracted = model.CADViTCandidate().eval()
        historical = self.historical["CADViTCandidate"]().eval()
        historical.load_state_dict(extracted.state_dict(), strict=True)
        inputs = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            actual = extracted(inputs)
            expected = historical(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
