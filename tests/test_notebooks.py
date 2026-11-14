"""Structural/safety regression for the eight read-only reproduction notebooks."""

import ast
import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import unittest
from PIL import Image


DIRECTORY = Path(__file__).resolve().parents[1] / "reproduction" / "notebooks"
NUMBERED = [
    "01_preprocessing.ipynb", "02_architecture.ipynb", "03_training_protocol.ipynb",
    "04_classification_evaluation.ipynb", "05_ablation_and_reduction.ipynb",
    "06_balancing_analysis.ipynb", "07_error_analysis.ipynb", "08_gradcam.ipynb",
]


class NotebookTests(unittest.TestCase):
    def test_notebook_json_code_safety_and_saved_reference_images(self):
        images = {}
        for name in NUMBERED:
            with self.subTest(name=name):
                notebook = json.loads((DIRECTORY / name).read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                self.assertGreaterEqual(notebook["nbformat_minor"], 5)
                ids = [cell["id"] for cell in notebook["cells"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(cell["cell_type"] in {"code", "markdown"}
                                    for cell in notebook["cells"]))
                count = 0
                sources = []
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    sources.append(source)
                    tree = ast.parse(source, filename=name)
                    dangerous = {"run_experiment", "final_test", "train", "gpu_memory_check"}
                    self.assertFalse(any(isinstance(node, ast.Call)
                                         and isinstance(node.func, ast.Name)
                                         and node.func.id in dangerous
                                         for node in ast.walk(tree)))
                    if cell["execution_count"] is None:
                        # The guarded real-image Grad-CAM cell is intentionally
                        # unexecuted and has no saved image-level output.
                        self.assertEqual(name, "08_gradcam.ipynb")
                        self.assertIn("CAD_VIT_RUN_GRADCAM", source)
                        self.assertEqual(cell["outputs"], [])
                    for output in cell["outputs"]:
                        if "image/png" in output.get("data", {}):
                            raw = base64.b64decode(output["data"]["image/png"], validate=True)
                            self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
                            with Image.open(BytesIO(raw)) as image:
                                image.verify()
                            count += 1
                self.assertIn("reproduction.cadvit", "\n".join(sources))
                images[name] = count
        self.assertEqual(images["04_classification_evaluation.ipynb"], 3)
        self.assertEqual(images["06_balancing_analysis.ipynb"], 2)

    @unittest.skipUnless(importlib.util.find_spec("nbformat"), "nbformat unavailable locally")
    def test_nbformat_schema_when_installed(self):
        import nbformat

        for name in NUMBERED:
            with self.subTest(name=name):
                nbformat.validate(nbformat.read(DIRECTORY / name, as_version=4))


if __name__ == "__main__":
    unittest.main()
