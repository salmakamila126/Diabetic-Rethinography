"""Pixel regression tests for preprocessing extracted from the archived notebook."""

import ast
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

if cv2 is not None:
    from reproduction.cadvit import preprocessing
else:
    preprocessing = None


NAMES = ("PREPROCESS_CONFIG", "preprocess_fundus", "BG_V2_CONFIG", "preprocess_fundus_v2")


def historical_nodes():
    notebook = json.loads((ROOT / "reproduction/archive/CAD_ViT_original_105_cells.ipynb").read_text(encoding="utf-8"))
    matching = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def preprocess_fundus(image," in "".join(cell["source"])
    ]
    if len(matching) != 1:
        raise AssertionError("Expected one historical preprocessing cell")
    nodes = {}
    for node in ast.parse(matching[0]).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                nodes[target.id] = node
    return nodes


class PreprocessingSourceTests(unittest.TestCase):
    def test_extracted_definitions_match_historical_ast(self):
        historical = historical_nodes()
        extracted = {}
        source = (ROOT / "reproduction/cadvit/preprocessing.py").read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                extracted[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    extracted[target.id] = node
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(
                    ast.dump(historical[name], include_attributes=False),
                    ast.dump(extracted[name], include_attributes=False),
                )


@unittest.skipUnless(cv2 is not None, "OpenCV (cv2) is not installed in this Python environment")
class PreprocessingPixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = [historical_nodes()[name] for name in NAMES]
        module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        cls.historical = {"np": np, "cv2": cv2, "ImageOps": ImageOps}
        exec(compile(module, "CAD_ViT_original_105_cells.ipynb:preprocessing", "exec"), cls.historical)

    def assert_pixel_equal(self, make_image):
        for name in ("preprocess_fundus", "preprocess_fundus_v2"):
            with self.subTest(function=name):
                expected = self.historical[name](make_image())
                actual = getattr(preprocessing, name)(make_image())
                self.assertTrue(np.array_equal(actual, expected), name)
                self.assertEqual(actual.shape, (512, 512, 3))
                self.assertEqual(actual.dtype, np.dtype("uint8"))
                self.assertGreaterEqual(int(actual.min()), 0)
                self.assertLessEqual(int(actual.max()), 255)

    def test_rgb_color_size_and_black_border_variations(self):
        rng = np.random.default_rng(42)
        for height, width, border in ((35, 53, 4), (79, 41, 7), (96, 128, 10)):
            with self.subTest(size=(height, width), border=border):
                array = np.zeros((height, width, 3), dtype=np.uint8)
                array[border:-border, border:-border] = rng.integers(
                    32, 256, size=(height - 2 * border, width - 2 * border, 3), dtype=np.uint8
                )
                self.assert_pixel_equal(lambda: Image.fromarray(array))

    def test_conversion_from_grayscale_and_rgba(self):
        gray = np.full((43, 59), 100, dtype=np.uint8)
        gray[:5, :] = 0
        rgba = np.zeros((59, 43, 4), dtype=np.uint8)
        rgba[6:-6, 6:-6] = (180, 40, 120, 255)
        self.assert_pixel_equal(lambda: Image.fromarray(gray))
        self.assert_pixel_equal(lambda: Image.fromarray(rgba))

    def test_exif_orientation(self):
        array = np.zeros((37, 61, 3), dtype=np.uint8)
        array[4:-4, 4:-4] = (30, 120, 220)
        array[8:18, 35:53] = (230, 60, 35)
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="JPEG", exif=exif)
        encoded = stream.getvalue()
        make_image = lambda: Image.open(io.BytesIO(encoded))
        self.assertEqual(make_image().getexif().get(274), 6)
        self.assert_pixel_equal(make_image)
        with make_image() as image:
            oriented = ImageOps.exif_transpose(image)
            expected = self.historical["preprocess_fundus"](oriented)
        actual = preprocessing.preprocess_fundus(make_image())
        self.assertTrue(np.array_equal(actual, expected))

    def test_all_dark_image_preserves_historical_error(self):
        for name in ("preprocess_fundus", "preprocess_fundus_v2"):
            with self.subTest(function=name):
                for implementation in (self.historical[name], getattr(preprocessing, name)):
                    with self.assertRaises(ValueError):
                        implementation(Image.new("RGB", (32, 24), color=(0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
