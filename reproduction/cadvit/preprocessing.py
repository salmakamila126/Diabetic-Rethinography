"""Historical CAD-ViT fundus preprocessing extracted without behavioral changes."""

import cv2
import numpy as np
from PIL import ImageOps


PREPROCESS_CONFIG = {"image_size": 512, "crop_threshold": 10, "gaussian_kernel": 5,
                     "gaussian_sigma": 1.0}  # implementation assumptions except image_size


def preprocess_fundus(image, config=PREPROCESS_CONFIG):
    """§III.A: crop -> INTER_AREA resize -> Gaussian blur; no raw-file writes."""
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points = cv2.findNonZero((gray > config["crop_threshold"]).astype(np.uint8))
    if points is None:
        raise ValueError("Gambar seluruhnya gelap; crop tidak dapat ditentukan")
    x, y, w, h = cv2.boundingRect(points)
    cropped = rgb[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (config["image_size"], config["image_size"]),
                         interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (config["gaussian_kernel"],)*2,
                            sigmaX=config["gaussian_sigma"], sigmaY=config["gaussian_sigma"])

BG_V2_CONFIG = {"background_sigma": 10.0, "alpha": 4.0, "beta": -4.0,
                "offset": 128.0}  # literature-guided; coefficients are implementation assumptions


def preprocess_fundus_v2(image, config=PREPROCESS_CONFIG, bg_config=BG_V2_CONFIG):
    """V1 CAD-ViT operations, then float32 Ben Graham enhancement; no raw-file writes."""
    smoothed = preprocess_fundus(image, config=config)
    pixels = smoothed.astype(np.float32)
    background = cv2.GaussianBlur(pixels, (0, 0),
                                  sigmaX=bg_config["background_sigma"],
                                  sigmaY=bg_config["background_sigma"])
    enhanced = bg_config["alpha"] * pixels + bg_config["beta"] * background + bg_config["offset"]
    return np.clip(enhanced, 0.0, 255.0).astype(np.uint8)
