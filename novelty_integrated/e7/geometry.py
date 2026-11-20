"""Geometry-only, label-free global/patch views from original EXIF-corrected RGB."""
import math
import numpy as np
from PIL import Image
import torch
from novelty import foundation_augmentation as augmentation
from novelty_integrated.e7 import config


def letterbox(image: Image.Image, size: int) -> Image.Image:
    if image.mode != "RGB" or min(image.size) < 1 or size < 1:
        raise ValueError("Require nonempty RGB image and positive output size")
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                           Image.Resampling.BICUBIC)
    result = Image.new("RGB", (size, size), (0, 0, 0))
    result.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return result


def fov_box(rgb: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError("FOV geometry requires uint8 RGB")
    mask = np.max(arr, axis=2) > config.FOV_THRESHOLD
    height, width = mask.shape
    if mask.sum() < 0.05 * mask.size:
        return (0, 0, width, height)
    ys, xs = np.where(mask)
    margin_x = math.ceil(width * config.FOV_MARGIN_FRACTION)
    margin_y = math.ceil(height * config.FOV_MARGIN_FRACTION)
    return (max(0, int(xs.min()) - margin_x), max(0, int(ys.min()) - margin_y),
            min(width, int(xs.max()) + 1 + margin_x),
            min(height, int(ys.max()) + 1 + margin_y))


def patch_boxes(rgb: Image.Image) -> tuple[tuple[int,int,int,int], tuple[int,int,int,int]]:
    x0, y0, x1, y1 = fov_box(rgb)
    width = x1 - x0
    height = y1 - y0
    patch_width = min(width, max(1, math.ceil(width * config.PATCH_WIDTH_FRACTION)))
    patch_height = min(height, max(1, math.ceil(height * config.PATCH_WIDTH_FRACTION)))
    center_y = y0 + (height - patch_height) // 2
    left = (x0, center_y, x0 + patch_width, center_y + patch_height)
    right = (x1 - patch_width, center_y, x1, center_y + patch_height)
    if min(left[2]-left[0], right[2]-right[0], patch_height) < 1:
        raise ValueError("Empty local patch")
    return left, right


def pil_views(rgb: Image.Image, *, training=False, seed=42, epoch=None, image_id=None):
    if rgb.mode != "RGB":
        raise ValueError("Cached image must already be EXIF-corrected RGB")
    if training:
        if epoch is None or image_id is None:
            raise ValueError("Training requires deterministic keyed augmentation identity")
        rgb, _ = augmentation.apply(rgb, seed=seed, epoch=epoch, image_id=image_id)
    elif epoch is not None:
        raise ValueError("Validation never receives an augmentation epoch")
    boxes = patch_boxes(rgb)
    return (letterbox(rgb, config.GLOBAL_SIZE),
            [letterbox(rgb.crop(box), config.PATCH_SIZE) for box in boxes], boxes)


def model_views(rgb: Image.Image, processor, *, training=False, seed=42, epoch=None,
                image_id=None):
    global_view, patches, boxes = pil_views(rgb, training=training, seed=seed,
                                             epoch=epoch, image_id=image_id)
    # Override only geometric resize/crop; official rescale and normalization remain active.
    global_pixels = processor(images=[global_view], do_resize=False,
                              return_tensors="pt")["pixel_values"]
    patch_pixels = processor(images=patches, do_resize=False,
                             return_tensors="pt")["pixel_values"]
    if (not isinstance(global_pixels, torch.Tensor) or not isinstance(patch_pixels, torch.Tensor)
            or tuple(global_pixels.shape) != (1,3,config.GLOBAL_SIZE,config.GLOBAL_SIZE)
            or tuple(patch_pixels.shape) != (2,3,config.PATCH_SIZE,config.PATCH_SIZE)
            or not bool(torch.isfinite(global_pixels).all())
            or not bool(torch.isfinite(patch_pixels).all())):
        raise ValueError("Official processor returned unexpected E7 tensors")
    return global_pixels[0], patch_pixels, boxes
