"""Optional E7 validation inference tools; no model or test-data loading here."""
import math
import numpy as np
from PIL import ImageOps
import torch
from torch.nn import functional as F
from novelty_integrated.e7.geometry import model_views

CLASS_ORDER = (0, 1, 2, 3, 4)


def _probabilities(array):
    values = np.asarray(array, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 5 or len(values) == 0
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Expected normalized finite Nx5 probabilities")
    return values


def ensemble(predictions, weights, *, split="val"):
    """Fuse precomputed probabilities after strict identity and order checks."""
    if split not in ("val", "calibration") or not predictions or len(predictions) != len(weights):
        raise ValueError("Only aligned validation/calibration predictions can be fused")
    w = np.asarray(weights, dtype=np.float64)
    if (w.shape != (len(predictions),) or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Ensemble weights must be nonnegative and sum to one")
    first = predictions[0]
    identity = (first["dataset"], first["manifest_sha256"], tuple(first["image_ids"]),
                tuple(first["class_order"]), first["split"])
    if identity[-2] != CLASS_ORDER or identity[-1] != split or len(set(identity[2])) != len(identity[2]):
        raise ValueError("Prediction identity or class order differs")
    arrays = []
    for item in predictions:
        candidate = (item["dataset"], item["manifest_sha256"], tuple(item["image_ids"]),
                     tuple(item["class_order"]), item["split"])
        digest = item.get("checkpoint_sha256")
        if (candidate != identity or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Ensemble split, order, classes, or checkpoint provenance differs")
        values = _probabilities(item["probabilities"])
        if len(values) != len(identity[2]):
            raise ValueError("Ensemble sample count differs")
        arrays.append(values)
    return np.tensordot(w, np.stack(arrays), axes=(0, 0))


def temperature_probabilities(logits, temperature):
    """Apply a prespecified positive temperature to categorical logits."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    values = torch.as_tensor(logits, dtype=torch.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not bool(torch.isfinite(values).all()):
        raise ValueError("Expected finite Nx5 categorical logits")
    return F.softmax(values / temperature, dim=1).numpy()


def select_temperature(logits, labels, *, split, sample_ids, forbidden_ids,
                       candidates=(0.75, 1.0, 1.25)):
    """Fit only on a disjoint, explicitly marked non-test calibration split."""
    if split != "calibration":
        raise ValueError("Temperature fitting requires a held-out calibration split")
    if (len(sample_ids) != len(labels) or len(set(sample_ids)) != len(sample_ids)
            or set(sample_ids) & set(forbidden_ids)):
        raise ValueError("Calibration IDs duplicate or overlap validation/test exclusions")
    if not candidates or any(not math.isfinite(t) or t <= 0 for t in candidates):
        raise ValueError("Calibration candidates must be positive and finite")
    values = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(labels, dtype=torch.long)
    if (values.ndim != 2 or values.shape != (len(targets), 5) or len(targets) < 1
            or bool(((targets < 0) | (targets > 4)).any())):
        raise ValueError("Calibration logits/labels differ")
    scores = [(float(F.cross_entropy(values / t, targets)), t) for t in candidates]
    if any(not math.isfinite(loss) for loss, _ in scores):
        raise ValueError("Nonfinite calibration loss")
    return min(scores)[1]


@torch.inference_mode()
def tta_probabilities(model, processor, rgb, device, *, horizontal=True, vertical=False):
    """Transform whole original RGB first so global and local views stay aligned."""
    transforms = [rgb]
    if horizontal:
        transforms.append(ImageOps.mirror(rgb))
    if vertical:
        transforms.append(ImageOps.flip(rgb))
    model.eval()
    probabilities = []
    for image in transforms:
        global_pixels, patches, _ = model_views(image, processor)
        logits, _, _ = model(global_pixels[None].to(device), patches[None].to(device))
        probabilities.append(logits.float().softmax(1).cpu())
    return torch.stack(probabilities).mean(0)[0]
