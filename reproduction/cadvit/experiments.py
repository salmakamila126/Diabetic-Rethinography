"""Controlled CAD-ViT Table 1--5 runner; no work runs on import.

Model/preprocessing are imported from the historical extraction. The new run
directory and frozen protocol are always outside this Git repository.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reproduction.cadvit.evaluation import classification_metrics, file_sha256, validate_manifest
from reproduction.cadvit.model import (ARCH_CONFIG, CADViTCandidate,
                                       CoordinateAttention, MultiScaleFeatureBlock)


CAMPAIGN = "cadvit_tables_1_5_paper_v1"
MODEL_SOURCE = Path(__file__).with_name("model.py")
PREPROCESS_SOURCE = Path(__file__).with_name("preprocessing.py")


@dataclass(frozen=True)
class Protocol:
    seed: int = 42  # implementation assumption
    epochs: int = 80  # paper Section IV.B
    learning_rate: float = 0.000186  # paper Section IV.B
    effective_batch_size: int = 16  # paper Section IV.B
    micro_batch_size: int = 16
    accumulation: int = 1
    preprocessing: str = "paper_described_historical_v1"
    augment_train: bool = True
    balancing: str = "none"  # no undocumented sampler/relabeling
    optimizer: str = "AdamW"
    weight_decay: float = 0.0
    loss: str = "CrossEntropyLoss"
    amp: bool = True
    grad_clip_norm: float = 1.0
    checkpoint_metric: str = "validation_macro_f1"
    checkpoint_tiebreak: str = "validation_loss"
    num_workers: int = 0  # deterministic resume

    def validate(self):
        if (self.seed != 42 or self.epochs != 80 or self.learning_rate != 0.000186
                or self.effective_batch_size != 16 or self.micro_batch_size < 1
                or self.accumulation < 1
                or self.micro_batch_size * self.accumulation != 16
                or self.preprocessing != "paper_described_historical_v1"
                or not self.augment_train or self.balancing != "none"
                or self.optimizer != "AdamW" or self.weight_decay != 0.0
                or self.loss != "CrossEntropyLoss" or self.grad_clip_norm != 1.0
                or self.checkpoint_metric != "validation_macro_f1"
                or self.checkpoint_tiebreak != "validation_loss" or self.num_workers != 0):
            raise ValueError("Protokol kampanye berubah; jangan campur run yang tidak setara")


@dataclass(frozen=True)
class Variant:
    name: str
    dataset: str
    task: str  # five_class or binary_head
    reduction: int = 16
    use_mfb: bool = True
    use_attention: bool = True

    def model_config(self):
        return {**ARCH_CONFIG, "reduction": self.reduction, "use_mfb": self.use_mfb,
                "use_attention": self.use_attention,
                "num_classes": 2 if self.task == "binary_head" else 5}


VARIANTS = {
    **{f"aptos_r{r}": Variant(f"aptos_r{r}", "APTOS2019", "five_class", r)
       for r in (2, 4, 8, 16, 32)},
    "aptos_baseline": Variant("aptos_baseline", "APTOS2019", "five_class", 16, False, False),
    "aptos_mfb": Variant("aptos_mfb", "APTOS2019", "five_class", 16, True, False),
    "aptos_attention": Variant("aptos_attention", "APTOS2019", "five_class", 16, False, True),
    "aptos_binary_head": Variant("aptos_binary_head", "APTOS2019", "binary_head"),
    "ddr_full": Variant("ddr_full", "DDR", "five_class"),
}
TABLES = {
    "Table 1": tuple(f"aptos_r{r}" for r in (2, 4, 8, 16, 32)),
    "Table 2": ("aptos_binary_head",),
    "Table 3": ("aptos_baseline", "aptos_mfb", "aptos_attention", "aptos_r16"),
    "Table 4": ("aptos_r16",),
    "Table 5": ("ddr_full",),
}
PAPER_COMPARISONS = {
    "APTOS2019": [
        ("SDMAE", 2023, 83.1, None, None, None, None),
        ("ADCNet", 2023, 83.4, None, 69.6, 67.7, 67.0),
        ("TLSPOFECOC", 2023, 82.1, 95.5, 82.1, 82.1, 82.1),
        ("Model based on GNN", 2023, 85.5, None, None, None, 84.9),
        ("TMILv4", 2024, 85.6, None, None, 73.7, None),
        ("CAD-ViT paper", 2025, 88.6, 96.7, 88.4, 88.6, 88.4),
    ],
    "DDR": [
        ("AABNet", 2023, 77.2, None, 63.8, 58.8, 59.6),
        ("CLANet", 2023, 80.3, None, None, None, None),
        ("CRA-Net", 2024, 83.1, None, None, None, None),
        ("Deep integrative network", 2024, 83.9, None, 83.3, 83.9, 83.3),
        ("MDGNet", 2024, 81.3, None, 63.9, 56.9, 59.2),
        ("CAD-ViT paper", 2025, 84.8, 90.3, 84.8, 84.8, 84.4),
    ],
}


def validate_variant_matrix():
    """Prove Table 1 changes only r and Table 3 only the two ablation flags."""
    baseline = VARIANTS["aptos_r16"].model_config()
    for key in TABLES["Table 1"]:
        config = VARIANTS[key].model_config()
        changed = {name for name in baseline if baseline[name] != config[name]}
        if changed != ({"reduction"} if key != "aptos_r16" else set()):
            raise AssertionError(f"Table 1 changed unexpected fields: {key}: {changed}")
    expected = {"aptos_baseline": (False, False), "aptos_mfb": (True, False),
                "aptos_attention": (False, True), "aptos_r16": (True, True)}
    for key, flags in expected.items():
        config = VARIANTS[key].model_config()
        if (config["use_mfb"], config["use_attention"]) != flags:
            raise AssertionError(f"Wrong Table 3 flags: {key}")
        if any(config[name] != baseline[name] for name in baseline
               if name not in {"use_mfb", "use_attention"}):
            raise AssertionError(f"Table 3 changed a non-ablation field: {key}")
        model = CADViTCandidate(config)
        mfb = [m for m in model.modules() if isinstance(m, MultiScaleFeatureBlock)]
        attention = [m for m in model.modules() if isinstance(m, CoordinateAttention)]
        if len(mfb) != (3 if flags[0] else 0) or len(attention) != (4 if flags[1] else 0):
            raise AssertionError(f"Ablation flags did not change real modules: {key}")
    return True


def core_macs(config, size=512):
    """Analytic conv/linear/attention MACs for batch=1; excludes BN/softmax/etc.

    One MAC = one multiply-accumulate. Two arithmetic FLOPs per MAC is also
    reported. This is an explicitly incomplete estimate, not paper's profiler.
    """
    if size < 32 or size % 32 or size % config["patch_size"]:
        raise ValueError("Input size must be a positive multiple of 32")
    details = {}

    def conv(h, w, cin, cout, kernel, groups=1):
        return h * w * cout * (cin // groups) * kernel * kernel

    c = config["stage_channels"]
    h = w = size // 2
    current = config["stem_channels"]
    details["stem"] = conv(h, w, 3, current, 3)
    steps = [("IRB-1", c[0], 1, False), ("IRB+A-1", c[1], 2, True),
             ("IRB-2", c[2], 1, False), ("IRB+A-2", c[3], 2, True),
             ("EMB-1", c[4], 1, False), ("IRB+A-3", c[5], 2, True),
             ("EMB-2", c[6], 1, False), ("IRB+A-4", c[7], 2, True),
             ("EMB-3", c[8], 1, False)]
    for name, out, stride, has_attention in steps:
        if name.startswith("EMB"):
            if out != current:
                raise ValueError("EMB channels do not match preceding block")
            mfb = 4 * conv(h, w, current, current, 3) + conv(h, w, 4 * current, current, 3)
            patch = config["patch_size"]
            n = (h // patch) * (w // patch)
            p = patch * patch
            dim = current
            ffn = dim * config["mlp_ratio"]
            transformer_layer = p * n * (4 * dim * dim + 2 * dim * ffn) + 2 * p * n * n * dim
            details[name] = (mfb if config["use_mfb"] else 0) + (
                config["transformer_depth"] * transformer_layer
                + conv(h, w, dim, dim, 1) + conv(h, w, dim, dim, 3))
        else:
            hidden = current * config["expansion"]
            after_h, after_w = h // stride, w // stride
            if h % stride or w % stride:
                raise ValueError("Noninteger feature map")
            operations = conv(h, w, current, hidden, 1)
            operations += conv(after_h, after_w, hidden, hidden, 3, groups=hidden)
            if has_attention and config["use_attention"]:
                reduced = max(1, hidden // config["reduction"])
                operations += 2 * (after_h + after_w) * hidden * reduced
            operations += conv(after_h, after_w, hidden, out, 1)
            details[name] = operations
            h, w = after_h, after_w
            current = out
    details["head"] = conv(h, w, current, config["head_channels"], 1)
    details["classifier"] = config["head_channels"] * config["num_classes"]
    macs = sum(details.values())
    return {"input_shape": [1, 3, size, size], "core_macs": macs,
            "gmac": macs / 1e9, "arithmetic_gflops_2_per_mac": 2 * macs / 1e9,
            "convention": "conv/linear/attention matmul only; excludes BN, activations, softmax, pooling, bias and data pipeline",
            "by_block_macs": details}


def binary_metrics(y_true, y_pred):
    """True rows and predicted columns; 0=No DR, 1=DR."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Empty or mismatched binary predictions")
    cm = [[0, 0], [0, 0]]
    for actual, predicted in zip(y_true, y_pred):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("Binary labels must be 0 or 1")
        cm[actual][predicted] += 1
    n = len(y_true)
    classes = {}
    for label, name in ((0, "No DR"), (1, "DR")):
        tp, fn = cm[label][label], sum(cm[label]) - cm[label][label]
        fp = sum(cm[i][label] for i in (0, 1) if i != label)
        tn = n - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else None
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[name] = {"support": tp + fn, "precision": precision, "recall": recall,
                         "specificity": specificity, "f1": f1,
                         "one_vs_rest_accuracy": (tp + tn) / n}
    return {"n": n, "accuracy": (cm[0][0] + cm[1][1]) / n,
            "f1_macro": sum(x["f1"] for x in classes.values()) / 2,
            "specificity_macro": sum(x["specificity"] for x in classes.values()) / 2,
            "confusion_matrix": cm, "per_class": classes,
            "orientation": "rows=true No DR,DR; columns=predicted No DR,DR"}


def campaign_root(artifact_root):
    root = Path(artifact_root).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if root == repo or repo in root.parents:
        raise ValueError("Artifact root must be outside Git repository")
    return root / "experiments" / CAMPAIGN


def protocol_path(artifact_root):
    return campaign_root(artifact_root) / "protocol.json"


def create_protocol(artifact_root, micro_batch_size=16):
    if micro_batch_size < 1 or 16 % micro_batch_size:
        raise ValueError("Micro batch must divide effective batch 16")
    protocol = replace(Protocol(), micro_batch_size=micro_batch_size,
                       accumulation=16 // micro_batch_size)
    protocol.validate()
    path = protocol_path(artifact_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"protocol": asdict(protocol), "classification":
                   "paper-guided implementation assumptions; BG10 excluded"}, stream, indent=2)
    return path


def load_protocol(artifact_root):
    path = protocol_path(artifact_root)
    content = json.loads(path.read_text(encoding="utf-8"))
    protocol = Protocol(**content["protocol"])
    protocol.validate()
    return protocol, file_sha256(path)


def artifact_paths(artifact_root, variant, data_root=None):
    root = Path(artifact_root).expanduser().resolve()
    dataset_root = (Path(data_root).expanduser().resolve() if data_root
                    else root / "datasets" / variant.dataset)
    return {"manifest": root / "cad_vit_runs" / "manifests" / f"{variant.dataset}_seed42.csv",
            "dataset_root": dataset_root,
            "run_dir": campaign_root(root) / variant.name}


def preflight(artifact_root, variant_name, data_root=None):
    """Read metadata only; neither train nor test image is opened."""
    if variant_name not in VARIANTS:
        raise ValueError("Unknown variant")
    validate_variant_matrix()
    variant = VARIANTS[variant_name]
    protocol, protocol_sha = load_protocol(artifact_root)
    paths = artifact_paths(artifact_root, variant, data_root)
    manifest = paths["manifest"]
    checked = validate_manifest(manifest, variant.dataset, paths["dataset_root"],
                                file_sha256(manifest))
    dependencies = {}
    for name in ("torch", "torchvision", "cv2", "PIL", "numpy"):
        try:
            importlib.import_module(name)
            dependencies[name] = True
        except (ImportError, RuntimeError):
            dependencies[name] = False
    if not all(dependencies.values()):
        raise RuntimeError(f"Runtime dependencies missing; no installation performed: {dependencies}")
    return {"variant": variant.name, "dataset": variant.dataset, "task": variant.task,
            "manifest_sha256": checked["sha256"], "split_sizes": checked["split_sizes"],
            "protocol_sha256": protocol_sha, "model_sha256": file_sha256(MODEL_SOURCE),
            "preprocessing_sha256": file_sha256(PREPROCESS_SOURCE),
            "run_dir": str(paths["run_dir"]), "dataset_root": str(paths["dataset_root"]),
            "manifest": str(manifest), "dependencies": dependencies,
            "micro_batch_size": protocol.micro_batch_size,
            "accumulation": protocol.accumulation,
            "note": "No test image opened; original 5-class manifest is retained for binary label mapping"}


def dataset_for(rows, root, split, task, augment):
    """Only the requested split is instantiated; test never enters training."""
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from reproduction.cadvit.preprocessing import preprocess_fundus

    chosen = [row for row in rows if row["split"] == split]
    transform = (transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, shear=10,
                                interpolation=InterpolationMode.BILINEAR, fill=0),
    ]) if split == "train" and augment else None)

    class FundusSplit(Dataset):
        def __len__(self):
            return len(chosen)

        def __getitem__(self, index):
            row = chosen[index]
            path = (Path(root) / PurePosixPath(row["image_path"])).resolve()
            if Path(root).resolve() not in path.parents:
                raise ValueError("Image path escapes dataset root")
            with Image.open(path) as image:
                pixels = preprocess_fundus(image)
            if transform is not None:
                pixels = np.asarray(transform(Image.fromarray(pixels)))
            label = int(row["label"])
            if task == "binary_head":
                label = int(label > 0)
            tensor = torch.from_numpy(np.array(pixels, copy=True, order="C")).permute(2, 0, 1).float() / 255
            return tensor, label

    return FundusSplit()


def split_loader(rows, root, split, task, protocol, epoch=0):
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    generator = torch.Generator().manual_seed(protocol.seed + epoch)
    return DataLoader(dataset_for(rows, root, split, task, protocol.augment_train),
                      batch_size=protocol.micro_batch_size, shuffle=split == "train",
                      num_workers=0, generator=generator, pin_memory=torch.cuda.is_available())


def score(model, loader, device, task):
    model.eval()
    truth, predicted, loss_sum, count = [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            if logits.shape != (len(y), 2 if task == "binary_head" else 5):
                raise ValueError("Unexpected logit shape")
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite logits")
            loss = criterion(logits, y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite evaluation loss")
            loss_sum += float(loss) * len(y)
            count += len(y)
            truth.extend(int(v) for v in y.tolist())
            predicted.extend(int(v) for v in logits.argmax(1).cpu().tolist())
    if not count:
        raise ValueError("Evaluation split empty")
    metrics = (binary_metrics(truth, predicted) if task == "binary_head"
               else classification_metrics(truth, predicted))
    metrics["loss"] = loss_sum / count
    return metrics


def train_epoch(model, loader, optimizer, scaler, device, protocol):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    n, loss_sum, skipped, consecutive_skips = 0, 0.0, 0, 0
    amp_enabled = protocol.amp and device.type == "cuda"
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        window_start = (step // protocol.accumulation) * protocol.accumulation
        window_end = min(window_start + protocol.accumulation, len(loader))
        window_examples = min(len(loader.dataset) - window_start * protocol.micro_batch_size,
                              (window_end - window_start) * protocol.micro_batch_size)
        scaler.scale(loss * (len(y) / window_examples)).backward()
        loss_sum += float(loss) * len(y)
        n += len(y)
        if (step + 1) % protocol.accumulation and step + 1 < len(loader):
            continue
        scaler.unscale_(optimizer)
        finite_gradients = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                               if p.grad is not None)
        if finite_gradients:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.grad_clip_norm,
                                                  error_if_nonfinite=True)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient norm")
        elif not amp_enabled:
            raise FloatingPointError("Nonfinite FP32 gradients")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not finite_gradients or scaler.get_scale() < scale_before:
            skipped += 1
            consecutive_skips += 1
            if consecutive_skips > 3:
                raise FloatingPointError("More than 3 consecutive skipped AMP updates")
        else:
            consecutive_skips = 0
        optimizer.zero_grad(set_to_none=True)
    if n != len(loader.dataset):
        raise ValueError("Training epoch incomplete")
    return {"loss": loss_sum / n, "skipped_optimizer_steps": skipped}


def _atomic_save(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_history(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def train(artifact_root, variant_name, data_root=None, resume=False):
    """Manual GPU/CPU training; never instantiates or reads the test split."""
    report = preflight(artifact_root, variant_name, data_root)
    variant = VARIANTS[variant_name]
    protocol, _ = load_protocol(artifact_root)
    run_dir = Path(report["run_dir"])
    record = {"variant": asdict(variant), "model_config": variant.model_config(),
              "protocol": asdict(protocol), "protocol_sha256": report["protocol_sha256"],
              "manifest_sha256": report["manifest_sha256"],
              "model_sha256": report["model_sha256"],
              "preprocessing_sha256": report["preprocessing_sha256"],
              "paper": "10.1109/ACCESS.2025.3572455; candidate architecture; assumptions documented"}
    record = json.loads(json.dumps(record))  # tuple stage widths become stable JSON lists
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    history_path = run_dir / "history.jsonl"
    if json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != record:
        raise ValueError("Run config differs; do not resume mixed protocols")
    with Path(report["manifest"]).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CADViTCandidate(variant.model_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.learning_rate,
                                  weight_decay=protocol.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=protocol.amp and device.type == "cuda")
    start, best_f1, best_loss = 0, -1.0, float("inf")
    if resume:
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if state.get("record") != record:
            raise ValueError("Checkpoint record differs")
        history = _read_history(history_path)
        if [item["epoch"] for item in history] != list(range(1, state["epoch"] + 2)):
            raise ValueError("History and last.pt epoch disagree; inspect manually")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["epoch"] + 1
        best_f1, best_loss = state["best_f1"], state["best_loss"]
    else:
        random.seed(protocol.seed)
        np.random.seed(protocol.seed)
        torch.manual_seed(protocol.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(protocol.seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    for epoch in range(start, protocol.epochs):
        start_time = time.monotonic()
        train_loader = split_loader(rows, report["dataset_root"], "train", variant.task,
                                    protocol, epoch)
        val_loader = split_loader(rows, report["dataset_root"], "val", variant.task, protocol)
        train_result = train_epoch(model, train_loader, optimizer, scaler, device, protocol)
        val = score(model, val_loader, device, variant.task)
        improved = (val["f1_macro"] > best_f1 or
                    (val["f1_macro"] == best_f1 and val["loss"] < best_loss))
        if improved:
            best_f1, best_loss = val["f1_macro"], val["loss"]
        entry = {"epoch": epoch + 1, "train_loss": train_result["loss"],
                 "skipped_optimizer_steps": train_result["skipped_optimizer_steps"],
                 "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                 "val_macro_f1": val["f1_macro"],
                 "val_macro_specificity": val["specificity_macro"],
                 "epoch_seconds": time.monotonic() - start_time}
        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "record": record, "best_f1": best_f1, "best_loss": best_loss,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        _atomic_save(run_dir / "last.pt", state)
        if improved:
            _atomic_save(run_dir / "best.pt", state)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps({"variant": variant_name, **entry}), flush=True)
    return run_dir


def final_test(artifact_root, variant_name, data_root=None):
    """Manual one-time test; never used to select variants or checkpoints."""
    report = preflight(artifact_root, variant_name, data_root)
    variant = VARIANTS[variant_name]
    protocol, _ = load_protocol(artifact_root)
    run_dir = Path(report["run_dir"])
    result_path = run_dir / "test_metrics.json"
    if result_path.exists():
        raise FileExistsError("Test already evaluated; do not retune")
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if state.get("record") != record or last.get("record") != record or last.get("epoch") != 79:
        raise ValueError("Checkpoint/config mismatch or 80 epochs incomplete")
    history = _read_history(run_dir / "history.jsonl")
    if [x["epoch"] for x in history] != list(range(1, 81)):
        raise ValueError("Training history incomplete")
    best = max(history, key=lambda x: (x["val_macro_f1"], -x["val_loss"]))
    if state["epoch"] + 1 != best["epoch"] or not math.isclose(
            state["best_f1"], best["val_macro_f1"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Best checkpoint not selected by validation macro-F1")
    if file_sha256(report["manifest"]) != report["manifest_sha256"]:
        raise ValueError("Manifest changed after preflight")
    with Path(report["manifest"]).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CADViTCandidate(variant.model_config()).to(device)
    model.load_state_dict(state["model"], strict=True)
    metrics = score(model, split_loader(rows, report["dataset_root"], "test",
                                        variant.task, protocol), device, variant.task)
    payload = {"dataset": variant.dataset, "variant": variant.name, "split": "test",
               "selected_epoch": best["epoch"], "selection": "validation_macro_f1",
               "manifest_sha256": report["manifest_sha256"],
               "protocol_sha256": report["protocol_sha256"],
               "model_sha256": report["model_sha256"], "metrics": metrics}
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return payload


def comparison_rows(dataset, result_path, artifact_root):
    """Paper rows are quoted, never represented as rerun third-party work."""
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    required_variant = "aptos_r16" if dataset == "APTOS2019" else "ddr_full"
    run_dir = campaign_root(artifact_root) / required_variant
    if Path(result_path).resolve() != (run_dir / "test_metrics.json").resolve():
        raise ValueError("Comparison requires the controlled run result path")
    record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    _, current_protocol_sha = load_protocol(artifact_root)
    if (record.get("variant") != asdict(VARIANTS[required_variant])
            or record.get("protocol_sha256") != current_protocol_sha
            or record.get("manifest_sha256") != payload.get("manifest_sha256")
            or payload.get("protocol_sha256") != current_protocol_sha):
        raise ValueError("Result does not match frozen run configuration and manifest")
    if (payload.get("dataset") != dataset or payload.get("variant") != required_variant
            or payload.get("split") != "test" or not payload.get("manifest_sha256")
            or not payload.get("protocol_sha256")):
        raise ValueError("Result provenance differs from controlled full-model run")
    metrics = payload["metrics"]
    own = ("Independent candidate, not author architecture", 2026,
           100 * metrics["accuracy"], 100 * metrics["specificity_macro"],
           100 * metrics["precision_macro"], 100 * metrics["recall_macro"],
           100 * metrics["f1_macro"])
    return [dict(zip(("method", "year", "accuracy", "specificity", "precision", "recall", "f1"), row))
            for row in PAPER_COMPARISONS[dataset] + [own]]


def matrix(artifact_root=None):
    result = []
    for table, names in TABLES.items():
        for name in names:
            status = "missing" if artifact_root is None or not protocol_path(artifact_root).is_file() else "needs-training"
            if artifact_root is not None:
                run_dir = campaign_root(artifact_root) / name
                if (run_dir / "test_metrics.json").is_file():
                    status = "needs-validation"  # metadata still must be checked
                elif (run_dir / "last.pt").is_file():
                    status = "needs-validation"
            result.append({"table": table, "variant": name, "dataset": VARIANTS[name].dataset,
                           "status": status})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("matrix", "init", "preflight", "dry-run", "train", "test", "compare"))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--execute-final-test", action="store_true")
    parser.add_argument("--dataset", choices=("APTOS2019", "DDR"))
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    if args.command != "matrix" and args.command != "dry-run" and args.artifact_root is None:
        parser.error("--artifact-root is required")
    if args.command in {"preflight", "dry-run", "train", "test"} and not args.variant:
        parser.error("--variant is required")
    if args.command == "matrix":
        output = matrix(args.artifact_root)
    elif args.command == "init":
        output = {"protocol": str(create_protocol(args.artifact_root, args.micro_batch))}
    elif args.command == "dry-run":
        validate_variant_matrix()
        variant = VARIANTS[args.variant]
        model = CADViTCandidate(variant.model_config()).cpu().eval()
        with torch.inference_mode():
            synthetic_output = model(torch.zeros(1, 3, 64, 64))
        if not bool(torch.isfinite(synthetic_output).all()):
            raise FloatingPointError("Synthetic CPU forward produced nonfinite logits")
        output = {"variant": variant.name, "model_config": variant.model_config(),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "flops_estimate": core_macs(variant.model_config()),
                  "synthetic_cpu_forward": {"input_shape": [1, 3, 64, 64],
                                            "output_shape": list(synthetic_output.shape),
                                            "finite": True},
                  "note": "No dataset, GPU training, or final test accessed"}
    elif args.command == "preflight":
        output = preflight(args.artifact_root, args.variant, args.data_root)
    elif args.command == "train":
        if not args.execute_training:
            parser.error("train requires --execute-training")
        output = {"run_dir": str(train(args.artifact_root, args.variant, args.data_root, args.resume))}
    elif args.command == "test":
        if not args.execute_final_test:
            parser.error("test requires --execute-final-test")
        output = final_test(args.artifact_root, args.variant, args.data_root)
    else:
        if args.dataset is None or args.result is None:
            parser.error("compare requires --dataset and --result")
        output = comparison_rows(args.dataset, args.result, args.artifact_root)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
