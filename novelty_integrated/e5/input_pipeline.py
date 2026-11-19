"""Optional B1-equivalent, bounded E5 input transport; train/validation only."""

from collections import OrderedDict
import multiprocessing
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from novelty import foundation_augmentation as augment
from novelty import preof_phase1 as phase


class DecodedRGB:
    """Per-process bounded RAM LRU. Every miss verifies the frozen source hash."""

    def __init__(self, max_bytes=0, *, shared_counts=None, offset=0):
        if max_bytes < 0:
            raise ValueError("Decoded-image RAM limit must be nonnegative")
        self.max_bytes = max_bytes
        self._items = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = 0
        self.shared_counts, self.offset = shared_counts, offset

    def _count(self, slot):
        if self.shared_counts is not None:
            with self.shared_counts.get_lock():
                self.shared_counts[self.offset + slot] += 1

    def get(self, row):
        if row["split"] not in {"train", "val"}:
            raise ValueError("E5 input cache forbids test rows")
        path = Path(row["path"])
        key = (str(path), row["sha256"])
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        stored = self._items.get(key)
        if stored is not None and stored[0] == fingerprint:
            self._items.move_to_end(key)
            self.hits += 1
            self._count(0)
            return Image.fromarray(stored[1].copy(), mode="RGB")
        if stored is not None:
            self._bytes -= stored[1].nbytes
            del self._items[key]
        phase.require_hash(path, row["sha256"])
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != fingerprint:
            raise ValueError("Source image changed during verified decode")
        self.misses += 1
        self._count(1)
        if self.max_bytes:
            array = np.ascontiguousarray(np.asarray(rgb))
            if array.nbytes <= self.max_bytes:
                while self._items and self._bytes + array.nbytes > self.max_bytes:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= old[1].nbytes
                self._items[key] = (fingerprint, array)
                self._bytes += array.nbytes
        return rgb

    @property
    def resident_bytes(self):
        return self._bytes


class Rows(Dataset):
    def __init__(self, rows):
        if any(row["split"] not in {"train", "val"} for row in rows):
            raise ValueError("E5 DataLoader forbids test rows")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class EpochBatches(Sampler):
    """Match foundation_adaptation._epochs_batches order and final batch."""

    def __init__(self, count, micro_batch, seed, *, training):
        self.count, self.micro_batch, self.seed = count, micro_batch, seed
        self.training, self.epoch = training, 1

    def __iter__(self):
        order = (np.random.default_rng(self.seed + self.epoch).permutation(self.count)
                 if self.training else np.arange(self.count))
        for start in range(0, self.count, self.micro_batch):
            yield order[start:start + self.micro_batch].tolist()

    def __len__(self):
        return (self.count + self.micro_batch - 1) // self.micro_batch


class B1Collator:
    def __init__(self, processor, *, training, seed, epoch, decoded_max_bytes=0,
                 validation_cache=None, shared_counts=None):
        self.processor = processor
        self.training = training
        self.seed = seed
        self.epoch = epoch
        self.decoded = DecodedRGB(decoded_max_bytes, shared_counts=shared_counts,
                                  offset=0 if training else 2)
        self.validation_cache = validation_cache
        self.shared_counts = shared_counts

    def __call__(self, rows):
        if self.training and any(row["split"] != "train" for row in rows):
            raise ValueError("B1 augmentation is train-only")
        if not self.training and any(row["split"] != "val" for row in rows):
            raise ValueError("Validation cache is val-only")
        if self.validation_cache is not None:
            result = self.validation_cache.batch(rows)
            if self.shared_counts is not None:
                with self.shared_counts.get_lock():
                    self.shared_counts[2] += len(rows)
            return result
        images = []
        for row in rows:
            rgb = self.decoded.get(row)
            if self.training:
                rgb, _ = augment.apply(rgb, seed=self.seed, epoch=self.epoch.value,
                                       image_id=row["image_id"])
            images.append(rgb)
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        if (not isinstance(pixels, torch.Tensor)
                or tuple(pixels.shape) != (len(rows), 3, 224, 224)
                or not torch.isfinite(pixels).all()):
            raise ValueError("Official processor produced invalid E5 tensor")
        return pixels, torch.tensor([row["label"] for row in rows], dtype=torch.long)


class InputLoaders:
    def __init__(self, rows, processor, *, seed, micro_batch, num_workers=0,
                 persistent_workers=False, prefetch_factor=2, pin_memory=False,
                 decoded_cache_mib=0, validation_cache=None):
        if num_workers < 0 or prefetch_factor < 1 or decoded_cache_mib < 0:
            raise ValueError("Invalid E5 DataLoader resource settings")
        if persistent_workers and not num_workers:
            raise ValueError("Persistent workers require num_workers > 0")
        self.epoch = multiprocessing.Value("i", 1)
        self.counts = multiprocessing.Array("q", [0, 0, 0, 0])
        self.train_sampler = EpochBatches(len(rows["train"]), micro_batch, seed,
                                          training=True)
        val_sampler = EpochBatches(len(rows["val"]), micro_batch, seed, training=False)
        common = {"num_workers": num_workers, "pin_memory": pin_memory,
                  "persistent_workers": persistent_workers}
        if num_workers:
            common["prefetch_factor"] = prefetch_factor
        self.train = DataLoader(Rows(rows["train"]), batch_sampler=self.train_sampler,
                                collate_fn=B1Collator(processor, training=True, seed=seed,
                                                      epoch=self.epoch,
                                                      decoded_max_bytes=decoded_cache_mib * 1048576,
                                                      shared_counts=self.counts),
                                **common)
        self.val = DataLoader(Rows(rows["val"]), batch_sampler=val_sampler,
                              collate_fn=B1Collator(processor, training=False, seed=seed,
                                                    epoch=self.epoch,
                                                    validation_cache=validation_cache,
                                                    shared_counts=self.counts), **common)

    def cache_counts(self):
        with self.counts.get_lock():
            values = list(self.counts[:])
        return dict(zip(("train_rgb_hits", "train_rgb_misses",
                         "validation_cache_reads_or_rgb_hits", "validation_rgb_misses"), values))

    def train_epoch(self, epoch):
        self.train_sampler.epoch = epoch
        with self.epoch.get_lock():
            self.epoch.value = epoch
        return iter(self.train)

    def validation(self):
        return iter(self.val)
