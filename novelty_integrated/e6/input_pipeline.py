"""B1-equivalent train loader using immutable decoded RGB disk cache."""
import multiprocessing
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from novelty import foundation_augmentation as augment
from novelty_integrated.e5.input_pipeline import Rows, EpochBatches, B1Collator


class TrainCollator:
    def __init__(self, processor, cache, *, seed, epoch):
        self.processor, self.seed, self.epoch = processor, seed, epoch
        self.cache = cache

    def __call__(self, rows):
        if not rows or any(row["split"] != "train" for row in rows):
            raise ValueError("E6 train cache accepts training rows only")
        images = []
        for row in rows:
            rgb = Image.fromarray(np.asarray(self.cache.read(row)).copy(), mode="RGB")
            image, _ = augment.apply(rgb, seed=self.seed, epoch=self.epoch.value,
                                     image_id=row["image_id"])
            images.append(image)
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        if (not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (len(rows), 3, 224, 224)
                or not bool(torch.isfinite(pixels).all())):
            raise ValueError("Official E6 processor output invalid")
        return pixels, torch.tensor([row["label"] for row in rows], dtype=torch.long)


class InputLoaders:
    def __init__(self, rows, processor, train_cache, *, seed, micro_batch,
                 num_workers=0, persistent_workers=False, prefetch_factor=2,
                 pin_memory=False, validation_cache=None):
        if num_workers < 0 or prefetch_factor < 1 or (persistent_workers and not num_workers):
            raise ValueError("Invalid worker configuration")
        self.epoch = multiprocessing.Value("i", 1)
        self.train_sampler = EpochBatches(len(rows["train"]), micro_batch, seed, training=True)
        val_sampler = EpochBatches(len(rows["val"]), micro_batch, seed, training=False)
        common = {"num_workers": num_workers, "persistent_workers": persistent_workers,
                  "pin_memory": pin_memory}
        if num_workers:
            common["prefetch_factor"] = prefetch_factor
        self.train = DataLoader(Rows(rows["train"]), batch_sampler=self.train_sampler,
                                collate_fn=TrainCollator(processor, train_cache, seed=seed, epoch=self.epoch), **common)
        self.val = DataLoader(Rows(rows["val"]), batch_sampler=val_sampler,
                              collate_fn=B1Collator(processor, training=False, seed=seed,
                                                    epoch=self.epoch, validation_cache=validation_cache),
                              **common)

    def train_epoch(self, epoch):
        self.train_sampler.epoch = epoch
        with self.epoch.get_lock():
            self.epoch.value = epoch
        return iter(self.train)

    def validation(self):
        return iter(self.val)
