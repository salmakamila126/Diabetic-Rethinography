"""Disk-backed train/validation input loader with E5 batch ordering."""
import multiprocessing
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from novelty_integrated.e5.input_pipeline import Rows, EpochBatches
from novelty_integrated.e7 import config, geometry


class ViewCollator:
    def __init__(self, processor, cache, *, training, seed, epoch):
        self.processor, self.cache = processor, cache
        self.training, self.seed, self.epoch = training, seed, epoch

    def __call__(self, rows):
        split = "train" if self.training else "val"
        if not rows or any(row["split"] != split for row in rows):
            raise ValueError("E7 collator forbids cross-split or empty batches")
        global_images, local_images, labels, ids = [], [], [], []
        for row in rows:
            rgb = Image.fromarray(np.asarray(self.cache.read(row)).copy(), mode="RGB")
            global_view, patches, _ = geometry.pil_views(rgb, training=self.training,
                seed=self.seed, epoch=self.epoch.value if self.training else None,
                image_id=row["image_id"] if self.training else None)
            global_images.append(global_view)
            local_images.extend(patches)
            labels.append(row["label"])
            ids.append(row["image_id"])
        common = {"do_resize": False, "return_tensors": "pt"}
        global_pixels = self.processor(images=global_images, **common)["pixel_values"]
        local_pixels = self.processor(images=local_images, **common)["pixel_values"]
        batch = len(rows)
        if (tuple(global_pixels.shape) != (batch,3,config.GLOBAL_SIZE,config.GLOBAL_SIZE)
                or tuple(local_pixels.shape) != (2*batch,3,config.PATCH_SIZE,config.PATCH_SIZE)
                or not bool(torch.isfinite(global_pixels).all())
                or not bool(torch.isfinite(local_pixels).all())):
            raise ValueError("E7 processor output shape/finiteness differs")
        return (global_pixels, local_pixels.reshape(batch,2,3,config.PATCH_SIZE,config.PATCH_SIZE),
                torch.tensor(labels, dtype=torch.long), ids)


class InputLoaders:
    def __init__(self, rows, processor, train_cache, val_cache, *, seed, micro_batch,
                 num_workers=0, persistent_workers=False, prefetch_factor=2, pin_memory=False):
        if num_workers < 0 or prefetch_factor < 1 or (persistent_workers and not num_workers):
            raise ValueError("Invalid E7 worker configuration")
        self.epoch = multiprocessing.Value("i", 1)
        self.train_sampler = EpochBatches(len(rows["train"]), micro_batch, seed, training=True)
        val_sampler = EpochBatches(len(rows["val"]), micro_batch, seed, training=False)
        common = {"num_workers": num_workers, "persistent_workers": persistent_workers,
                  "pin_memory": pin_memory}
        if num_workers:
            common["prefetch_factor"] = prefetch_factor
        self.train = DataLoader(Rows(rows["train"]), batch_sampler=self.train_sampler,
            collate_fn=ViewCollator(processor, train_cache, training=True,
                                   seed=seed, epoch=self.epoch),
            generator=torch.Generator().manual_seed(seed), **common)
        self.val = DataLoader(Rows(rows["val"]), batch_sampler=val_sampler,
            collate_fn=ViewCollator(processor, val_cache, training=False,
                                   seed=seed, epoch=self.epoch),
            generator=torch.Generator().manual_seed(seed + 1), **common)

    def train_epoch(self, epoch):
        self.train_sampler.epoch = epoch
        with self.epoch.get_lock():
            self.epoch.value = epoch
        return iter(self.train)

    def validation(self):
        return iter(self.val)
