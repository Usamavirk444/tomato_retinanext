# src/dataset/yolo_dataset.py
"""
v3: adds pin_memory + persistent_workers to the dataloader.
Functionally identical to v1 otherwise — the dataset class itself
is unchanged.

Why these matter on a CUDA box:
  pin_memory=True       -> faster CPU->GPU transfers when num_workers>0
  persistent_workers    -> avoids worker process startup cost between epochs
  prefetch_factor=2     -> each worker prefetches 2 batches ahead
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.utils.yolo_parser import parse_yolo_txt
from src.utils.general import collate_fn


class YoloTomatoDataset(Dataset):
    """PyTorch Dataset for YOLO format tomato disease annotations."""

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        print(f"  Loaded {len(self.image_files)} images from {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path     = os.path.join(self.images_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        w, h  = image.size

        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path     = os.path.join(self.labels_dir, label_filename)

        annotation = parse_yolo_txt(label_path, w, h)

        boxes  = torch.tensor(annotation["boxes"],  dtype=torch.float32)
        labels = torch.tensor(annotation["labels"], dtype=torch.int64)

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "filename": img_filename,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def build_yolo_dataloader(
    images_dir, labels_dir, transforms,
    batch_size=8, shuffle=True, num_workers=0,
    pin_memory=None, persistent_workers=None
):
    """
    v3.1: Windows-safe defaults.

    Windows has a hard limit on the number of shared file mappings
    (error 1455 = ERROR_COMMITMENT_LIMIT). Keeping workers alive across
    epochs (persistent_workers=True) accumulates mappings until the
    process can't create new ones, which crashes training mid-run.

    Fix: on Windows, persistent_workers=False (workers restart each
    epoch — slightly slower but stable) and prefetch_factor=1.
    On macOS / Linux, the original v3 defaults stay in place.
    """
    import sys
    is_windows = sys.platform.startswith("win")

    dataset = YoloTomatoDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transforms=transforms
    )

    if pin_memory is None:
        pin_memory = (num_workers > 0 and torch.cuda.is_available())
    if persistent_workers is None:
        persistent_workers = (num_workers > 0 and not is_windows)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = 1 if is_windows else 2

    return DataLoader(dataset, **kwargs)
