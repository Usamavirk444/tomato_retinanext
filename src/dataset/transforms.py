# src/dataset/transforms.py

import torch
import torchvision.transforms.functional as F
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std  = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        new_h, new_w   = self.size
        image = F.resize(image, self.size)

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes    = target["boxes"].clone()
            x_scale  = new_w / orig_w
            y_scale  = new_h / orig_h
            boxes[:, 0] *= x_scale
            boxes[:, 1] *= y_scale
            boxes[:, 2] *= x_scale
            boxes[:, 3] *= y_scale
            target["boxes"] = boxes

        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, _ = image.size
            image    = F.hflip(image)
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes          = target["boxes"].clone()
                boxes[:, 0], boxes[:, 2] = (
                    width - boxes[:, 2],
                    width - boxes[:, 0]
                )
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, height = image.size
            image     = F.vflip(image)
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes          = target["boxes"].clone()
                boxes[:, 1], boxes[:, 3] = (
                    height - boxes[:, 3],
                    height - boxes[:, 1]
                )
                target["boxes"] = boxes
        return image, target


class RandomRotation:
    """Random 90/180/270 degree rotation."""
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.choice([90, 180, 270])
            w, h  = image.size
            image = F.rotate(image, angle, expand=True)

            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                new_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    if angle == 90:
                        new_boxes.append([ymin, w - xmax, ymax, w - xmin])
                    elif angle == 180:
                        new_boxes.append([w - xmax, h - ymax, w - xmin, h - ymin])
                    elif angle == 270:
                        new_boxes.append([h - ymax, xmin, h - ymin, xmax])
                target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)

        return image, target


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4,
                 saturation=0.4, hue=0.15):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(
            image, 1 + random.uniform(-self.brightness, self.brightness)
        )
        image = F.adjust_contrast(
            image, 1 + random.uniform(-self.contrast, self.contrast)
        )
        image = F.adjust_saturation(
            image, 1 + random.uniform(-self.saturation, self.saturation)
        )
        image = F.adjust_hue(
            image, random.uniform(-self.hue, self.hue)
        )
        return image, target


class RandomGrayscale:
    """Convert to grayscale with probability — helps with color invariance."""
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.rgb_to_grayscale(image, num_output_channels=3)
        return image, target


class RandomScale:
    """
    Randomly scale image between min_scale and max_scale
    and adjust bounding boxes accordingly.
    """
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h  = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = F.resize(image, (new_h, new_w))

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes       = target["boxes"].clone()
            boxes[:, 0] *= scale
            boxes[:, 1] *= scale
            boxes[:, 2] *= scale
            boxes[:, 3] *= scale
            target["boxes"] = boxes

        return image, target


class RandomCrop:
    """
    Random crop that preserves at least min_overlap of each box.
    """
    def __init__(self, prob=0.5, min_overlap=0.5):
        self.prob        = prob
        self.min_overlap = min_overlap

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        w, h = image.size
        for _ in range(10):  # max attempts
            crop_w = random.randint(int(w * 0.7), w)
            crop_h = random.randint(int(h * 0.7), h)
            left   = random.randint(0, w - crop_w)
            top    = random.randint(0, h - crop_h)

            if "boxes" not in target or len(target["boxes"]) == 0:
                image = F.crop(image, top, left, crop_h, crop_w)
                return image, target

            boxes   = target["boxes"].clone()
            crop_box = torch.tensor(
                [[left, top, left + crop_w, top + crop_h]],
                dtype=torch.float32
            )

            # Check overlap
            inter_xmin = torch.max(boxes[:, 0], crop_box[:, 0])
            inter_ymin = torch.max(boxes[:, 1], crop_box[:, 1])
            inter_xmax = torch.min(boxes[:, 2], crop_box[:, 2])
            inter_ymax = torch.min(boxes[:, 3], crop_box[:, 3])

            inter_w = (inter_xmax - inter_xmin).clamp(min=0)
            inter_h = (inter_ymax - inter_ymin).clamp(min=0)
            inter   = inter_w * inter_h

            box_area = (boxes[:, 2] - boxes[:, 0]) * \
                       (boxes[:, 3] - boxes[:, 1])
            overlap  = inter / box_area.clamp(min=1e-6)

            # Keep boxes with sufficient overlap
            keep = overlap >= self.min_overlap
            if keep.sum() == 0:
                continue

            image       = F.crop(image, top, left, crop_h, crop_w)
            boxes       = boxes[keep]
            boxes[:, 0] = (boxes[:, 0] - left).clamp(min=0)
            boxes[:, 1] = (boxes[:, 1] - top).clamp(min=0)
            boxes[:, 2] = (boxes[:, 2] - left).clamp(max=crop_w)
            boxes[:, 3] = (boxes[:, 3] - top).clamp(max=crop_h)

            target["boxes"]  = boxes
            target["labels"] = target["labels"][keep]
            return image, target

        return image, target


class RandomErasing:
    """
    Randomly erase a rectangular region — helps with occlusion robustness.
    Applied after ToTensor.
    """
    def __init__(self, prob=0.2, scale=(0.02, 0.1)):
        self.prob  = prob
        self.scale = scale

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        if not isinstance(image, torch.Tensor):
            return image, target

        _, h, w  = image.shape
        area     = h * w
        erase_area = random.uniform(*self.scale) * area
        aspect   = random.uniform(0.3, 3.0)
        eh       = int(np.sqrt(erase_area / aspect))
        ew       = int(np.sqrt(erase_area * aspect))

        if eh >= h or ew >= w:
            return image, target

        top  = random.randint(0, h - eh)
        left = random.randint(0, w - ew)
        image[:, top:top+eh, left:left+ew] = torch.rand(
            3, eh, ew, device=image.device
        )
        return image, target


def get_train_transforms(size=(512, 512)):
    """
    Full augmentation pipeline for training.
    Stronger augmentation for better generalization on
    imbalanced agricultural dataset.
    """
    return Compose([
        RandomScale(min_scale=0.8, max_scale=1.2),
        RandomCrop(prob=0.5, min_overlap=0.5),
        Resize(size=size),
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.3),
        RandomRotation(prob=0.3),
        ColorJitter(brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.15),
        RandomGrayscale(prob=0.1),
        ToTensor(),
        RandomErasing(prob=0.2, scale=(0.02, 0.1)),
        Normalize(),
    ])


def get_val_transforms(size=(512, 512)):
    """
    Minimal pipeline for validation and testing.
    No augmentation — only resize and normalize.
    """
    return Compose([
        Resize(size=size),
        ToTensor(),
        Normalize(),
    ])
