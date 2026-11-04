# src/utils/yolo_parser.py
# Parse YOLO format annotations (.txt files)
# Format: class_id cx cy width height (normalized 0-1)

import os
from typing import Dict, List
from PIL import Image
from src.utils.label_map import YOLO_LABEL_MAP


def parse_yolo_txt(txt_path: str, img_width: int, img_height: int) -> Dict:
    """
    Parse a YOLO .txt annotation file and convert to absolute coordinates.

    Args:
        txt_path:   Path to .txt annotation file
        img_width:  Image width in pixels
        img_height: Image height in pixels

    Returns:
        Dict with boxes (xyxy), labels, class_names
    """
    boxes       = []
    labels      = []
    class_names = []

    if not os.path.exists(txt_path):
        return {"boxes": boxes, "labels": labels, "class_names": class_names}

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            yolo_class = int(float(parts[0]))  # handle both "7" and "7.0" formats
            cx = float(parts[1])
            cy = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])

            # Convert normalized xywh to absolute xyxy
            xmin = (cx - w / 2) * img_width
            ymin = (cy - h / 2) * img_height
            xmax = (cx + w / 2) * img_width
            ymax = (cy + h / 2) * img_height

            # Clip to image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width,  xmax)
            ymax = min(img_height, ymax)

            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert YOLO class to PyTorch label
            label = YOLO_LABEL_MAP.get(yolo_class, -1)
            if label == -1:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            class_names.append(str(yolo_class))

    return {
        "boxes":       boxes,
        "labels":      labels,
        "class_names": class_names,
    }


def get_image_size(img_path: str):
    """Get image width and height."""
    with Image.open(img_path) as img:
        return img.width, img.height
