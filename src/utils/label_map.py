# src/utils/label_map.py
# clean_yolo_dataset — 9 foreground classes (YOLO 0-indexed)
#
# For torchvision models (SSDLite, RetinaNet):
#   background = 0  (reserved by PyTorch)
#   YOLO class i  →  PyTorch label  i + 1  (1-indexed, 1–9)
#   NUM_CLASSES   = 10  (9 foreground + 1 background)
#
# Class weights derived from training set annotation counts:
#   Dominant class: Healthy (8,582 instances)
#   Weight = dominant_count / class_count  →  upsamples rare classes

CLASSES = [
    "Early_Blight",           # YOLO 0  → PyTorch 1
    "Healthy",                # YOLO 1  → PyTorch 2
    "Late_Blight",            # YOLO 2  → PyTorch 3
    "Leaf_Miners",            # YOLO 3  → PyTorch 4
    "Leaf_Mold",              # YOLO 4  → PyTorch 5
    "Mosaic_Virus",           # YOLO 5  → PyTorch 6
    "Septoria",               # YOLO 6  → PyTorch 7
    "Spider_Mites",           # YOLO 7  → PyTorch 8
    "Yellow_Leaf_Curl_Virus", # YOLO 8  → PyTorch 9
]

# Background is class 0 (PyTorch convention for detection models)
NUM_CLASSES = len(CLASSES) + 1   # 10  (9 foreground + 1 background)

# YOLO class index (0-8) → PyTorch label (1-9)
YOLO_LABEL_MAP = {i: i + 1 for i in range(len(CLASSES))}

# Class name → PyTorch label
LABEL_MAP = {name: idx + 1 for idx, name in enumerate(CLASSES)}

# PyTorch label → class name
REVERSE_LABEL_MAP = {idx + 1: name for idx, name in enumerate(CLASSES)}

# Display-friendly names (spaces instead of underscores)
DISPLAY_NAMES = {
    1: "Early Blight",
    2: "Healthy",
    3: "Late Blight",
    4: "Leaf Miners",
    5: "Leaf Mold",
    6: "Mosaic Virus",
    7: "Septoria",
    8: "Spider Mites",
    9: "Yellow Leaf Curl Virus",
}

# ── Class weights (WeightedRandomSampler for SSDLite / hybrid_ssd) ────────────
# Train annotation counts (clean_yolo_dataset):
#   Early Blight          4,061   Septoria              5,056
#   Healthy               8,582   Spider Mites          4,069
#   Late Blight           4,690   Yellow Leaf Curl      7,307
#   Leaf Miners           3,866
#   Leaf Mold             5,961   ← Healthy is dominant (8,582)
#   Mosaic Virus          3,651
#
# Weight = dominant_count / class_count
# PyTorch label 1-9 (YOLO 0-8 shifted by 1)
CLASS_WEIGHTS = {
    1:  8582 / 4061,   # Early Blight            →  2.11
    2:  8582 / 8582,   # Healthy                 →  1.00  (dominant)
    3:  8582 / 4690,   # Late Blight             →  1.83
    4:  8582 / 3866,   # Leaf Miners             →  2.22
    5:  8582 / 5961,   # Leaf Mold               →  1.44
    6:  8582 / 3651,   # Mosaic Virus            →  2.35
    7:  8582 / 5056,   # Septoria                →  1.70
    8:  8582 / 4069,   # Spider Mites            →  2.11
    9:  8582 / 7307,   # Yellow Leaf Curl Virus  →  1.17
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_label_from_yolo(yolo_class: int) -> int:
    """Convert YOLO class index (0-8) to PyTorch label (1-9)."""
    return YOLO_LABEL_MAP.get(yolo_class, -1)


def get_display_name(label: int) -> str:
    return DISPLAY_NAMES.get(label, "Unknown")


def get_class_name(label: int) -> str:
    return REVERSE_LABEL_MAP.get(label, "Unknown")
