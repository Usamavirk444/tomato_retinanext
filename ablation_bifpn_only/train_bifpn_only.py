"""
Ablation: RetinaNet — ConvNeXt-Tiny + BiFPN×3  (NO CBAM)
==========================================================
Isolates the contribution of BiFPN alone in the ablation table:

    R50 + FPN           → 0.8969  (done)
    ConvNeXt + FPN      → 0.9497  (done)
    ConvNeXt + BiFPN    → ???     ← THIS SCRIPT
    ConvNeXt + BiFPN+CBAM → 0.9511 (done — Hybrid v3)

Outputs go to NEW folders — nothing in retinanet_hybrid/ is touched:
    outputs/checkpoints/retinanet_convnext_bifpn/seed_<n>/
    outputs/logs/retinanet_convnext_bifpn/seed_<n>/

Usage (run from project root):
    python ablation_bifpn_only/train_bifpn_only.py --seed 42
    python ablation_bifpn_only/train_bifpn_only.py --seed 123
    python ablation_bifpn_only/train_bifpn_only.py --seed 456

    # Chain all 3:
    python ablation_bifpn_only/train_bifpn_only.py --seed 42  && \
    python ablation_bifpn_only/train_bifpn_only.py --seed 123 && \
    python ablation_bifpn_only/train_bifpn_only.py --seed 456
"""

import os
import sys
import argparse
import ssl
import urllib.request

# SSL bypass (university MITM proxy)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    _ctx = ssl._create_unverified_context()
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
    )
except Exception:
    pass

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.utils.general import set_seed, get_device
from src.dataset.yolo_dataset import build_yolo_dataloader
from src.dataset.transforms import get_train_transforms, get_val_transforms
from src.training.trainer import Trainer

from retinanet_convnext_bifpn import build_retinanet_convnext_bifpn


MODEL_NAME = "retinanet_convnext_bifpn"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_header(seed):
    print(f"\n{'#' * 60}")
    print(f"  RetinaNet — ConvNeXt-T + BiFPN×3  (NO CBAM)")
    print(f"  Ablation variant  |  seed {seed}")
    print(f"{'#' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 456])
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) \
        if not os.path.isabs(args.config) else args.config
    config = load_config(config_path)

    set_seed(args.seed)
    device = get_device()

    tr   = config["training"]
    data = config["data"]
    img_size = tuple(tr["image_size"])

    # ── Hyperparameters — identical to retinanet_hybrid / convnext_baseline ──
    lr             = 1e-4
    batch_size     = 8
    epochs         = 70
    patience       = 20
    warmup_epochs  = 5
    weight_decay   = 1e-4

    print_header(args.seed)
    print(f"  Data   : {data['train_images']}")
    print(f"  Device : {device}")
    print(f"  HP     : lr={lr}  batch={batch_size}  epochs={epochs}  patience={patience}")

    is_windows = sys.platform.startswith("win")
    nw = 0 if is_windows else 2
    if is_windows:
        print("  Windows detected — using num_workers=0")

    train_loader = build_yolo_dataloader(
        images_dir=os.path.join(PROJECT_ROOT, data["train_images"]),
        labels_dir=os.path.join(PROJECT_ROOT, data["train_labels"]),
        transforms=get_train_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
    )
    val_loader = build_yolo_dataloader(
        images_dir=os.path.join(PROJECT_ROOT, data["val_images"]),
        labels_dir=os.path.join(PROJECT_ROOT, data["val_labels"]),
        transforms=get_val_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
    )

    model = build_retinanet_convnext_bifpn(tr["num_classes"], pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)

    # ── Output dirs — isolated, nothing else is touched ──────────────────────
    checkpoints_dir = os.path.join(
        PROJECT_ROOT, "outputs", "checkpoints", MODEL_NAME, f"seed_{args.seed}")
    logs_dir = os.path.join(
        PROJECT_ROOT, "outputs", "logs", MODEL_NAME, f"seed_{args.seed}")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={
            "model_name":               MODEL_NAME,
            "epochs":                   epochs,
            "learning_rate":            lr,
            "early_stopping_patience":  patience,
            "warmup_epochs":            warmup_epochs,
            "num_classes":              tr["num_classes"],
            "label_offset":             -1,
            "use_ema":                  True,
            "output": {
                "checkpoints_dir": checkpoints_dir,
                "logs_dir":        logs_dir,
            },
        },
    )

    trainer.fit(train_loader, val_loader)
    print(f"\n  Done. Best checkpoint → {checkpoints_dir}/{MODEL_NAME}_best.pth")
    print(f"  Training log        → {logs_dir}/")


if __name__ == "__main__":
    main()
