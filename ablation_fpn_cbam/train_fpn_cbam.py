"""
Ablation: RetinaNet — ConvNeXt-Tiny + standard FPN + CBAM  (NO BiFPN)
=======================================================================
Missing ablation cell added to address peer-reviewer Critical comment.

Ablation table (S = stage):
    S1   ResNet-50 + FPN                  → done ✓
    S2   ConvNeXt-T + FPN                 → done ✓
    S2.5 ConvNeXt-T + FPN + CBAM         ← THIS SCRIPT
    S3   ConvNeXt-T + BiFPN              → done ✓
    S4   ConvNeXt-T + BiFPN + CBAM       → done ✓  (RetinaNeXt)

Outputs go to NEW folders — NOTHING in any existing folder is touched:
    outputs/checkpoints/retinanet_convnext_fpn_cbam/seed_<n>/
    outputs/logs/retinanet_convnext_fpn_cbam/seed_<n>/

Hyperparameters are IDENTICAL to all other ablation variants for fairness.

Usage (run from project root, i.e. the tomato_new folder):
    python ablation_fpn_cbam/train_fpn_cbam.py --seed 42
    python ablation_fpn_cbam/train_fpn_cbam.py --seed 123
    python ablation_fpn_cbam/train_fpn_cbam.py --seed 456

    # Chain all 3 seeds sequentially:
    python ablation_fpn_cbam/train_fpn_cbam.py --seed 42  && \\
    python ablation_fpn_cbam/train_fpn_cbam.py --seed 123 && \\
    python ablation_fpn_cbam/train_fpn_cbam.py --seed 456
"""

import os
import sys
import argparse
import ssl
import urllib.request

# SSL bypass (university MITM proxy — only fires on first weight download)
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

# ── Path setup — makes src/ importable and this folder importable ──────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.utils.general        import set_seed, get_device
from src.dataset.yolo_dataset import build_yolo_dataloader
from src.dataset.transforms   import get_train_transforms, get_val_transforms
from src.training.trainer     import Trainer

from retinanet_convnext_fpn_cbam import build_retinanet_convnext_fpn_cbam


MODEL_NAME = "retinanet_convnext_fpn_cbam"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_header(seed: int) -> None:
    print(f"\n{'#' * 60}")
    print(f"  RetinaNet — ConvNeXt-T + FPN + CBAM  (NO BiFPN)")
    print(f"  Ablation S2.5  |  seed {seed}")
    print(f"{'#' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RetinaNet ConvNeXt+FPN+CBAM ablation (S2.5)")
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="Path to experiment YAML (relative to project root)")
    parser.add_argument("--seed", type=int, required=True,
                        choices=[42, 123, 456],
                        help="Random seed (must match the other ablation runs)")
    args = parser.parse_args()

    config_path = (os.path.join(PROJECT_ROOT, args.config)
                   if not os.path.isabs(args.config) else args.config)
    config = load_config(config_path)

    set_seed(args.seed)
    device = get_device()

    tr       = config["training"]
    data     = config["data"]
    img_size = tuple(tr["image_size"])

    # ── Hyperparameters — IDENTICAL to all other ablation variants ─────────────
    lr             = 1e-4
    batch_size     = 8
    epochs         = 70
    patience       = 20
    warmup_epochs  = 5
    weight_decay   = 1e-4

    print_header(args.seed)
    print(f"  Data   : {data['train_images']}")
    print(f"  Device : {device}")
    print(f"  HP     : lr={lr}  batch={batch_size}  epochs={epochs}  "
          f"patience={patience}  warmup={warmup_epochs}")

    # Windows / Linux num_workers
    is_windows = sys.platform.startswith("win")
    nw = 0 if is_windows else 2
    if is_windows:
        print("  Windows detected — using num_workers=0")

    # ── Data loaders ───────────────────────────────────────────────────────────
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

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_retinanet_convnext_fpn_cbam(
        num_classes=tr["num_classes"], pretrained=True)
    model.to(device)

    # ── Optimizer + scheduler — identical to all other ablation variants ────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)

    # ── Output dirs — NEW location, no collision with any existing outputs ──────
    checkpoints_dir = os.path.join(
        PROJECT_ROOT, "outputs", "checkpoints", MODEL_NAME, f"seed_{args.seed}")
    logs_dir = os.path.join(
        PROJECT_ROOT, "outputs", "logs", MODEL_NAME, f"seed_{args.seed}")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir,        exist_ok=True)

    # ── Trainer ────────────────────────────────────────────────────────────────
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
            "label_offset":             -1,   # RetinaNet uses 0..N-1 internally
            "use_ema":                  True,
            "output": {
                "checkpoints_dir": checkpoints_dir,
                "logs_dir":        logs_dir,
            },
        },
    )

    trainer.fit(train_loader, val_loader)
    print(f"\n  Training complete.")
    print(f"  Best checkpoint → {checkpoints_dir}/{MODEL_NAME}_best.pth")
    print(f"  Training log    → {logs_dir}/")
    print(f"\n  Next step: run evaluate_fpn_cbam.py --seed {args.seed}")


if __name__ == "__main__":
    main()
