# train_all.py
# ─────────────────────────────────────────────────────────────
# Train any of the 5 models (4 baselines + 1 hybrid) across 3 seeds.
# Auto-detects device: CUDA on Windows/Linux GPU, MPS on Apple Silicon, else CPU.
#
# Usage:
#   python train_all.py --model ssdlite          --seed 42
#   python train_all.py --model yolov8n          --seed 42
#   python train_all.py --model yolov8s          --seed 42
#   python train_all.py --model retinanet        --seed 42
#   python train_all.py --model retinanet_hybrid --seed 42
#   python train_all.py --model yolov8n                       # all 3 seeds
#   python train_all.py --model all                           # 3 baselines × 3 seeds
# ─────────────────────────────────────────────────────────────

# ── MPS CONFIG — set BEFORE torch is imported ────────────────
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]  = "0.1"

# ── SSL bypass for university MITM proxy ─────────────────────
# University networks sometimes intercept HTTPS with self-signed
# certs, breaking torch.hub downloads (CERTIFICATE_VERIFY_FAILED).
# We bypass cert verification globally — torchvision still verifies
# the SHA256 hash of the downloaded file, so MITM tampering would
# still be caught at the integrity check.
import ssl as _ssl
import urllib.request as _urlreq
try:
    _ssl._create_default_https_context = _ssl._create_unverified_context
    _ctx = _ssl._create_unverified_context()
    _urlreq.install_opener(
        _urlreq.build_opener(_urlreq.HTTPSHandler(context=_ctx))
    )
except Exception:
    pass

# ── TAL PATCH — moves TAL assigner to CPU, results back to MPS
# Fixes: Ultralytics boolean-indexing crash on MPS in tal.py
import torch

# ── Multiprocessing sharing strategy (macOS only) ───────────────
# macOS: default 'file_descriptor' strategy exhausts the per-process
#        256-fd limit after many epochs ("Too many open files").
#        'file_system' strategy uses one shared-memory file per tensor
#        and avoids fd exhaustion.
# Windows: 'file_system' strategy on Windows triggers shared file-
#        mapping accumulation and crashes with error code 1455
#        (ERROR_COMMITMENT_LIMIT) after ~15 epochs. The Windows
#        default IPC mechanism is fine — leave it alone.
import sys as _sys
import torch.multiprocessing as _mp
if _sys.platform == "darwin":
    try:
        _mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass  # already set by another import
# ────────────────────────────────────────────────────────────────

from ultralytics.utils.tal import TaskAlignedAssigner
_orig_tal_forward = TaskAlignedAssigner.forward

def _patched_tal_forward(self, pd_scores, pd_bboxes, anc_points,
                          gt_labels, gt_bboxes, mask_gt):
    device = pd_scores.device
    if device.type == "mps":
        pd_scores  = pd_scores.cpu()
        pd_bboxes  = pd_bboxes.cpu()
        anc_points = anc_points.cpu()
        gt_labels  = gt_labels.cpu()
        gt_bboxes  = gt_bboxes.cpu()
        mask_gt    = mask_gt.cpu()
    res = _orig_tal_forward(self, pd_scores, pd_bboxes, anc_points,
                             gt_labels, gt_bboxes, mask_gt)
    if device.type == "mps":
        res = tuple(x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in res)
    return res

TaskAlignedAssigner.forward = _patched_tal_forward
# ─────────────────────────────────────────────────────────────

import argparse
import yaml
import gc

from src.utils.general import set_seed, get_device, ensure_dir


# ── Helpers ──────────────────────────────────────────────────
def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_yolo_device():
    """
    Return the device string Ultralytics YOLO accepts.
    CUDA   → 0          (NVIDIA GPU index)
    MPS    → 'mps'      (Apple Silicon)
    else   → 'cpu'
    Important: do NOT hardcode 'mps' — on Windows/CUDA machines that
    string is invalid and Ultralytics silently falls back to CPU,
    which is what caused YOLOv8n to crawl at 1.1 it/s instead of 8-15.
    """
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_hp(config, model_key, key):
    """Return per-model override if present, else fall back to training defaults."""
    model_cfg = config["models"].get(model_key, {})
    if key in model_cfg:
        return model_cfg[key]
    return config["training"][key]


def apply_smoke_test(config):
    """
    Force a 2-epoch sanity-check run.
    Purpose: verify the full training pipeline end-to-end (data loading,
    forward pass, TAL patch on MPS, loss, checkpointing, mAP eval, plots)
    in ~10 min per model instead of waiting hours to discover a crash.
    Output paths are redirected under outputs/smoke_test/... so real
    runs never collide with or get clobbered by a smoke test.
    """
    # Override shared defaults
    config["training"]["epochs"]        = 2
    config["training"]["patience"]      = 2
    config["training"]["warmup_epochs"] = 1

    # Override any per-model values that shadow the shared defaults,
    # and redirect every output directory under outputs/smoke_test/
    for model_cfg in config["models"].values():
        for k in ("epochs", "patience", "warmup_epochs"):
            if k in model_cfg:
                model_cfg[k] = {"epochs": 2, "patience": 2, "warmup_epochs": 1}[k]
        for k in ("checkpoints_dir", "logs_dir", "results_dir"):
            if k in model_cfg and model_cfg[k].startswith("outputs/"):
                model_cfg[k] = model_cfg[k].replace(
                    "outputs/", "outputs/smoke_test/", 1
                )

    print("\n" + "!"*60)
    print("  SMOKE TEST MODE — epochs=2, patience=2, warmup=1")
    print("  Outputs redirected to outputs/smoke_test/...")
    print("  Use this to verify the pipeline runs end-to-end.")
    print("!"*60 + "\n")
    return config


def print_header(model_name, seed):
    print(f"\n{'#'*60}")
    print(f"  {model_name}  |  seed {seed}")
    print(f"{'#'*60}")


# ── 1. SSDLite — MobileNetV3-Large ───────────────────────────
def train_ssdlite(config, seed):
    from src.models.ssdlite import build_ssdlite_vanilla
    from src.dataset.yolo_dataset import build_yolo_dataloader
    from src.dataset.transforms import get_train_transforms, get_val_transforms
    from src.training.trainer import Trainer

    set_seed(seed)
    device   = get_device()          # → cuda / mps / cpu
    tr       = config["training"]
    data     = config["data"]
    cfg      = config["models"]["ssdlite"]
    img_size = tuple(tr["image_size"])

    lr           = get_hp(config, "ssdlite", "lr")
    batch_size   = get_hp(config, "ssdlite", "batch_size")
    epochs       = get_hp(config, "ssdlite", "epochs")
    patience     = get_hp(config, "ssdlite", "patience")
    warmup_ep    = get_hp(config, "ssdlite", "warmup_epochs")
    weight_decay = get_hp(config, "ssdlite", "weight_decay")

    print_header("SSDLite MobileNetV3-Large", seed)
    print(f"  Data   : {data['train_images']}")
    print(f"  Device : {device}")
    print(f"  HP     : lr={lr}  batch={batch_size}  epochs={epochs}  patience={patience}")

    train_loader = build_yolo_dataloader(
        images_dir=data["train_images"],
        labels_dir=data["train_labels"],
        transforms=get_train_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # 2 is safer on macOS vs fd exhaustion
    )
    val_loader = build_yolo_dataloader(
        images_dir=data["val_images"],
        labels_dir=data["val_labels"],
        transforms=get_val_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = build_ssdlite_vanilla(tr["num_classes"], pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_ep, eta_min=1e-7)

    trainer = Trainer(
        model=model, optimizer=optimizer,
        scheduler=scheduler, device=device,
        config={
            "model_name":              cfg["name"],
            "epochs":                  epochs,
            "learning_rate":           lr,
            "early_stopping_patience": patience,
            "warmup_epochs":           warmup_ep,
            "num_classes":             tr["num_classes"],
            "output": {
                "checkpoints_dir": f"{cfg['checkpoints_dir']}/seed_{seed}",
                "logs_dir":        f"{cfg['logs_dir']}/seed_{seed}",
            },
        }
    )
    trainer.fit(train_loader, val_loader)


# ── 2. YOLOv8n — anchor-free nano ────────────────────────────
def train_yolov8n(config, seed):
    from ultralytics import YOLO

    set_seed(seed)
    tr  = config["training"]
    cfg = config["models"]["yolov8n"]

    lr           = get_hp(config, "yolov8n", "lr")
    batch_size   = get_hp(config, "yolov8n", "batch_size")
    epochs       = get_hp(config, "yolov8n", "epochs")
    patience     = get_hp(config, "yolov8n", "patience")
    warmup_ep    = get_hp(config, "yolov8n", "warmup_epochs")
    weight_decay = get_hp(config, "yolov8n", "weight_decay")

    yolo_device = get_yolo_device()

    print_header("YOLOv8n (anchor-free, nano)", seed)
    print(f"  Data   : {config['data']['yaml']}")
    print(f"  Device : {yolo_device}  | AMP: True")
    print(f"  HP     : lr={lr}  batch={batch_size}  epochs={epochs}  patience={patience}")

    project_path = os.path.abspath(cfg["checkpoints_dir"])
    ensure_dir(project_path)

    model = YOLO("yolov8n.pt")
    model.train(
        data=config["data"]["yaml"],
        epochs=epochs,
        imgsz=tr["image_size"][0],
        batch=batch_size,
        optimizer="AdamW",
        lr0=lr,
        lrf=get_hp(config, "yolov8n", "lrf"),
        weight_decay=weight_decay,
        warmup_epochs=warmup_ep,
        patience=patience,
        cos_lr=True,
        cls=2.0,
        seed=seed,
        device=yolo_device,
        amp=True,
        cache="disk",   # disable RAM caching — not enough RAM on VM
        workers=2,     # reduced from 4 to save RAM
        project=project_path,
        name=f"seed_{seed}",
        exist_ok=True,
        verbose=True,
    )


# ── 4. RetinaNet — ResNet50-FPN v2, Focal Loss ───────────────
def train_retinanet(config, seed, key="retinanet", box_loss="giou"):
    """
    key      which block of configs/experiment.yaml -> models: to read, so
             a variant writes its checkpoints, logs and results somewhere
             of its own and never overwrites a published run.
    box_loss "giou" reproduces the published S1. "l1" matches S2-S5.

    Everything else — data, transforms, optimizer, scheduler, warm-up,
    label offset, seeds — is shared, so the two runs differ in exactly
    one thing.
    """
    from src.models.retinanet import build_retinanet
    from src.dataset.yolo_dataset import build_yolo_dataloader
    from src.dataset.transforms import get_train_transforms, get_val_transforms
    from src.training.trainer import Trainer

    set_seed(seed)
    device   = get_device()          # → cuda / mps / cpu
    tr       = config["training"]
    data     = config["data"]
    cfg      = config["models"][key]
    img_size = tuple(tr["image_size"])

    lr           = get_hp(config, key, "lr")
    batch_size   = get_hp(config, key, "batch_size")
    epochs       = get_hp(config, key, "epochs")
    patience     = get_hp(config, key, "patience")
    warmup_ep    = get_hp(config, key, "warmup_epochs")
    weight_decay = get_hp(config, key, "weight_decay")

    print_header(f"RetinaNet ResNet50-FPN v2 (Focal Loss, box={box_loss})", seed)
    print(f"  Data   : {data['train_images']}")
    print(f"  Device : {device}")
    print(f"  HP     : lr={lr}  batch={batch_size}  epochs={epochs}  patience={patience}")

    train_loader = build_yolo_dataloader(
        images_dir=data["train_images"],
        labels_dir=data["train_labels"],
        transforms=get_train_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # 2 is safer on macOS vs fd exhaustion
    )
    val_loader = build_yolo_dataloader(
        images_dir=data["val_images"],
        labels_dir=data["val_labels"],
        transforms=get_val_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = build_retinanet(tr["num_classes"], pretrained=True,
                            box_loss=box_loss)

    # Fail before the first epoch rather than after several hours.
    assert model.head.regression_head._loss_type == box_loss, (
        f"expected box loss {box_loss!r}, got "
        f"{model.head.regression_head._loss_type!r}")
    print(f"  VERIFIED: head.regression_head._loss_type = "
          f"{model.head.regression_head._loss_type!r}")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_ep, eta_min=1e-7)

    trainer = Trainer(
        model=model, optimizer=optimizer,
        scheduler=scheduler, device=device,
        config={
            "model_name":              cfg["name"],
            "epochs":                  epochs,
            "learning_rate":           lr,
            "early_stopping_patience": patience,
            "warmup_epochs":           warmup_ep,
            "num_classes":             tr["num_classes"],
            # RetinaNet's sigmoid focal loss has no background class:
            # model expects labels 0..N-1. Our data is 1..N (YOLO_LABEL_MAP).
            # Shift labels down by 1 going INTO the model; evaluate_map
            # shifts predictions back up to match GT for mAP.
            "label_offset":            -1,
            "output": {
                "checkpoints_dir": f"{cfg['checkpoints_dir']}/seed_{seed}",
                "logs_dir":        f"{cfg['logs_dir']}/seed_{seed}",
            },
        }
    )
    trainer.fit(train_loader, val_loader)


# ── 4b. RetinaNet S1 with L1 box loss (confound-free baseline) ──
def train_retinanet_l1(config, seed):
    """
    S1 rebuilt with the same L1 regression loss that S2-S5 use.

    The published S1 used GIoU, because retinanet_resnet50_fpn_v2()
    overrides the head default, while every ConvNeXt stage calls
    RetinaNet(...) directly and keeps L1. That made the S1 -> S2 step
    change the backbone and the regression loss at once. This run
    removes the second change.

    Writes to outputs/{checkpoints,logs,results}/retinanet_l1/ so the
    published retinanet/ results are untouched.
    """
    return train_retinanet(config, seed, key="retinanet_l1", box_loss="l1")


# ── 5. RetinaNet-Hybrid — BiFPN + CBAM on RetinaNet ──────────
def train_retinanet_hybrid(config, seed):
    """
    Train the hybrid: RetinaNet-ResNet50 with BiFPN neck + CBAM attention.
    Uses the SAME data pipeline, optimizer, scheduler, label_offset, and
    hyperparameters as the retinanet baseline — only the architecture
    differs. This guarantees a fair comparison.
    """
    from src.models.hybrid import build_retinanet_hybrid
    from src.dataset.yolo_dataset import build_yolo_dataloader
    from src.dataset.transforms import get_train_transforms, get_val_transforms
    from src.training.trainer import Trainer

    set_seed(seed)
    device   = get_device()          # → cuda / mps / cpu
    tr       = config["training"]
    data     = config["data"]
    cfg      = config["models"]["retinanet_hybrid"]
    img_size = tuple(tr["image_size"])

    lr           = get_hp(config, "retinanet_hybrid", "lr")
    batch_size   = get_hp(config, "retinanet_hybrid", "batch_size")
    epochs       = get_hp(config, "retinanet_hybrid", "epochs")
    patience     = get_hp(config, "retinanet_hybrid", "patience")
    warmup_ep    = get_hp(config, "retinanet_hybrid", "warmup_epochs")
    weight_decay = get_hp(config, "retinanet_hybrid", "weight_decay")

    print_header("RetinaNet-Hybrid (BiFPN + CBAM)", seed)
    print(f"  Data   : {data['train_images']}")
    print(f"  Device : {device}")
    print(f"  HP     : lr={lr}  batch={batch_size}  epochs={epochs}  patience={patience}")

    # ── DataLoader worker count ──
    # On Windows, hybrid v3 (ConvNeXt + BiFPN×3 + CBAM) crashes with
    # error 1455 (ERROR_COMMITMENT_LIMIT) when num_workers > 0 because
    # per-batch shared-memory mappings outpace what Windows can sustain.
    # Forcing num_workers=0 means data loads in the main process — no
    # multiprocessing, no shared mappings, zero chance of crash.
    # Costs ~10-15% wall-clock per epoch on Windows. On non-Windows
    # platforms we keep num_workers=2 for speed.
    import sys as _sys
    _hybrid_nw = 0 if _sys.platform.startswith("win") else 2

    train_loader = build_yolo_dataloader(
        images_dir=data["train_images"],
        labels_dir=data["train_labels"],
        transforms=get_train_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=_hybrid_nw,
    )
    val_loader = build_yolo_dataloader(
        images_dir=data["val_images"],
        labels_dir=data["val_labels"],
        transforms=get_val_transforms(size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=_hybrid_nw,
    )

    model = build_retinanet_hybrid(tr["num_classes"], pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_ep, eta_min=1e-7)

    trainer = Trainer(
        model=model, optimizer=optimizer,
        scheduler=scheduler, device=device,
        config={
            "model_name":              cfg["name"],
            "epochs":                  epochs,
            "learning_rate":           lr,
            "early_stopping_patience": patience,
            "warmup_epochs":           warmup_ep,
            "num_classes":             tr["num_classes"],
            # Same label-offset rule as the RetinaNet baseline:
            # sigmoid focal loss → no background class → labels 0..N-1.
            "label_offset":            -1,
            "output": {
                "checkpoints_dir": f"{cfg['checkpoints_dir']}/seed_{seed}",
                "logs_dir":        f"{cfg['logs_dir']}/seed_{seed}",
            },
        }
    )
    trainer.fit(train_loader, val_loader)


# ── Entry point ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train base models for tomato leaf disease detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all.py --model ssdlite          --seed 42
  python train_all.py --model yolov8n          --seed 42
  python train_all.py --model yolov8s          --seed 42
  python train_all.py --model retinanet        --seed 42
  python train_all.py --model retinanet_hybrid --seed 42
  python train_all.py --model yolov8n                       # all 3 seeds
  python train_all.py --model all                           # 3 baselines × 3 seeds
  # Sanity-check the whole pipeline in ~10 min/model (epochs=2, separate outputs)
  python train_all.py --model all --seed 42 --smoke-test
        """
    )
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--model",  required=True,
                        choices=["ssdlite", "yolov8n", "retinanet",
                                 "retinanet_l1", "retinanet_hybrid", "all"],
                        help="Which model to train")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Single seed override (default: all 3 seeds from config)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a 2-epoch sanity check with separate outputs "
                             "(outputs/smoke_test/...). Use before burning a full "
                             "multi-hour training run.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_test:
        config = apply_smoke_test(config)

    seeds  = [args.seed] if args.seed is not None else config["experiment"]["seeds"]

    train_fns = {
        "ssdlite":          train_ssdlite,
        "yolov8n":          train_yolov8n,
        "retinanet":        train_retinanet,
        "retinanet_l1":     train_retinanet_l1,
        "retinanet_hybrid": train_retinanet_hybrid,
    }

    # --model all runs only the 3 base models. retinanet_l1 is deliberately
    # excluded: it is a variant of retinanet, not a fifth baseline, and must
    # be requested explicitly.
    base_models = ["ssdlite", "yolov8n", "retinanet"]
    models_to_run = base_models if args.model == "all" else [args.model]

    print(f"\n{'='*60}")
    print(f"  Tomato Leaf Disease — Base Model Training")
    print(f"  Models : {models_to_run}")
    print(f"  Seeds  : {seeds}")
    print(f"  Config : {args.config}")
    print(f"  CUDA   : {torch.cuda.is_available()}  | "
          f"MPS    : {torch.backends.mps.is_available()}")
    print(f"{'='*60}")

    for model_name in models_to_run:
        for seed in seeds:
            train_fns[model_name](config, seed)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"\n{'='*60}")
    print(f"  All training runs complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()