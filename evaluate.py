# evaluate.py

import argparse
import yaml
import torch
import os
import sys
import gc
import pandas as pd
from src.dataset.yolo_dataset import build_yolo_dataloader
from src.dataset.transforms import get_val_transforms
from src.evaluation.evaluator import Evaluator
from src.utils.general import get_device, load_model_for_eval


def load_config(path):
    # Force UTF-8 — Windows defaults to GBK on some locales and crashes
    # on the unicode characters in our config (em-dashes, arrows, etc.).
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_name, num_classes):
    """
    Build the BASELINE architecture used for the comparative study.
    SSDLite here is the vanilla torchvision build — NOT the BiFPN hybrid —
    because the trainer saves a vanilla-SSDLite state_dict. Using the hybrid
    builder at eval time would produce a silent state_dict mismatch.
    """
    if model_name == "ssdlite":
        from src.models.ssdlite import build_ssdlite_vanilla
        return build_ssdlite_vanilla(num_classes, pretrained=False)
    elif model_name == "retinanet":
        from src.models.retinanet import build_retinanet
        return build_retinanet(num_classes, pretrained=False)
    elif model_name == "retinanet_l1":
        # Same architecture and same state_dict layout as `retinanet`;
        # _loss_type is not a parameter and has no effect at inference.
        # Built with box_loss="l1" anyway so the printed summary matches
        # how the checkpoint was trained.
        from src.models.retinanet import build_retinanet
        return build_retinanet(num_classes, pretrained=False, box_loss="l1")
    elif model_name == "retinanet_hybrid":
        # Hybrid uses the SAME label convention as RetinaNet baseline
        # (sigmoid focal loss, foreground 0..N-1). Build with
        # pretrained=False at eval time — checkpoint contains everything.
        from src.models.hybrid import build_retinanet_hybrid
        return build_retinanet_hybrid(num_classes, pretrained=False)
    elif model_name == "yolov8n":
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")  # handled separately
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_checkpoint_path(config, model_name, seed, checkpoint=None):
    """
    Resolve the checkpoint to evaluate.

    FIX B (Phase D.3). `--checkpoint` used to be parsed and then ignored:
    the function always rebuilt the default path, so a reviewer asking for a
    specific file silently got a different one. That is exactly the hazard
    in the S3 seed-456 case, where two checkpoints differ by 3.45 pp.

    `checkpoint` now takes precedence when given, and a missing file is a
    hard error rather than a fall-back. With `checkpoint=None` the behaviour
    is byte-for-byte what it was before.

    The trainer saves as {cfg['name']}_best.pth, where cfg['name'] is the
    full torchvision/config name (e.g. ssdlite_mobilenet_v3_large), not the
    short alias.
    """
    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"--checkpoint was given but the file does not exist:\n"
                f"  {checkpoint}\n"
                f"Refusing to fall back to the default path, which would "
                f"evaluate a different set of weights.")
        return checkpoint

    cfg = config["models"][model_name]
    ckpt_dir = f"{cfg['checkpoints_dir']}/seed_{seed}"
    # Trainer writes: {cfg['name']}_best.pth
    return os.path.join(ckpt_dir, f"{cfg['name']}_best.pth")


def evaluate_yolov8n(config, seed):
    from ultralytics import YOLO
    tr   = config["training"]
    cfg  = config["models"]["yolov8n"]
    data = config["data"]

    ckpt = f"{cfg['checkpoints_dir']}/seed_{seed}/weights/best.pt"
    if not os.path.exists(ckpt):
        print(f"  ERROR: No checkpoint at {ckpt}")
        return

    print(f"\n{'='*55}")
    print(f"  Evaluating YOLOv8n | Seed {seed}")
    print(f"  Checkpoint: {ckpt}")
    print(f"{'='*55}")

    # Pick the correct Ultralytics device string for the current host.
    # CUDA  -> int 0  (NVIDIA GPU)
    # MPS   -> "mps"  (Apple Silicon)
    # else  -> "cpu"
    # Do NOT hardcode "mps" — on Windows/CUDA boxes that string is
    # invalid and Ultralytics silently falls back to CPU.
    if torch.cuda.is_available():
        yolo_device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        yolo_device = "mps"
    else:
        yolo_device = "cpu"

    model   = YOLO(ckpt)
    results = model.val(
        data=data["yaml"],
        imgsz=tr["image_size"][0],
        batch=tr["batch_size"],
        split="test",
        device=yolo_device,
        verbose=True,
    )

    map50    = float(results.box.map50)
    map5095  = float(results.box.map)
    per_class_ap     = results.box.ap50.tolist() if hasattr(results.box, 'ap50') else []
    per_class_recall = results.box.r.tolist()    if hasattr(results.box, 'r')    else []

    class_names = [
        "Early Blight", "Healthy", "Late Blight", "Leaf Miner",
        "Leaf Mold", "Mosaic Virus", "Septoria", "Spider Mites",
        "Yellow Leaf Curl Virus"
    ]

    results_dir = f"{cfg['results_dir']}"
    os.makedirs(results_dir, exist_ok=True)

    # Long format (one row per class) — original format preserved
    rows = []
    for i, name in enumerate(class_names):
        ap     = per_class_ap[i]     if i < len(per_class_ap)     else 0.0
        recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
        rows.append({"class": name, "AP50": round(ap*100, 2), "Recall": round(recall*100, 2)})

    df = pd.DataFrame(rows)
    df["model"]   = "yolov8n"
    df["seed"]    = seed
    df["mAP50"]   = round(map50   * 100, 2)
    df["mAP5095"] = round(map5095 * 100, 2)

    out_path = os.path.join(results_dir, f"yolov8n_seed_{seed}_eval_results.csv")
    df.to_csv(out_path, index=False)

    # Wide format — matches SSDLite/RetinaNet column structure for generate_plots.py
    wide = {"model": f"yolov8n_seed_{seed}", "mAP50": round(map50*100, 2), "mAP5095": round(map5095*100, 2)}
    for i, name in enumerate(class_names):
        safe   = name.replace(" ", "_")
        ap     = per_class_ap[i]     if i < len(per_class_ap)     else 0.0
        recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
        wide[f"AP_{safe}"]     = round(ap     * 100, 2)
        wide[f"Recall_{safe}"] = round(recall * 100, 2)

    wide_path = os.path.join(results_dir, f"yolov8n_seed_{seed}_eval_wide.csv")
    pd.DataFrame([wide]).to_csv(wide_path, index=False)

    print(f"\n  mAP@50     : {map50*100:.2f}%")
    print(f"  mAP@50:95  : {map5095*100:.2f}%")
    print(f"\n  Per-Class Recall:")
    for i, name in enumerate(class_names):
        r = per_class_recall[i] if i < len(per_class_recall) else 0.0
        print(f"    {name:<30} Recall: {r*100:.2f}%")
    print(f"\n  Results saved  → {out_path}")
    print(f"  Wide results   → {wide_path}")


def evaluate_pytorch_model(config, model_name, seed, checkpoint=None):
    device     = get_device()
    tr         = config["training"]
    data       = config["data"]
    cfg        = config["models"][model_name]
    img_size   = tuple(tr["image_size"])
    num_classes = tr["num_classes"]

    print(f"\n{'='*55}")
    print(f"  Evaluating {model_name} | Seed {seed}")
    print(f"{'='*55}")

    ckpt_path = get_checkpoint_path(config, model_name, seed, checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: No checkpoint at {ckpt_path}")
        return
    if checkpoint is not None:
        print(f"  Using EXPLICIT checkpoint (--checkpoint): {ckpt_path}")

    test_loader = build_yolo_dataloader(
        images_dir=data["test_images"],
        labels_dir=data["test_labels"],
        transforms=get_val_transforms(size=img_size),
        batch_size=tr["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = build_model(model_name, num_classes)
    model, epoch, loss = load_model_for_eval(model, ckpt_path, device)
    model.to(device)

    results_dir = f"{cfg['results_dir']}"
    os.makedirs(results_dir, exist_ok=True)

    evaluator = Evaluator(
        num_classes=num_classes,
        model_name=f"{model_name}_seed_{seed}",
        results_dir=results_dir,
    )
    is_retinanet = "retinanet" in model_name
    evaluator.evaluate(model, test_loader, device, is_retinanet=is_retinanet)

    gc.collect()
    # FIX A (Phase D.3). The original line was:
    #     torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    # hasattr() is True on a Windows/Linux CUDA build because torch.mps
    # exists as a module, but the MPS backend does not, so the call raised
    #     RuntimeError: Cannot execute emptyCache() without MPS backend
    # AFTER the results CSV had been written. No metric was affected, but it
    # aborted any multi-seed loop after the first seed. Guarding on actual
    # backend availability is behaviour-preserving: this only frees a
    # memory cache.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/experiment.yaml")
    parser.add_argument("--model",      required=True,
                        choices=["ssdlite", "yolov8n", "retinanet",
                                 "retinanet_l1", "retinanet_hybrid", "all"])
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="evaluate this exact checkpoint file instead of "
                             "the default path from the config. Errors if the "
                             "file is missing; never falls back silently. "
                             "Use with a single --model and --seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds  = [args.seed] if args.seed else config["experiment"]["seeds"]

    models_to_run = (
        ["ssdlite", "yolov8n", "retinanet", "retinanet_hybrid"]
        if args.model == "all" else [args.model]
    )

    for model_name in models_to_run:
        for seed in seeds:
            print(f"\n{'#'*55}")
            print(f"  {model_name} | seed {seed}")
            print(f"{'#'*55}")
            if model_name == "yolov8n":
                if args.checkpoint:
                    sys.exit("--checkpoint is not supported for yolov8n; "
                             "its weights are resolved by Ultralytics.")
                evaluate_yolov8n(config, seed)
            else:
                evaluate_pytorch_model(config, model_name, seed,
                                       checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()