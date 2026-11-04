"""
Confusion matrix script — runs a trained model on val (or test) split,
matches predictions to ground truth, and builds a 10x10 confusion matrix
(9 classes + 1 'background').

Supports all four model types:
  - ssdlite
  - yolov8n
  - retinanet
  - retinanet_hybrid

Matching rule (YOLO-style):
  • For each predicted box, find the highest-IoU GT box.
  • If best IoU >= iou_thresh (default 0.5):
      - cell [GT_class, pred_class] += 1
        (diagonal cell if classes match -> correct; off-diagonal -> confusion)
  • If best IoU < iou_thresh:
      - cell [background, pred_class] += 1  (FP, predicted but not in any GT)
  • For each GT box with no matching prediction:
      - cell [GT_class, background] += 1   (FN, missed by model)

Usage:
    python confusion_matrix.py --model retinanet_hybrid --seed 42 --split test
    python confusion_matrix.py --model yolov8n --seed 42 --split test --conf 0.25

Output:
    outputs/results/<model>/seed_<n>/confusion_matrix_<split>.png   (heatmap)
    outputs/results/<model>/seed_<n>/confusion_matrix_<split>.csv   (raw counts)
    outputs/results/<model>/seed_<n>/confusion_matrix_<split>_norm.png  (row-normalised)
"""

import os
import sys
import argparse
import csv

import numpy as np
import torch
from torchvision.ops import box_iou

HERE = os.path.dirname(os.path.abspath(__file__))
# moved from paper_files/ into scripts/eval/ - walk up to the repo root
PROJECT_ROOT = HERE
while not os.path.exists(os.path.join(PROJECT_ROOT, "train_all.py")) \
        and os.path.dirname(PROJECT_ROOT) != PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.label_map import CLASSES, DISPLAY_NAMES

# Short labels for the matrix (paper-friendly)
CLASS_LABELS_SHORT = [
    "Early Blight", "Healthy", "Late Blight", "Leaf Miners", "Leaf Mold",
    "Mosaic Virus", "Septoria", "Spider Mites", "YLCV",
]
BG_LABEL = "background"
NUM_FG = 9
MATRIX_SIZE = NUM_FG + 1  # 10x10


def build_confusion_matrix(predictions, targets, iou_thresh=0.5, conf_thresh=0.05):
    """
    Build a 10x10 confusion matrix.

    Args:
        predictions: list of dicts, one per image, each with:
            "boxes":  (N, 4) tensor xyxy
            "labels": (N,)   tensor in 1..9 (PyTorch label space)
            "scores": (N,)   tensor
        targets: list of dicts, one per image, each with:
            "boxes":  (M, 4) tensor xyxy
            "labels": (M,)   tensor in 1..9
        iou_thresh: IoU threshold for matching (default 0.5)
        conf_thresh: drop predictions with score < this (default 0.05)

    Returns:
        cm: np.ndarray of shape (10, 10), int
            rows = ground-truth class (1..9 then background as index 0 in
                   1-indexed terms; we use index 9 in the array)
            cols = predicted class (same layout)
        Layout: cm[i, j] = number of times GT class (i+1) was predicted
                as class (j+1). Index 9 (last) is 'background'.
    """
    cm = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.int64)
    # Index mapping: foreground class label 1..9 -> matrix index 0..8
    # background -> matrix index 9
    BG_IDX = NUM_FG  # = 9

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        # Apply confidence threshold
        keep = pred_scores >= conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]

        # Edge cases
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            # All predictions are FPs against background
            for p_lbl in pred_labels:
                cm[BG_IDX, int(p_lbl) - 1] += 1
            continue
        if len(pred_boxes) == 0:
            # All GT boxes are missed (FN -> background)
            for g_lbl in gt_labels:
                cm[int(g_lbl) - 1, BG_IDX] += 1
            continue

        # Compute IoU between every pred and every GT
        ious = box_iou(pred_boxes, gt_boxes).numpy()  # (P, G)

        # Track which GT boxes have been claimed (one prediction each)
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # Sort predictions by descending IoU best-match strength so
        # high-quality matches get to claim their GT first
        pred_best_iou = ious.max(axis=1) if ious.size > 0 else np.zeros(len(pred_boxes))
        pred_order = np.argsort(-pred_best_iou)

        for pi in pred_order:
            best_g = ious[pi].argmax()
            best_iou = ious[pi, best_g]
            p_lbl = int(pred_labels[pi]) - 1  # 0..8

            if best_iou >= iou_thresh and not gt_matched[best_g]:
                # Matched a GT box
                g_lbl = int(gt_labels[best_g]) - 1  # 0..8
                cm[g_lbl, p_lbl] += 1
                gt_matched[best_g] = True
            else:
                # Either IoU too low or GT already claimed -> FP / background
                cm[BG_IDX, p_lbl] += 1

        # Any unmatched GT boxes are FNs (missed)
        for gi, matched in enumerate(gt_matched):
            if not matched:
                g_lbl = int(gt_labels[gi]) - 1
                cm[g_lbl, BG_IDX] += 1

    return cm


# ── Inference + prediction collection ─────────────────────────
@torch.no_grad()
def collect_predictions_torchvision(model, dataloader, device, label_offset=0):
    """For SSDLite, RetinaNet, Hybrid."""
    predictions = []
    targets_all = []
    model.eval()
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for o, t in zip(outputs, targets):
            predictions.append({
                "boxes":  o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": (o["labels"].cpu().long() - label_offset),
            })
            targets_all.append({
                "boxes":  t["boxes"].cpu(),
                "labels": t["labels"].cpu().long(),
            })
    return predictions, targets_all


@torch.no_grad()
def collect_predictions_yolov8(weights_path, images_dir, labels_dir, device):
    """For YOLOv8n via Ultralytics."""
    from ultralytics import YOLO
    from src.dataset.yolo_dataset import YoloTomatoDataset
    from src.dataset.transforms import get_val_transforms
    from PIL import Image as _Image

    model = YOLO(weights_path)
    val_ds = YoloTomatoDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transforms=get_val_transforms(size=(640, 640)),
    )

    predictions = []
    targets_all = []
    for idx in range(len(val_ds)):
        _, target = val_ds[idx]
        img_path = os.path.join(images_dir, val_ds.image_files[idx])

        results = model.predict(img_path, device=device,
                                conf=0.05, iou=0.5, verbose=False)
        r = results[0]
        pred_boxes = r.boxes.xyxy.cpu()
        pred_scores = r.boxes.conf.cpu()
        # Ultralytics labels are 0..8; shift to 1..9 to match GT
        pred_labels = (r.boxes.cls.cpu().long() + 1)

        # Rescale predictions to 640x640 to match GT boxes
        with _Image.open(img_path) as im:
            orig_w, orig_h = im.size
        sx = 640.0 / orig_w
        sy = 640.0 / orig_h
        if len(pred_boxes) > 0:
            pred_boxes = pred_boxes.clone()
            pred_boxes[:, [0, 2]] *= sx
            pred_boxes[:, [1, 3]] *= sy

        predictions.append({
            "boxes":  pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        })
        targets_all.append({
            "boxes":  target["boxes"],
            "labels": target["labels"].long(),
        })

    return predictions, targets_all


# ── Model loaders (same as evaluate_per_class.py) ─────────────
def load_model(model_name, seed, device):
    if model_name == "yolov8n":
        weights_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "yolov8n",
            f"seed_{seed}", "weights", "best.pt"
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLOv8n weights not found at {weights_path}")
        return weights_path, 0, "yolo"

    elif model_name == "ssdlite":
        from src.models.ssdlite import build_ssdlite_vanilla
        ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "ssdlite",
            f"seed_{seed}", "ssdlite_mobilenet_v3_large_best.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SSDLite checkpoint not found at {ckpt_path}")
        model = build_ssdlite_vanilla(num_classes=10, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, 0, "torchvision"

    elif model_name == "retinanet":
        from src.models.retinanet import build_retinanet
        ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "retinanet",
            f"seed_{seed}", "retinanet_resnet50_fpn_v2_best.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"RetinaNet checkpoint not found at {ckpt_path}")
        model = build_retinanet(num_classes=10, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, -1, "torchvision"

    elif model_name == "retinanet_hybrid":
        from src.models.hybrid import build_retinanet_hybrid
        ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "retinanet_hybrid",
            f"seed_{seed}", "retinanet_hybrid_resnet50_bifpn_cbam_best.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Hybrid checkpoint not found at {ckpt_path}")
        model = build_retinanet_hybrid(num_classes=10, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, -1, "torchvision"

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Plotting ───────────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path, title, normalize=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CLASS_LABELS_SHORT + [BG_LABEL]

    if normalize:
        # Row-normalise: each row sums to 1 (percentage of each true class)
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot = np.divide(cm, row_sums,
                                out=np.zeros_like(cm, dtype=float),
                                where=row_sums > 0)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm.astype(float)
        fmt = "d"
        vmax = cm.max() if cm.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Annotate cells
    threshold = (vmax + (cm_plot.min() if cm_plot.size else 0)) / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            v = cm_plot[i, j]
            if normalize:
                text = f"{v:.2f}" if v >= 0.005 else ""
            else:
                text = f"{int(v)}" if v > 0 else ""
            color = "white" if v > threshold else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["ssdlite", "yolov8n", "retinanet", "retinanet_hybrid"])
    parser.add_argument("--seed", required=True, type=int,
                        choices=[42, 123, 456])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for matching predictions to GT")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Confidence threshold for keeping predictions")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model} | seed {args.seed} | split {args.split}")
    print(f"Thresholds: IoU={args.iou_thresh}, conf={args.conf_thresh}")

    split_images = os.path.join(PROJECT_ROOT, "data", "clean_yolo_dataset",
                                args.split, "images")
    split_labels = os.path.join(PROJECT_ROOT, "data", "clean_yolo_dataset",
                                args.split, "labels")
    if not os.path.isdir(split_images):
        raise FileNotFoundError(f"Split images dir not found: {split_images}")

    print("\nLoading model ...")
    model_or_path, label_offset, kind = load_model(args.model, args.seed, device)

    print(f"Running inference on {args.split} split ...")
    if kind == "yolo":
        preds, tgts = collect_predictions_yolov8(
            model_or_path, split_images, split_labels, device
        )
    else:
        from src.dataset.yolo_dataset import build_yolo_dataloader
        from src.dataset.transforms import get_val_transforms
        loader = build_yolo_dataloader(
            images_dir=split_images,
            labels_dir=split_labels,
            transforms=get_val_transforms(size=(640, 640)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        preds, tgts = collect_predictions_torchvision(
            model_or_path, loader, device, label_offset=label_offset
        )

    print(f"Collected predictions on {len(preds)} images.")

    print("\nBuilding confusion matrix ...")
    cm = build_confusion_matrix(preds, tgts,
                                iou_thresh=args.iou_thresh,
                                conf_thresh=args.conf_thresh)

    # ── Print to stdout ──
    print(f"\nConfusion matrix ({args.model}, seed {args.seed}, {args.split}):")
    labels = CLASS_LABELS_SHORT + [BG_LABEL]
    header_label = "GT \\ Pred"
    header = f"{header_label:<16}" + "".join(f"{l[:8]:>10}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = f"{row_label[:14]:<16}"
        for j in range(len(labels)):
            row_str += f"{cm[i, j]:>10d}"
        print(row_str)

    # ── Save outputs ──
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "results", args.model,
                           f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # CSV (raw counts)
    csv_path = os.path.join(out_dir, f"confusion_matrix_{args.split}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["GT \\ Pred"] + labels)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + cm[i].tolist())
    print(f"\nSaved CSV : {csv_path}")

    # PNG (raw counts heatmap)
    png_path = os.path.join(out_dir, f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(
        cm, png_path,
        title=f"{args.model} (seed {args.seed}, {args.split}) — raw counts",
        normalize=False
    )
    print(f"Saved PNG : {png_path}")

    # PNG (row-normalised heatmap — easier to read for the paper)
    png_norm_path = os.path.join(out_dir, f"confusion_matrix_{args.split}_norm.png")
    plot_confusion_matrix(
        cm, png_norm_path,
        title=f"{args.model} (seed {args.seed}, {args.split}) — row-normalised",
        normalize=True
    )
    print(f"Saved PNG : {png_norm_path}  (normalised)")

    # ── Quick diagnostic summary ──
    diag = np.diag(cm[:NUM_FG, :NUM_FG]).sum()
    total_classified = cm[:NUM_FG, :NUM_FG].sum()
    accuracy = diag / max(total_classified, 1)
    print(f"\nDiagonal sum: {diag}  /  All foreground matches: {total_classified}  "
          f"(class accuracy = {accuracy:.3f})")
    print(f"Total FPs (background row): {cm[NUM_FG, :NUM_FG].sum()}")
    print(f"Total FNs (background col): {cm[:NUM_FG, NUM_FG].sum()}")


if __name__ == "__main__":
    main()