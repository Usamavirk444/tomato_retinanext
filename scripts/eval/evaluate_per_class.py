"""
Per-class evaluation script — runs a trained model on val (or test) split
and reports per-class AP@50, AP@50:95, precision, recall.

Supports all four model types:
  - ssdlite
  - yolov8n
  - retinanet
  - retinanet_hybrid

For all models, the SAME mAP computation is used (the _ap_per_class
function from the project's trainer) so results are strictly comparable
across architectures.

Usage:
    python evaluate_per_class.py --model retinanet_hybrid --seed 42 --split val
    python evaluate_per_class.py --model yolov8n --seed 42 --split test

Output:
    Prints a per-class table to stdout
    Saves CSV at outputs/results/<model>/seed_<n>/per_class_<split>.csv
"""

import os
import sys
import argparse
import csv

import numpy as np
import torch
from torchvision.ops import box_iou

# ── Path setup so we can import src.* ──────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
# moved from paper_files/ into scripts/eval/ - walk up to the repo root
PROJECT_ROOT = HERE
while not os.path.exists(os.path.join(PROJECT_ROOT, "train_all.py")) \
        and os.path.dirname(PROJECT_ROOT) != PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.label_map import CLASSES, DISPLAY_NAMES

CLASS_NAMES = CLASSES  # 9 foreground classes (YOLO 0..8)


# ── Compute per-class AP from the same TP/FP/conf arrays used in trainer.py
def _compute_ap_101(recall, precision):
    """COCO-style 101-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at_r = precision[recall >= thr]
        ap += p_at_r.max() if len(p_at_r) else 0.0
    return ap / 101.0


def per_class_metrics(tp, conf, pred_cls, target_cls, num_classes):
    """
    Compute per-class AP@50, AP@50:95, precision, recall.

    Args:
        tp:         (n_iou_thresholds, n_preds) bool — was this prediction
                    a true positive at each IoU threshold?
        conf:       (n_preds,) float — prediction confidences
        pred_cls:   (n_preds,) int — predicted class label (1..9)
        target_cls: (n_targets,) int — GT class label (1..9)
        num_classes: int — number of foreground classes (9)

    Returns:
        dict: per-class metrics keyed by class label (1..9):
            "ap50", "ap5095", "precision", "recall", "n_gt", "n_pred"
    """
    sort_idx = np.argsort(-conf)
    tp = tp[:, sort_idx]
    conf = conf[sort_idx]
    pred_cls = pred_cls[sort_idx]

    out = {}
    for c in range(1, num_classes + 1):  # labels 1..9
        n_gt = int((target_cls == c).sum())
        mask = pred_cls == c
        n_pred = int(mask.sum())

        if n_gt == 0:
            out[c] = dict(ap50=0.0, ap5095=0.0,
                          precision=0.0, recall=0.0,
                          n_gt=0, n_pred=n_pred)
            continue

        if n_pred == 0:
            out[c] = dict(ap50=0.0, ap5095=0.0,
                          precision=0.0, recall=0.0,
                          n_gt=n_gt, n_pred=0)
            continue

        tp_c = tp[:, mask]
        fpc = (1 - tp_c).cumsum(1)
        tpc = tp_c.cumsum(1)
        recall_c = tpc / (n_gt + 1e-16)
        precision_c = tpc / (tpc + fpc + 1e-16)

        ap50 = _compute_ap_101(recall_c[0], precision_c[0])
        ap5095 = float(np.mean([
            _compute_ap_101(recall_c[i], precision_c[i])
            for i in range(tp_c.shape[0])
        ]))

        # Precision / recall at the end of the PR curve (i.e. with all preds)
        precision_final = float(precision_c[0][-1])
        recall_final = float(recall_c[0][-1])

        out[c] = dict(
            ap50=float(ap50),
            ap5095=float(ap5095),
            precision=precision_final,
            recall=recall_final,
            n_gt=n_gt,
            n_pred=n_pred,
        )

    return out


# ── Inference + collection ─────────────────────────────────────
@torch.no_grad()
def collect_predictions_torchvision(model, dataloader, device, label_offset=0):
    """For torchvision detection models (SSDLite, RetinaNet, Hybrid)."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    n_thr = len(iou_thresholds)
    model.eval()

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_tgt_cls = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].cpu()
            pred_scores = output["scores"].cpu()
            pred_labels = (output["labels"].cpu().numpy() - label_offset)

            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu().numpy()

            all_tgt_cls.extend(gt_labels.tolist())

            if len(pred_boxes) == 0:
                continue

            all_conf.extend(pred_scores.numpy().tolist())
            all_pred_cls.extend(pred_labels.tolist())

            if len(gt_boxes) == 0:
                all_tp.extend([np.zeros(n_thr, dtype=bool)] * len(pred_boxes))
                continue

            iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
            gt_matched = np.zeros((n_thr, len(gt_boxes)), dtype=bool)

            for pi in range(len(pred_boxes)):
                tp_flags = np.zeros(n_thr, dtype=bool)
                if iou_matrix.shape[1] == 0:
                    all_tp.append(tp_flags)
                    continue

                best_iou_idx = iou_matrix[pi].argmax()
                best_iou = iou_matrix[pi, best_iou_idx]

                for ti, thr in enumerate(iou_thresholds):
                    if (best_iou >= thr
                            and pred_labels[pi] == gt_labels[best_iou_idx]
                            and not gt_matched[ti, best_iou_idx]):
                        tp_flags[ti] = True
                        gt_matched[ti, best_iou_idx] = True

                all_tp.append(tp_flags)

    return (np.array(all_tp).T if all_tp else np.zeros((n_thr, 0), dtype=bool),
            np.array(all_conf),
            np.array(all_pred_cls),
            np.array(all_tgt_cls))


@torch.no_grad()
def collect_predictions_yolov8(weights_path, images_dir, labels_dir, device):
    """
    For YOLOv8n via Ultralytics. We use the same mAP code, just need
    to extract predictions in the right format.

    The dataset is loaded directly via YoloTomatoDataset so labels match
    the 1..9 PyTorch convention used by all other models in this codebase.
    """
    from ultralytics import YOLO
    from src.dataset.yolo_dataset import YoloTomatoDataset
    from src.dataset.transforms import get_val_transforms

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    n_thr = len(iou_thresholds)

    model = YOLO(weights_path)

    # We need the same GT labels in the same 1..9 space.
    # Use the project's YoloTomatoDataset which already shifts 0..8 to 1..9.
    val_ds = YoloTomatoDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transforms=get_val_transforms(size=(640, 640)),
    )

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_tgt_cls = []

    for idx in range(len(val_ds)):
        _, target = val_ds[idx]
        img_path = os.path.join(images_dir, val_ds.image_files[idx])

        results = model.predict(img_path, device=device,
                                conf=0.05, iou=0.5, verbose=False)
        r = results[0]

        # Predictions in xyxy at the original resolution
        pred_boxes = r.boxes.xyxy.cpu()
        pred_scores = r.boxes.conf.cpu().numpy()
        # Ultralytics labels are 0..8; shift to 1..9 to match GT space
        pred_labels = (r.boxes.cls.cpu().numpy().astype(int) + 1)

        # Rescale predictions to 640×640 so they match the GT boxes
        # (which were resized to 640×640 by val_transforms).
        from PIL import Image as _Image
        with _Image.open(img_path) as im:
            orig_w, orig_h = im.size
        sx = 640.0 / orig_w
        sy = 640.0 / orig_h
        if len(pred_boxes) > 0:
            pred_boxes = pred_boxes.clone()
            pred_boxes[:, [0, 2]] *= sx
            pred_boxes[:, [1, 3]] *= sy

        gt_boxes = target["boxes"]
        gt_labels = target["labels"].numpy()

        all_tgt_cls.extend(gt_labels.tolist())

        if len(pred_boxes) == 0:
            continue

        all_conf.extend(pred_scores.tolist())
        all_pred_cls.extend(pred_labels.tolist())

        if len(gt_boxes) == 0:
            all_tp.extend([np.zeros(n_thr, dtype=bool)] * len(pred_boxes))
            continue

        iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
        gt_matched = np.zeros((n_thr, len(gt_boxes)), dtype=bool)

        for pi in range(len(pred_boxes)):
            tp_flags = np.zeros(n_thr, dtype=bool)
            if iou_matrix.shape[1] == 0:
                all_tp.append(tp_flags)
                continue

            best_iou_idx = iou_matrix[pi].argmax()
            best_iou = iou_matrix[pi, best_iou_idx]

            for ti, thr in enumerate(iou_thresholds):
                if (best_iou >= thr
                        and pred_labels[pi] == gt_labels[best_iou_idx]
                        and not gt_matched[ti, best_iou_idx]):
                    tp_flags[ti] = True
                    gt_matched[ti, best_iou_idx] = True

            all_tp.append(tp_flags)

    return (np.array(all_tp).T if all_tp else np.zeros((n_thr, 0), dtype=bool),
            np.array(all_conf),
            np.array(all_pred_cls),
            np.array(all_tgt_cls))


# ── Model loaders ──────────────────────────────────────────────
def load_model(model_name, seed, device):
    """Load a trained checkpoint. Returns (model, label_offset, model_kind)."""
    if model_name == "yolov8n":
        # For YOLOv8n we return the weights path, not a torch model
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
        return model, 0, "torchvision"   # SSDLite: labels 1..9, bg at 0

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
        # RetinaNet expects label 0..8 internally; we shift by -1 during training
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


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["ssdlite", "yolov8n", "retinanet", "retinanet_hybrid"])
    parser.add_argument("--seed", required=True, type=int,
                        choices=[42, 123, 456])
    parser.add_argument("--split", default="val",
                        choices=["val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model} | seed {args.seed} | split {args.split}")

    # ── Locate the split's images/labels ──
    split_images = os.path.join(PROJECT_ROOT, "data", "clean_yolo_dataset",
                                args.split, "images")
    split_labels = os.path.join(PROJECT_ROOT, "data", "clean_yolo_dataset",
                                args.split, "labels")
    if not os.path.isdir(split_images):
        raise FileNotFoundError(f"Split images dir not found: {split_images}")

    # ── Load model + collect predictions ──
    print("\nLoading model ...")
    model_or_path, label_offset, kind = load_model(args.model, args.seed, device)

    print(f"Running inference on {args.split} split ...")
    if kind == "yolo":
        tp, conf, pred_cls, tgt_cls = collect_predictions_yolov8(
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
        tp, conf, pred_cls, tgt_cls = collect_predictions_torchvision(
            model_or_path, loader, device, label_offset=label_offset
        )

    print(f"Collected {len(conf)} predictions over {len(tgt_cls)} GT boxes.")

    # ── Per-class metrics ──
    metrics = per_class_metrics(tp, conf, pred_cls, tgt_cls, num_classes=9)

    # ── Print table ──
    print(f"\nPer-class metrics ({args.model}, seed {args.seed}, split {args.split}):")
    print(f"  {'Class':<26} {'AP@50':>7} {'AP@50:95':>9} {'Prec':>7} {'Rec':>7} {'#GT':>6} {'#Pred':>7}")
    rows = []
    for c in range(1, 10):
        d = metrics[c]
        name = DISPLAY_NAMES[c]
        print(f"  {name:<26} {d['ap50']:>7.4f} {d['ap5095']:>9.4f} "
              f"{d['precision']:>7.4f} {d['recall']:>7.4f} "
              f"{d['n_gt']:>6d} {d['n_pred']:>7d}")
        rows.append((name, d['ap50'], d['ap5095'], d['precision'], d['recall'],
                     d['n_gt'], d['n_pred']))

    mean_ap50 = np.mean([metrics[c]['ap50'] for c in range(1, 10)])
    mean_ap5095 = np.mean([metrics[c]['ap5095'] for c in range(1, 10)])
    print(f"  {'─'*70}")
    print(f"  {'OVERALL (mean over classes)':<26} {mean_ap50:>7.4f} {mean_ap5095:>9.4f}")

    # ── Save CSV ──
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "results", args.model,
                           f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"per_class_{args.split}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "ap50", "ap5095", "precision", "recall",
                    "n_gt", "n_pred"])
        for r in rows:
            w.writerow(r)
        w.writerow(["OVERALL_mean_over_classes", mean_ap50, mean_ap5095, "", "", "", ""])
    print(f"\nSaved per-class CSV: {csv_path}")


if __name__ == "__main__":
    main()
