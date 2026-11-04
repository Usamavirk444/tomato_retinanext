"""
Cross-dataset evaluation #3 — YOLOv8 Oriented Bounding Box (OBB) format.

Standalone, read-only. Evaluates the four trained models on the
Tomatoes_diseases.v1i.yolov8-obb dataset (2,396 images, 11 classes, OBB
labels). Nine of the 11 classes overlap with our training set — much
higher overlap than the previous two cross-datasets.

OBB → axis-aligned conversion: each label line is
    class x1 y1 x2 y2 x3 y3 x4 y4
where the four (x,y) pairs are the four corners of the rotated rectangle,
normalised to [0,1]. We compute the tight axis-aligned circumscribing
rectangle (min/max of the four corners) for compatibility with our
axis-aligned detectors. This is a standard conversion documented in the
methods section as: "Oriented bounding boxes were converted to their tight
axis-aligned circumscribing rectangles."

Class mapping (new dataset class index → our PyTorch 1..9 label):
    0  Bacterial Spot             → SKIP (not in our training)
    1  Early Blight               → 1
    2  Healthy                    → 2
    3  Iron Deficiency            → SKIP (not in our training)
    4  Late Blight                → 3
    5  Leaf Mold                  → 5
    6  Leaf_Miner                 → 4
    7  Mosaic Virus               → 6
    8  Septoria                   → 7
    9  Spider Mites               → 8
    10 Yellow Leaf Curl Virus     → 9

Usage:
    python cross_eval/cross_dataset_3_eval.py --model all --seed 42

Output:
    outputs/cross_eval_3/<model>/seed_<n>/per_class_cross.csv
    outputs/cross_eval_3/cross_dataset_3_summary.csv

DOES NOT modify any training code or checkpoint. Reads only.
"""

import os
import sys
import argparse
import csv

import numpy as np
import torch
from torchvision.ops import box_iou
from PIL import Image

# Path setup
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE) if HERE.endswith("cross_eval") else HERE
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Class mapping
OBB_INDEX_TO_OURS = {
    1:  ("Early Blight",            1),
    2:  ("Healthy",                 2),
    4:  ("Late Blight",             3),
    5:  ("Leaf Mold",               5),
    6:  ("Leaf_Miner",              4),
    7:  ("Mosaic Virus",            6),
    8:  ("Septoria",                7),
    9:  ("Spider Mites",            8),
    10: ("Yellow Leaf Curl Virus",  9),
}
OUR_DISPLAY = {
    1: "Early Blight", 2: "Healthy", 3: "Late Blight",
    4: "Leaf Miners", 5: "Leaf Mold", 6: "Mosaic Virus",
    7: "Septoria", 8: "Spider Mites", 9: "Yellow Leaf Curl Virus",
}
OVERLAP_PYTORCH_LABELS = sorted(set(v[1] for v in OBB_INDEX_TO_OURS.values()))


# ── mAP utilities ──────────────────────────────────────────────
def _compute_ap_101(recall, precision):
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at_r = precision[recall >= thr]
        ap += p_at_r.max() if len(p_at_r) else 0.0
    return ap / 101.0


def per_class_metrics(tp, conf, pred_cls, target_cls, class_labels):
    sort_idx = np.argsort(-conf)
    tp = tp[:, sort_idx]
    conf = conf[sort_idx]
    pred_cls = pred_cls[sort_idx]

    out = {}
    for c in class_labels:
        n_gt = int((target_cls == c).sum())
        mask = pred_cls == c
        n_pred = int(mask.sum())

        if n_gt == 0 or n_pred == 0:
            out[c] = dict(ap50=0.0, ap5095=0.0, precision=0.0, recall=0.0,
                          n_gt=n_gt, n_pred=n_pred)
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

        out[c] = dict(
            ap50=float(ap50), ap5095=float(ap5095),
            precision=float(precision_c[0][-1]),
            recall=float(recall_c[0][-1]),
            n_gt=n_gt, n_pred=n_pred,
        )
    return out


# ── OBB label parser ───────────────────────────────────────────
def parse_obb_label_file(path):
    """
    Parse a YOLOv8 OBB label file. Each row:
        class x1 y1 x2 y2 x3 y3 x4 y4   (all normalised to [0, 1])
    Returns list of (our_label, x1_n, y1_n, x2_n, y2_n) — axis-aligned bbox
    in normalised coordinates. Boxes belonging to non-overlap classes
    are skipped.
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            parts = ln.strip().split()
            if len(parts) < 9:
                continue
            try:
                cls = int(float(parts[0]))
                coords = [float(x) for x in parts[1:9]]
            except (TypeError, ValueError):
                continue
            if cls not in OBB_INDEX_TO_OURS:
                continue
            our_label = OBB_INDEX_TO_OURS[cls][1]
            xs = coords[::2]
            ys = coords[1::2]
            x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
            x2, y2 = min(1.0, max(xs)), min(1.0, max(ys))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((our_label, x1, y1, x2, y2))
    return out


def load_cross_dataset_3(root):
    """
    Walk train/valid/test splits. Returns list of (img_path, orig_w, orig_h,
    gt_list) where gt_list is [(x1,y1,x2,y2, our_label), ...] in pixel
    coordinates scaled to a uniform 640×640 reference (matching what the
    detection pipeline uses).
    """
    items = []
    skipped = 0
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, fn)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")

            normalised_boxes = parse_obb_label_file(lbl_path)
            if not normalised_boxes:
                skipped += 1
                continue

            # Get image size; scale boxes to 640×640
            try:
                with Image.open(img_path) as im:
                    orig_w, orig_h = im.size
            except Exception:
                skipped += 1
                continue

            gt = []
            for lbl, x1n, y1n, x2n, y2n in normalised_boxes:
                gt.append([x1n * 640, y1n * 640, x2n * 640, y2n * 640, lbl])
            items.append((img_path, orig_w, orig_h, gt))
    return items, skipped


# ── Model loaders (READ-ONLY) ──────────────────────────────────
def load_model(model_name, seed, device):
    if model_name == "yolov8n":
        weights_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "yolov8n",
            f"seed_{seed}", "weights", "best.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLOv8n weights not found: {weights_path}")
        return weights_path, 0, "yolo"

    elif model_name == "ssdlite":
        from src.models.ssdlite import build_ssdlite_vanilla
        ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "ssdlite",
            f"seed_{seed}", "ssdlite_mobilenet_v3_large_best.pth")
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
            f"seed_{seed}", "retinanet_resnet50_fpn_v2_best.pth")
        model = build_retinanet(num_classes=10, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, -1, "torchvision"

    elif model_name == "retinanet_l1":
        # Same architecture as `retinanet`; only the training-time box
        # regression loss differed, which has no effect at inference.
        from src.models.retinanet import build_retinanet
        ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "retinanet_l1",
            f"seed_{seed}", "retinanet_resnet50_fpn_v2_l1_best.pth")
        model = build_retinanet(num_classes=10, pretrained=False, box_loss="l1")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, -1, "torchvision"

    elif model_name == "retinanet_hybrid":
        from src.models.hybrid import build_retinanet_hybrid
        ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "retinanet_hybrid",
            f"seed_{seed}",
            "retinanet_hybrid_resnet50_bifpn_cbam_best.pth")
        model = build_retinanet_hybrid(num_classes=10, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, -1, "torchvision"

    raise ValueError(f"Unknown model: {model_name}")


# ── Inference + TP/FP collection ───────────────────────────────
def _collect_torchvision(model, items, device, label_offset=0):
    from torchvision.transforms.functional import to_tensor, normalize, resize
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    n_thr = len(iou_thresholds)
    model.eval()

    all_tp, all_conf, all_pred_cls, all_tgt_cls = [], [], [], []

    for img_path, orig_w, orig_h, gt in items:
        img = Image.open(img_path).convert("RGB")
        img = resize(img, (640, 640))
        t = to_tensor(img)
        t = normalize(t, mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)).to(device)

        with torch.no_grad():
            outputs = model([t])
        o = outputs[0]

        pred_boxes = o["boxes"].cpu()
        pred_scores = o["scores"].cpu().numpy()
        pred_labels = (o["labels"].cpu().numpy() - label_offset)

        keep = np.array([lbl in OVERLAP_PYTORCH_LABELS for lbl in pred_labels])
        if keep.sum() == 0:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = np.zeros(0)
            pred_labels = np.zeros(0, dtype=int)
        else:
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        gt_arr = np.array(gt)
        gt_boxes = torch.tensor(gt_arr[:, :4], dtype=torch.float32)
        gt_labels = gt_arr[:, 4].astype(int)

        all_tgt_cls.extend(gt_labels.tolist())
        if len(pred_boxes) == 0:
            continue

        all_conf.extend(pred_scores.tolist())
        all_pred_cls.extend(pred_labels.tolist())

        iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
        gt_matched = np.zeros((n_thr, len(gt_boxes)), dtype=bool)
        for pi in range(len(pred_boxes)):
            tp_flags = np.zeros(n_thr, dtype=bool)
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
            np.array(all_conf), np.array(all_pred_cls), np.array(all_tgt_cls))


def _collect_yolov8(weights_path, items, device):
    from ultralytics import YOLO
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    n_thr = len(iou_thresholds)

    model = YOLO(weights_path)

    all_tp, all_conf, all_pred_cls, all_tgt_cls = [], [], [], []
    for img_path, orig_w, orig_h, gt in items:
        results = model.predict(img_path,
                                device=device.index if device.type == "cuda" else "cpu",
                                conf=0.05, iou=0.5, verbose=False, imgsz=640)
        r = results[0]
        pred_boxes = r.boxes.xyxy.cpu()
        pred_scores = r.boxes.conf.cpu().numpy()
        pred_labels = (r.boxes.cls.cpu().numpy().astype(int) + 1)

        # Predictions are at 640x640 internally; GT already scaled to 640x640
        keep = np.array([lbl in OVERLAP_PYTORCH_LABELS for lbl in pred_labels])
        if keep.sum() == 0:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = np.zeros(0)
            pred_labels = np.zeros(0, dtype=int)
        else:
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        gt_arr = np.array(gt)
        gt_boxes = torch.tensor(gt_arr[:, :4], dtype=torch.float32)
        gt_labels = gt_arr[:, 4].astype(int)

        all_tgt_cls.extend(gt_labels.tolist())
        if len(pred_boxes) == 0:
            continue

        all_conf.extend(pred_scores.tolist())
        all_pred_cls.extend(pred_labels.tolist())

        iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
        gt_matched = np.zeros((n_thr, len(gt_boxes)), dtype=bool)
        for pi in range(len(pred_boxes)):
            tp_flags = np.zeros(n_thr, dtype=bool)
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
            np.array(all_conf), np.array(all_pred_cls), np.array(all_tgt_cls))


def run_one(model_name, seed, device, items, results_root):
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name} (seed {seed}) on cross_dataset_3")
    print(f"{'=' * 60}")

    model_or_path, label_offset, kind = load_model(model_name, seed, device)
    if kind == "yolo":
        tp, conf, pred_cls, tgt_cls = _collect_yolov8(model_or_path, items, device)
    else:
        tp, conf, pred_cls, tgt_cls = _collect_torchvision(
            model_or_path, items, device, label_offset=label_offset)

    metrics = per_class_metrics(tp, conf, pred_cls, tgt_cls,
                                OVERLAP_PYTORCH_LABELS)

    print(f"Predictions: {len(conf)}   GT boxes: {len(tgt_cls)}")
    print(f"  {'Class':<25} {'AP@50':>7} {'AP@50:95':>9} {'Prec':>7} {'Rec':>7} {'#GT':>6} {'#Pred':>7}")
    rows = []
    for c in OVERLAP_PYTORCH_LABELS:
        d = metrics[c]
        name = OUR_DISPLAY[c]
        print(f"  {name:<25} {d['ap50']:>7.4f} {d['ap5095']:>9.4f} "
              f"{d['precision']:>7.4f} {d['recall']:>7.4f} "
              f"{d['n_gt']:>6d} {d['n_pred']:>7d}")
        rows.append((name, d['ap50'], d['ap5095'], d['precision'],
                     d['recall'], d['n_gt'], d['n_pred']))

    mean_ap50 = float(np.mean([metrics[c]['ap50'] for c in OVERLAP_PYTORCH_LABELS]))
    mean_ap5095 = float(np.mean([metrics[c]['ap5095'] for c in OVERLAP_PYTORCH_LABELS]))
    print(f"  {'-' * 70}")
    print(f"  {'OVERALL (overlap classes)':<25} {mean_ap50:>7.4f} {mean_ap5095:>9.4f}")

    out_dir = os.path.join(results_root, model_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "per_class_cross.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "ap50", "ap5095", "precision", "recall",
                    "n_gt", "n_pred"])
        for r in rows:
            w.writerow(r)
        w.writerow(["OVERALL_mean_overlap", mean_ap50, mean_ap5095,
                    "", "", "", ""])
    print(f"Saved: {csv_path}")

    return {"model": model_name, "seed": seed,
            "mean_ap50": mean_ap50, "mean_ap5095": mean_ap5095,
            "per_class": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["ssdlite", "yolov8n", "retinanet",
                                 "retinanet_l1", "retinanet_hybrid", "all"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset-root", default=None,
                        help="Path to cross_dataset_3 folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = args.dataset_root or os.path.join(
        PROJECT_ROOT, "data", "cross_dataset_3")
    print(f"Dataset root: {dataset_root}")

    items, skipped = load_cross_dataset_3(dataset_root)
    n_gt = sum(len(it[3]) for it in items)
    print(f"Loaded {len(items)} images with {n_gt} overlap-class GT boxes "
          f"(OBB converted to axis-aligned)")
    print(f"Skipped {skipped} images (no overlap-class labels or no image)")
    print(f"Overlap classes: " +
          ", ".join(OUR_DISPLAY[c] for c in OVERLAP_PYTORCH_LABELS))

    results_root = os.path.join(PROJECT_ROOT, "outputs", "cross_eval_3")
    os.makedirs(results_root, exist_ok=True)

    models = (["ssdlite", "yolov8n", "retinanet", "retinanet_hybrid"]
              if args.model == "all" else [args.model])

    summaries = []
    for m in models:
        try:
            res = run_one(m, args.seed, device, items, results_root)
            summaries.append(res)
        except Exception as e:
            print(f"[FAIL] {m}: {e}")
            summaries.append({"model": m, "seed": args.seed,
                              "mean_ap50": 0.0, "mean_ap5095": 0.0,
                              "per_class": {}})

    if len(summaries) > 1:
        summary_path = os.path.join(results_root, "cross_dataset_3_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "seed",
                        "cross_mean_ap50", "cross_mean_ap5095"]
                       + [f"ap50_{OUR_DISPLAY[c]}" for c in OVERLAP_PYTORCH_LABELS])
            for s in summaries:
                row = [s["model"], s["seed"], s["mean_ap50"], s["mean_ap5095"]]
                for c in OVERLAP_PYTORCH_LABELS:
                    row.append(s["per_class"].get(c, {}).get("ap50", 0.0))
                w.writerow(row)
        print(f"\nSummary saved: {summary_path}")

        print(f"\n{'=' * 80}")
        print("CROSS-DATASET_3 SUMMARY (mean over 9 overlap classes)")
        print(f"{'=' * 80}")
        print(f"  {'Model':<28} {'cross mAP@50':>14} {'cross mAP@50:95':>17}")
        for s in summaries:
            print(f"  {s['model']:<28} {s['mean_ap50']:>14.4f} "
                  f"{s['mean_ap5095']:>17.4f}")


if __name__ == "__main__":
    main()
