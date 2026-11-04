"""
eval_yolov8n_torchmetrics.py
────────────────────────────
Re-evaluates the trained YOLOv8n models (seeds 42, 123, 456) using the
same TorchMetrics MeanAveragePrecision implementation (v1.8.2, COCO-style
101-point interpolation) used for SSDLite, RetinaNet, and RetinaNeXt.

This script is READ-ONLY with respect to the existing project — it does
NOT modify any existing file, checkpoint, or result. Outputs are written
to a brand-new directory:
    outputs/results/yolov8n_torchmetrics/seed_<N>/

Usage (run from tomato_vm/ root):
    python eval_yolov8n_torchmetrics.py              # all three seeds
    python eval_yolov8n_torchmetrics.py --seed 42    # single seed

Why this script exists:
    The original evaluation used the Ultralytics native validator
    (model.val()), which computes mAP via a different internal pipeline
    than TorchMetrics.  To ensure all four models are evaluated on an
    identical metric basis, this script runs YOLOv8n inference and feeds
    the raw predictions directly into TorchMetrics, exactly as is done
    for the three torchvision models.
"""

import argparse
import csv
import os
import sys

import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ── Allow `from src.*` imports when run from tomato_vm/ root ──────────────────
# --- repository-root resolution -------------------------------------
# This script was moved from the project root into a scripts/ sub-
# folder when the reproducibility repository was built. The line
# below walks up until it finds the repo root (the folder holding
# train_all.py) so every path it builds resolves exactly as it did
# in the original layout. Behaviour is unchanged.
HERE = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(HERE, "train_all.py")) \
        and os.path.dirname(HERE) != HERE:
    HERE = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.utils.label_map import DISPLAY_NAMES  # PyTorch label 1-9 → display name

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 640          # all images are 640×640
NUM_FG     = 9            # foreground classes
SEEDS      = [42, 123, 456]

# Defaults point at the full test split. main() overrides these when
# --test-dir / --out-root are given, so the identical TorchMetrics path can be
# run against the leakage-free subset. Keeping one code path for both means the
# full-test and clean-subset YOLOv8n numbers are computed the same way, and are
# therefore comparable to each other and to the three torchvision models.
DATA_ROOT  = os.path.join(HERE, "data", "clean_yolo_dataset", "test")
TEST_IMGS  = os.path.join(DATA_ROOT, "images")
TEST_LBLS  = os.path.join(DATA_ROOT, "labels")

CKPT_TMPL  = os.path.join(
    HERE, "outputs", "checkpoints", "yolov8n", "seed_{seed}", "weights", "best.pt"
)
OUT_ROOT   = os.path.join(HERE, "outputs", "results", "yolov8n_torchmetrics")
OUT_TMPL   = os.path.join(OUT_ROOT, "seed_{seed}")

# Score threshold identical to the torchvision Evaluator default
SCORE_THRESH = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────────

def yolo_label_to_xyxy(label_path, img_w=IMG_SIZE, img_h=IMG_SIZE):
    """
    Parse a YOLO-format label file and return:
        boxes  : float32 tensor (N, 4) in pixel xyxy
        labels : int64  tensor (N,)   in 1-indexed PyTorch convention (1-9)
    Returns empty tensors if the file is absent or blank.
    """
    if not os.path.exists(label_path):
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,),   dtype=torch.int64))

    boxes, labels = [], []
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = (float(p) for p in parts[:5])
            cx *= img_w;  cy *= img_h
            bw *= img_w;  bh *= img_h
            x1 = cx - bw / 2;  y1 = cy - bh / 2
            x2 = cx + bw / 2;  y2 = cy + bh / 2
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)   # YOLO 0-8 -> PyTorch 1-9

    if not boxes:
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,),   dtype=torch.int64))

    return (torch.tensor(boxes,  dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64))


def ultralytics_to_torchmetrics(result):
    """
    Convert one Ultralytics Results object to a TorchMetrics prediction dict.
        boxes  : float32 (N, 4) pixel xyxy
        scores : float32 (N,)
        labels : int64  (N,)   1-indexed (1-9) to match other models
    """
    boxes  = result.boxes.xyxy.cpu().float()
    scores = result.boxes.conf.cpu().float()
    labels = (result.boxes.cls.cpu().long() + 1)   # YOLO 0-8 -> PyTorch 1-9

    if len(scores) > 0:
        keep   = scores >= SCORE_THRESH
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    if len(boxes) == 0:
        return {
            "boxes":  torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,),   dtype=torch.float32),
            "labels": torch.zeros((0,),   dtype=torch.int64),
        }
    return {"boxes": boxes, "scores": scores, "labels": labels}


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_seed(seed, batch_size=8):
    """Run TorchMetrics mAP evaluation for one seed. Returns the metric dict."""
    from ultralytics import YOLO

    ckpt = CKPT_TMPL.format(seed=seed)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if torch.cuda.is_available():
        device_str = "0"
    elif (hasattr(torch.backends, "mps")
          and torch.backends.mps.is_available()):
        device_str = "mps"
    else:
        device_str = "cpu"

    print(f"\n{'='*60}")
    print(f"  YOLOv8n  |  seed {seed}  |  TorchMetrics evaluation")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Device     : {device_str}")
    print(f"{'='*60}")

    model = YOLO(ckpt)

    img_files = sorted(
        f for f in os.listdir(TEST_IMGS)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"  Test images: {len(img_files)}")

    # Two accumulators over the SAME predictions.
    #
    #   metric   : default iou_thresholds 0.50:0.05:0.95
    #              -> map_per_class is per-class AP@0.5:0.95
    #   metric50 : iou_thresholds fixed to [0.5]
    #              -> map_per_class is per-class AP@0.5
    #
    # The earlier version of this script used only the first accumulator and
    # wrote its map_per_class into the "ap50" column. That column therefore
    # held AP@0.5:0.95 values, and "ap5095" was filled with -1. Adding the
    # second accumulator costs no extra inference — both are updated from the
    # same batch of predictions.
    metric   = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric50 = MeanAveragePrecision(iou_type="bbox", class_metrics=True,
                                    iou_thresholds=[0.5])

    n_batches = (len(img_files) + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        batch_files = img_files[b_idx * batch_size : (b_idx + 1) * batch_size]

        img_paths = [os.path.join(TEST_IMGS, f) for f in batch_files]
        lbl_paths = [
            os.path.join(TEST_LBLS, os.path.splitext(f)[0] + ".txt")
            for f in batch_files
        ]

        results = model.predict(
            source=img_paths,
            imgsz=IMG_SIZE,
            device=device_str,
            verbose=False,
            save=False,
            conf=0.001,   # must be very low so TorchMetrics sees the full
                          # prediction set and can compute the true PR curve;
                          # Ultralytics predict() default of 0.25 silently
                          # discards low-confidence TPs before we ever see them
            iou=0.5,      # NMS IoU threshold — same as evaluate_per_class.py
        )

        preds   = [ultralytics_to_torchmetrics(r) for r in results]
        targets = []
        for r, lp in zip(results, lbl_paths):
            h, w = r.orig_shape          # actual image dims (H, W) — may be 512 or 640
            gt_boxes, gt_labels = yolo_label_to_xyxy(lp, img_w=w, img_h=h)
            targets.append({"boxes": gt_boxes, "labels": gt_labels})

        metric.update(preds, targets)
        metric50.update(preds, targets)

        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == n_batches:
            print(f"    [{b_idx + 1:4d}/{n_batches}] batches processed")

    print("  Computing metrics ...")
    computed   = metric.compute()
    computed50 = metric50.compute()
    return computed, computed50


def save_results(computed, computed50, seed):
    """Save per_class_test.csv in the same format as the other models."""
    out_dir = OUT_TMPL.format(seed=seed)
    os.makedirs(out_dir, exist_ok=True)

    # per-class AP@0.5 comes from the [0.5]-only accumulator;
    # per-class AP@0.5:0.95 comes from the default accumulator.
    per_class_ap50 = computed50.get("map_per_class",     None)
    per_class_ap   = computed.get("map_per_class",       None)
    per_class_rec  = computed.get("mar_100_per_class",   None)
    class_ids      = computed.get("classes",             None)
    class_ids50    = computed50.get("classes",           None)

    # Both accumulators see identical predictions and targets, so their class
    # ordering must agree. Fail loudly rather than silently misalign columns.
    if class_ids is not None and class_ids50 is not None:
        assert torch.equal(class_ids, class_ids50), \
            "class ordering differs between the two accumulators"

    map50   = computed["map_50"].item()
    map5095 = computed["map"].item()

    print(f"\n  mAP@0.5      : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95 : {map5095:.4f}  ({map5095*100:.2f}%)")
    print(f"\n  Per-class:")
    print(f"  {'Class':<26} {'AP50':>7} {'AP50-95':>8} {'Recall':>8}")

    rows = []
    for i in range(len(per_class_ap50)):
        label     = int(class_ids[i].item()) if class_ids is not None else i + 1
        name      = DISPLAY_NAMES.get(label, f"class_{label}")
        ap50_val  = per_class_ap50[i].item()
        ap95_val  = per_class_ap[i].item() if per_class_ap is not None else -1
        rec_val   = per_class_rec[i].item() if per_class_rec is not None else -1
        print(f"  {name:<26} {ap50_val:>7.4f} {ap95_val:>8.4f} {rec_val:>8.4f}")
        rows.append({
            "class":     name,
            "ap50":      round(ap50_val, 6),
            "ap5095":    round(ap95_val, 6),
            "precision": -1,
            "recall":    round(rec_val, 6),
            "n_gt":      -1,
            "n_pred":    -1,
        })

    # Sanity check: the mean of the per-class AP@0.5 values should land on the
    # headline mAP@0.5. TorchMetrics averages over classes that have at least
    # one ground-truth box, so a small gap is possible but a large one is not.
    mean_pc50 = sum(r["ap50"] for r in rows) / len(rows)
    print(f"\n  mean of per-class AP@0.5 : {mean_pc50*100:.2f}%"
          f"   (headline mAP@0.5 {map50*100:.2f}%)")
    if abs(mean_pc50 - map50) > 0.005:
        print("  ** WARNING: per-class mean and headline mAP@0.5 disagree "
              "by more than 0.5 pp — investigate before using in the paper **")

    csv_path = os.path.join(out_dir, "per_class_test.csv")
    fieldnames = ["class", "ap50", "ap5095", "precision", "recall", "n_gt", "n_pred"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({
            "class":     "OVERALL_mAP50",
            "ap50":      round(map50,   6),
            "ap5095":    round(map5095, 6),
            "precision": "",
            "recall":    "",
            "n_gt":      "",
            "n_pred":    "",
        })

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "mAP50", "mAP50_95"])
        w.writerow([seed, round(map50, 6), round(map5095, 6)])

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {summary_path}")
    return map50, map5095


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8n with TorchMetrics (new output dir, no existing files touched)"
    )
    parser.add_argument(
        "--seed", type=int, choices=SEEDS, default=None,
        help="Single seed to evaluate (default: all three seeds)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Inference batch size (default: 8)"
    )
    parser.add_argument(
        "--test-dir", default=None,
        help="Test split to evaluate, containing images/ and labels/. "
             "Default: data/clean_yolo_dataset/test. "
             "For the leakage-free subset: data/clean_test_only/test"
    )
    parser.add_argument(
        "--out-root", default=None,
        help="Where to write results. Default: "
             "outputs/results/yolov8n_torchmetrics. Give a different path when "
             "using --test-dir so the full-test results are not overwritten."
    )
    args = parser.parse_args()

    # ── redirect the split / output if asked ─────────────────────────────────
    global DATA_ROOT, TEST_IMGS, TEST_LBLS, OUT_ROOT, OUT_TMPL

    if args.test_dir:
        DATA_ROOT = os.path.abspath(args.test_dir)
        TEST_IMGS = os.path.join(DATA_ROOT, "images")
        TEST_LBLS = os.path.join(DATA_ROOT, "labels")
        for d in (TEST_IMGS, TEST_LBLS):
            if not os.path.isdir(d):
                sys.exit(f"ERROR: not found: {d}")
        if not args.out_root:
            sys.exit("ERROR: --test-dir given without --out-root.\n"
                     "Refusing to write clean-subset results into "
                     "outputs/results/yolov8n_torchmetrics, which holds the "
                     "published full-test numbers.")

    if args.out_root:
        OUT_ROOT = os.path.abspath(args.out_root)
        OUT_TMPL = os.path.join(OUT_ROOT, "seed_{seed}")

    n_imgs = len([f for f in os.listdir(TEST_IMGS)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"\n  Test split : {TEST_IMGS}  ({n_imgs:,} images)")
    print(f"  Output     : {OUT_ROOT}")

    seeds_to_run = [args.seed] if args.seed else SEEDS

    all_map50   = []
    all_map5095 = []

    for seed in seeds_to_run:
        computed, computed50 = evaluate_seed(seed, batch_size=args.batch_size)
        map50, map5095 = save_results(computed, computed50, seed)
        all_map50.append(map50)
        all_map5095.append(map5095)

    if len(seeds_to_run) > 1:
        mean50   = float(np.mean(all_map50))
        std50    = float(np.std(all_map50,   ddof=1))
        mean5095 = float(np.mean(all_map5095))
        std5095  = float(np.std(all_map5095, ddof=1))
        print(f"\n{'='*60}")
        print(f"  YOLOv8n (TorchMetrics) — final summary")
        print(f"  mAP@0.5      : {mean50*100:.2f}% ± {std50*100:.2f}%")
        print(f"  mAP@0.5:0.95 : {mean5095*100:.2f}% ± {std5095*100:.2f}%")
        print(f"{'='*60}")

        os.makedirs(OUT_ROOT, exist_ok=True)
        cross_path = os.path.join(OUT_ROOT, "cross_seed_summary.csv")
        with open(cross_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "seed_42", "seed_123", "seed_456", "mean", "std"])
            w.writerow(["mAP50"]   + [f"{v:.6f}" for v in all_map50]
                       + [f"{mean50:.6f}", f"{std50:.6f}"])
            w.writerow(["mAP5095"] + [f"{v:.6f}" for v in all_map5095]
                       + [f"{mean5095:.6f}", f"{std5095:.6f}"])
        print(f"  Cross-seed summary: {cross_path}")


if __name__ == "__main__":
    main()