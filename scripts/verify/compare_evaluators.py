"""
compare_evaluators.py
─────────────────────
Runs the in-domain evaluator (TorchMetrics, src/evaluation/evaluator.py) and
the external evaluator (custom COCO-style, cross_eval/cross_dataset_3_eval.py)
over THE SAME predictions and THE SAME ground truth, so any difference in the
reported mAP comes from the evaluation protocol and nothing else.

Why this matters
    Table 2 subtracts an external mAP from an in-domain mAP. That subtraction
    is only a measurement of domain shift if both numbers were computed the
    same way. They were not: the two evaluators differ in at least four
    places (confidence floor, matching order, maxDets, zero-GT handling).
    This script measures how much those differences are worth, in pp, on
    identical data.

What it does NOT do
    No training. No checkpoint is written. No existing result file, script or
    config is modified. Predictions are generated once, in memory, and fed to
    both evaluators. Outputs go to a new folder of their own.

Usage
    python compare_evaluators.py --model retinanet_hybrid --seed 42 --limit 100
    python compare_evaluators.py --model retinanet --seed 42 --limit 100
    python compare_evaluators.py --model retinanet_l1 --seed 42 --limit 100

    --limit 0 runs the whole test split (slow).

Outputs  (created fresh, nothing overwritten)
    outputs/evaluator_comparison/<model>_seed<n>_per_class.csv
    outputs/evaluator_comparison/<model>_seed<n>_report.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

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
CROSS_DIR = os.path.join(HERE, "cross_eval")
if CROSS_DIR not in sys.path:
    sys.path.insert(0, CROSS_DIR)

from src.dataset.yolo_dataset import build_yolo_dataloader          # noqa: E402
from src.dataset.transforms import get_val_transforms                # noqa: E402
from src.utils.general import get_device                             # noqa: E402

# The external evaluator's own helpers, imported — not copied — so this
# script cannot drift from the code that produced the published numbers.
import cross_dataset_3_eval as cross                                 # noqa: E402


# ── model construction, mirroring evaluate.py ───────────────────────
def build(model_name: str, num_classes: int):
    if model_name == "retinanet":
        from src.models.retinanet import build_retinanet
        return build_retinanet(num_classes, pretrained=False), -1
    if model_name == "retinanet_l1":
        from src.models.retinanet import build_retinanet
        return build_retinanet(num_classes, pretrained=False,
                               box_loss="l1"), -1
    if model_name == "retinanet_hybrid":
        from src.models.hybrid import build_retinanet_hybrid
        return build_retinanet_hybrid(num_classes, pretrained=False), -1
    if model_name == "ssdlite":
        from src.models.ssdlite import build_ssdlite_vanilla
        return build_ssdlite_vanilla(num_classes, pretrained=False), 0
    raise SystemExit(f"unsupported --model {model_name!r}. "
                     "yolov8n is excluded: it is not a torchvision model and "
                     "its external path already applies conf=0.05, a "
                     "different comparison.")


def checkpoint_path(cfg, model_name, seed):
    m = cfg["models"][model_name]
    return os.path.join(HERE, m["checkpoints_dir"], f"seed_{seed}",
                        f"{m['name']}_best.pth")


# ── TorchMetrics side, same settings as src/evaluation/evaluator.py ──
def torchmetrics_map(preds, targets, score_threshold=0.01):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    for p, t in zip(preds, targets):
        keep = p["scores"] >= score_threshold
        metric.update(
            [{"boxes": p["boxes"][keep], "labels": p["labels"][keep],
              "scores": p["scores"][keep]}],
            [{"boxes": t["boxes"], "labels": t["labels"]}])
    return metric.compute()


# ── external side: reuse cross_dataset_3_eval's own AP code ─────────
def external_map(preds, targets, class_labels, score_threshold=0.0):
    """Replicates _collect_torchvision + per_class_metrics exactly, but over
    predictions handed in rather than regenerated, so both evaluators see the
    identical detection set."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    n_thr = len(iou_thresholds)
    all_tp, all_conf, all_pred_cls, all_tgt_cls = [], [], [], []

    for p, t in zip(preds, targets):
        keep = p["scores"] >= score_threshold
        pred_boxes = p["boxes"][keep]
        pred_scores = p["scores"][keep].numpy()
        pred_labels = p["labels"][keep].numpy()

        gt_boxes = t["boxes"]
        gt_labels = t["labels"].numpy()
        all_tgt_cls.extend(gt_labels.tolist())

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            if len(pred_boxes):
                all_conf.extend(pred_scores.tolist())
                all_pred_cls.extend(pred_labels.tolist())
                all_tp.extend([np.zeros(n_thr, dtype=bool)] * len(pred_boxes))
            continue

        all_conf.extend(pred_scores.tolist())
        all_pred_cls.extend(pred_labels.tolist())

        from torchvision.ops import box_iou
        iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
        gt_matched = np.zeros((n_thr, len(gt_boxes)), dtype=bool)
        # NOTE: best-IoU-only matching, in prediction order — this is the
        # published external rule, kept verbatim.
        for pi in range(len(pred_boxes)):
            tp_flags = np.zeros(n_thr, dtype=bool)
            best_idx = iou_matrix[pi].argmax()
            best_iou = iou_matrix[pi, best_idx]
            for ti, thr in enumerate(iou_thresholds):
                if (best_iou >= thr
                        and pred_labels[pi] == gt_labels[best_idx]
                        and not gt_matched[ti, best_idx]):
                    tp_flags[ti] = True
                    gt_matched[ti, best_idx] = True
            all_tp.append(tp_flags)

    tp = np.array(all_tp).T if all_tp else np.zeros((n_thr, 0), dtype=bool)
    return cross.per_class_metrics(tp, np.array(all_conf),
                                   np.array(all_pred_cls),
                                   np.array(all_tgt_cls), class_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["retinanet", "retinanet_l1",
                             "retinanet_hybrid", "ssdlite"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=100,
                    help="number of test images; 0 = all")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # encoding is explicit: the VM runs a GBK locale, where Python's default
    # open() cannot read this UTF-8 config.
    cfg = yaml.safe_load(open(os.path.join(HERE, args.config),
                              encoding="utf-8"))
    tr, data = cfg["training"], cfg["data"]
    num_classes = tr["num_classes"]
    device = torch.device(args.device) if args.device else get_device()

    ckpt = checkpoint_path(cfg, args.model, args.seed)
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt}")

    model, label_offset = build(args.model, num_classes)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob.get("model_state_dict",
                                   blob.get("model", blob)))
    model.to(device).eval()

    loader = build_yolo_dataloader(
        images_dir=os.path.join(HERE, data["test_images"]),
        labels_dir=os.path.join(HERE, data["test_labels"]),
        transforms=get_val_transforms(size=tuple(tr["image_size"])),
        batch_size=4, shuffle=False, num_workers=0)

    # ── generate predictions ONCE ───────────────────────────────────
    preds, targets, n = [], [], 0
    print(f"collecting predictions ({'all' if not args.limit else args.limit} images)...")
    with torch.no_grad():
        for images, tgts in loader:
            out = model([im.to(device) for im in images])
            for o, t in zip(out, tgts):
                # shift model labels into dataset space (1..9), as evaluate.py
                # does via is_retinanet
                preds.append({"boxes": o["boxes"].cpu(),
                              "scores": o["scores"].cpu(),
                              "labels": (o["labels"].cpu() - label_offset)})
                targets.append({"boxes": t["boxes"].cpu(),
                                "labels": t["labels"].cpu()})
                n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    print(f"  {n} images, {sum(len(p['scores']) for p in preds)} raw detections")

    labels_present = sorted({int(x) for t in targets
                             for x in t["labels"].tolist()})
    class_labels = cross.OVERLAP_PYTORCH_LABELS

    # ── run both evaluators on those same predictions ───────────────
    print("running TorchMetrics (score>=0.01, COCO matching, maxDets=100)...")
    tm = torchmetrics_map(preds, targets, score_threshold=0.01)

    print("running external evaluator (no score floor, best-IoU matching)...")
    ex_raw = external_map(preds, targets, class_labels, score_threshold=0.0)

    print("running external evaluator with a 0.01 score floor...")
    ex_001 = external_map(preds, targets, class_labels, score_threshold=0.01)

    def ext_mean(d, key):
        return float(np.mean([d[c][key] for c in class_labels]))

    tm50, tm5095 = float(tm["map_50"]), float(tm["map"])
    ex50, ex5095 = ext_mean(ex_raw, "ap50"), ext_mean(ex_raw, "ap5095")
    ex50b, ex5095b = ext_mean(ex_001, "ap50"), ext_mean(ex_001, "ap5095")

    outdir = os.path.join(HERE, "outputs", "evaluator_comparison")
    os.makedirs(outdir, exist_ok=True)
    stem = f"{args.model}_seed{args.seed}"

    # per-class CSV
    tm_ap = tm.get("map_per_class")
    tm_cls = tm.get("classes")
    tm_by_class = {}
    if tm_ap is not None and tm_cls is not None and len(tm_ap) > 0:
        for i, c in enumerate(tm_cls.tolist()):
            tm_by_class[int(c)] = float(tm_ap[i])

    csv_path = os.path.join(outdir, f"{stem}_per_class.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "n_gt",
                    "torchmetrics_ap5095", "external_ap50",
                    "external_ap5095", "external_ap50_thr0.01"])
        for c in class_labels:
            w.writerow([c, cross.OUR_DISPLAY.get(c, c), ex_raw[c]["n_gt"],
                        round(tm_by_class.get(c, float("nan")), 5),
                        round(ex_raw[c]["ap50"], 5),
                        round(ex_raw[c]["ap5095"], 5),
                        round(ex_001[c]["ap50"], 5)])

    lines = [
        "EVALUATOR COMPARISON — identical predictions, identical ground truth",
        "=" * 72,
        f"model            {args.model}  seed {args.seed}",
        f"checkpoint       {ckpt}",
        f"images           {n}",
        f"raw detections   {sum(len(p['scores']) for p in preds)}",
        f"classes averaged {class_labels}",
        f"classes with GT  {labels_present}",
        "",
        "HEADLINE",
        "-" * 72,
        f"  TorchMetrics  mAP@0.5      {tm50*100:8.3f}",
        f"  External      mAP@0.5      {ex50*100:8.3f}    diff {(ex50-tm50)*100:+7.3f} pp",
        f"  External@0.01 mAP@0.5      {ex50b*100:8.3f}    diff {(ex50b-tm50)*100:+7.3f} pp",
        "",
        f"  TorchMetrics  mAP@0.5:0.95 {tm5095*100:8.3f}",
        f"  External      mAP@0.5:0.95 {ex5095*100:8.3f}    diff {(ex5095-tm5095)*100:+7.3f} pp",
        f"  External@0.01 mAP@0.5:0.95 {ex5095b*100:8.3f}    diff {(ex5095b-tm5095)*100:+7.3f} pp",
        "",
        "PROTOCOL DIFFERENCES BEING MEASURED",
        "-" * 72,
        "  confidence floor   TorchMetrics 0.01 | external none (0.0)",
        "                     the External@0.01 row isolates this one effect",
        "  matching           TorchMetrics: COCO, greedy by descending score,",
        "                       matched within class",
        "                     external: single best-IoU GT per prediction, in",
        "                       prediction order; if that GT is taken the",
        "                       prediction is a FP even when another GT qualifies",
        "  maxDets            TorchMetrics 100 per image | external unlimited",
        "  zero-GT / zero-pred TorchMetrics drops the class from the mean",
        "                     external scores it 0.0 and keeps it in the mean",
        "  averaging          both: per-class AP then unweighted mean",
        "",
        "READING THIS",
        "-" * 72,
        "  Both columns use the same detections on the same images, so every",
        "  pp of difference is protocol, not model and not domain. Subtracting",
        "  an external mAP from a TorchMetrics mAP carries that difference as",
        "  a bias. A difference under ~0.5 pp means the published drops are",
        "  usable as stated; larger means they are upper bounds and should be",
        "  described comparatively across models rather than as absolute",
        "  degradations.",
    ]
    txt_path = os.path.join(outdir, f"{stem}_report.txt")
    open(txt_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {csv_path}\nwrote {txt_path}")


if __name__ == "__main__":
    main()
