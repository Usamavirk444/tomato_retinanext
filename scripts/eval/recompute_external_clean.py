"""
recompute_external_clean.py
───────────────────────────
Corrects the external mAP of the overlap-free benchmark
(data/cross_dataset_3_clean) WITHOUT re-running any model.

Why
  cross_eval/cross_dataset_3_eval.py :: per_class_metrics() sets AP = 0 for a
  class with no ground truth (n_gt == 0) and run_one() then averages all nine
  classes. In the clean set Mosaic Virus has n_gt = 0, so a meaningless 0 is
  averaged into every model's mAP. COCO / TorchMetrics exclude such classes.

Why no re-run is needed
  per_class_metrics() computes each class's AP only from that class's own
  predictions (pred_cls == c) and its own ground truth (target_cls == c).
  The stored per-class AP values are therefore already correct; only the
  final average is wrong. This script re-averages them over the classes with
  n_gt > 0. IoU threshold, matching, confidence handling and the 101-point AP
  are unchanged because the per-class values are reused as stored.

Reads (read-only)
  outputs/cross_eval_3_clean/<model>/seed_<n>/per_class_cross.csv   (24 files)
  data/cross_dataset_3_clean/*/labels                               (count check)
Writes (new folder; refuses to overwrite)
  outputs/cross_eval_3_clean_corrected/per_class_cross.csv
  outputs/cross_eval_3_clean_corrected/summary_cross.csv
  outputs/cross_eval_3_clean_corrected/summary_cross_mean_std.csv
  outputs/cross_eval_3_clean_corrected/validation_report.txt

Usage (from the project root):  python recompute_external_clean.py
"""
import csv
import os
import statistics
import sys

# --- repository-root resolution -------------------------------------
# This script was moved from the project root into a scripts/ sub-
# folder when the reproducibility repository was built. The line
# below walks up until it finds the repo root (the folder holding
# train_all.py) so every path it builds resolves exactly as it did
# in the original layout. Behaviour is unchanged.
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ROOT, "train_all.py")) \
        and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
SRC = os.path.join(ROOT, "outputs", "cross_eval_3_clean")
DST = os.path.join(ROOT, "outputs", "cross_eval_3_clean_corrected")
DATA = os.path.join(ROOT, "data", "cross_dataset_3_clean")

MODELS = [
    ("ssdlite", "SSDLite-MobileNetV3"),
    ("yolov8n", "YOLOv8n"),
    ("retinanet", "S1 RetinaNet-R50-FPN-v2 (GIoU, published)"),
    ("retinanet_l1", "S1-L1 RetinaNet-R50-FPN-v2 (L1)"),
    ("retinanet_convnext_baseline", "S2 ConvNeXt-T + FPN"),
    ("retinanet_convnext_fpn_cbam", "S3 ConvNeXt-T + FPN + CBAM"),
    ("retinanet_convnext_bifpn", "S4 ConvNeXt-T + BiFPN x3"),
    ("retinanet_hybrid", "S5 RetinaNeXt"),
]
SEEDS = [42, 123, 456]
CLASSES = ["Early Blight", "Healthy", "Late Blight", "Leaf Miners", "Leaf Mold",
           "Mosaic Virus", "Septoria", "Spider Mites", "Yellow Leaf Curl Virus"]
# External OBB class index -> evaluated (same mapping as cross_dataset_3_eval.py)
OBB_OVERLAP = {1, 2, 4, 5, 6, 7, 8, 9, 10}
EXPECTED_IMAGES = 805
EXPECTED_GT = 7307

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def count_evaluable_images():
    """Re-applies load_cross_dataset_3()'s rule: an image is evaluated when it
    has >= 1 valid box of an overlap class (non-degenerate after OBB->AABB)."""
    n_img, n_box = 0, 0
    for split in ("train", "valid", "test"):
        idir = os.path.join(DATA, split, "images")
        ldir = os.path.join(DATA, split, "labels")
        if not os.path.isdir(idir):
            continue
        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lp = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
            boxes = 0
            if os.path.exists(lp):
                for ln in open(lp):
                    p = ln.split()
                    if len(p) < 9:
                        continue
                    try:
                        c = int(float(p[0]))
                        co = [float(x) for x in p[1:9]]
                    except ValueError:
                        continue
                    if c not in OBB_OVERLAP:
                        continue
                    xs, ys = co[::2], co[1::2]
                    if min(1, max(xs)) <= max(0, min(xs)) or min(1, max(ys)) <= max(0, min(ys)):
                        continue
                    boxes += 1
            if boxes:
                n_img += 1
                n_box += boxes
    return n_img, n_box


def main():
    if os.path.exists(DST) and os.listdir(DST):
        sys.exit(f"ERROR: {DST} already has files - refusing to overwrite.")
    os.makedirs(DST, exist_ok=True)

    per_class_rows, summary_rows = [], []
    gt_profile = {}          # (model, seed) -> tuple of n_gt per class
    missing = []

    for model, label in MODELS:
        for seed in SEEDS:
            f = os.path.join(SRC, model, f"seed_{seed}", "per_class_cross.csv")
            if not os.path.isfile(f):
                missing.append(f)
                continue
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
            cls_rows = [r for r in rows if not r["class"].startswith("OVERALL")]
            overall = [r for r in rows if r["class"].startswith("OVERALL")]
            names = [r["class"] for r in cls_rows]
            check(f"{model} s{seed}: 9 classes, no duplicates, expected names",
                  names == CLASSES, f"got {names}")

            ap50_all = [float(r["ap50"]) for r in cls_rows]
            ap95_all = [float(r["ap5095"]) for r in cls_rows]
            if overall:
                check(f"{model} s{seed}: stored OVERALL = 9-class mean (source consistent)",
                      abs(float(overall[0]["ap50"]) - statistics.mean(ap50_all)) < 1e-9)

            pos = [r for r in cls_rows if int(r["n_gt"]) > 0]
            zero = [r["class"] for r in cls_rows if int(r["n_gt"]) == 0]
            gt_profile[(model, seed)] = tuple(int(r["n_gt"]) for r in cls_rows)

            map50 = statistics.mean(float(r["ap50"]) for r in pos)
            map95 = statistics.mean(float(r["ap5095"]) for r in pos)
            for r in cls_rows:
                inc = int(r["n_gt"]) > 0
                per_class_rows.append({
                    "model": model, "label": label, "seed": seed, "class": r["class"],
                    "n_gt": int(r["n_gt"]), "n_pred": int(r["n_pred"]),
                    "ap50": f"{float(r['ap50']):.6f}" if inc else "N/A",
                    "ap50_95": f"{float(r['ap5095']):.6f}" if inc else "N/A",
                    "precision_at_last_det": f"{float(r['precision']):.6f}" if inc else "N/A",
                    "recall_at_last_det": f"{float(r['recall']):.6f}" if inc else "N/A",
                    "included_in_map": "yes" if inc else "no (zero ground truth)",
                })
            summary_rows.append({
                "model": model, "label": label, "seed": seed,
                "n_classes_in_map": len(pos), "excluded_classes": ";".join(zero) or "none",
                "map50": f"{map50 * 100:.4f}", "map50_95": f"{map95 * 100:.4f}",
                "uncorrected_9class_map50": f"{statistics.mean(ap50_all) * 100:.4f}",
                "uncorrected_9class_map50_95": f"{statistics.mean(ap95_all) * 100:.4f}",
                "n_gt_total": sum(int(r["n_gt"]) for r in cls_rows),
            })
            # corrected mAP equals the arithmetic mean of the GT-positive APs
            check(f"{model} s{seed}: corrected mAP50 = mean of {len(pos)} GT-positive APs",
                  abs(map50 - sum(float(r["ap50"]) for r in pos) / len(pos)) < 1e-12)

    check("all 24 model/seed files present", not missing, "; ".join(missing))
    check("8 configurations x 3 seeds summarised", len(summary_rows) == 24, str(len(summary_rows)))
    keys = [(r["model"], r["seed"], r["class"]) for r in per_class_rows]
    check("216 per-class records, no duplicates", len(keys) == 216 and len(set(keys)) == 216,
          f"{len(keys)} rows, {len(set(keys))} unique")

    profiles = set(gt_profile.values())
    check("identical ground truth (per-class n_gt) in every run", len(profiles) == 1, str(profiles))
    prof = next(iter(profiles)) if profiles else ()
    check(f"total GT boxes = {EXPECTED_GT} in every run",
          all(sum(p) == EXPECTED_GT for p in gt_profile.values()))
    mi = CLASSES.index("Mosaic Virus")
    check("Mosaic Virus has n_gt = 0 and is excluded from the mean",
          prof and prof[mi] == 0 and all(r["excluded_classes"] == "Mosaic Virus" for r in summary_rows))
    check("8 GT-positive classes used in every run",
          all(r["n_classes_in_map"] == 8 for r in summary_rows))

    n_img, n_box = count_evaluable_images()
    check(f"dataset folder yields {EXPECTED_IMAGES} evaluable images / {EXPECTED_GT} boxes "
          f"under the evaluator's loading rule", n_img == EXPECTED_IMAGES and n_box == EXPECTED_GT,
          f"{n_img} images, {n_box} boxes")

    # write outputs
    with open(os.path.join(DST, "per_class_cross.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader(); w.writerows(per_class_rows)
    with open(os.path.join(DST, "summary_cross.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    agg = []
    for model, label in MODELS:
        s = [r for r in summary_rows if r["model"] == model]
        v50 = [float(r["map50"]) for r in s]
        v95 = [float(r["map50_95"]) for r in s]
        agg.append({"model": model, "label": label, "n_seeds": len(s),
                    "seeds": ";".join(str(r["seed"]) for r in s),
                    "map50_mean": f"{statistics.mean(v50):.2f}",
                    "map50_sd": f"{statistics.stdev(v50):.2f}",
                    "map50_95_mean": f"{statistics.mean(v95):.2f}",
                    "map50_95_sd": f"{statistics.stdev(v95):.2f}",
                    "n_images": EXPECTED_IMAGES, "n_gt_boxes": EXPECTED_GT,
                    "classes_in_map": 8, "excluded": "Mosaic Virus (0 GT)",
                    "sd_type": "sample (n-1)"})
        check(f"{model}: summary from 3 seed values", len(s) == 3)
    with open(os.path.join(DST, "summary_cross_mean_std.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)

    failed = [c for c in checks if not c[1]]
    with open(os.path.join(DST, "validation_report.txt"), "w", encoding="utf-8") as fh:
        for n, ok, d in checks:
            fh.write(f"{'PASS' if ok else 'FAIL'}  {n}{('  -- ' + d) if (d and not ok) else ''}\n")
        fh.write(f"\n{len(checks) - len(failed)} passed, {len(failed)} failed\n")

    print(f"{'Model':<44} {'mAP@0.5':>16} {'mAP@0.5:0.95':>16}")
    for a in agg:
        print(f"{a['label']:<44} {a['map50_mean']:>8} +/- {a['map50_sd']:<5} "
              f"{a['map50_95_mean']:>8} +/- {a['map50_95_sd']:<5}")
    print(f"\nChecks: {len(checks) - len(failed)} passed, {len(failed)} failed")
    for n, ok, d in failed:
        print(f"  FAIL {n} {d}")
    print(f"Output: {DST}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
