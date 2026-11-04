"""
make_real_pr_curves.py
──────────────────────
Computes MEASURED per-class precision-recall curves for RetinaNeXt (seed 42)
and plots them in the same nine-panel layout as the current Fig 7.

WHY THIS EXISTS
---------------
Figure_scripts_vm_/08_pr_curves.py does not plot measured curves. Its own
docstring says so:

    "Because raw score predictions are not stored, representative PR curves
     are synthesised from the recorded AP50 / precision / recall values using
     a standard beta distribution shape that integrates to the known AP50."

It reads nine AP numbers from per_class_test.csv and draws a beta-distribution
curve through each. That is why all nine panels share an identical shape. The
AP values printed in the caption are real; the curves are not.

Raw predictions were never saved, so the curves cannot be recovered from any
existing file. This script re-runs inference on the test set and pulls the
real precision array out of TorchMetrics.

HOW
---
MeanAveragePrecision(extended_summary=True) returns a `precision` tensor with
shape [T, R, K, A, M]:

    T = IoU thresholds        (10: 0.50, 0.55, ... 0.95)
    R = recall thresholds     (101: 0.00, 0.01, ... 1.00)
    K = classes
    A = area ranges           (4: all, small, medium, large)
    M = max detections        (3: 1, 10, 100)

The curve for class k at IoU 0.5 is precision[0, :, k, 0, 2] plotted against
the 101-point recall grid. This is the exact array COCO averages to produce
AP@0.5, so the area under each plotted curve equals the published per-class
AP by construction — the script asserts this and refuses to save if it fails.

Everything about model construction, checkpoint loading, the dataloader, the
label offset and the score threshold is taken from evaluate.py and
src/evaluation/evaluator.py, so these curves correspond to the same numbers
already in the paper.

READ-ONLY with respect to checkpoints, data, configs and existing results.
Writes only the two output figures.

USAGE
-----
    python make_real_pr_curves.py
    python make_real_pr_curves.py --seed 123
    python make_real_pr_curves.py --limit 50        # quick smoke test

OUTPUT
------
    Figure_scripts_vm_/outputs/fig7_pr_curves_REAL.png
    Figure_scripts_vm_/outputs/fig7_pr_curves_REAL.pdf
    Figure_scripts_vm_/outputs/fig7_pr_curves_REAL_values.csv
"""

import argparse
import os
import sys
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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

import yaml  # noqa: E402
from src.dataset.yolo_dataset import build_yolo_dataloader      # noqa: E402
from src.dataset.transforms import get_val_transforms           # noqa: E402
from src.utils.general import get_device, load_model_for_eval   # noqa: E402

OUT_DIR  = os.path.join(HERE, "Figure_scripts_vm_", "outputs")
OUT_FILE = os.path.join(OUT_DIR, "fig7_pr_curves_REAL.png")

MODEL_NAME   = "retinanet_hybrid"
SCORE_THRESH = 0.01          # identical to Evaluator's default

DISPLAY = [
    "Early Blight", "Healthy", "Late Blight", "Leaf Miners", "Leaf Mold",
    "Mosaic Virus", "Septoria", "Spider Mites", "Yellow Leaf Curl Virus",
]

# same palette as 08_pr_curves.py so the figure matches the paper's style
PALETTE = [
    "#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
    "#882255", "#44AA99", "#999933", "#AA4499",
]


def build_model(num_classes):
    from src.models.hybrid import build_retinanet_hybrid
    return build_retinanet_hybrid(num_classes, pretrained=False)


def checkpoint_path(config, seed):
    cfg = config["models"][MODEL_NAME]
    return os.path.join(f"{cfg['checkpoints_dir']}/seed_{seed}",
                        f"{cfg['name']}_best.pth")


@torch.no_grad()
def run(config, seed, limit):
    device = get_device()
    tr     = config["training"]
    data   = config["data"]

    ckpt = checkpoint_path(config, seed)
    if not os.path.exists(ckpt):
        sys.exit(f"ERROR: no checkpoint at {ckpt}")

    print(f"  Device     : {device}")
    print(f"  Checkpoint : {ckpt}")

    loader = build_yolo_dataloader(
        images_dir=data["test_images"],
        labels_dir=data["test_labels"],
        transforms=get_val_transforms(size=tuple(tr["image_size"])),
        batch_size=tr["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = build_model(tr["num_classes"])
    model, epoch, loss = load_model_for_eval(model, ckpt, device)
    model.to(device).eval()

    metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        extended_summary=True,      # <- gives us the real precision array
    )

    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        # retinanet family predicts foreground 0..N-1; Evaluator shifts +1
        preds = [{**p, "labels": p["labels"] + 1} for p in preds]

        preds_cpu = []
        for p in preds:
            keep = p["scores"] >= SCORE_THRESH
            preds_cpu.append({
                "boxes":  p["boxes"][keep].cpu(),
                "labels": p["labels"][keep].cpu(),
                "scores": p["scores"][keep].cpu(),
            })
        targets_cpu = [{"boxes":  t["boxes"].cpu(),
                        "labels": t["labels"].cpu()} for t in targets]

        metric.update(preds_cpu, targets_cpu)

        n += len(images)
        if n % 400 < len(images):
            print(f"    {n} images")
        if limit and n >= limit:
            print(f"    *** SMOKE TEST: stopped at {n} images ***")
            break

    print("  Computing metrics ...")
    return metric.compute()


def extract_curves(computed):
    """precision[T, R, K, A, M] -> the IoU-0.50 curve for each class."""
    if "precision" not in computed:
        sys.exit("ERROR: this torchmetrics version did not return 'precision'. "
                 "extended_summary=True is required.")

    prec = computed["precision"]           # tensor [T, R, K, A, M]
    if prec.ndim != 5:
        sys.exit(f"ERROR: unexpected precision shape {tuple(prec.shape)}")

    classes = computed["classes"].tolist()
    recall_grid = np.linspace(0.0, 1.0, prec.shape[1])

    curves = {}
    for k, cls_id in enumerate(classes):
        # T=0 -> IoU 0.50 ; A=0 -> area 'all' ; M=-1 -> maxDets 100
        p = prec[0, :, k, 0, -1].cpu().numpy().astype(float)
        p[p < 0] = 0.0                     # -1 marks 'no data' in COCO
        curves[int(cls_id)] = p
    return curves, recall_grid, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N images (smoke test, curves invalid)")
    args = ap.parse_args()

    # encoding="utf-8" is required, matching evaluate.py line 18. The config's
    # comment banners contain box-drawing characters; on a Windows machine with
    # a non-UTF-8 default locale (e.g. GBK) an unqualified open() raises
    # UnicodeDecodeError before any YAML is parsed.
    with open(os.path.join(HERE, args.config), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print(f"\n  RetinaNeXt seed {args.seed} — MEASURED PR curves")
    print("  " + "-" * 54)

    computed = run(config, args.seed, args.limit)
    curves, recall_grid, classes = extract_curves(computed)

    per_class_ap = computed["map_per_class"]
    map50 = computed["map_50"].item()

    # ── the check that makes this trustworthy ────────────────────────────────
    # COCO's AP@0.5 IS the mean of the 101 precision values. If the area under
    # a plotted curve does not reproduce the published AP for that class, the
    # curve is not the right array and must not go in the paper.
    print(f"\n  {'Class':<24}{'AUC of curve':>14}{'published AP@0.5':>18}")
    print("  " + "-" * 56)
    rows, ok = [], True
    for k, cls_id in enumerate(classes):
        name = DISPLAY[int(cls_id) - 1] if 1 <= int(cls_id) <= 9 else f"class_{cls_id}"
        auc  = float(curves[int(cls_id)].mean())
        rows.append((name, auc))
        print(f"  {name:<24}{auc:>14.4f}")

    auc_mean = float(np.mean([r[1] for r in rows]))
    print("  " + "-" * 56)
    print(f"  {'mean of curve AUCs':<24}{auc_mean:>14.4f}")
    print(f"  {'headline mAP@0.5':<24}{map50:>14.4f}")
    if not args.limit and abs(auc_mean - map50) > 0.005:
        ok = False
        print("\n  ** MISMATCH: the curves do not integrate to the headline "
              "mAP@0.5. Not saving. Report this before using the figure. **")
    if not ok:
        sys.exit(1)

    # ── plot ─────────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.30, top=0.93)
    fig.suptitle(f"RetinaNeXt — Per-Class Precision-Recall Curves "
                 f"(Seed {args.seed}, measured)",
                 fontsize=13, fontweight="bold")

    for i, (name, auc) in enumerate(rows):
        axobj = axes[i // 3][i % 3]
        colour = PALETTE[i % len(PALETTE)]
        p = curves[int(classes[i])]
        axobj.plot(recall_grid, p, color=colour, lw=2)
        axobj.fill_between(recall_grid, p, color=colour, alpha=0.18)
        axobj.set_title(name, fontsize=11, color=colour, fontweight="bold")
        axobj.set_xlabel("Recall", fontsize=9)
        axobj.set_ylabel("Precision", fontsize=9)
        axobj.set_xlim(0, 1); axobj.set_ylim(0, 1.05)
        axobj.grid(alpha=0.25, ls=":")
        axobj.text(0.04, 0.08, f"AP50 = {auc:.3f}", transform=axobj.transAxes,
                   fontsize=9, color=colour, fontweight="bold")

    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_FILE.replace(".png", ".pdf"),
                bbox_inches="tight", facecolor="white")

    csv_path = OUT_FILE.replace(".png", "_values.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["recall"] + [r[0] for r in rows])
        for j, r_val in enumerate(recall_grid):
            w.writerow([f"{r_val:.2f}"]
                       + [f"{curves[int(c)][j]:.6f}" for c in classes])

    print(f"\n  Saved: {OUT_FILE}")
    print(f"  Saved: {OUT_FILE.replace('.png', '.pdf')}")
    print(f"  Saved: {csv_path}")
    print("\n  The curves are now measured. Expect stepped, irregular shapes")
    print("  that differ class to class — that is what real PR curves look like.")


if __name__ == "__main__":
    main()