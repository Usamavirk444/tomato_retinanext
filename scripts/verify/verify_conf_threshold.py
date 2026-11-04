"""
verify_conf_threshold.py
────────────────────────
Determines which confidence threshold produced the confusion matrices used
in the paper.

The question: paper_files/confusion_matrix.py declares two different
defaults —

    line  58   def build_confusion_matrix(..., conf_thresh=0.05)
    line 344   parser.add_argument("--conf-thresh", default=0.25)

— so a run that did not pass --conf-thresh explicitly used 0.25, while the
R pipeline's documentation (R_diagrams/00_setup.R:421-424) states 0.05.
No record of the original command survives. This script settles it by
rebuilding the matrix at both thresholds and diffing against the stored CSV.

WHY NOT JUST RUN confusion_matrix.py TWICE
    It writes to outputs/results/<model>/seed_<n>/confusion_matrix_test.csv
    with no way to redirect, so running it would destroy the very file we
    are trying to verify. This script imports its functions instead:
    build_confusion_matrix, load_model and collect_predictions_torchvision
    are used verbatim, so the arithmetic is identical to the original run.

    paper_files/confusion_matrix.py is NOT modified, imported-and-patched,
    or executed as a program. Nothing under outputs/results/ is written.

Predictions are collected ONCE and both thresholds applied to that same set,
so the only variable is conf_thresh.

Usage
    python verify_conf_threshold.py                       # hybrid, seed 42
    python verify_conf_threshold.py --model retinanet --seed 123
    python verify_conf_threshold.py --extra-thresh 0.1 0.3

Outputs  (new folder, nothing overwritten)
    outputs/conf_threshold_check/<model>_seed<n>_cm_conf<t>.csv
    outputs/conf_threshold_check/<model>_seed<n>_report.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

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
for p in (HERE, os.path.join(HERE, "paper_files")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Imported, never edited. These are the exact functions that built the
# published matrices.
# paper_files/ has no __init__.py, so load the module by file path rather
# than as a package. This neither creates nor requires any new file.
import importlib.util                                  # noqa: E402

_cm_path = os.path.join(HERE, "paper_files", "confusion_matrix.py")
_spec = importlib.util.spec_from_file_location("pf_confusion_matrix", _cm_path)
_cmmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cmmod)                        # noqa: E402

build_confusion_matrix = _cmmod.build_confusion_matrix
collect_predictions_torchvision = _cmmod.collect_predictions_torchvision
load_model = _cmmod.load_model
CLASS_LABELS_SHORT = _cmmod.CLASS_LABELS_SHORT
BG_LABEL = _cmmod.BG_LABEL


def read_stored(path):
    """Read a stored confusion_matrix_test.csv into an int array, ignoring
    the header row and the first column of row labels."""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    header = rows[0][1:]
    data = np.array([[int(float(x)) for x in r[1:]] for r in rows[1:]],
                    dtype=np.int64)
    return data, header, [r[0] for r in rows[1:]]


def write_cm(path, cm, labels):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["GT \\ Pred"] + labels)
        for i, lbl in enumerate(labels):
            w.writerow([lbl] + [int(v) for v in cm[i]])


def diff_report(name, cm, stored, labels):
    """Element-by-element comparison. Returns (n_mismatch, lines)."""
    lines = []
    if cm.shape != stored.shape:
        return -1, [f"  {name}: SHAPE MISMATCH {cm.shape} vs {stored.shape}"]
    d = cm.astype(np.int64) - stored
    n = int((d != 0).sum())
    tot_cm, tot_st = int(cm.sum()), int(stored.sum())
    lines.append(f"  {name}")
    lines.append(f"    mismatched cells : {n} of {d.size}")
    lines.append(f"    total detections : rebuilt {tot_cm}  stored {tot_st}  "
                 f"diff {tot_cm - tot_st:+d}")
    if n:
        idx = np.argwhere(d != 0)
        lines.append(f"    first {min(len(idx), 15)} differing cells "
                     f"(GT row -> Pred col : rebuilt vs stored):")
        for i, j in idx[:15]:
            lines.append(f"      {labels[i]:<24} -> {labels[j]:<24} "
                         f"{int(cm[i, j]):>6} vs {int(stored[i, j]):>6}")
        if len(idx) > 15:
            lines.append(f"      ... {len(idx) - 15} more")
    else:
        lines.append("    EXACT MATCH")
    return n, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="retinanet_hybrid",
                    choices=["ssdlite", "retinanet", "retinanet_hybrid"],
                    help="yolov8n excluded: different prediction path")
    ap.add_argument("--seed", type=int, default=42, choices=[42, 123, 456])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--extra-thresh", type=float, nargs="*", default=[],
                    help="additional thresholds to try if neither matches")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    stored_path = os.path.join(HERE, "outputs", "results", args.model,
                               f"seed_{args.seed}",
                               f"confusion_matrix_{args.split}.csv")
    if not os.path.exists(stored_path):
        raise SystemExit(f"stored matrix not found: {stored_path}")
    stored, header, row_labels = read_stored(stored_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model} seed {args.seed} split {args.split}")
    print(f"Stored : {stored_path}")

    split_images = os.path.join(HERE, "data", "clean_yolo_dataset",
                                args.split, "images")
    split_labels = os.path.join(HERE, "data", "clean_yolo_dataset",
                                args.split, "labels")
    if not os.path.isdir(split_images):
        raise SystemExit(f"split images not found: {split_images}")

    model, label_offset, kind = load_model(args.model, args.seed, device)
    if kind == "yolo":
        raise SystemExit("yolov8n uses a different prediction path; "
                         "not comparable here.")

    from src.dataset.yolo_dataset import build_yolo_dataloader
    from src.dataset.transforms import get_val_transforms
    loader = build_yolo_dataloader(
        images_dir=split_images, labels_dir=split_labels,
        transforms=get_val_transforms(size=(640, 640)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    print("\nCollecting predictions once ...")
    preds, tgts = collect_predictions_torchvision(
        model, loader, device, label_offset=label_offset)
    print(f"  {len(preds)} images, "
          f"{sum(len(p['scores']) for p in preds)} raw detections")

    labels = CLASS_LABELS_SHORT + [BG_LABEL]
    outdir = os.path.join(HERE, "outputs", "conf_threshold_check")
    os.makedirs(outdir, exist_ok=True)
    stem = f"{args.model}_seed{args.seed}"

    thresholds = [0.05, 0.25] + list(args.extra_thresh)
    results, lines = {}, []
    lines.append("CONFIDENCE-THRESHOLD VERIFICATION")
    lines.append("=" * 72)
    lines.append(f"model      {args.model}  seed {args.seed}  split {args.split}")
    lines.append(f"stored     {stored_path}")
    lines.append(f"iou_thresh {args.iou_thresh}")
    lines.append(f"images     {len(preds)}")
    lines.append(f"stored total detections {int(stored.sum())}")
    lines.append("")

    for t in thresholds:
        print(f"\nBuilding matrix at conf_thresh={t} ...")
        cm = build_confusion_matrix(preds, tgts,
                                    iou_thresh=args.iou_thresh,
                                    conf_thresh=t)
        write_cm(os.path.join(outdir, f"{stem}_cm_conf{t}.csv"), cm, labels)
        n, rep = diff_report(f"conf_thresh = {t}", cm, stored, labels)
        results[t] = n
        lines.extend(rep)
        lines.append("")

    exact = [t for t, n in results.items() if n == 0]
    lines.append("CONCLUSION")
    lines.append("-" * 72)
    if len(exact) == 1:
        lines.append(f"  The stored matrix was produced with "
                     f"conf_thresh = {exact[0]}.")
        lines.append("  Use that value in the figure caption.")
    elif len(exact) > 1:
        lines.append(f"  More than one threshold reproduces the matrix "
                     f"exactly: {exact}.")
        lines.append("  That means no detection falls between them; either "
                     "value is numerically correct, so quote the one that "
                     "was actually passed if it can be established.")
    else:
        best = min(results, key=lambda k: results[k])
        lines.append("  NO threshold reproduces the stored matrix exactly.")
        lines.append(f"  Closest is {best} with {results[best]} differing cells.")
        lines.append("  Something other than the confidence threshold "
                     "differs. Things to check next, in order:")
        lines.append("    - a different checkpoint (EMA vs raw weights)")
        lines.append("    - a different split or a changed dataset copy")
        lines.append("    - a different iou_thresh: rerun with "
                     "--extra-thresh and --iou-thresh")
        lines.append("    - torchvision version affecting NMS or box_iou")
        lines.append("  Do not assume a value; the caption should omit the "
                     "threshold until this is resolved.")

    report = os.path.join(outdir, f"{stem}_report.txt")
    open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {report}")
    print(f"wrote per-threshold matrices to {outdir}")
    print(f"\nstored file untouched: {stored_path}")


if __name__ == "__main__":
    main()
