"""
filter_external_evaluable.py
────────────────────────────
Stage 2 of the external benchmark filter: 1,051 kept images -> 805 evaluable.

WHY THIS SCRIPT EXISTS
    Stage 1 (scripts/data/build_clean_crossdataset.py) removes the 1,345
    images that share a source photograph with training, leaving 1,051. The
    paper then reports 805 evaluable images with 7,307 boxes over eight
    classes. That second stage was never a separate script in the original
    project: it happened implicitly inside the evaluator, which skips any
    image with no box of an overlap class.

    Running it implicitly is fine for evaluation but leaves the 805 figure
    unverifiable without loading a model. This script applies the identical
    rule explicitly, writes the manifest, and checks the counts.

THE RULE, taken verbatim from cross_eval/cross_dataset_3_eval.py
    An image is evaluable when it has at least one box that is
      (a) of a class in OBB_OVERLAP = {1,2,4,5,6,7,8,9,10}
          i.e. one of the nine classes shared with training. Class 0 is
          Bacterial Spot, which was never trained, so Bacterial-Spot-only
          images drop out here; and
      (b) non-degenerate after the oriented box is converted to an
          axis-aligned rectangle and clipped to the image:
              min(1, max(xs)) > max(0, min(xs))  and the same for ys.

    Both conditions are copied from `load_cross_dataset_3()` so this script
    cannot drift from the evaluator.

MOSAIC VIRUS
    Has zero ground-truth boxes remaining after stage 1, so it is excluded
    from the external mAP, leaving eight evaluable classes. This script
    reports it rather than assuming it.

Usage
    python scripts/data/filter_external_evaluable.py \
        --dataset data/cross_dataset_3_clean

    --write-manifest   also write manifests/external_evaluable_805.txt
    --out DIR          write the manifest somewhere else (for testing)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.exists(os.path.join(REPO, "train_all.py")) \
        and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)

# identical to cross_eval/cross_dataset_3_eval.py
OBB_OVERLAP = {1, 2, 4, 5, 6, 7, 8, 9, 10}
OBB_TO_OURS = {1: 1, 2: 2, 4: 3, 5: 5, 6: 4, 7: 6, 8: 7, 9: 8, 10: 9}
OUR_DISPLAY = {1: "Early Blight", 2: "Healthy", 3: "Late Blight",
               4: "Leaf Miners", 5: "Leaf Mold", 6: "Mosaic Virus",
               7: "Septoria", 8: "Spider Mites", 9: "Yellow Leaf Curl Virus"}

EXPECTED_INPUT = 1051
EXPECTED_EVALUABLE = 805
EXPECTED_BOXES = 7307
EXPECTED_CLASSES = 8


def parse_obb_line(line):
    """Return (class_index, kept) applying the evaluator's degeneracy test."""
    p = line.split()
    if len(p) < 9:
        return None, False
    try:
        c = int(float(p[0]))
        co = [float(x) for x in p[1:9]]
    except ValueError:
        return None, False
    if c not in OBB_OVERLAP:
        return c, False
    xs, ys = co[::2], co[1::2]
    if min(1, max(xs)) <= max(0, min(xs)) or min(1, max(ys)) <= max(0, min(ys)):
        return c, False            # degenerate after OBB -> AABB + clip
    return c, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=os.path.join(REPO, "data", "cross_dataset_3_clean"))
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "manifests"))
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        sys.exit(f"dataset not found: {args.dataset}\n"
                 "Run scripts/data/build_clean_crossdataset.py first.")

    total = evaluable = boxes = 0
    per_class = Counter()
    excluded_bs_only = 0
    excluded_degenerate = 0
    excluded_other = 0
    kept_names = []

    for split in ("train", "valid", "test"):
        idir = os.path.join(args.dataset, split, "images")
        ldir = os.path.join(args.dataset, split, "labels")
        if not os.path.isdir(idir):
            continue
        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            total += 1
            lp = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
            n_ok = 0
            classes_seen = set()
            had_overlap_class = False
            if os.path.exists(lp):
                for ln in open(lp, encoding="utf-8"):
                    c, ok = parse_obb_line(ln)
                    if c is None:
                        continue
                    classes_seen.add(c)
                    if c in OBB_OVERLAP:
                        had_overlap_class = True
                    if ok:
                        n_ok += 1
                        per_class[OBB_TO_OURS[c]] += 1
            if n_ok:
                evaluable += 1
                boxes += n_ok
                kept_names.append(f"{split}/{fn}")
            elif classes_seen and classes_seen <= {0}:
                excluded_bs_only += 1
            elif had_overlap_class:
                excluded_degenerate += 1
            else:
                excluded_other += 1

    # ── report ──────────────────────────────────────────────────────
    print("external benchmark, stage 2")
    print(f"  dataset               {args.dataset}")
    print(f"  input images          {total:>6}   (expected {EXPECTED_INPUT})")
    print(f"  evaluable images      {evaluable:>6}   (expected {EXPECTED_EVALUABLE})")
    print(f"  ground-truth boxes    {boxes:>6}   (expected {EXPECTED_BOXES})")
    print()
    print(f"  excluded, Bacterial-Spot-only   {excluded_bs_only:>5}")
    print(f"  excluded, all boxes degenerate  {excluded_degenerate:>5}")
    print(f"  excluded, no usable label       {excluded_other:>5}")
    print()
    print("  per-class ground truth")
    zero = []
    for i in sorted(OUR_DISPLAY):
        n = per_class.get(i, 0)
        flag = "   <- excluded from mAP (zero GT)" if n == 0 else ""
        if n == 0:
            zero.append(OUR_DISPLAY[i])
        print(f"    {i} {OUR_DISPLAY[i]:<24} {n:>6,}{flag}")
    n_classes = sum(1 for i in OUR_DISPLAY if per_class.get(i, 0) > 0)
    print(f"\n  classes with ground truth  {n_classes}   "
          f"(expected {EXPECTED_CLASSES})")
    if zero:
        print(f"  zero-GT classes: {', '.join(zero)}")

    checks = [("input images", total, EXPECTED_INPUT),
              ("evaluable images", evaluable, EXPECTED_EVALUABLE),
              ("ground-truth boxes", boxes, EXPECTED_BOXES),
              ("classes in mAP", n_classes, EXPECTED_CLASSES)]
    print()
    ok = True
    for name, got, exp in checks:
        good = got == exp
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<22} {got} vs {exp}")

    if args.write_manifest:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "external_evaluable_805.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(kept_names) + "\n")
        print(f"\n  wrote {p}  ({len(kept_names)} entries)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
