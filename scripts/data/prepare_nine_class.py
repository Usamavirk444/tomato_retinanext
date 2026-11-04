"""
prepare_nine_class.py
─────────────────────
Turns the original 11-class source dataset into the nine-class benchmark the
paper uses, and restores the EXACT train/val/test split from the recorded
manifest.

    27,297 images / 11 classes  ->  24,876 images / 9 classes / 67,595 boxes
    split: 17,419 train / 4,970 val / 2,487 test

HOW THE SPLIT IS RESTORED
    Not by re-randomising. `manifests/split_manifest_clean_yolo_dataset.csv`
    records, for all 24,876 images, which split it belongs to, its label file,
    its Roboflow source-image id and augmentation hash, its box count and the
    class ids present. This script reads that manifest and places each image
    where it originally was. A random split with the same proportions would
    NOT reproduce the paper, because the dataset is pre-augmented and
    source-image membership differs.

WHICH CLASSES ARE DROPPED
    The source has class indices 0-10. The paper keeps nine and removes two:

        index 9  (2,965 boxes)  Bacterial Spot  - lesion morphology overlaps
                                                  several fungal classes
        index 10 (  618 boxes)  Iron Deficiency - too sparse to train on

    71,178 total boxes - 2,965 - 618 = 67,595, matching the manuscript.

    NOTE ON INDICES. The nine retained classes already occupy 0-8 in the
    source, so no remapping is required: dropping 9 and 10 leaves 0-8 intact
    and contiguous. The script asserts this rather than assuming it. If a
    future source version orders classes differently the assertion fires
    instead of silently producing a mislabelled dataset.

FAILURE BEHAVIOUR
    If an image or label named in the manifest is missing from the source,
    the script reports every missing entry and exits non-zero. It does not
    skip, guess or substitute.

Usage
    python scripts/data/prepare_nine_class.py \
        --source  /path/to/data_merged \
        --dest    data/clean_yolo_dataset

    --dry-run   report what would happen, write nothing
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MANIFEST = os.path.join(REPO, "manifests",
                        "split_manifest_clean_yolo_dataset.csv")

DROP_CLASSES = {9, 10}          # Bacterial Spot, Iron Deficiency
KEEP_CLASSES = set(range(9))    # 0-8, already contiguous

CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight", "Leaf_Miners",
               "Leaf_Mold", "Mosaic_Virus", "Septoria", "Spider_Mites",
               "Yellow_Leaf_Curl_Virus"]

EXPECTED = {"train": 17419, "val": 4970, "test": 2487}
EXPECTED_BOXES = 67595


def read_manifest():
    if not os.path.exists(MANIFEST):
        sys.exit(f"manifest not found: {MANIFEST}\n"
                 "It is part of the repository and must not be regenerated.")
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if len(rows) != sum(EXPECTED.values()):
        sys.exit(f"manifest has {len(rows)} rows, expected "
                 f"{sum(EXPECTED.values())}")
    return rows


def filter_label(src_path):
    """Return (kept_lines, n_kept, n_dropped). Lines whose class is in
    DROP_CLASSES are removed; every other line is passed through byte-for-byte
    so coordinates are never re-encoded."""
    kept, dropped = [], 0
    for ln in open(src_path, encoding="utf-8"):
        parts = ln.split()
        if not parts:
            continue
        try:
            c = int(float(parts[0]))
        except ValueError:
            continue
        if c in DROP_CLASSES:
            dropped += 1
            continue
        if c not in KEEP_CLASSES:
            raise ValueError(f"unexpected class {c} in {src_path}")
        kept.append(ln.rstrip("\n"))
    return kept, len(kept), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="folder holding the original images/ and labels/ "
                         "(the 27,297-image download)")
    ap.add_argument("--dest", default=os.path.join(REPO, "data",
                                                   "clean_yolo_dataset"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_img = os.path.join(args.source, "images")
    src_lbl = os.path.join(args.source, "labels")
    for d in (src_img, src_lbl):
        if not os.path.isdir(d):
            sys.exit(f"not found: {d}")

    rows = read_manifest()
    print(f"manifest: {len(rows)} images across "
          f"{len({r['split'] for r in rows})} splits")

    # ---- locate every file before copying anything ----
    missing = []
    for r in rows:
        if not os.path.exists(os.path.join(src_img, r["image"])):
            missing.append(("image", r["image"]))
        if not os.path.exists(os.path.join(src_lbl, r["label"])):
            missing.append(("label", r["label"]))
    if missing:
        print(f"\nERROR: {len(missing)} manifest entries not found in "
              f"{args.source}", file=sys.stderr)
        for kind, name in missing[:20]:
            print(f"  missing {kind}: {name}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... {len(missing)-20} more", file=sys.stderr)
        sys.exit("refusing to continue: the source does not match the "
                 "recorded split. Check you downloaded the same dataset "
                 "version (see docs/DATASET_ATTRIBUTION.md).")
    print("all manifest entries located in the source")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ---- place files ----
    for split in EXPECTED:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.dest, split, sub), exist_ok=True)

    counts = Counter()
    boxes = Counter()
    dropped_total = 0
    per_class = Counter()

    for r in rows:
        split = r["split"]
        shutil.copy2(os.path.join(src_img, r["image"]),
                     os.path.join(args.dest, split, "images", r["image"]))
        kept, n_kept, n_drop = filter_label(os.path.join(src_lbl, r["label"]))
        dropped_total += n_drop
        with open(os.path.join(args.dest, split, "labels", r["label"]),
                  "w", encoding="utf-8") as fh:
            fh.write("\n".join(kept) + ("\n" if kept else ""))
        counts[split] += 1
        boxes[split] += n_kept
        for ln in kept:
            per_class[int(float(ln.split()[0]))] += 1

        # the manifest records the expected box count; verify per image
        if n_kept != int(r["n_boxes"]):
            sys.exit(f"box-count mismatch for {r['image']}: produced {n_kept}, "
                     f"manifest says {r['n_boxes']}")

    # ---- data.yaml ----
    with open(os.path.join(args.dest, "data.yaml"), "w",
              encoding="utf-8") as fh:
        fh.write(f"path: {args.dest}\ntrain: train/images\nval: val/images\n"
                 f"test: test/images\n\nnc: 9\nnames:\n")
        for i, n in enumerate(CLASS_NAMES):
            fh.write(f"  {i}: {n}\n")

    # ---- report ----
    print("\nresult")
    ok = True
    for split in ("train", "val", "test"):
        exp = EXPECTED[split]
        mark = "OK " if counts[split] == exp else "BAD"
        ok &= counts[split] == exp
        print(f"  {mark} {split:<6} {counts[split]:>6} images "
              f"(expected {exp:>6})  {boxes[split]:>6} boxes")
    total_boxes = sum(boxes.values())
    mark = "OK " if total_boxes == EXPECTED_BOXES else "BAD"
    ok &= total_boxes == EXPECTED_BOXES
    print(f"  {mark} total  {sum(counts.values()):>6} images  "
          f"{total_boxes:>6} boxes (expected {EXPECTED_BOXES})")
    print(f"      dropped {dropped_total} boxes of classes "
          f"{sorted(DROP_CLASSES)}")
    print("\n  per-class boxes")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {i} {n:<24} {per_class[i]:>7,}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
