"""
build_clean_crossdataset.py
───────────────────────────
Builds data/cross_dataset_3_clean/: the external benchmark (Roboflow
"Tomatoes_diseases", 2,396 images) with every image removed that also
appears in our own dataset (data/clean_yolo_dataset, any split).

An external image is REMOVED when either
  1. its source file name (Roboflow name without ".rf.<hash>") matches an
     image in our dataset                                  -> 1,273 images
  2. it is a renamed copy: 16x16 greyscale thumbnail differs from one of
     our images (incl. flips / 90-degree rotations) by < 1 grey level
                                                           ->    72 images
Removed: 1,345 images.  Kept: 1,051 (train 1,002 / valid 41 / test 8).

The keep-list was computed once and is read from
    outputs/audit_external_overlap/kept_external_images.txt
The removal evidence is in
    outputs/audit_external_overlap/removed_external_images.csv

Only creates new paths. Refuses to overwrite data/cross_dataset_3_clean.
Never touches data/cross_dataset_3 or anything else.

Usage (from the project root):
    python build_clean_crossdataset.py
"""
import os
import shutil
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
SRC = os.path.join(ROOT, "data", "cross_dataset_3")
DST = os.path.join(ROOT, "data", "cross_dataset_3_clean")
KEEP = os.path.join(ROOT, "outputs", "audit_external_overlap",
                    "kept_external_images.txt")
EXPECTED = 1051


def main():
    if not os.path.isdir(SRC):
        sys.exit(f"ERROR: source not found: {SRC}")
    if not os.path.isfile(KEEP):
        sys.exit(f"ERROR: keep-list not found: {KEEP}")
    if os.path.exists(DST):
        sys.exit(f"ERROR: {DST} already exists - refusing to overwrite.")

    keep = [ln.strip() for ln in open(KEEP, encoding="utf-8") if ln.strip()]
    if len(keep) != EXPECTED:
        sys.exit(f"ERROR: keep-list has {len(keep)} entries, expected {EXPECTED}.")

    missing = []
    for rel in keep:
        split, fname = rel.split("/", 1)
        img = os.path.join(SRC, split, "images", fname)
        lbl = os.path.join(SRC, split, "labels", os.path.splitext(fname)[0] + ".txt")
        if not os.path.isfile(img) or not os.path.isfile(lbl):
            missing.append(rel)
    if missing:
        sys.exit(f"ERROR: {len(missing)} listed files are missing in {SRC}, "
                 f"e.g. {missing[:3]}. Nothing was copied.")

    counts = {}
    for rel in keep:
        split, fname = rel.split("/", 1)
        stem = os.path.splitext(fname)[0]
        for sub, name in (("images", fname), ("labels", stem + ".txt")):
            os.makedirs(os.path.join(DST, split, sub), exist_ok=True)
            shutil.copy2(os.path.join(SRC, split, sub, name),
                         os.path.join(DST, split, sub, name))
        counts[split] = counts.get(split, 0) + 1
    shutil.copy2(os.path.join(SRC, "data.yaml"), os.path.join(DST, "data.yaml"))
    with open(os.path.join(DST, "README_CLEAN.txt"), "w", encoding="utf-8") as fh:
        fh.write("Overlap-free subset of data/cross_dataset_3 "
                 "(built by build_clean_crossdataset.py).\n"
                 f"Kept {len(keep)} of 2396 images: {counts}.\n"
                 "See outputs/audit_external_overlap/ for the evidence.\n")

    print(f"Built {DST}")
    print(f"  kept {len(keep)} images: {counts}")


if __name__ == "__main__":
    main()
