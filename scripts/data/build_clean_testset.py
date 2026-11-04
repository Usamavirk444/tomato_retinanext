"""
build_clean_testset.py
──────────────────────
Builds a leakage-free version of the test split and everything needed to
evaluate on it, then prints the evaluation commands.

WHY THIS EXISTS
---------------
The source imagery was already augmented before the 70:20:10 split was
made, and the split was performed at the image level. Augmented variants
of one source photograph therefore landed in different splits: 554 of the
2,487 test images (22.3%) share a source photograph with the training set.
Those images do not measure generalisation, because the model saw the same
leaf during training.

This script identifies the 1,933 test images whose source photograph does
NOT appear in training, and assembles a self-contained split from them so
the existing evaluation pipeline can be pointed at it unchanged.

Roboflow encodes the augmentation in the filename:

    IMG_0421_jpg.rf.a3f9c1e08b7d4e2f9c1a5b3d7e9f2a4c.jpg
    └──── source ────┘   └──────── augmentation hash ────────┘

Stripping ".rf.<hash>" collapses every variant of one photograph to a
single key. Two files with the same key are the same leaf.

WHAT IT TOUCHES
---------------
Creates only new paths. It will refuse to overwrite an existing output
directory unless --force is given. It never modifies the original
dataset, the checkpoints, the configs, or any existing result.

    data/clean_test_only/test/images/     <- 1,933 images (copied)
    data/clean_test_only/test/labels/     <- their labels
    data/clean_test_only/data.yaml
    configs/experiment_cleantest.yaml
    outputs/clean_test/clean_test_images.txt
    outputs/clean_test/leaked_test_images.txt

USAGE
-----
    # look before you leap — reports the split, writes nothing
    python build_clean_testset.py --dry-run

    # build everything
    python build_clean_testset.py

    # rebuild over a previous run
    python build_clean_testset.py --force

Then run the 12 evaluations it prints at the end.
"""

import argparse
import os
import re
import shutil
import sys
import collections

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

SRC_DATASET = os.path.join(HERE, "data", "clean_yolo_dataset")
OUT_DATASET = os.path.join(HERE, "data", "clean_test_only")
SRC_CONFIG  = os.path.join(HERE, "configs", "experiment.yaml")
OUT_CONFIG  = os.path.join(HERE, "configs", "experiment_cleantest.yaml")
LIST_DIR    = os.path.join(HERE, "outputs", "clean_test")

IMG_EXT = (".jpg", ".jpeg", ".png")

CLASS_NAMES = [
    "Early_Blight", "Healthy", "Late_Blight", "Leaf_Miners", "Leaf_Mold",
    "Mosaic_Virus", "Septoria", "Spider_Mites", "Yellow_Leaf_Curl_Virus",
]

DISPLAY = [
    "Early Blight", "Healthy", "Late Blight", "Leaf Miners", "Leaf Mold",
    "Mosaic Virus", "Septoria", "Spider Mites", "Yellow Leaf Curl Virus",
]

MODELS = ["ssdlite", "yolov8n", "retinanet", "retinanet_hybrid"]
SEEDS  = [42, 123, 456]

# published full-test numbers, for comparison after the run
BASELINE = {
    "ssdlite":          "80.24%",
    "yolov8n":          "87.65%",
    "retinanet":        "89.16%",
    "retinanet_hybrid": "94.42%",
}


# ── source-key logic ─────────────────────────────────────────────────────────

def source_stem(filename):
    """Strip the Roboflow augmentation hash so every augmented variant of one
    source photograph collapses to the same key."""
    s = re.sub(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", filename, flags=re.I)
    return re.sub(r"\.(jpg|jpeg|png)$", "", s, flags=re.I)


def listdir_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def split_test_set():
    train_dir = os.path.join(SRC_DATASET, "train", "images")
    test_dir  = os.path.join(SRC_DATASET, "test",  "images")

    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: not found: {d}\n"
                     f"Run this script from the project root (D:\\virk\\tomato).")

    train_keys = {source_stem(f) for f in listdir_images(train_dir)}
    test_files = listdir_images(test_dir)

    clean  = [f for f in test_files if source_stem(f) not in train_keys]
    leaked = [f for f in test_files if source_stem(f) in train_keys]
    return clean, leaked, test_files


def annotation_counts(files):
    lbl_dir = os.path.join(SRC_DATASET, "test", "labels")
    per = collections.Counter()
    missing = 0
    for f in files:
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(lbl):
            missing += 1
            continue
        with open(lbl) as fh:
            for line in fh:
                parts = line.split()
                if parts:
                    per[int(parts[0])] += 1
    return per, missing


def report(clean, leaked, test_files):
    n = len(test_files)
    print()
    print("  LEAKAGE ANALYSIS OF THE TEST SPLIT")
    print("  " + "-" * 52)
    print(f"  test images total          {n:>6,}")
    print(f"  share a source with train  {len(leaked):>6,}   ({len(leaked)/n*100:.1f}%)")
    print(f"  leakage-free subset        {len(clean):>6,}   ({len(clean)/n*100:.1f}%)")

    per, missing = annotation_counts(clean)
    print()
    print("  annotations in the leakage-free subset:")
    for i, name in enumerate(DISPLAY):
        print(f"    {name:<24} {per[i]:>6,}")
    print(f"    {'TOTAL':<24} {sum(per.values()):>6,}")
    if missing:
        print(f"  note: {missing} image(s) had no label file")
    if not clean:
        sys.exit("\nERROR: no clean images found — check the filename pattern.")


# ── building ─────────────────────────────────────────────────────────────────

def write_lists(clean, leaked):
    os.makedirs(LIST_DIR, exist_ok=True)
    for name, files in (("clean_test_images.txt", clean),
                        ("leaked_test_images.txt", leaked)):
        path = os.path.join(LIST_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            for f in sorted(files):
                fh.write(f + "\n")
        print(f"  wrote {len(files):>6,} names -> {path}")


def copy_subset(clean, force):
    img_out = os.path.join(OUT_DATASET, "test", "images")
    lbl_out = os.path.join(OUT_DATASET, "test", "labels")

    if os.path.exists(OUT_DATASET):
        if not force:
            sys.exit(f"\nERROR: {OUT_DATASET} already exists.\n"
                     f"Re-run with --force to rebuild it.")
        shutil.rmtree(OUT_DATASET)

    os.makedirs(img_out)
    os.makedirs(lbl_out)

    img_src = os.path.join(SRC_DATASET, "test", "images")
    lbl_src = os.path.join(SRC_DATASET, "test", "labels")

    copied_img = copied_lbl = 0
    for f in clean:
        shutil.copy2(os.path.join(img_src, f), os.path.join(img_out, f))
        copied_img += 1
        stem = os.path.splitext(f)[0] + ".txt"
        s = os.path.join(lbl_src, stem)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(lbl_out, stem))
            copied_lbl += 1

    print(f"  copied {copied_img:,} images and {copied_lbl:,} labels")
    if copied_img != len(clean):
        sys.exit("ERROR: image count mismatch after copy")
    return img_out


def write_data_yaml():
    """One dataset root, three relative splits. train/val still point at the
    original data so Ultralytics can validate the file; only test differs."""
    data_root = os.path.join(HERE, "data").replace("\\", "/")
    path = os.path.join(OUT_DATASET, "data.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {data_root}\n")
        fh.write("train: clean_yolo_dataset/train/images\n")
        fh.write("val: clean_yolo_dataset/val/images\n")
        fh.write("test: clean_test_only/test/images\n")
        fh.write("\nnc: 9\nnames:\n")
        for i, n in enumerate(CLASS_NAMES):
            fh.write(f"  {i}: {n}\n")
    print(f"  wrote {path}")


def write_config():
    """Copy experiment.yaml, changing the three test-split keys AND every
    results_dir.

    The results_dir rewrite is not optional. evaluate.py writes its output to
    cfg['results_dir'], which in experiment.yaml is outputs/results/<model> —
    the directory holding the published full-test results behind Tables 1, 2
    and 3. Without this rewrite, evaluating on the clean subset would silently
    overwrite them. Redirecting to outputs/results_cleantest/<model> keeps the
    two sets of numbers side by side, which is what the paper needs anyway.
    """
    if not os.path.exists(SRC_CONFIG):
        sys.exit(f"ERROR: not found: {SRC_CONFIG}")

    # exactly-once keys
    repl = {
        "test_images:": '  test_images:  "data/clean_test_only/test/images"',
        "test_labels:": '  test_labels:  "data/clean_test_only/test/labels"',
        "yaml:":        '  yaml:         "data/clean_test_only/data.yaml"',
    }
    hits = dict.fromkeys(repl, 0)

    # results_dir appears once per model — redirect each, preserving the model
    # name so outputs stay separated per model.
    res_pat = re.compile(r'^(\s*)results_dir:\s*"?outputs/results/([^"\s]+)"?\s*$')
    n_results = 0

    out = []
    for line in open(SRC_CONFIG, encoding="utf-8").read().splitlines():
        m = res_pat.match(line)
        if m:
            indent, model = m.group(1), m.group(2)
            out.append(f'{indent}results_dir:      "outputs/results_cleantest/{model}"')
            n_results += 1
            continue

        stripped = line.strip()
        for key, new in repl.items():
            if stripped.startswith(key):
                out.append(new)
                hits[key] += 1
                break
        else:
            out.append(line)

    for key, n in hits.items():
        if n != 1:
            sys.exit(f"ERROR: expected exactly one '{key}' line in "
                     f"experiment.yaml, found {n}. Not writing the config.")
    if n_results == 0:
        sys.exit("ERROR: no results_dir lines matched. Refusing to write the "
                 "config — evaluating with it could overwrite outputs/results/.")

    with open(OUT_CONFIG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"  wrote {OUT_CONFIG}")
    print(f"        3 test-split keys redirected to data/clean_test_only")
    print(f"        {n_results} results_dir keys redirected to outputs/results_cleantest/")

    # prove it: no surviving reference to the published results directory
    text = open(OUT_CONFIG, encoding="utf-8").read()
    if 'outputs/results/' in text:
        sys.exit("ERROR: the written config still references outputs/results/ — "
                 "delete it and report this before running any evaluation.")
    print("        verified: no reference to outputs/results/ remains")


def print_commands():
    print()
    print("  NEXT: evaluate the four models on the leakage-free subset.")
    print("  No retraining — same checkpoints, fewer test images.")
    print()
    print("  PowerShell:")
    print()
    print('    foreach ($m in "ssdlite","yolov8n","retinanet","retinanet_hybrid") {')
    print("      foreach ($s in 42,123,456) {")
    print("        python evaluate.py --config configs/experiment_cleantest.yaml "
          "--model $m --seed $s")
    print("      }")
    print("    }")
    print()
    print("  Results land in outputs/results_cleantest/<model>/ — a NEW tree.")
    print("  Your published outputs/results/ is not touched.")
    print()
    print("  Compare each result against the published full-test number:")
    for m in MODELS:
        print(f"    {m:<20} {BASELINE[m]}")
    print()
    print("  The drop from those numbers is the measured leakage effect.")
    print("  A drop of a few points is normal and expected — report both.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build a leakage-free test split from the existing test set.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split and stop, writing nothing")
    ap.add_argument("--force", action="store_true",
                    help="rebuild data/clean_test_only if it already exists")
    ap.add_argument("--config-only", action="store_true",
                    help="rewrite configs/experiment_cleantest.yaml only, "
                         "leaving the already-copied images alone")
    args = ap.parse_args()

    if args.config_only:
        print("\n  --config-only: rewriting the config, touching nothing else")
        print("  " + "-" * 52)
        write_config()
        print_commands()
        return

    clean, leaked, test_files = split_test_set()
    report(clean, leaked, test_files)

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return

    print()
    print("  BUILDING")
    print("  " + "-" * 52)
    write_lists(clean, leaked)
    copy_subset(clean, args.force)
    write_data_yaml()
    write_config()
    print_commands()


if __name__ == "__main__":
    main()