"""
Dataset-preparation tests using small synthetic fixtures in temp directories.

These exercise the real scripts' logic against data we construct, so the
behaviour is checked without the 3.3 GB download and without touching
anything in the repository.

    python -m pytest tests/test_dataset_preparation.py -v
    python tests/test_dataset_preparation.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(f"{name}: {detail}")
    return cond


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── class remapping ─────────────────────────────────────────────────
def test_class_removal_and_indices():
    """Classes 9 and 10 are dropped; 0-8 pass through untouched."""
    prep = load("scripts/data/prepare_nine_class.py", "prep")
    check("drops exactly classes 9 and 10", prep.DROP_CLASSES == {9, 10},
          str(prep.DROP_CLASSES))
    check("keeps 0-8", prep.KEEP_CLASSES == set(range(9)))
    check("no index overlap", not (prep.DROP_CLASSES & prep.KEEP_CLASSES))
    check("9 class names", len(prep.CLASS_NAMES) == 9,
          str(len(prep.CLASS_NAMES)))

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.txt")
        open(p, "w").write(
            "0 0.5 0.5 0.2 0.2\n"      # keep
            "9 0.1 0.1 0.1 0.1\n"      # drop  (Bacterial Spot)
            "8 0.9 0.9 0.1 0.1\n"      # keep
            "10 0.2 0.2 0.1 0.1\n")    # drop  (Iron Deficiency)
        kept, n_kept, n_drop = prep.filter_label(p)
        check("2 kept, 2 dropped", (n_kept, n_drop) == (2, 2),
              f"{n_kept},{n_drop}")
        check("kept classes are 0 and 8",
              [l.split()[0] for l in kept] == ["0", "8"])
        check("coordinates passed through byte-for-byte",
              kept[0] == "0 0.5 0.5 0.2 0.2")

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "b.txt")
        open(p, "w").write("11 0.5 0.5 0.2 0.2\n")   # out of range
        try:
            prep.filter_label(p)
            check("unexpected class raises", False, "no exception")
        except ValueError:
            check("unexpected class raises", True)


# ── missing-file detection ──────────────────────────────────────────
def test_missing_file_detection():
    """prepare_nine_class must refuse to proceed, not guess."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        os.makedirs(os.path.join(src, "images"))
        os.makedirs(os.path.join(src, "labels"))
        # deliberately empty: every manifest entry is missing
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "data",
                                          "prepare_nine_class.py"),
             "--source", src, "--dest", os.path.join(td, "out"), "--dry-run"],
            capture_output=True, text=True)
        check("exits non-zero when files are missing", r.returncode != 0,
              f"rc={r.returncode}")
        check("names the missing files",
              "missing image" in r.stderr or "not found in" in r.stderr)
        check("refuses rather than guessing", "refusing to continue" in r.stderr)
        check("wrote nothing", not os.path.exists(os.path.join(td, "out")))


# ── split manifest integrity ────────────────────────────────────────
def test_split_manifest_fields():
    p = os.path.join(REPO, "manifests", "split_manifest_clean_yolo_dataset.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    need = {"split", "image", "label", "source_image_id",
            "augmentation_id", "n_boxes", "classes"}
    check("manifest has all required columns", need <= set(rows[0]),
          str(need - set(rows[0])))
    check("every image name unique",
          len({r["image"] for r in rows}) == len(rows))
    check("label stem matches image stem",
          all(os.path.splitext(r["image"])[0] == os.path.splitext(r["label"])[0]
              for r in rows[:500]))
    check("source ids recorded (pre-augmentation grouping)",
          len({r["source_image_id"] for r in rows}) < len(rows),
          "no augmentation grouping found")
    bad = [r for r in rows if int(r["n_boxes"]) < 1]
    check("no image has zero boxes", not bad, f"{len(bad)} with 0")
    cls = {int(c) for r in rows for c in r["classes"].split(";") if c}
    check("only classes 0-8 present", cls == set(range(9)), str(sorted(cls)))


# ── external filtering ──────────────────────────────────────────────
def test_external_filter_rule():
    """The stage-2 rule must match the evaluator's, including degeneracy."""
    f = load("scripts/data/filter_external_evaluable.py", "extfilt")
    check("overlap classes exclude 0 (Bacterial Spot)",
          0 not in f.OBB_OVERLAP and len(f.OBB_OVERLAP) == 9,
          str(sorted(f.OBB_OVERLAP)))

    c, ok = f.parse_obb_line("1 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5")
    check("valid overlap box kept", (c, ok) == (1, True), f"{c},{ok}")
    c, ok = f.parse_obb_line("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5")
    check("Bacterial Spot box rejected", (c, ok) == (0, False), f"{c},{ok}")
    c, ok = f.parse_obb_line("1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1")
    check("degenerate box rejected", ok is False)
    c, ok = f.parse_obb_line("1 0.5 0.5")
    check("malformed line rejected", (c, ok) == (None, False))
    check("expected counts encoded",
          (f.EXPECTED_INPUT, f.EXPECTED_EVALUABLE, f.EXPECTED_BOXES,
           f.EXPECTED_CLASSES) == (1051, 805, 7307, 8))


# ── evaluation output format ────────────────────────────────────────
def test_reference_csv_formats():
    ref = os.path.join(REPO, "reference_results")
    p = os.path.join(ref, "in_domain", "retinanet_hybrid",
                     "retinanet_hybrid_seed_42_eval_results.csv")
    if os.path.exists(p):
        r = next(csv.DictReader(open(p, encoding="utf-8")))
        need = {"model", "mAP_50", "mAP_50_95"}
        check("in-domain CSV has the expected fields", need <= set(r),
              str(need - set(r)))
        check("mAP_50 is a 0-1 fraction", 0 <= float(r["mAP_50"]) <= 1)
    else:
        check("in-domain reference CSV present", False, p)

    p = os.path.join(ref, "external", "summary_cross_mean_std.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    need = {"model", "map50_mean", "map50_sd", "n_images", "n_gt_boxes",
            "classes_in_map", "sd_type"}
    check("external summary has the expected fields", need <= set(rows[0]),
          str(need - set(rows[0])))
    check("external summary covers 8 configurations", len(rows) == 8,
          str(len(rows)))
    check("sample SD is declared",
          all("sample" in r["sd_type"] for r in rows))


# ── figure/table input dependencies ─────────────────────────────────
def test_figure_inputs_exist():
    mk = load("scripts/figures/make_manuscript_figures.py", "mk")
    checks = mk.check_sources()
    check("six manuscript figures declared", len(checks) == 6, str(len(checks)))
    missing = [n for n, ok, _ in checks if not ok]
    # Figure 1 needs the dataset, which is not shipped - that is expected
    unexpected = [m for m in missing if m != "Figure1_dataset_composition"]
    check("all figure sources present except the dataset-dependent one",
          not unexpected, str(unexpected))


def test_table_inputs_exist():
    bt = load("scripts/eval/build_tables.py", "bt")
    t1, _ = bt.table1()
    check("Table 1 builds, 4 rows", t1 and len(t1) == 4,
          str(len(t1) if t1 else 0))
    t2, _ = bt.table2()
    check("Table 2 builds, 6 stages", t2 and len(t2) == 6,
          str(len(t2) if t2 else 0))
    check("Table 2 has no 'n/a' in-domain cells",
          all(r["In-domain mAP@0.5"] != "n/a" for r in t2))
    t3, _ = bt.table3()
    check("Table 3 builds, 8 models", t3 and len(t3) == 8,
          str(len(t3) if t3 else 0))
    check("Table 3 excludes Mosaic Virus (zero GT)",
          t3 and "Mosaic Virus" not in t3[0])


def test_superseded_data_not_default():
    """No script may read the superseded May results table."""
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "audit_history",
                                                "__pycache__", "tests")]
        for f in files:
            # this test file names the string itself; skip it
            if f.endswith((".py", ".R")):
                p = os.path.join(root, f)
                if "SUPERSEDED" in open(p, encoding="utf-8",
                                        errors="ignore").read():
                    hits.append(os.path.relpath(p, REPO))
    check("no script references superseded data", not hits, str(hits))
    check("superseded table is outside reference_results",
          os.path.exists(os.path.join(REPO, "audit_history",
                                      "main_results_table_SUPERSEDED_2026-05.csv"))
          and not os.path.exists(os.path.join(
              REPO, "reference_results", "in_domain",
              "main_results_table_SUPERSEDED_2026-05.csv")))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"\n{name}")
            try:
                fn()
            except Exception as e:                       # noqa: BLE001
                check(f"{name} raised", False, f"{type(e).__name__}: {e}")
    print(f"\n{'='*64}\n  {len(failures)} failure(s)")
    for f in failures:
        print("   -", f)
    sys.exit(1 if failures else 0)
