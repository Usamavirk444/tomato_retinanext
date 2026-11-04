"""
make_manuscript_figures.py
──────────────────────────
Generates the six manuscript figures into one clean directory, under their
FINAL manuscript names.

The underlying scripts are unchanged and keep their historical filenames —
`01_main_results.R` writes `Fig04_main_results.svg`, and so on. Renaming
them in place would break the R pipeline and lose the link to the analysis
history. Instead this script runs them and copies the outputs to the names
the paper uses.

    Figure 1  Dataset composition          figures/R/06_dataset_eda.R
    Figure 2  RetinaNeXt architecture      figures/architecture/make_fig03_svg.py
    Figure 3  Four detectors, five metrics figures/R/01_main_results.R
    Figure 4  Per-class AP                 figures/R/02_per_class.R
    Figure 5  Confusion matrices           figures/R/07_confusion_and_errors.R
    Figure 6  Training curves S1-S5        figures/R/05_training_and_pr.R

No metric is computed or hardcoded here. Every figure comes from its own
script reading its own source data; this is orchestration only.

Usage
    python scripts/figures/make_manuscript_figures.py
    python scripts/figures/make_manuscript_figures.py --out figures/final
    python scripts/figures/make_manuscript_figures.py --skip-r   # Fig 2 only
    python scripts/figures/make_manuscript_figures.py --check    # verify
                                                                 # sources, run
                                                                 # nothing
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.exists(os.path.join(REPO, "train_all.py")) \
        and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)

R_DIR = os.path.join(REPO, "figures", "R")
ARCH_DIR = os.path.join(REPO, "figures", "architecture")
R_OUT = os.path.join(R_DIR, "figures")        # where 00_setup.R writes

# final name -> (producing script, its output stem, required source)
FIGURES = [
    ("Figure1_dataset_composition", "R", "06_dataset_eda.R",
     "Fig01_dataset_analysis",
     "prepared dataset at data/clean_yolo_dataset/ (raw YOLO labels)"),
    ("Figure2_retinanext_architecture", "PY", "make_fig03_svg.py",
     "Fig03_RetinaNeXt_architecture",
     "none - generated from the model description in the script"),
    ("Figure3_main_results", "R", "01_main_results.R",
     "Fig04_main_results",
     "figures/source_csv/results/main_results_table.csv"),
    ("Figure4_per_class_ap", "R", "02_per_class.R",
     "Fig08_per_class_ap_circular",
     "figures/source_csv/results/*/per_class_test.csv"),
    ("Figure5_confusion_matrices", "R", "07_confusion_and_errors.R",
     "Fig09_confusion_matrices",
     "figures/source_csv/results/*/confusion_matrix_test.csv"),
    ("Figure6_training_curves", "R", "05_training_and_pr.R",
     "FigS1_training_curves",
     "figures/source_csv/logs/*/seed_*/*history.csv"),
]

EXTS = ("svg", "png", "pdf")


def check_sources():
    """Confirm each figure's inputs exist. Returns list of (fig, ok, detail)."""
    out = []
    csvdir = os.path.join(REPO, "figures", "source_csv")
    checks = {
        "Figure1_dataset_composition":
            os.path.isdir(os.path.join(REPO, "data", "clean_yolo_dataset")),
        "Figure2_retinanext_architecture":
            os.path.exists(os.path.join(ARCH_DIR, "make_fig03_svg.py")),
        "Figure3_main_results":
            os.path.exists(os.path.join(csvdir, "results",
                                        "main_results_table.csv")),
        "Figure4_per_class_ap":
            bool(_glob(os.path.join(csvdir, "results"), "per_class_test.csv")),
        "Figure5_confusion_matrices":
            bool(_glob(os.path.join(csvdir, "results"),
                       "confusion_matrix_test.csv")),
        "Figure6_training_curves":
            bool(_glob(os.path.join(csvdir, "logs"), "history.csv")),
    }
    for name, _, script, _, src in FIGURES:
        out.append((name, checks.get(name, False), src))
    return out


def _glob(root, suffix):
    hits = []
    for dirpath, _, files in os.walk(root):
        hits += [os.path.join(dirpath, f) for f in files if f.endswith(suffix)]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "final"))
    ap.add_argument("--skip-r", action="store_true",
                    help="skip the R figures (useful without R installed)")
    ap.add_argument("--check", action="store_true",
                    help="verify sources and exit without running anything")
    args = ap.parse_args()

    print("source check")
    all_ok = True
    for name, ok, src in check_sources():
        print(f"  {'OK  ' if ok else 'MISS'} {name:<36} {src}")
        all_ok &= ok
    if args.check:
        # Figure 1 reads the prepared dataset, which is not shipped. Its
        # absence is expected in a fresh clone and must not be reported as
        # a repository fault; any OTHER missing source is a real problem.
        missing = [n for n, ok, _ in check_sources() if not ok]
        only_dataset = missing == ["Figure1_dataset_composition"]
        if only_dataset:
            print("\n--check: 5 of 6 sources present. Figure 1 needs the "
                  "prepared dataset\n           (scripts/data/prepare_nine_class.py); "
                  "this is expected in a fresh clone.")
        elif missing:
            print(f"\n--check: missing sources: {missing}")
        else:
            print("\n--check: all sources present")
        sys.exit(0 if (all_ok or only_dataset) else 1)

    os.makedirs(args.out, exist_ok=True)

    # ── run the R pipeline once; run_all.R sources 01..08 in order ──
    if not args.skip_r:
        if shutil.which("Rscript") is None:
            print("\nRscript not found - skipping R figures. Install R >= 4.2 "
                  "with tidyverse, patchwork and svglite, or pass --skip-r.")
        else:
            print("\nrunning figures/R/run_all.R ...")
            r = subprocess.run(["Rscript", os.path.join(R_DIR, "run_all.R")],
                               cwd=R_DIR, capture_output=True, text=True)
            sys.stdout.write(r.stdout[-2000:])
            if r.returncode != 0:
                sys.stderr.write(r.stderr[-2000:])
                print("  R pipeline reported a failure; continuing so that "
                      "any figures it did produce are still collected.")

    # ── Figure 2, the architecture diagram ──────────────────────────
    print("\nrunning figures/architecture/make_fig03_svg.py ...")
    r = subprocess.run([sys.executable, "make_fig03_svg.py"],
                       cwd=ARCH_DIR, capture_output=True, text=True)
    print("  " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip()
                  else f"exit {r.returncode}"))

    # ── collect under final names ───────────────────────────────────
    print(f"\ncollecting into {args.out}")
    made, missing = [], []
    for final, kind, script, stem, _ in FIGURES:
        srcdir = ARCH_DIR if kind == "PY" else R_OUT
        got = False
        for ext in EXTS:
            p = os.path.join(srcdir, f"{stem}.{ext}")
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(args.out, f"{final}.{ext}"))
                got = True
        (made if got else missing).append((final, stem))
        print(f"  {'OK  ' if got else 'MISS'} {final:<36} <- {stem}")

    print(f"\n  {len(made)} of {len(FIGURES)} figures collected")
    if missing:
        print("  missing:")
        for f, s in missing:
            print(f"    {f}  (expected {s}.svg)")
    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()
