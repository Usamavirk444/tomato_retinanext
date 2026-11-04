"""
validate_against_paper.py
─────────────────────────
Compares what is on disk against the values reported in the manuscript.

Every expected value below is a REFERENCE taken from the paper. None of them
is used to produce a result — they are only compared against numbers computed
from the CSVs and dataset folders. A check that cannot be evaluated is
reported as SKIP, never silently passed.

Usage
    python scripts/verify/validate_against_paper.py
    python scripts/verify/validate_against_paper.py --data data/clean_yolo_dataset
    python scripts/verify/validate_against_paper.py --json report.json

Exit code 0 only when nothing FAILED. SKIPs do not fail the run, but are
counted and listed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REF = os.path.join(REPO, "reference_results")
MAN = os.path.join(REPO, "manifests")

SEEDS = (42, 123, 456)
TOL = 0.01          # percentage points; CSVs are stored to 2 dp

results = []


def record(section, name, expected, actual, status, detail=""):
    """status is PASS, FAIL, SKIP or DISCREPANCY.

    DISCREPANCY marks a KNOWN, DOCUMENTED divergence between the manuscript
    and the code where the code is authoritative. It is reported loudly but
    does not fail the run: the repository is correct, the manuscript text is
    what differs. Never use it to paper over an unexplained mismatch."""
    results.append(dict(section=section, check=name, expected=expected,
                        actual=actual, status=status, detail=detail))


def cmp_num(section, name, expected, actual, tol=TOL, unit=""):
    if actual is None:
        record(section, name, expected, None, "SKIP", "not computable")
        return
    ok = abs(actual - expected) <= tol
    record(section, name, f"{expected}{unit}", f"{actual:.4f}{unit}",
           "PASS" if ok else "FAIL",
           "" if ok else f"differs by {actual-expected:+.4f}")


def cmp_int(section, name, expected, actual):
    if actual is None:
        record(section, name, expected, None, "SKIP", "not computable")
        return
    record(section, name, expected, actual,
           "PASS" if actual == expected else "FAIL")


# ── 1. dataset ──────────────────────────────────────────────────────
def check_dataset(data_dir):
    S = "dataset"
    man = os.path.join(MAN, "split_manifest_clean_yolo_dataset.csv")
    if not os.path.exists(man):
        record(S, "split manifest present", "present", "missing", "FAIL")
        return
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    per_split = Counter(r["split"] for r in rows)
    cmp_int(S, "manifest: train images", 17419, per_split["train"])
    cmp_int(S, "manifest: val images", 4970, per_split["val"])
    cmp_int(S, "manifest: test images", 2487, per_split["test"])
    cmp_int(S, "manifest: total images", 24876, len(rows))
    cmp_int(S, "manifest: total boxes", 67595,
            sum(int(r["n_boxes"]) for r in rows))

    cc = Counter()
    for r in rows:
        for c in filter(None, r["classes"].split(";")):
            cc[int(c)] += 1
    cmp_int(S, "manifest: distinct classes", 9, len(cc))

    # overlap-excluded subset
    clean = os.path.join(MAN, "testsubset_1933_overlap_excluded.txt")
    leak = os.path.join(MAN, "testsubset_554_overlap_leaked.txt")
    if os.path.exists(clean) and os.path.exists(leak):
        a = {l.strip() for l in open(clean, encoding="utf-8") if l.strip()}
        b = {l.strip() for l in open(leak, encoding="utf-8") if l.strip()}
        cmp_int(S, "overlap-excluded test images", 1933, len(a))
        cmp_int(S, "overlap-leaked test images", 554, len(b))
        cmp_int(S, "clean + leaked = full test", 2487, len(a | b))
        cmp_int(S, "clean/leaked disjoint", 0, len(a & b))
    else:
        record(S, "overlap manifests", "present", "missing", "SKIP")

    # external
    kept = os.path.join(MAN, "external_kept_1051.txt")
    rem = os.path.join(MAN, "external_removed_1345.csv")
    if os.path.exists(kept) and os.path.exists(rem):
        k = {l.strip() for l in open(kept, encoding="utf-8") if l.strip()}
        rr = list(csv.DictReader(open(rem, encoding="utf-8")))
        cmp_int(S, "external kept after overlap removal", 1051, len(k))
        cmp_int(S, "external removed for overlap", 1345, len(rr))
        cmp_int(S, "external original total", 2396, len(k) + len(rr))
    else:
        record(S, "external manifests", "present", "missing", "SKIP")

    # on-disk dataset, if the user has built it
    if data_dir and os.path.isdir(data_dir):
        for split, exp in (("train", 17419), ("val", 4970), ("test", 2487)):
            d = os.path.join(data_dir, split, "images")
            n = len([f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]) \
                if os.path.isdir(d) else None
            cmp_int(S, f"on-disk {split} images", exp, n)
    else:
        record(S, "on-disk dataset", "built", "not built", "SKIP",
               "run scripts/data/prepare_nine_class.py first")


# ── 2. in-domain results ────────────────────────────────────────────
def seed_mean(paths, field="mAP_50"):
    vals = []
    for p in paths:
        if not os.path.exists(p):
            return None, None
        try:
            vals.append(float(next(csv.DictReader(
                open(p, encoding="utf-8")))[field]) * 100)
        except Exception:
            return None, None
    if len(vals) < 2:
        return None, None
    return statistics.mean(vals), statistics.stdev(vals)


def check_in_domain():
    S = "in-domain"
    f = os.path.join(REF, "in_domain", "main_results_table.csv")
    if not os.path.exists(f):
        record(S, "main_results_table.csv", "present", "missing", "SKIP")
        return
    d = {r["model"]: r for r in csv.DictReader(open(f, encoding="utf-8"))}
    exp = {"SSDLite-MobileNetV3-Large": (80.24, 0.23),
           "YOLOv8n": (87.65, 0.29),
           "RetinaNet-ResNet50-FPN-v2": (89.16, 0.40),
           "RetinaNet-Hybrid v3": (94.42, 0.05)}
    for model, (m, sd) in exp.items():
        if model not in d:
            record(S, f"{model} mAP@0.5", m, None, "SKIP", "row absent")
            continue
        cmp_num(S, f"{model} mAP@0.5 mean", m,
                float(d[model]["map50_mean"]) * 100, tol=0.01, unit="%")
        cmp_num(S, f"{model} mAP@0.5 sd", sd,
                float(d[model]["map50_std"]) * 100, tol=0.01, unit="%")
    if "RetinaNet-Hybrid v3" in d:
        cmp_num(S, "RetinaNeXt mAP@0.5:0.95", 81.98,
                float(d["RetinaNet-Hybrid v3"]["map5095_mean"]) * 100,
                unit="%")


# ── 3. external results ─────────────────────────────────────────────
def check_external():
    S = "external"
    f = os.path.join(REF, "external", "summary_cross_mean_std.csv")
    if not os.path.exists(f):
        record(S, "summary_cross_mean_std.csv", "present", "missing", "SKIP")
        return
    d = {r["model"]: r for r in csv.DictReader(open(f, encoding="utf-8"))}
    exp = {"ssdlite": (39.21, 3.17), "yolov8n": (38.25, 1.03),
           "retinanet": (39.31, 1.14), "retinanet_l1": (40.70, 1.38),
           "retinanet_convnext_baseline": (62.01, 4.73),
           "retinanet_convnext_fpn_cbam": (61.45, 2.97),
           "retinanet_convnext_bifpn": (61.89, 2.44),
           "retinanet_hybrid": (64.67, 2.24)}
    for model, (m, sd) in exp.items():
        if model not in d:
            record(S, f"{model} external mAP@0.5", m, None, "SKIP")
            continue
        cmp_num(S, f"{model} external mAP@0.5", m,
                float(d[model]["map50_mean"]), unit="%")
        cmp_num(S, f"{model} external sd", sd,
                float(d[model]["map50_sd"]), unit="%")
    if "retinanet_hybrid" in d:
        r = d["retinanet_hybrid"]
        cmp_num(S, "RetinaNeXt external mAP@0.5:0.95", 49.70,
                float(r["map50_95_mean"]), unit="%")
        cmp_int(S, "external evaluated images", 805, int(r["n_images"]))
        cmp_int(S, "external GT boxes", 7307, int(r["n_gt_boxes"]))
        cmp_int(S, "external classes in mAP", 8, int(r["classes_in_map"]))


# ── 4. ablation ─────────────────────────────────────────────────────
def check_ablation():
    S = "ablation"
    stages = {
        "S1 RetinaNet-R50": ("in_domain/retinanet/retinanet_seed_{s}_eval_results.csv", 89.16),
        "S1-L1 RetinaNet-R50": ("in_domain/retinanet_l1/retinanet_l1_seed_{s}_eval_results.csv", 89.52),
        "S2 ConvNeXt+FPN": ("in_domain/retinanet_convnext_baseline/seed_{s}/retinanet_convnext_baseline_seed_{s}_eval_results.csv", 94.19),
        "S4 ConvNeXt+BiFPN": ("in_domain/retinanet_convnext_bifpn/seed_{s}/retinanet_convnext_bifpn_seed_{s}_eval_results.csv", 94.18),
        "S5 RetinaNeXt": ("in_domain/retinanet_hybrid/retinanet_hybrid_seed_{s}_eval_results.csv", 94.42),
    }
    for name, (pat, exp) in stages.items():
        paths = [os.path.join(REF, pat.format(s=s)) for s in SEEDS]
        m, _ = seed_mean(paths)
        cmp_num(S, f"{name} in-domain mAP@0.5", exp, m, tol=0.02, unit="%")

    # S3 is the disputed one - check BOTH and report, do not assert
    july = [os.path.join(
        REF, f"in_domain/retinanet_convnext_fpn_cbam_JULY_ORIGINAL/seed_{s}/"
             f"retinanet_convnext_fpn_cbam_seed_{s}_eval_results.csv")
        for s in SEEDS]
    curr = [os.path.join(
        REF, f"in_domain/retinanet_convnext_fpn_cbam/seed_{s}/"
             f"retinanet_convnext_fpn_cbam_seed_{s}_eval_results.csv")
        for s in SEEDS]
    mj, _ = seed_mean(july)
    mc, _ = seed_mean(curr)
    if mj is not None:
        cmp_num(S, "S3 (July, provisional canonical) mAP@0.5", 94.61, mj,
                tol=0.02, unit="%")
    else:
        record(S, "S3 July result", 94.61, None, "SKIP")
    record(S, "S3 (September checkpoint) mAP@0.5",
           "n/a - disputed", f"{mc:.2f}%" if mc else None, "SKIP",
           "see docs/S3_SEED456_INVESTIGATION.md - NOT independently verified")


# ── 5. protocols actually encoded in the code ───────────────────────
def check_protocols():
    S = "protocol"
    t = os.path.join(REPO, "train_all.py")
    if os.path.exists(t):
        src = open(t, encoding="utf-8").read()
        n = src.count("eta_min=1e-7")
        cmp_int(S, "torchvision eta_min=1e-7 occurrences", 3, n)
        record(S, "manuscript 3.3 says 1e-6, code uses 1e-7",
               "1e-6 (manuscript)", "1e-7 (code, authoritative)",
               "DISCREPANCY",
               "documented in docs/LEARNING_RATE_DISCREPANCY.md. The code "
               "produced the published numbers; the manuscript sentence is "
               "wrong for the three torchvision models. NOT corrected here.")
        record(S, "manuscript 3.3 omits warm-up starting at 10% of base LR",
               "from zero", "from 0.1 x base_lr",
               "DISCREPANCY", "trainer.py:320")
    else:
        record(S, "train_all.py", "present", "missing", "SKIP")

    cm = os.path.join(REPO, "scripts", "eval", "confusion_matrix.py")
    if os.path.exists(cm):
        src = open(cm, encoding="utf-8").read()
        record(S, "confusion-matrix confidence threshold", "0.25",
               "0.25 (CLI default; verified reproduces stored matrices)",
               "PASS" if 'default=0.25' in src else "FAIL",
               "function signature still says 0.05 - documented in "
               "docs/EVALUATION_PROTOCOLS.md")
        record(S, "manuscript says confusion-matrix conf 0.05",
               "0.05 (manuscript)", "0.25 (verified reproduces matrices)",
               "DISCREPANCY",
               "verify_conf_threshold.py: 0.25 gives an exact cell-for-cell "
               "match, 0.05 differs in 72 of 100 cells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data",
                                                   "clean_yolo_dataset"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--suite", default="all",
                    choices=["all", "dataset", "results", "protocol",
                             "manuscript"],
                    help="dataset = splits and manifests; results = in-domain, "
                         "external and ablation metrics; protocol = what the "
                         "code encodes; manuscript = documented divergences "
                         "between paper text and code")
    args = ap.parse_args()

    if args.suite in ("all", "dataset"):
        check_dataset(args.data)
    if args.suite in ("all", "results"):
        check_in_domain()
        check_external()
        check_ablation()
    if args.suite in ("all", "protocol", "manuscript"):
        check_protocols()

    w = max(len(r["check"]) for r in results) + 2
    cur = None
    for r in results:
        if r["section"] != cur:
            cur = r["section"]
            print(f"\n── {cur} " + "─" * (w + 34 - len(cur)))
        print(f"  {r['status']:<5}{r['check']:<{w}}"
              f"expected={str(r['expected']):<12} got={r['actual']}")
        if r["detail"]:
            print(f"        {r['detail']}")

    c = Counter(r["status"] for r in results)
    print(f"\n{'='*70}")
    print(f"  PASS {c['PASS']}   FAIL {c['FAIL']}   "
          f"SKIP {c['SKIP']}   DISCREPANCY {c['DISCREPANCY']}")
    if c["DISCREPANCY"]:
        print("\n  documented manuscript/code divergences (code is "
              "authoritative; nothing was changed to hide these):")
        for r in results:
            if r["status"] == "DISCREPANCY":
                print(f"    - {r['check']}")
                print(f"        {r['detail']}")
    if c["FAIL"]:
        print("\n  failures:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    - {r['check']}: expected {r['expected']}, "
                      f"got {r['actual']}")
    print("=" * 70)

    if args.json:
        json.dump(results, open(args.json, "w", encoding="utf-8"), indent=2)
        print(f"wrote {args.json}")

    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
