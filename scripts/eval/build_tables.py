"""
build_tables.py
───────────────
Regenerates manuscript Tables 1, 2 and 3 from the authoritative result CSVs.

WHY NOT build_results_table.py
    That script (kept, unmodified in behaviour) reads the per-epoch training
    histories and reports the best VALIDATION mAP. For RetinaNeXt seed 42 it
    gives 0.9497 — the validation figure — whereas the manuscript's 94.38 is
    the TEST figure. It is useful for monitoring training; it does not
    produce Table 1.

    This script reads the test-set evaluation CSVs instead, which is what the
    manuscript reports.

SOURCES, all shipped in reference_results/
    Table 1  reference_results/in_domain/main_results_table.csv
             (the October 2026 table; the superseded May one lives in
              audit_history/ and is never read by default)
    Table 2  in-domain per-seed CSVs  +  external summary_cross_mean_std.csv
    Table 3  reference_results/external/per_class_cross.csv

NO VALUE IS HARDCODED. Every number is computed from the CSVs. Expected
values appear only in scripts/verify/validate_against_paper.py, which
compares rather than produces.

Usage
    python scripts/eval/build_tables.py                  # print all three
    python scripts/eval/build_tables.py --table 2
    python scripts/eval/build_tables.py --out tables/    # also write CSV/MD
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.exists(os.path.join(REPO, "train_all.py")) \
        and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
REF = os.path.join(REPO, "reference_results")
SEEDS = (42, 123, 456)


def read_rows(path):
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path, encoding="utf-8")))


def seed_stats(pattern, field="mAP_50"):
    """mean and sample SD over the three seeds, or (None, None)."""
    vals = []
    for s in SEEDS:
        p = pattern.format(s=s)
        if not os.path.exists(p):
            return None, None
        try:
            vals.append(float(next(csv.DictReader(
                open(p, encoding="utf-8")))[field]) * 100)
        except Exception:
            return None, None
    return statistics.mean(vals), statistics.stdev(vals)


# ── Table 1 ─────────────────────────────────────────────────────────
def table1():
    f = os.path.join(REF, "in_domain", "main_results_table.csv")
    rows = read_rows(f)
    if rows is None:
        return None, f"missing {f}"
    label = {"SSDLite-MobileNetV3-Large": "SSDLite-MobileNetV3",
             "YOLOv8n": "YOLOv8n",
             "RetinaNet-ResNet50-FPN-v2": "RetinaNet-ResNet50-FPN-v2",
             "RetinaNet-Hybrid v3": "RetinaNeXt (Ours)"}
    out = []
    for r in rows:
        out.append({
            "Model": label.get(r["model"], r["model"]),
            "mAP@0.5": f"{float(r['map50_mean'])*100:.2f}% ± "
                       f"{float(r['map50_std'])*100:.2f}%",
            "mAP@0.5:0.95": f"{float(r['map5095_mean'])*100:.2f}% ± "
                            f"{float(r['map5095_std'])*100:.2f}%",
        })
    return out, f


# ── Table 2 ─────────────────────────────────────────────────────────
def table2():
    stages = [
        ("S1", "RetinaNet-R50-FPN-v2 (GIoU)", "retinanet",
         "in_domain/retinanet/retinanet_seed_{s}_eval_results.csv"),
        ("S1-L1", "RetinaNet-R50-FPN-v2 (L1)", "retinanet_l1",
         "in_domain/retinanet_l1/retinanet_l1_seed_{s}_eval_results.csv"),
        ("S2", "ConvNeXt-T + FPN", "retinanet_convnext_baseline",
         "in_domain/retinanet_convnext_baseline/seed_{s}/"
         "retinanet_convnext_baseline_seed_{s}_eval_results.csv"),
        ("S3", "ConvNeXt-T + FPN + CBAM", "retinanet_convnext_fpn_cbam",
         "in_domain/retinanet_convnext_fpn_cbam_JULY_ORIGINAL/seed_{s}/"
         "retinanet_convnext_fpn_cbam_seed_{s}_eval_results.csv"),
        ("S4", "ConvNeXt-T + BiFPN x3", "retinanet_convnext_bifpn",
         "in_domain/retinanet_convnext_bifpn/seed_{s}/"
         "retinanet_convnext_bifpn_seed_{s}_eval_results.csv"),
        ("S5", "ConvNeXt-T + BiFPN x3 + CBAM (RetinaNeXt)", "retinanet_hybrid",
         "in_domain/retinanet_hybrid/retinanet_hybrid_seed_{s}_eval_results.csv"),
    ]
    ext = {r["model"]: r for r in
           (read_rows(os.path.join(REF, "external",
                                   "summary_cross_mean_std.csv")) or [])}
    out, notes = [], []
    for tag, name, key, pat in stages:
        m, sd = seed_stats(os.path.join(REF, pat))
        e = ext.get(key)
        em = float(e["map50_mean"]) if e else None
        esd = float(e["map50_sd"]) if e else None
        out.append({
            "Stage": tag, "Configuration": name,
            "In-domain mAP@0.5": f"{m:.2f}% ± {sd:.2f}%" if m else "n/a",
            "External mAP@0.5": f"{em:.2f}% ± {esd:.2f}%" if em else "n/a",
            "Drop (pp)": f"{m-em:.2f}" if (m and em) else "n/a",
        })
        if tag == "S3":
            notes.append("S3 uses the July 2026 seed-456 evaluation; the "
                         "September checkpoint is disputed and NOT "
                         "independently verified. See "
                         "docs/S3_SEED456_INVESTIGATION.md")
    return out, notes


# ── Table 3 ─────────────────────────────────────────────────────────
def table3():
    f = os.path.join(REF, "external", "per_class_cross.csv")
    rows = read_rows(f)
    if rows is None:
        return None, f"missing {f}"
    label = {"ssdlite": "SSDLite", "yolov8n": "YOLOv8n",
             "retinanet": "S1 RetinaNet-R50", "retinanet_l1": "S1-L1",
             "retinanet_convnext_baseline": "S2",
             "retinanet_convnext_fpn_cbam": "S3",
             "retinanet_convnext_bifpn": "S4",
             "retinanet_hybrid": "RetinaNeXt"}
    by = {}
    for r in rows:
        if r["ap50"] == "N/A":
            continue
        by.setdefault(r["model"], {}).setdefault(r["class"], []).append(
            float(r["ap50"]) * 100)
    classes = [c for c in dict.fromkeys(r["class"] for r in rows)
               if any(c in v for v in by.values())]
    out = []
    for model in label:
        if model not in by:
            continue
        row = {"Model": label[model]}
        means = []
        for c in classes:
            v = by[model].get(c)
            if v and len(v) == 3:
                m, sd = statistics.mean(v), statistics.stdev(v)
                row[c] = f"{m:.2f} ± {sd:.2f}"
                means.append(m)
            else:
                row[c] = "—"
        row["External mAP@0.5"] = f"{statistics.mean(means):.2f}" if means else "—"
        out.append(row)
    return out, f


def show(title, rows, note=""):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    if not rows:
        print(f"  UNAVAILABLE: {note}")
        return
    cols = list(rows[0])
    w = {c: max(len(c), max(len(str(r.get(c, ''))) for r in rows)) + 2
         for c in cols}
    print("".join(c.ljust(w[c]) for c in cols))
    print("-" * sum(w.values()))
    for r in rows:
        print("".join(str(r.get(c, '')).ljust(w[c]) for c in cols))
    if note:
        print(f"\n  source: {note}" if note.startswith(REPO) or "/" in note
              else f"\n  note: {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--out", default=None, help="directory for CSV output")
    args = ap.parse_args()

    produced = {}
    if args.table in (None, 1):
        t, src = table1()
        show("TABLE 1  mAP@0.5 and mAP@0.5:0.95, four detectors, seeds 42/123/456",
             t, src)
        produced[1] = t
    if args.table in (None, 2):
        t, notes = table2()
        show("TABLE 2  Ablation: in-domain and external mAP@0.5", t)
        for n in notes:
            print(f"  NOTE: {n}")
        produced[2] = t
    if args.table in (None, 3):
        t, src = table3()
        show("TABLE 3  External per-class AP@0.5, 805 images, 8 evaluable classes",
             t, src)
        produced[3] = t

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for n, rows in produced.items():
            if not rows:
                continue
            p = os.path.join(args.out, f"table{n}.csv")
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
