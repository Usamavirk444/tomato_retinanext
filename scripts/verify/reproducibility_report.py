"""
reproducibility_report.py
─────────────────────────
Produces a machine-readable comparison of manuscript values against archived
CSVs and, where available, freshly computed results.

EVIDENCE LEVELS — the point of this script

    A  INDEPENDENTLY REPRODUCED
       A model checkpoint was loaded and run over the dataset in this
       environment, and the metric recomputed from its predictions.

    B  RECONSTRUCTED FROM ARCHIVED CSVs
       The number was read or re-aggregated from an evaluation CSV produced
       by an earlier run. This confirms the manuscript transcribes its own
       results correctly. It does NOT confirm the model produces them.

    C  STATIC / CODE INSPECTION
       Confirmed by reading code or counting files; no model involved.

Level B is NOT reproduction. The distinction is enforced here: a value is
only ever marked A when a fresh evaluation is supplied via --fresh.

TOLERANCES
    metric_pp   0.01  percentage points. Archived CSVs store mAP to 4 decimal
                      places as a 0-1 fraction, i.e. 0.01 pp. Anything inside
                      that is rounding, not disagreement.
    count       0     exact. Image and box counts are integers; an
                      off-by-one is a real error, never rounding.
    fresh_pp    0.10  percentage points, for a freshly computed metric against
                      its archive. Wider than metric_pp because a different
                      torch/torchvision build, CPU vs GPU, and non-deterministic
                      kernels can shift mAP in the third decimal. A difference
                      above 0.10 pp indicates something substantive, not noise.

Usage
    python scripts/verify/reproducibility_report.py
    python scripts/verify/reproducibility_report.py --out reports/
    python scripts/verify/reproducibility_report.py --fresh fresh_results.json

`--fresh` takes {"<model>_seed<N>": {"mAP_50": 0.9438, ...}, ...} produced by
a GPU run, and promotes those rows to level A.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
while not os.path.exists(os.path.join(REPO, "train_all.py")) \
        and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
REF = os.path.join(REPO, "reference_results")
MAN = os.path.join(REPO, "manifests")
SEEDS = (42, 123, 456)

TOL = {"metric_pp": 0.01, "count": 0, "fresh_pp": 0.10}

# checkpoint hashes, from manifests/checkpoint_inventory.csv
HASH = {}


def load_hashes():
    p = os.path.join(MAN, "checkpoint_inventory.csv")
    if not os.path.exists(p):
        return
    for r in csv.DictReader(open(p, encoding="utf-8")):
        HASH.setdefault(f"{r['model']}_seed{r['seed']}", []).append(
            (r["sha256"], r["required_for_paper"]))


def ck(model, seed):
    """Return the required-for-paper hash for a model/seed, else ''."""
    for h, req in HASH.get(f"{model}_seed{seed}", []):
        if req.startswith("YES"):
            return h[:16]
    return ""


rows = []


def add(item, manuscript, archived, fresh, level, protocol,
        checkpoint="", tol_key="metric_pp", note=""):
    diff = None
    status = "UNVERIFIED"
    if archived is not None and manuscript is not None:
        try:
            diff = abs(float(archived) - float(manuscript))
            status = "PASS" if diff <= TOL[tol_key] else "FAIL"
        except (TypeError, ValueError):
            status = "PASS" if str(archived) == str(manuscript) else "FAIL"
    if fresh is not None and manuscript is not None:
        d2 = abs(float(fresh) - float(manuscript))
        status = "PASS" if d2 <= TOL["fresh_pp"] else "FAIL"
        diff = d2
    rows.append(dict(item=item, manuscript=manuscript, archived=archived,
                     fresh=fresh,
                     abs_diff=None if diff is None else round(diff, 4),
                     status=status, evidence_level=level, protocol=protocol,
                     checkpoint_sha256=checkpoint,
                     tolerance=TOL[tol_key], note=note))


def seed_mean(pattern, field="mAP_50"):
    v = []
    for s in SEEDS:
        p = pattern.format(s=s)
        if not os.path.exists(p):
            return None
        try:
            v.append(float(next(csv.DictReader(
                open(p, encoding="utf-8")))[field]) * 100)
        except Exception:
            return None
    return statistics.mean(v)


def build(fresh_data):
    load_hashes()
    F = fresh_data or {}

    def fresh_mean(model, field="mAP_50"):
        v = []
        for s in SEEDS:
            k = f"{model}_seed{s}"
            if k not in F:
                return None
            v.append(float(F[k][field]) * 100)
        return statistics.mean(v)

    # ── Table 1 ────────────────────────────────────────────────────
    t1 = {r["model"]: r for r in csv.DictReader(
        open(os.path.join(REF, "in_domain", "main_results_table.csv"),
             encoding="utf-8"))}
    MAIN = [("SSDLite-MobileNetV3-Large", "ssdlite", 80.24, 65.03),
            ("YOLOv8n", "yolov8n", 87.65, 70.85),
            ("RetinaNet-ResNet50-FPN-v2", "retinanet", 89.16, 77.03),
            ("RetinaNet-Hybrid v3", "retinanet_hybrid", 94.42, 81.98)]
    for key, mdl, m50, m5095 in MAIN:
        r = t1.get(key)
        fr = fresh_mean(mdl)
        add(f"Table 1: {mdl} mAP@0.5", m50,
            round(float(r["map50_mean"]) * 100, 4) if r else None, fr,
            "A" if fr is not None else "B",
            "TorchMetrics, conf>=0.01, COCO IoU sweep", ck(mdl, 42))
        add(f"Table 1: {mdl} mAP@0.5:0.95", m5095,
            round(float(r["map5095_mean"]) * 100, 4) if r else None,
            fresh_mean(mdl, "mAP_50_95"),
            "A" if fresh_mean(mdl, "mAP_50_95") is not None else "B",
            "TorchMetrics, conf>=0.01, COCO IoU sweep", ck(mdl, 42))

    # ── Table 2 ────────────────────────────────────────────────────
    ext = {r["model"]: r for r in csv.DictReader(
        open(os.path.join(REF, "external", "summary_cross_mean_std.csv"),
             encoding="utf-8"))}
    STAGES = [
        ("S1", "retinanet", 89.16, 39.31,
         "in_domain/retinanet/retinanet_seed_{s}_eval_results.csv", ""),
        ("S1-L1", "retinanet_l1", 89.52, 40.70,
         "in_domain/retinanet_l1/retinanet_l1_seed_{s}_eval_results.csv", ""),
        ("S2", "retinanet_convnext_baseline", 94.19, 62.01,
         "in_domain/retinanet_convnext_baseline/seed_{s}/"
         "retinanet_convnext_baseline_seed_{s}_eval_results.csv", ""),
        ("S3", "retinanet_convnext_fpn_cbam", 94.61, 61.45,
         "in_domain/retinanet_convnext_fpn_cbam_JULY_ORIGINAL/seed_{s}/"
         "retinanet_convnext_fpn_cbam_seed_{s}_eval_results.csv",
         "July seed-456 checkpoint, PROVISIONALLY canonical, "
         "NOT independently re-evaluated"),
        ("S4", "retinanet_convnext_bifpn", 94.18, 61.89,
         "in_domain/retinanet_convnext_bifpn/seed_{s}/"
         "retinanet_convnext_bifpn_seed_{s}_eval_results.csv", ""),
        ("S5", "retinanet_hybrid", 94.42, 64.67,
         "in_domain/retinanet_hybrid/retinanet_hybrid_seed_{s}_eval_results.csv",
         ""),
    ]
    for tag, mdl, m_in, m_ex, pat, note in STAGES:
        a = seed_mean(os.path.join(REF, pat))
        fr = fresh_mean(mdl)
        add(f"Table 2: {tag} in-domain mAP@0.5", m_in,
            None if a is None else round(a, 4), fr,
            "A" if fr is not None else "B",
            "TorchMetrics, conf>=0.01", ck(mdl, 456), note=note)
        e = ext.get(mdl)
        add(f"Table 2: {tag} external mAP@0.5", m_ex,
            float(e["map50_mean"]) if e else None, None, "B",
            "custom COCO-style, 101-pt AP, best-IoU matching, "
            "8 classes, 805 images", ck(mdl, 42), note=note)

    # ── dataset, level C ───────────────────────────────────────────
    man = list(csv.DictReader(open(os.path.join(
        MAN, "split_manifest_clean_yolo_dataset.csv"), encoding="utf-8")))
    from collections import Counter
    c = Counter(r["split"] for r in man)
    for name, exp, got in [("train images", 17419, c["train"]),
                           ("val images", 4970, c["val"]),
                           ("test images", 2487, c["test"]),
                           ("total images", 24876, len(man)),
                           ("total boxes", 67595,
                            sum(int(r["n_boxes"]) for r in man))]:
        add(f"Dataset: {name}", exp, got, None, "C",
            "file and manifest count", tol_key="count")

    for name, exp, path in [
            ("overlap-excluded test", 1933, "testsubset_1933_overlap_excluded.txt"),
            ("overlap-leaked test", 554, "testsubset_554_overlap_leaked.txt"),
            ("external kept", 1051, "external_kept_1051.txt"),
            ("external evaluable", 805, "external_evaluable_805.txt")]:
        p = os.path.join(MAN, path)
        n = len([l for l in open(p, encoding="utf-8") if l.strip()]) \
            if os.path.exists(p) else None
        add(f"Dataset: {name}", exp, n, None, "C", "manifest line count",
            tol_key="count")

    e = ext.get("retinanet_hybrid")
    if e:
        add("Dataset: external GT boxes", 7307, int(e["n_gt_boxes"]), None,
            "C", "evaluator record", tol_key="count")
        add("Dataset: external classes in mAP", 8, int(e["classes_in_map"]),
            None, "C", "Mosaic Virus excluded, zero GT", tol_key="count")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "reports"))
    ap.add_argument("--fresh", default=None,
                    help="JSON of freshly evaluated metrics; promotes rows "
                         "to evidence level A")
    args = ap.parse_args()

    fresh = json.load(open(args.fresh, encoding="utf-8")) if args.fresh else None
    build(fresh)

    from collections import Counter
    lv = Counter(r["evidence_level"] for r in rows)
    st = Counter(r["status"] for r in rows)

    print(f"{'item':<46}{'paper':>9}{'archived':>10}{'fresh':>9}"
          f"{'diff':>8}  {'st':<6}{'lvl'}")
    print("-" * 97)
    for r in rows:
        f = "-" if r["fresh"] is None else f"{r['fresh']:.2f}"
        a = "-" if r["archived"] is None else (
            f"{r['archived']:.2f}" if isinstance(r["archived"], float)
            else str(r["archived"]))
        d = "-" if r["abs_diff"] is None else f"{r['abs_diff']:.3f}"
        print(f"{r['item']:<46}{str(r['manuscript']):>9}{a:>10}{f:>9}"
              f"{d:>8}  {r['status']:<6}{r['evidence_level']}")

    print(f"\n{'='*97}")
    print(f"  status   PASS {st['PASS']}   FAIL {st['FAIL']}   "
          f"UNVERIFIED {st['UNVERIFIED']}")
    print(f"  evidence A (independently reproduced) {lv['A']}   "
          f"B (archived CSV) {lv['B']}   C (static) {lv['C']}")
    if not lv["A"]:
        print("\n  NO VALUE IS INDEPENDENTLY REPRODUCED.")
        print("  Every metric is level B: read back from archived evaluation")
        print("  CSVs. That confirms the manuscript transcribes its own")
        print("  results faithfully, and nothing more. Independent")
        print("  reproduction needs the checkpoints and a GPU; supply the")
        print("  results with --fresh to promote rows to level A.")
    print("=" * 97)

    os.makedirs(args.out, exist_ok=True)
    meta = dict(generated=datetime.now(timezone.utc).isoformat(),
                tolerances=TOL, counts=dict(status=st, evidence=lv),
                rows=rows)
    j = os.path.join(args.out, "reproducibility_report.json")
    json.dump(meta, open(j, "w", encoding="utf-8"), indent=2)
    cpath = os.path.join(args.out, "reproducibility_report.csv")
    with open(cpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {j}\nwrote {cpath}")
    sys.exit(1 if st["FAIL"] else 0)


if __name__ == "__main__":
    main()
