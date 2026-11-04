"""
Aggregate all training results into the paper's main results table.

Reads every history CSV under outputs/logs/<model>/seed_<n>/, picks the
best epoch by mAP@50, and computes mean ± std across the 3 seeds.

Output:
  Prints a Markdown table to stdout
  Saves CSV at outputs/results/main_results_table.csv

For YOLOv8n, also extracts the best mAP from results.csv (Ultralytics format).

Usage:
    python build_results_table.py
"""

import os
import csv
import glob
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# moved from paper_files/ into scripts/eval/ - walk up to the repo root
PROJECT_ROOT = HERE
while not os.path.exists(os.path.join(PROJECT_ROOT, "train_all.py")) \
        and os.path.dirname(PROJECT_ROOT) != PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# Default to the live training outputs, exactly as in the original project.
# RETINANEXT_LOGS_DIR lets a reader point at the shipped reference copies
# instead, so Table 1 can be rebuilt without retraining:
#
#   RETINANEXT_LOGS_DIR=reference_results/logs \
#       python scripts/eval/build_results_table.py
#
# Unset, behaviour is identical to the original script.
LOGS_DIR = os.environ.get(
    "RETINANEXT_LOGS_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "logs"))
if not os.path.isabs(LOGS_DIR):
    LOGS_DIR = os.path.join(PROJECT_ROOT, LOGS_DIR)
YOLO_CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")

MODELS = {
    "ssdlite":          "SSDLite-MobileNetV3-Large",
    "yolov8n":          "YOLOv8n",
    "retinanet":        "RetinaNet-ResNet50-FPN-v2",
    "retinanet_hybrid": "RetinaNet-Hybrid v3",
}
SEEDS = [42, 123, 456]


def read_torchvision_history(csv_path):
    """Read a training history CSV (from our Trainer) and return best row."""
    if not os.path.exists(csv_path):
        return None
    best = None
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                map50 = float(row.get("map50", 0))
                map5095 = float(row.get("map50_95", 0))
            except (TypeError, ValueError):
                continue
            if best is None or map50 > best["map50"]:
                best = {"map50": map50, "map50_95": map5095}
    return best


def read_yolo_results(seed):
    """Read Ultralytics' results.csv and return best row by mAP50."""
    # Ultralytics writes results to outputs/checkpoints/yolov8n/seed_X/results.csv
    candidates = [
        os.path.join(YOLO_CKPT_DIR, "yolov8n", f"seed_{seed}", "results.csv"),
        os.path.join(YOLO_CKPT_DIR, "yolov8n", f"seed_{seed}", "results", "results.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            best = None
            with open(path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                # Ultralytics column names contain spaces; strip them
                for row in reader:
                    row = {k.strip(): v for k, v in row.items()}
                    try:
                        map50 = float(row.get("metrics/mAP50(B)", row.get("metrics/mAP_0.5", 0)))
                        map5095 = float(row.get("metrics/mAP50-95(B)", row.get("metrics/mAP_0.5:0.95", 0)))
                    except (TypeError, ValueError):
                        continue
                    if best is None or map50 > best["map50"]:
                        best = {"map50": map50, "map50_95": map5095}
            return best
    return None


def find_torchvision_history(model_key, seed):
    """Find the history CSV for a given torchvision model + seed."""
    candidates = glob.glob(os.path.join(
        LOGS_DIR, model_key, f"seed_{seed}", "*_history.csv"))
    return candidates[0] if candidates else None


def mean_std(xs):
    import statistics
    if len(xs) == 0:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def main():
    rows = []
    print("Reading results ...\n")

    for key, label in MODELS.items():
        seed_results = {}
        for seed in SEEDS:
            if key == "yolov8n":
                r = read_yolo_results(seed)
            else:
                csv_path = find_torchvision_history(key, seed)
                r = read_torchvision_history(csv_path) if csv_path else None
            if r is not None:
                seed_results[seed] = r
                print(f"  {label:<32} seed {seed}: "
                      f"mAP50={r['map50']:.4f} mAP50-95={r['map50_95']:.4f}")
            else:
                print(f"  {label:<32} seed {seed}: [missing]")

        map50_vals  = [r["map50"]    for r in seed_results.values()]
        map5095_vals = [r["map50_95"] for r in seed_results.values()]

        m50_mean, m50_std    = mean_std(map50_vals)
        m95_mean, m95_std    = mean_std(map5095_vals)
        rows.append({
            "key":   key,
            "label": label,
            "n_seeds": len(seed_results),
            "seeds_done": sorted(seed_results.keys()),
            "map50_mean":   m50_mean,
            "map50_std":    m50_std,
            "map5095_mean": m95_mean,
            "map5095_std":  m95_std,
            "per_seed":     seed_results,
        })

    # ── Print Markdown table ──
    print("\n" + "="*82)
    print("MAIN RESULTS TABLE")
    print("="*82)
    print()
    print("| Model | Seeds | mAP@50 (mean ± std) | mAP@50:95 (mean ± std) |")
    print("|---|---|---|---|")
    for r in rows:
        if r["n_seeds"] == 0:
            print(f"| {r['label']} | 0/3 | — | — |")
            continue
        seeds_done_str = "/".join(str(s) for s in r["seeds_done"])
        if r["n_seeds"] >= 2:
            map50_str = f"{r['map50_mean']:.4f} ± {r['map50_std']:.4f}"
            map95_str = f"{r['map5095_mean']:.4f} ± {r['map5095_std']:.4f}"
        else:
            map50_str = f"{r['map50_mean']:.4f}"
            map95_str = f"{r['map5095_mean']:.4f}"
        print(f"| {r['label']} | {r['n_seeds']}/3 ({seeds_done_str}) | "
              f"{map50_str} | {map95_str} |")

    # ── Save CSV ──
    out_path = os.path.join(PROJECT_ROOT, "outputs", "results",
                            "main_results_table.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seeds_done", "n_seeds",
                    "map50_mean", "map50_std", "map5095_mean", "map5095_std",
                    "seed_42_map50", "seed_42_map5095",
                    "seed_123_map50", "seed_123_map5095",
                    "seed_456_map50", "seed_456_map5095"])
        for r in rows:
            row = [r["label"],
                   "/".join(str(s) for s in r["seeds_done"]),
                   r["n_seeds"],
                   r["map50_mean"],
                   r["map50_std"],
                   r["map5095_mean"],
                   r["map5095_std"]]
            for seed in SEEDS:
                if seed in r["per_seed"]:
                    row.extend([r["per_seed"][seed]["map50"],
                                r["per_seed"][seed]["map50_95"]])
                else:
                    row.extend(["", ""])
            w.writerow(row)
    print(f"\nSaved CSV: {out_path}")

    # ── Compute headline deltas (Hybrid vs best baseline) ──
    print("\n" + "="*82)
    print("HEADLINE DELTAS (for paper abstract)")
    print("="*82)
    hybrid = next((r for r in rows if r["key"] == "retinanet_hybrid"), None)
    if hybrid is None or hybrid["n_seeds"] == 0:
        print("Hybrid v3 results not available yet.")
        return

    for r in rows:
        if r["key"] == "retinanet_hybrid" or r["n_seeds"] == 0:
            continue
        d50  = (hybrid["map50_mean"]   - r["map50_mean"])   * 100
        d95  = (hybrid["map5095_mean"] - r["map5095_mean"]) * 100
        print(f"  Hybrid v3 vs {r['label']:<32} "
              f"+{d50:>5.2f}pp @mAP50  |  +{d95:>5.2f}pp @mAP50:95")


if __name__ == "__main__":
    main()
