"""
cross_dataset_ablation.py
─────────────────────────
Cross-dataset evaluation of the ABLATION stages on the external benchmark.

Place at the PROJECT ROOT (D:\\virk\\tomato\\cross_dataset_ablation.py).

WHAT THIS ANSWERS
-----------------
Section 4.4 reports that BiFPN gives no in-domain accuracy gain (S4 is
statistically identical to S2), and concludes that its role is confined to
training stability. That conclusion rests only on in-domain numbers. This
script evaluates the same stages on the external benchmark, where BiFPN's
multi-scale fusion is far more likely to matter, and turns

    "BiFPN's role is confined to feature fusion within the complete
     architecture"

into a measured claim rather than an inference.

STAGES
------
    S2  retinanet_convnext_baseline   ConvNeXt-T + FPN            (no CBAM)
    S4  retinanet_convnext_bifpn      ConvNeXt-T + BiFPN x3       (no CBAM)
    S5  retinanet_hybrid              complete RetinaNeXt          <- control

S2 vs S4 isolates the neck: identical backbone, head, anchors and training,
differing only in FPN vs BiFPN. Their gap on the external benchmark is BiFPN's
transfer contribution.

S3 (ConvNeXt-T + FPN + CBAM) is NOT included. Its checkpoints were never
saved — outputs/checkpoints/retinanet_convnext_fpn_cbam/ is empty — so it
cannot be evaluated without retraining. This is why the paper already states
"S3 was not itself evaluated cross-domain, so no claim is made regarding its
transfer behavior." That sentence stays true after this run.

HOW IT WORKS
------------
Imports cross_eval/cross_dataset_3_eval.py and reuses its dataset loader,
inference loop, label mapping and metric code without modification. It adds
only two model builders, a device fallback, and a separate output directory.
The original script is not touched, and outputs/cross_eval_3/ is not written
to, so the published S1 and S5 numbers cannot be overwritten.

REFERENCE (already in the paper, from outputs/cross_eval_3)
-----------------------------------------------------------
    S1  RetinaNet baseline      57.91% +/- 2.46
    S5  RetinaNeXt              80.55% +/- 1.16

Seed 42 alone: retinanet_hybrid = 0.7955. The control run must reproduce that
number exactly, otherwise the pipeline is not being driven correctly and no
result from this script should be used.

USAGE
-----
    # 1. smoke test, ~1 min — proves loading and inference work
    python cross_dataset_ablation.py --model retinanet_hybrid --seed 42 --limit 20

    # 2. control — MUST print 0.7955. Do not skip.
    python cross_dataset_ablation.py --model retinanet_hybrid --seed 42

    # 3. the real run — S2 and S4 across three seeds
    python cross_dataset_ablation.py --model all --seed all

OUTPUT
------
    outputs/cross_eval_3_ablation/<model>/seed_<n>/per_class_cross.csv
    outputs/cross_eval_3_ablation/ablation_cross_summary.csv
    outputs/cross_eval_3_ablation/ablation_cross_aggregate.csv
"""

import argparse
import csv
import os
import statistics
import sys

import torch

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
CROSS_EVAL_DIR = os.path.join(HERE, "cross_eval")
for _p in (HERE, CROSS_EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cross_dataset_3_eval as base  # noqa: E402

# Capture the original loader BEFORE the monkeypatch below replaces
# base.load_model. Without this, the fall-through branch in our load_model()
# would call itself and recurse until the stack blows.
_ORIG_LOAD_MODEL = base.load_model

ABLATION_MODELS = ["retinanet_convnext_baseline",
                   "retinanet_convnext_fpn_cbam",
                   "retinanet_convnext_bifpn"]
CONTROL_MODEL   = "retinanet_hybrid"
ALL_SEEDS       = [42, 123, 456]

# Stage numbering follows the paper's current S1-S5 scheme (Table 4).
STAGE_LABEL = {
    "retinanet_convnext_baseline": "S2  ConvNeXt-T + FPN",
    "retinanet_convnext_fpn_cbam": "S3  ConvNeXt-T + FPN + CBAM",
    "retinanet_convnext_bifpn":    "S4  ConvNeXt-T + BiFPN x3",
    "retinanet_hybrid":            "S5  RetinaNeXt (control)",
}

CKPT = {
    "retinanet_convnext_baseline": (
        "retinanet_convnext_baseline", "retinanet_convnext_baseline_best.pth"),
    "retinanet_convnext_fpn_cbam": (
        "retinanet_convnext_fpn_cbam", "retinanet_convnext_fpn_cbam_best.pth"),
    "retinanet_convnext_bifpn": (
        "retinanet_convnext_bifpn", "retinanet_convnext_bifpn_best.pth"),
}


def have_checkpoints(model_name, seeds):
    """S3's weights were deleted after the original run. Until it is retrained
    there is nothing to evaluate, so it is skipped rather than failing."""
    if model_name not in CKPT:
        return True
    folder, fname = CKPT[model_name]
    return all(os.path.exists(os.path.join(
        HERE, "outputs", "checkpoints", folder, f"seed_{s}", fname))
        for s in seeds)


# ── Model loaders ────────────────────────────────────────────────────────────
# Both stages are torchvision RetinaNet models built with num_classes=10
# (9 foreground + background), exactly like retinanet_hybrid, so they take the
# same label_offset of -1.

def load_model(model_name, seed, device):
    """Extends base.load_model with the two ablation stages."""

    if model_name in CKPT:
        folder, fname = CKPT[model_name]
        path = os.path.join(HERE, "outputs", "checkpoints", folder,
                            f"seed_{seed}", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if model_name == "retinanet_convnext_baseline":
            from ablation_convnext_baseline.retinanet_convnext import (
                build_retinanet_convnext)
            model = build_retinanet_convnext(num_classes=10, pretrained=False)
        elif model_name == "retinanet_convnext_fpn_cbam":
            sys.path.insert(0, os.path.join(HERE, "ablation_fpn_cbam"))
            from retinanet_convnext_fpn_cbam import (
                build_retinanet_convnext_fpn_cbam)
            model = build_retinanet_convnext_fpn_cbam(
                num_classes=10, pretrained=False)
        else:
            from ablation_bifpn_only.retinanet_convnext_bifpn import (
                build_retinanet_convnext_bifpn)
            model = build_retinanet_convnext_bifpn(
                num_classes=10, pretrained=False, num_bifpn_repeats=3)

        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state)
        model.to(device).eval()
        return model, -1, "torchvision"

    return _ORIG_LOAD_MODEL(model_name, seed, device)


# base.run_one() resolves load_model from the base module's globals at call
# time, so replacing it here redirects run_one to the extended loader while
# leaving every other line of the original evaluation path untouched.
base.load_model = load_model


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(
        description="Cross-dataset evaluation of the ablation stages.")
    ap.add_argument("--model", default="all",
                    choices=ABLATION_MODELS + [CONTROL_MODEL, "all", "all+control"],
                    help="'all' = the two ablation stages; "
                         "'all+control' also re-runs RetinaNeXt as a check")
    ap.add_argument("--seed", default="all", help="42, 123, 456, or 'all'")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda. Default: auto")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N images (smoke test). 0 = all")
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"CPU threads: {torch.get_num_threads()}")

    root = args.dataset_root or os.path.join(HERE, "data", "cross_dataset_3")
    print(f"Dataset root: {root}")

    items, skipped = base.load_cross_dataset_3(root)
    if args.limit:
        items = items[:args.limit]
        print(f"*** SMOKE TEST: limited to {len(items)} images — "
              f"numbers are NOT comparable to the paper ***")
    n_gt = sum(len(it[3]) for it in items)
    print(f"Loaded {len(items)} images with {n_gt} overlap-class GT boxes")
    print(f"Skipped {skipped} images")

    # Separate directory. outputs/cross_eval_3/ holds the published S1 and S5
    # numbers and is never written to by this script.
    results_root = os.path.join(HERE, "outputs", "cross_eval_3_ablation")
    os.makedirs(results_root, exist_ok=True)

    if args.model == "all":
        models = list(ABLATION_MODELS)
    elif args.model == "all+control":
        models = ABLATION_MODELS + [CONTROL_MODEL]
    else:
        models = [args.model]

    seeds = ALL_SEEDS if str(args.seed).lower() == "all" else [int(args.seed)]

    # Skip any stage whose checkpoints are absent rather than failing halfway
    # through. S3 has no weights until it is retrained.
    runnable, skipped_stages = [], []
    for m in models:
        (runnable if have_checkpoints(m, seeds) else skipped_stages).append(m)
    for m in skipped_stages:
        print(f"\n  SKIPPING {STAGE_LABEL.get(m, m)} — no checkpoints for "
              f"seeds {seeds}")
        if m == "retinanet_convnext_fpn_cbam":
            print("           retrain with: "
                  "python ablation_fpn_cbam\\train_fpn_cbam.py --seed <n>")
    models = runnable
    if not models:
        sys.exit("\nNothing to run — no stage has checkpoints for these seeds.")

    print(f"\nPlanned runs: {len(models) * len(seeds)}")
    for m in models:
        print(f"  {STAGE_LABEL.get(m, m)}  seeds {seeds}")

    summaries = []
    for m in models:
        for s in seeds:
            try:
                summaries.append(base.run_one(m, s, device, items, results_root))
            except Exception as e:
                print(f"[FAIL] {m} seed {s}: {type(e).__name__}: {e}")

    if not summaries:
        print("\nNo runs completed.")
        return

    per_run = os.path.join(results_root, "ablation_cross_summary.csv")
    with open(per_run, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "stage", "seed", "cross_mean_ap50", "cross_mean_ap5095"]
                   + [f"ap50_{base.OUR_DISPLAY[c]}"
                      for c in base.OVERLAP_PYTORCH_LABELS])
        for r in summaries:
            w.writerow([r["model"], STAGE_LABEL.get(r["model"], ""), r["seed"],
                        r["mean_ap50"], r["mean_ap5095"]]
                       + [r["per_class"].get(c, {}).get("ap50", "")
                          for c in base.OVERLAP_PYTORCH_LABELS])
    print(f"\nSaved: {per_run}")

    agg = os.path.join(results_root, "ablation_cross_aggregate.csv")
    with open(agg, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "stage", "n_seeds", "cross_ap50_mean",
                    "cross_ap50_std", "cross_ap5095_mean", "cross_ap5095_std"])
        print(f"\n{'Stage':<32} {'n':>2} {'cross mAP@0.5':>18} "
              f"{'cross mAP@0.5:0.95':>20}")
        print("-" * 76)
        means = {}
        for m in models:
            v50 = [r["mean_ap50"] * 100 for r in summaries if r["model"] == m]
            v95 = [r["mean_ap5095"] * 100 for r in summaries if r["model"] == m]
            if not v50:
                continue
            m50, m95 = statistics.mean(v50), statistics.mean(v95)
            s50 = statistics.stdev(v50) if len(v50) > 1 else 0.0
            s95 = statistics.stdev(v95) if len(v95) > 1 else 0.0
            means[m] = m50
            w.writerow([m, STAGE_LABEL.get(m, m), len(v50),
                        f"{m50:.2f}", f"{s50:.2f}", f"{m95:.2f}", f"{s95:.2f}"])
            print(f"{STAGE_LABEL.get(m, m):<32} {len(v50):>2} "
                  f"{m50:>11.2f} +/- {s50:<4.2f} {m95:>13.2f} +/- {s95:<4.2f}")
    print(f"\nSaved: {agg}")

    print("\nReference, already in the paper (outputs/cross_eval_3):")
    print(f"  {'S1  RetinaNet baseline':<32}  3       57.91 +/- 2.46")
    print(f"  {'S5  RetinaNeXt':<32}  3       80.55 +/- 1.16")

    S2, S3, S4 = ("retinanet_convnext_baseline", "retinanet_convnext_fpn_cbam",
                  "retinanet_convnext_bifpn")
    S5_CROSS = 80.55        # published, outputs/cross_eval_3

    if S2 in means and S4 in means:
        gap = means[S4] - means[S2]
        print(f"\n  BiFPN's cross-domain contribution (S4 - S2): {gap:+.2f} pp")
        print("  In-domain the same comparison is -0.01 pp.")
        if gap > 1.0:
            print("  -> BiFPN helps transfer even though it does not help in-domain.")
            print("     Section 4.4's claim that its role is 'confined to feature")
            print("     fusion' should be updated to report this.")
        elif gap < -1.0:
            print("  -> BiFPN HURTS transfer. Report it; do not bury it.")
        else:
            print("  -> No meaningful cross-domain effect either. Section 4.4's")
            print("     current wording stands, now backed by measurement.")

    # S3 is the configuration that beats RetinaNeXt in-domain (94.61 vs 94.42).
    # Whether it also beats it cross-domain decides whether the BiFPN choice is
    # defensible. Report the answer either way.
    if S3 in means:
        d = S5_CROSS - means[S3]
        print(f"\n  S3 cross-domain : {means[S3]:.2f}")
        print(f"  S5 cross-domain : {S5_CROSS:.2f}  (published)")
        print(f"  S5 - S3         : {d:+.2f} pp")
        print(f"  In-domain S3 beats S5 by +0.19 pp (94.61 vs 94.42).")
        if d > 1.0:
            print("  -> S3 wins in-domain but loses cross-domain. This is the")
            print("     justification for keeping BiFPN: it buys transfer, not")
            print("     in-domain accuracy. Say so explicitly in Section 5.")
        elif d < -1.0:
            print("  -> S3 beats the full model in BOTH domains. The BiFPN choice")
            print("     is not supported by the evidence. Report this honestly and")
            print("     reconsider what the paper claims about the neck.")
        else:
            print("  -> S3 and S5 transfer equivalently. BiFPN is justified by")
            print("     stability alone, which is what Section 4.4 already says.")

    if args.limit:
        print("\n*** These came from a smoke test. Discard them. ***")


if __name__ == "__main__":
    main()