"""
eval_external_clean.py
──────────────────────
Re-runs the external (cross-dataset) evaluation on the OVERLAP-FREE subset
data/cross_dataset_3_clean/ (1,051 images; build it first with
build_clean_crossdataset.py).

Same models, checkpoints, matching code and AP code as the published runs:
  - cross_eval/cross_dataset_3_eval.py   (SSDLite, YOLOv8n, RetinaNet S1,
                                          RetinaNet-L1, RetinaNeXt S5)
  - cross_dataset_ablation.py            (S2, S3, S4)
Neither file is modified; both are imported.

Two differences only:
  1. dataset root  -> data/cross_dataset_3_clean
  2. output folder -> outputs/cross_eval_3_clean/   (new; existing
     outputs/cross_eval_3 and cross_eval_3_ablation are never written)

S3 seed 456: the checkpoint in outputs/checkpoints/retinanet_convnext_fpn_cbam
/seed_456 on the VM is the interrupted epoch-21 run. This script uses the
recovered epoch-68 checkpoint in outputs/checkpoints_recovered/ instead and
checks its SHA-256 before use.

Usage (from the project root):
    python eval_external_clean.py                    # all 8 models x 3 seeds
    python eval_external_clean.py --model retinanet_hybrid --seed 42
"""
import argparse
import csv
import hashlib
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
# cross_dataset_ablation.py was moved into scripts/eval/ in the same
# reorganisation, so that folder needs to be importable too.
for _p in (HERE, os.path.join(HERE, "cross_eval"),
           os.path.join(HERE, "scripts", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cross_dataset_3_eval as base          # noqa: E402
import cross_dataset_ablation as abl         # noqa: E402  (patches base.load_model)

DATA_ROOT = os.path.join(HERE, "data", "cross_dataset_3_clean")
OUT_ROOT = os.path.join(HERE, "outputs", "cross_eval_3_clean")
SEEDS = [42, 123, 456]
MODELS = ["ssdlite", "yolov8n", "retinanet", "retinanet_l1",
          "retinanet_convnext_baseline", "retinanet_convnext_fpn_cbam",
          "retinanet_convnext_bifpn", "retinanet_hybrid"]
LABEL = {
    "ssdlite": "SSDLite-MobileNetV3",
    "yolov8n": "YOLOv8n",
    "retinanet": "S1 RetinaNet-R50-FPN-v2 (GIoU, published)",
    "retinanet_l1": "S1-L1 RetinaNet-R50-FPN-v2 (L1)",
    "retinanet_convnext_baseline": "S2 ConvNeXt-T + FPN",
    "retinanet_convnext_fpn_cbam": "S3 ConvNeXt-T + FPN + CBAM",
    "retinanet_convnext_bifpn": "S4 ConvNeXt-T + BiFPN x3",
    "retinanet_hybrid": "S5 RetinaNeXt",
}
# Published external mAP@0.5 on all 2,396 images (for comparison only)
OLD = {"ssdlite": 56.58, "yolov8n": 56.89, "retinanet": 57.91,
       "retinanet_l1": 57.42, "retinanet_convnext_baseline": 78.47,
       "retinanet_convnext_fpn_cbam": 79.05, "retinanet_convnext_bifpn": 78.98,
       "retinanet_hybrid": 80.55}

S3_456 = os.path.join(HERE, "outputs", "checkpoints_recovered",
                      "retinanet_convnext_fpn_cbam", "seed_456",
                      "retinanet_convnext_fpn_cbam_best.pth")
S3_456_SHA = "0e9f3dcfcb53c816895a83aecb249d03d3ec6ddc136f9c353c16a7a0b80efc07"

_abl_load = abl.load_model


def load_model(model_name, seed, device):
    if model_name == "retinanet_convnext_fpn_cbam" and seed == 456:
        if not os.path.isfile(S3_456):
            raise FileNotFoundError(S3_456)
        h = hashlib.sha256()
        with open(S3_456, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != S3_456_SHA:
            raise RuntimeError("S3 seed 456 recovered checkpoint hash mismatch")
        sys.path.insert(0, os.path.join(HERE, "ablation_fpn_cbam"))
        from retinanet_convnext_fpn_cbam import build_retinanet_convnext_fpn_cbam
        model = build_retinanet_convnext_fpn_cbam(num_classes=10, pretrained=False)
        ckpt = torch.load(S3_456, map_location=device, weights_only=False)
        print(f"  [S3 seed 456] recovered checkpoint, epoch {ckpt.get('epoch')}")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        return model, -1, "torchvision"
    return _abl_load(model_name, seed, device)


base.load_model = load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--seed", default="all")
    args = ap.parse_args()

    if not os.path.isdir(DATA_ROOT):
        sys.exit(f"ERROR: {DATA_ROOT} not found. Run build_clean_crossdataset.py first.")

    models = MODELS if args.model == "all" else [args.model]
    seeds = SEEDS if str(args.seed) == "all" else [int(args.seed)]

    for m in models:
        for s in seeds:
            d = os.path.join(OUT_ROOT, m, f"seed_{s}")
            if os.path.isdir(d) and os.listdir(d):
                sys.exit(f"ERROR: {d} already has results - refusing to overwrite.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, skipped = base.load_cross_dataset_3(DATA_ROOT)
    n_gt = sum(len(it[3]) for it in items)
    print(f"Device {device} | dataset {DATA_ROOT}")
    print(f"Loaded {len(items)} images, {n_gt} GT boxes (skipped {skipped})")
    os.makedirs(OUT_ROOT, exist_ok=True)

    runs = []
    for m in models:
        for s in seeds:
            try:
                runs.append(base.run_one(m, s, device, items, OUT_ROOT))
            except Exception as e:
                print(f"[FAIL] {m} seed {s}: {type(e).__name__}: {e}")

    if not runs:
        return
    tag = "" if (args.model == "all" and str(args.seed) == "all") else "_partial"
    with open(os.path.join(OUT_ROOT, f"summary_per_run{tag}.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "seed", "map50", "map50_95"])
        for r in runs:
            w.writerow([r["model"], LABEL[r["model"]], r["seed"],
                        f"{r['mean_ap50']*100:.4f}", f"{r['mean_ap5095']*100:.4f}"])

    print(f"\n{'=' * 86}")
    print(f"OVERLAP-FREE EXTERNAL BENCHMARK ({len(items)} images) - mAP@0.5, mean +/- sample SD")
    print(f"{'=' * 86}")
    with open(os.path.join(OUT_ROOT, f"summary_aggregate{tag}.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "n_seeds", "map50_mean", "map50_sd",
                    "map50_95_mean", "map50_95_sd", "old_all_2396_map50"])
        for m in models:
            v = [r["mean_ap50"] * 100 for r in runs if r["model"] == m]
            v2 = [r["mean_ap5095"] * 100 for r in runs if r["model"] == m]
            if not v:
                continue
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            sd2 = statistics.stdev(v2) if len(v2) > 1 else 0.0
            w.writerow([m, LABEL[m], len(v), f"{statistics.mean(v):.2f}", f"{sd:.2f}",
                        f"{statistics.mean(v2):.2f}", f"{sd2:.2f}", OLD[m]])
            print(f"  {LABEL[m]:<44} {statistics.mean(v):6.2f} +/- {sd:4.2f}"
                  f"   (all 2,396: {OLD[m]:.2f})")
    print(f"\nSaved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
