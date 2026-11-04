"""
verify_box_loss.py
──────────────────
Checks which box-regression loss each detector actually ends up with
after construction. Run this before and after the S1 retrain.

Builds every model on CPU with no data, so it takes seconds and needs
no GPU. Exits non-zero if anything is not what the paper claims.

Expected:
    retinanet (published S1)     giou    <- torchvision factory override
    retinanet_l1 (new S1)        l1
    retinanet_hybrid (S5)        l1
    S2 ConvNeXt + FPN            l1
    S3 ConvNeXt + FPN + CBAM     l1
    S4 ConvNeXt + BiFPN          l1

Usage:
    python verify_box_loss.py
"""

import sys
import torch

NUM_CLASSES = 10          # 9 foreground + 1 background, as in experiment.yaml
rows, failures = [], []


def loss_type(model):
    return getattr(model.head.regression_head, "_loss_type", "<absent>")


def check(label, build, expect):
    try:
        model = build()
    except Exception as exc:                       # noqa: BLE001
        rows.append((label, "BUILD FAILED", expect, "ERROR"))
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        return
    got = loss_type(model)
    ok = (got == expect)
    rows.append((label, got, expect, "ok" if ok else "MISMATCH"))
    if not ok:
        failures.append(f"{label}: got {got!r}, expected {expect!r}")
    del model


# ── S1, as published ────────────────────────────────────────────────
from src.models.retinanet import build_retinanet

check("retinanet  (published S1)",
      lambda: build_retinanet(NUM_CLASSES, pretrained=False, box_loss="giou"),
      "giou")

# ── S1, rebuilt to match S2-S5 ──────────────────────────────────────
check("retinanet_l1  (new S1)",
      lambda: build_retinanet(NUM_CLASSES, pretrained=False, box_loss="l1"),
      "l1")

# ── S5, the proposed model ──────────────────────────────────────────
from src.models.hybrid.retinanet_hybrid import build_retinanet_hybrid

check("retinanet_hybrid  (S5)",
      lambda: build_retinanet_hybrid(NUM_CLASSES, pretrained=False),
      "l1")

# ── ablation stages S2-S4, if their modules are importable ──────────
import importlib

for label, module, fn in [
        ("S2  ConvNeXt + FPN",
         "ablation_convnext_baseline.retinanet_convnext", None),
        ("S3  ConvNeXt + FPN + CBAM",
         "ablation_fpn_cbam.retinanet_convnext_fpn_cbam", None),
        ("S4  ConvNeXt + BiFPN x3",
         "ablation_bifpn_only.retinanet_convnext_bifpn", None)]:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:                        # noqa: BLE001
        rows.append((label, f"import failed ({type(exc).__name__})", "l1", "skip"))
        continue
    builder = fn or next(
        (getattr(mod, n) for n in dir(mod)
         if n.startswith("build_") and callable(getattr(mod, n))), None)
    if builder is None:
        rows.append((label, "no build_* function found", "l1", "skip"))
        continue
    check(label, lambda b=builder: b(NUM_CLASSES, pretrained=False), "l1")


# ── report ──────────────────────────────────────────────────────────
w = max(len(r[0]) for r in rows) + 2
print()
print(f"{'model':<{w}}{'actual':<14}{'expected':<12}status")
print("-" * (w + 34))
for label, got, expect, status in rows:
    print(f"{label:<{w}}{got:<14}{expect:<12}{status}")
print()

if failures:
    print("FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("All checked models use the expected box regression loss.")
print("torch", torch.__version__)
import torchvision
print("torchvision", torchvision.__version__)
