"""
Sanity check before launching the 48-hour training run.

Verifies:
  1. Model builds without errors
  2. Forward pass works in train mode (returns finite loss dict)
  3. Backward pass produces finite gradients
  4. Forward pass works in eval mode (returns detection dicts)

If this passes in ~30 seconds, you're safe to launch training.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

from retinanet_convnext import build_retinanet_convnext


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    NUM_CLASSES = 10  # 9 fg + 1 bg

    print("\n[1/4] Building model ...")
    model = build_retinanet_convnext(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)
    print("       OK.")

    print("\n[2/4] Forward pass (train mode) ...")
    model.train()
    images = [torch.randn(3, 640, 640, device=device) for _ in range(2)]
    targets = []
    for _ in range(2):
        targets.append({
            "boxes": torch.tensor(
                [[100., 100., 300., 300.], [320., 320., 540., 540.]],
                device=device),
            "labels": torch.tensor([0, 1], dtype=torch.int64, device=device),
        })
    losses = model(images, targets)
    print(f"       Loss keys: {list(losses.keys())}")
    for k, v in losses.items():
        print(f"       {k:25s} {v.item():.4f}  finite={torch.isfinite(v).item()}")
    total = sum(losses.values())
    assert torch.isfinite(total), "Total loss is not finite!"

    print("\n[3/4] Backward pass ...")
    total.backward()
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    gn = torch.tensor(norms)
    print(f"       Grad norms — mean: {gn.mean():.4f}  max: {gn.max():.4f}  "
          f"finite: {torch.isfinite(gn).all().item()}")
    assert torch.isfinite(gn).all(), "Found NaN/inf gradients!"

    print("\n[4/4] Forward pass (eval mode) ...")
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    print(f"       Got {len(outputs)} detection dicts.")
    print(f"       First image — boxes: {outputs[0]['boxes'].shape}, "
          f"scores: {outputs[0]['scores'].shape}")

    print("\nAll checks passed. Safe to train.")


if __name__ == "__main__":
    main()
