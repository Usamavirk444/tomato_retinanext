"""
Standalone test-set evaluation for RetinaNet-ConvNeXt baseline (ablation).

Saves results to:
  outputs/results/retinanet_convnext_baseline/seed_<n>/per_class_test.csv

Usage (run from project root, i.e. the tomato_new folder):
  python ablation_convnext_baseline/evaluate_convnext.py --seed 42
  python ablation_convnext_baseline/evaluate_convnext.py --seed 123
  python ablation_convnext_baseline/evaluate_convnext.py --seed 456
"""

import argparse
import os
import sys
import gc
import torch

# ── path setup ────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

from src.dataset.yolo_dataset import build_yolo_dataloader
from src.dataset.transforms   import get_val_transforms
from src.evaluation.evaluator  import Evaluator
from src.utils.general         import get_device, load_model_for_eval
from retinanet_convnext        import build_retinanet_convnext

# ── constants (mirror experiment.yaml) ────────────────────────────────────────
MODEL_NAME   = "retinanet_convnext_baseline"
NUM_CLASSES  = 10          # 9 foreground + 1 background (same as RetinaNet)
IMAGE_SIZE   = (640, 640)
BATCH_SIZE   = 4           # lower than training to be safe on GPU memory
NUM_WORKERS  = 0

DATA_ROOT    = os.path.join(PROJECT_ROOT, "data", "clean_yolo_dataset")
TEST_IMAGES  = os.path.join(DATA_ROOT, "test", "images")
TEST_LABELS  = os.path.join(DATA_ROOT, "test", "labels")

CKPT_DIR     = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", MODEL_NAME)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "results",     MODEL_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        choices=[42, 123, 456],
                        help="Which seed checkpoint to evaluate")
    args = parser.parse_args()
    seed = args.seed

    device = get_device()
    print(f"\n{'='*55}")
    print(f"  Evaluating {MODEL_NAME} | Seed {seed}")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")

    # ── checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, f"seed_{seed}",
                             f"{MODEL_NAME}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at:\n  {ckpt_path}")
        sys.exit(1)
    print(f"  Checkpoint : {ckpt_path}")

    # ── data loader ───────────────────────────────────────────────────────────
    test_loader = build_yolo_dataloader(
        images_dir=TEST_IMAGES,
        labels_dir=TEST_LABELS,
        transforms=get_val_transforms(size=IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"  Test images: {len(test_loader.dataset)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_retinanet_convnext(num_classes=NUM_CLASSES, pretrained=False)
    model, epoch, loss = load_model_for_eval(model, ckpt_path, device)
    model.to(device)
    print(f"  Loaded from epoch {epoch}  (best val loss {loss:.4f})\n")

    # ── evaluate ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)

    evaluator = Evaluator(
        num_classes=NUM_CLASSES,
        model_name=f"{MODEL_NAME}_seed_{seed}",
        results_dir=results_dir,
    )
    evaluator.evaluate(model, test_loader, device, is_retinanet=True)

    print(f"\n  Results saved to: {results_dir}")
    print(f"  Look for: per_class_test.csv\n")

    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
