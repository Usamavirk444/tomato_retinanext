"""
Evaluate the RECOVERED seed 456 checkpoint (July run, epoch 68) for
RetinaNet ConvNeXt-T + FPN + CBAM, using evaluate_fpn_cbam.py unchanged.

- Reuses evaluate_fpn_cbam.main() so the test set, image size, batch size,
  model build and Evaluator are IDENTICAL to the seed 42 / 123 runs.
- Only two module paths are redirected:
      CKPT_DIR    -> outputs/checkpoints_recovered/retinanet_convnext_fpn_cbam
      RESULTS_DIR -> outputs/results_recovered/retinanet_convnext_fpn_cbam
- Verifies checkpoint SHA-256 and metadata BEFORE evaluating; aborts on mismatch.
- Refuses to run if the results folder already contains files.
- Never writes to any checkpoint.

Place in:  D:\\virk\\tomato\\ablation_fpn_cbam\\eval_recovered_seed456.py
Run from:  D:\\virk\\tomato
    python ablation_fpn_cbam\\eval_recovered_seed456.py
"""
import hashlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_fpn_cbam as ev  # noqa: E402  (unchanged original script)

SEED = 456
REC_CKPT_DIR = os.path.join(ev.PROJECT_ROOT, "outputs", "checkpoints_recovered", ev.MODEL_NAME)
REC_RESULTS_DIR = os.path.join(ev.PROJECT_ROOT, "outputs", "results_recovered", ev.MODEL_NAME)
CKPT_PATH = os.path.join(REC_CKPT_DIR, f"seed_{SEED}", f"{ev.MODEL_NAME}_best.pth")

EXPECTED_SHA256 = "0e9f3dcfcb53c816895a83aecb249d03d3ec6ddc136f9c353c16a7a0b80efc07"
EXPECTED = {
    "epoch": 68,
    "map50": 0.9547245236931102,
    "map50_95": 0.8273824291111405,
    "val_loss": 0.0843656398445799,
    "is_ema": True,
    "model_name": "retinanet_convnext_fpn_cbam",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify():
    print(f"\n[verify] Checkpoint : {CKPT_PATH}")
    if not os.path.isfile(CKPT_PATH):
        sys.exit(f"ERROR: recovered checkpoint not found:\n  {CKPT_PATH}")

    digest = sha256(CKPT_PATH)
    print(f"[verify] SHA-256    : {digest}")
    if digest != EXPECTED_SHA256:
        sys.exit("ERROR: SHA-256 mismatch - this is not the recovered July checkpoint.")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    ok = True
    for k, exp in EXPECTED.items():
        got = ckpt.get(k, "Not available")
        match = (abs(got - exp) < 1e-9) if isinstance(exp, float) and isinstance(got, float) else got == exp
        ok &= match
        print(f"[verify] {k:<11}: {got!r:<24} expected {exp!r:<24} {'OK' if match else 'MISMATCH'}")
    del ckpt
    if not ok:
        sys.exit("ERROR: metadata mismatch - aborting.")


def main():
    verify()

    out_dir = os.path.join(REC_RESULTS_DIR, f"seed_{SEED}")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"ERROR: results folder is not empty, refusing to overwrite:\n  {out_dir}")

    # Redirect only the two paths; every other setting comes from evaluate_fpn_cbam.py
    ev.CKPT_DIR = REC_CKPT_DIR
    ev.RESULTS_DIR = REC_RESULTS_DIR

    print("\n[settings] (taken unchanged from evaluate_fpn_cbam.py)")
    for name in ("MODEL_NAME", "NUM_CLASSES", "IMAGE_SIZE", "BATCH_SIZE",
                 "NUM_WORKERS", "TEST_IMAGES", "TEST_LABELS"):
        print(f"  {name:<12}: {getattr(ev, name)}")
    print(f"  {'RESULTS':<12}: {out_dir}")

    sys.argv = [sys.argv[0], "--seed", str(SEED)]
    ev.main()

    print(f"\n[done] Checkpoint used : {CKPT_PATH}")
    print(f"[done] Results folder  : {out_dir}")
    # Checkpoint unchanged after evaluation?
    print(f"[done] SHA-256 after   : {'unchanged' if sha256(CKPT_PATH) == EXPECTED_SHA256 else 'CHANGED!'}")


if __name__ == "__main__":
    main()