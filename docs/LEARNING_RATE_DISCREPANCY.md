# Learning-rate schedule: code vs manuscript

Read-only record. Neither the manuscript nor the training code has been changed.

## What the manuscript says

Section 3.3, verbatim:

> "At the start of training, the learning rate was increased linearly over the
> first 5 epochs (warm-up), after which it was decreased via a cosine annealing
> schedule until reaching a final value of 1×10⁻⁶."

## What the code does

### Torchvision models — SSDLite, RetinaNet-R50, RetinaNeXt

`train_all.py` lines 210, 350 and 459 are byte-identical:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs - warmup_ep, eta_min=1e-7)
```

| | value |
|---|---|
| optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| warm-up | 5 epochs |
| cosine `T_max` | `epochs - warmup_ep` = 70 − 5 = **65** |
| **minimum LR (`eta_min`)** | **1e-7** |

The same three lines serve every torchvision model; the ablation stages S2–S4
use their own training scripts but construct the identical scheduler
(`ablation_convnext_baseline/train_convnext_baseline.py:135-136`).

### YOLOv8n

`train_all.py:258-280` passes `lrf=0.01` and `cos_lr=True` to Ultralytics.
Final LR = `lr0 × lrf` = 1e-4 × 0.01 = **1e-6**.

`lrf` appears in `configs/experiment.yaml` but is read **only** by the
Ultralytics call. The torchvision path never consumes it.

### Warm-up shape

`src/training/trainer.py:318-322`

```python
def _warmup_lr(self, epoch):
    if epoch <= self.warmup_epochs:
        warmup_factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr * warmup_factor
```

Linear over 5 epochs, but starting at **10% of base LR**, not from zero.
At epoch 1 the LR is 2.8e-5, reaching 1e-4 at epoch 5.

Ordering, `trainer.py:442` and `:489-490`: warm-up is applied at the start of
every epoch; the cosine steps only when `epoch > warmup_epochs`. So epochs
1–5 are warm-up and epochs 6–70 are cosine.

## The discrepancies

**1. Minimum learning rate — genuine, one order of magnitude.**

| Model | Manuscript | Code |
|---|---|---|
| SSDLite | 1×10⁻⁶ | **1×10⁻⁷** |
| RetinaNet-R50 | 1×10⁻⁶ | **1×10⁻⁷** |
| RetinaNeXt | 1×10⁻⁶ | **1×10⁻⁷** |
| YOLOv8n | 1×10⁻⁶ | 1×10⁻⁶ ✓ |

The sentence is correct for one model of four. The 1e-6 figure corresponds to
the `lrf: 0.01` config key, which only YOLOv8n uses.

**2. Warm-up starting point — omission, not an error.**

"Increased linearly over the first 5 epochs" describes the shape correctly but
omits that it begins at 10% of base LR. Ultralytics runs its own
iteration-based warm-up with a separate bias LR, so YOLOv8n's curve differs
again.

## Impact on reproducibility

None on the results — the code is self-consistent and all reported numbers came
from it. The risk is to the *description*: a reader implementing from §3.3
would set `eta_min=1e-6` and not reproduce the torchvision runs exactly.

The reproducibility repository will encode **1e-7**, matching the code, and
carry this file so the divergence from §3.3 is documented rather than silent.

## Source locations

| Claim | File | Lines |
|---|---|---|
| `eta_min=1e-7`, SSDLite | `train_all.py` | 210 |
| `eta_min=1e-7`, RetinaNet | `train_all.py` | 350 |
| `eta_min=1e-7`, RetinaNeXt | `train_all.py` | 459 |
| `eta_min=1e-7`, S2 ablation | `ablation_convnext_baseline/train_convnext_baseline.py` | 135–136 |
| YOLOv8n `lrf`, `cos_lr` | `train_all.py` | 265, 269 |
| warm-up from 10% | `src/training/trainer.py` | 318–322 |
| warm-up / cosine ordering | `src/training/trainer.py` | 442, 489–490 |
| `lrf: 0.01` | `configs/experiment.yaml` | 51 |
