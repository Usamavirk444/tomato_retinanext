# RetinaNet-ConvNeXt-T Ablation Baseline

A standalone training pipeline to isolate "how much of the hybrid's gain comes
from the ConvNeXt-T backbone vs from BiFPN+CBAM."

## What this trains

A **RetinaNet** with:
- **ConvNeXt-Tiny backbone** (same as the hybrid)
- **Standard top-down FPN** (NOT BiFPN)
- **No CBAM attention**
- Default RetinaNet anchors (32-512 px)
- Same focal-loss head as the standard RetinaNet baseline

Total params: ~36 M (close to both your existing RetinaNet-ResNet50 (~36.5 M)
and Hybrid (~35.2 M)).

## What this answers

After 3 seeds × ~16 hours = ~48 hours, you'll be able to write:

> "ConvNeXt-Tiny backbone alone improves mAP@50 by **+X pp** over the
> RetinaNet-ResNet50 baseline. Our BiFPN+CBAM additions contribute the
> remaining **+Y pp**, for a total improvement of +5.4 pp."

That breaks the +5.4 pp into "backbone contribution" + "architectural
contribution" — exactly the ablation a Q1 reviewer wants to see.

## Files in this folder

| File | Purpose |
|---|---|
| `retinanet_convnext.py` | Model builder (ConvNeXt-T + plain FPN + standard RetinaNet head) |
| `sanity_check_convnext.py` | Pre-flight check (~30 seconds) before launching training |
| `train_convnext_baseline.py` | Standalone training script (1 seed per call) |

## Isolation guarantees

| Folder/file | Touched? |
|---|---|
| `src/models/*` | **Read-only import** of ConvNeXt body + Trainer |
| `outputs/checkpoints/retinanet_hybrid/*` | **NOT touched** |
| `outputs/checkpoints/retinanet/*` | **NOT touched** |
| `train_all.py`, `evaluate.py`, `cross_eval/`, `paper_files/` | **NOT touched** |
| `outputs/checkpoints/retinanet_convnext_baseline/` | **NEW** — written by this script |
| `outputs/logs/retinanet_convnext_baseline/` | **NEW** — written by this script |

You can delete this entire folder at any time and nothing else breaks.

## How to run on the VM

```
cd <repository root>
$env:TORCH_HOME = "<conda env>\.torch_cache"

# Step 1 — sanity check (~30 sec)
python ablation_convnext_baseline\sanity_check_convnext.py

# Step 2 — train 3 seeds in sequence (~48 hours total)
python ablation_convnext_baseline\train_convnext_baseline.py --seed 42 ; python ablation_convnext_baseline\train_convnext_baseline.py --seed 123 ; python ablation_convnext_baseline\train_convnext_baseline.py --seed 456
```

PowerShell's `;` chains the three runs — when seed 42 finishes, seed 123 auto-starts.

## After training finishes — adding to the comparison

If the teacher gives the green light, you'll want to compute per-class
metrics for this baseline (matching the other models). Add this to the
existing per-class eval script's model-loading switch:

```python
elif model_name == "retinanet_convnext_baseline":
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "ablation_convnext_baseline"))
    from retinanet_convnext import build_retinanet_convnext
    ckpt_path = os.path.join(
        PROJECT_ROOT, "outputs", "checkpoints", "retinanet_convnext_baseline",
        f"seed_{seed}", "retinanet_convnext_baseline_best.pth")
    model = build_retinanet_convnext(num_classes=10, pretrained=False)
    ...
```

Or I can write a fresh eval script in this folder when you have the
training results.

## Expected outcome — three scenarios

| Scenario | ConvNeXt-T baseline mAP@50 | Backbone contribution | Architecture contribution |
|---|---|---|---|
| ConvNeXt is the magic | 0.93-0.94 | +3 to +4 pp | +1.5 to +2 pp (modest) |
| Both contribute | 0.92-0.93 | +2 to +3 pp | +2.5 to +3 pp (decent) |
| BiFPN+CBAM is the magic | 0.89-0.91 | 0 to +2 pp | +4 to +5 pp (strong) |

Most likely the middle case based on prior literature. **All three scenarios
still publish.** The paper just frames the contribution differently.

## If the teacher says no

Delete this folder. Nothing else is affected. You only spent ~48h of GPU time
which is unavoidable to find out the answer.
