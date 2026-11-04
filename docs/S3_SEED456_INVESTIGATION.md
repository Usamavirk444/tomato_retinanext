# S3 (ConvNeXt-T + FPN + CBAM) seed 456 — two conflicting results

Read-only investigation. No file in `tomato_vm_clean/` was changed.

## The conflict

| Result CSV | mAP@0.5 | mAP@0.5:0.95 | CSV mtime |
|---|---|---|---|
| `results/retinanet_convnext_fpn_cbam/seed_456/` | **91.23** | 75.78 | 2026-11-02 14:02 |
| `results/retinanet_convnext_fpn_cbam_JULY_ORIGINAL/seed_456/` | **94.68** | 82.29 | 2026-07-03 15:58 |

Difference 3.45 pp. The manuscript's S3 = 94.61 ± 0.10 uses the July value.

## Two distinct checkpoints, confirmed by hash

| Path | SHA-256 | Size | mtime |
|---|---|---|---|
| `outputs/checkpoints_recovered/.../seed_456/` | `0e9f3dcfcb53c816895a83aecb249d03d3ec6ddc136f9c353c16a7a0b80efc07` | 434,057,442 | 2026-07-03 11:47 |
| `outputs/checkpoints/.../seed_456/` | `52ee4088a89fb3364aad8079c22c1d0b2e85f57ddbaa9841091712158d0ce741` | 434,057,442 | 2026-09-08 13:43 |

Same size, different weights. Not a copy.

## Decisive evidence — the validation-to-test relationship

Only ONE S3 seed-456 training log exists: 2026-07-03 12:40, 70 epochs,
best validation mAP@0.5 = 95.47%.

| Seed | Best val | Current test | July test | val − test |
|---|---|---|---|---|
| 42 | 95.43 | 94.49 | 94.49 | −0.94 |
| 123 | 95.39 | 94.65 | 94.65 | −0.74 |
| **456** | **95.47** | **91.23** | **94.68** | **−4.24** vs **−0.79** |

Seeds 42 and 123 show a consistent 0.7–0.9 pp validation-to-test gap. The July
seed-456 result fits that pattern (−0.79). The November result is −4.24 pp, a
five-fold outlier, from the model with the *highest* validation score.

This reasoning was carried out before checking which value the manuscript uses.

## Independent corroboration from older project copies

| Location | Checkpoint SHA-256 (12) | Result |
|---|---|---|
| `Tomato/tomato_vm/` | `0e9f3dcfcb53` | 94.68 |
| `tomato_vm_clean/outputs/checkpoints_recovered/` | `0e9f3dcfcb53` | 94.68 |
| `Tomato/tomato_vm_copy/` | checkpoint not retained | 94.68 |
| `tomato_vm_clean/outputs/checkpoints/` | `52ee4088a89f` | 91.23 |

`tomato_vm/` predates the September checkpoint and holds the July weights,
its training log, and its evaluation.

## Conclusion

The **July checkpoint `0e9f3dcf…` is the provisional canonical version.** It is
the only S3 seed-456 checkpoint with a corresponding training log, its test
result is consistent with its own validation curve and with the other two
seeds, and it is reproduced in three independent locations.

The September checkpoint `52ee4088…` exists in exactly one place, has **no
training log**, and produces an outlier. It is retained in the inventory and
has not been deleted.

## Unresolved uncertainty

The provenance of the September 8 checkpoint cannot be established from the
files. Possibilities include an interrupted or abandoned retrain, a partially
written save, or a run under a different configuration. Nothing on disk
records it.

## Pending verification

Not yet run. On the VM, evaluating the July checkpoint by explicit path and
confirming it yields 94.68 would close this end-to-end. The command must point
at `checkpoints_recovered/` explicitly — `evaluate.py --model
retinanet_convnext_fpn_cbam --seed 456` resolves to `checkpoints/` and would
silently load the September weights.

Status: **RESOLVED — INDEPENDENTLY REPRODUCED, 4 November 2026**

## Independent evaluation, Phase D.2

Run on the CUDA VM with `ablation_fpn_cbam/eval_recovered_seed456.py`, which
verifies the checkpoint before loading and aborts on any mismatch.

```
[verify] SHA-256    : 0e9f3dcfcb53c816895a83aecb249d03d3ec6ddc136f9c353c16a7a0b80efc07
[verify] epoch      : 68                  expected 68                  OK
[verify] map50      : 0.9547245236931102  expected 0.9547245236931102  OK
[verify] map50_95   : 0.8273824291111405  expected 0.8273824291111405  OK
[verify] val_loss   : 0.0843656398445799  expected 0.0843656398445799  OK
[verify] is_ema     : True                expected True                OK
[verify] model_name : retinanet_convnext_fpn_cbam                      OK

  Test images : 2487      Score threshold: 0.01      Device: cuda
  mAP @ IoU 0.50      : 0.9468
  mAP @ IoU 0.75      : 0.8701
  mAP @ IoU 0.50:0.95 : 0.8229
[done] SHA-256 after : unchanged
```

| | |
|---|---|
| Archived | 94.68 % |
| **Freshly evaluated** | **94.68 %** |
| Difference | **0.00 pp** |
| Evidence level | **A — independently reproduced** |

**S3 three-seed mean: (94.49 + 94.65 + 94.68) / 3 = 94.607 ≈ 94.61 ± 0.10**,
matching Table 2.

### Consequences

- The July checkpoint is no longer provisional. It is **confirmed canonical**.
- The September checkpoint `52ee4088…` produces 91.23 and is the outlier. It
  is retained in the inventory, marked not-required, and still has no
  training log explaining its origin.
- S3 (94.61) exceeds S5 (94.42) in-domain by 0.19 pp, so the abstract's
  statement that a reduced configuration omitting BiFPN scores marginally
  higher in-domain **is supported**.
- §4.4's reading — that BiFPN is justified by cross-domain transfer rather
  than in-domain accuracy — stands.
