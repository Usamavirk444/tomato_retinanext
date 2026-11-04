# Evaluation protocols

Three different evaluators produced the paper's numbers. They are NOT
interchangeable and have deliberately not been merged.

## 1. In-domain — TorchMetrics

`src/evaluation/evaluator.py`, driven by `evaluate.py`.

| Setting | Value | Location |
|---|---|---|
| Implementation | `torchmetrics.detection.MeanAveragePrecision` | `evaluator.py:22` |
| Confidence floor | **0.01** | `evaluator.py:13` |
| IoU | COCO sweep 0.50:0.05:0.95 | TorchMetrics default |
| maxDets | 100 per image | TorchMetrics default |
| Matching | COCO: greedy by descending score, within class | TorchMetrics |
| Zero-GT classes | dropped from the mean (returns −1) | TorchMetrics |
| Label handling | predictions shifted +1 for RetinaNet models | `evaluator.py:86-90` |

Used for: Table 1, Table 2 in-domain column, Figures 3 and 4.

## 2. External — custom COCO-style

`cross_eval/cross_dataset_3_eval.py`, aggregated by
`scripts/eval/recompute_external_clean.py`.

| Setting | Value | Location |
|---|---|---|
| Confidence floor | **none** for torchvision models; 0.05 for YOLOv8n | `cross_dataset_3_eval.py:337` |
| IoU | `np.arange(0.5,1.0,0.05)` | `:266` |
| AP | 101-point interpolation | `_compute_ap_101`, `:78` |
| Matching | single best-IoU GT per prediction, in prediction order | `:308-321` |
| maxDets | unlimited | — |
| Zero-GT classes | scored 0.0 and INCLUDED, unless excluded upstream | `per_class_metrics`, `:99` |
| Classes averaged | 8 (Mosaic Virus excluded, 0 GT after filtering) | `recompute_external_clean.py` |

Used for: Table 2 external column, Table 3.

### Are the two comparable?

**Yes, measured.** `scripts/verify/compare_evaluators.py` ran both over
identical predictions (RetinaNeXt seed 42, 100 images, 851 detections):

| | TorchMetrics | External | Difference |
|---|---|---|---|
| mAP@0.5 | 90.920 | 91.006 | **+0.086 pp** |
| mAP@0.5:0.95 | 75.396 | 75.445 | **+0.049 pp** |

Against in-domain-to-external drops of 13–31 pp, a 0.086 pp protocol bias is
negligible. The subtraction in Table 2 is sound as an absolute measurement.

The confidence-floor rows were identical, so no detection scored between 0.0
and 0.01; the entire difference comes from the matching rule and maxDets.

## 3. Confusion matrices — separate protocol again

`scripts/eval/confusion_matrix.py`.

| Setting | Value | Location |
|---|---|---|
| IoU | 0.50 | `:342` |
| **Confidence** | **0.25** | `:344` (CLI default) |
| Matching | greedy by **best IoU**, not by confidence | `:120-122` |
| Each GT claimed once | yes | `:129-133` |
| background **column** | unmatched GT = false negatives | `:139-141` |
| background **row** | unmatched predictions = false positives | `:134-136` |
| Normalisation | by **row** (ground-truth class) → recall | `figures/R/00_setup.R` |

### The 0.05 / 0.25 conflict — resolved

The function signature says `conf_thresh=0.05` (`:58`) while the CLI says
`default=0.25` (`:344`). `scripts/verify/verify_conf_threshold.py` settled it
by rebuilding the matrix at both thresholds and diffing against the stored
CSV:

```
conf_thresh = 0.05 : 72 of 100 cells differ, 21,422 vs 9,435 detections
conf_thresh = 0.25 :  0 of 100 cells differ, 9,435 vs 9,435   EXACT MATCH
```

**0.25 is the value that produced the published matrices.** The manuscript
and `figures/R/00_setup.R` both state 0.05; both are wrong.

Note also that matching is ordered by best IoU, not by confidence, so this is
not standard COCO matching and should not be described as such.

## Checkpoint selection

`src/training/trainer.py` saves `*_best.pth` on best validation mAP@0.5, with
EMA weights (decay 0.9999, ramp tau 2000) when `use_ema` is true, which is the
default. Early stopping patience 20 epochs. All evaluation uses `_best.pth`;
YOLOv8n's `last.pt` files exist but are unused.
