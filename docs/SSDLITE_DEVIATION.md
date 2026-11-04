# SSDLite mAP@0.5 deviation — investigation closed, cause identified

**Finding: this is not a failed reproduction. The published SSDLite mAP@0.5
was never produced by the TorchMetrics evaluator, so there was no
same-protocol number to reproduce.**

Fresh GPU run (A100, torch 2.6.0+cu124, torchvision 0.21.0+cu124), full
2,487-image test set:

| seed | fresh (TorchMetrics) | published | diff |
|---|---|---|---|
| 42 | 79.86 | 80.0385 | −0.18 |
| 123 | 80.17 | 80.4841 | −0.31 |
| 456 | 80.11 | 80.2004 | −0.09 |
| **mean** | **80.05** | **80.24** | **−0.19** |

---

## What the evidence shows

### Every other model has an archived TorchMetrics file. SSDLite does not.

`tomato_vm_clean/outputs/results/` contains `*_eval_results.csv` for
RetinaNet, RetinaNet-L1, all three ConvNeXt ablations and RetinaNeXt. Each
stores `mAP_50` at four decimal places, and each matches the published table
**exactly**:

```
retinanet        s42 0.8953  s123 0.8921  s456 0.8874   -> table: 0.8953 / 0.8921 / 0.8874
retinanet_hybrid s42 0.9438  s123 0.9448  s456 0.9440   -> table: 0.9438 / 0.9448 / 0.9440
```

For SSDLite the directory holds exactly one file — `seed_42`, at
`mAP_50 = 0.5751`. That is 22 pp below the published value and there is no
seed 123 or 456 at all. It is an early or broken run, not the source of the
published numbers.

### The published SSDLite numbers came from the custom per-class script

In `main_results_table.csv` the SSDLite entries carry full float precision
while the others are 4 dp:

```
SSDLite-MobileNetV3-Large   seed_42_map50 = 0.8003854708715455
RetinaNet-ResNet50-FPN-v2   seed_42_map50 = 0.8953
RetinaNet-Hybrid v3         seed_42_map50 = 0.9438
```

`0.8003854708715455` is the arithmetic mean of the nine `ap50` values in
`ssdlite/seed_42/per_class_test.csv`, reproduced to the last digit. The
4 dp entries are TorchMetrics values copied from the eval CSVs.

So Table 1 mixes two evaluation implementations: **SSDLite from the custom
per-class script, the other three from TorchMetrics.**

### Why SSDLite is the model where the two implementations diverge

The two differ in matching rule (best-IoU vs greedy-by-score), confidence
floor, and detection cap. Those differences bite hardest on a model with a
long low-confidence tail, which is exactly what SSDLite has (seed 42, test
set):

| model | predictions | GT boxes | pred/GT | mean precision |
|---|---|---|---|---|
| SSDLite | 101,134 | 6,861 | **14.7** | **0.081** |
| RetinaNet-R50 | 41,691 | 6,861 | 6.1 | 0.179 |
| RetinaNeXt | 21,195 | 6,861 | 3.1 | 0.377 |

SSDLite emits roughly five times as many predictions per ground-truth box as
RetinaNeXt and its mean precision at full recall is a fifth. The 0.19 pp gap
is an inter-implementation difference concentrated on the model most exposed
to it.

### torchvision is ruled out as the primary cause

RetinaNet, RetinaNet-L1 and all three ConvNeXt ablations run through the same
torchvision detection stack at the same version, and every one of them
reproduced to 4 dp. A torchvision postprocessing change would have moved them
too. (This does not exclude a small version effect inside the 0.19 pp, but it
cannot be the main term.)

### Not investigated to completion

Which specific implementation difference contributes how much of the 0.19 pp
was not decomposed — that would need both evaluators run on one stored set of
SSDLite predictions, which is outside this investigation's scope. The
provenance finding above is sufficient to explain why the numbers differ, and
no threshold, model setting or checkpoint was altered at any point.

---

## How to report this

The deviation itself is immaterial: 0.19 pp, smaller than the seed-to-seed SD
of several other rows, and it changes no ordering or conclusion. The abstract's
"outperforming SSDLite by 14.18 pp" becomes 14.37 pp on same-protocol numbers.

The **provenance inconsistency** is the part worth a footnote, because a
reviewer re-running `evaluate.py` will get 80.05, not 80.24:

> SSDLite mAP@0.5 was computed as the mean of per-class AP@0.5 from the
> per-class evaluation script, whereas the other detectors' values are
> TorchMetrics `map_50`. Evaluated under TorchMetrics, SSDLite attains
> 80.05% ± 0.16% mAP@0.5.

Two options, both defensible — your call:

1. **Keep 80.24, add the footnote.** Nothing in the manuscript changes except
   one clarifying sentence. Least disruption.
2. **Report 80.05 so all four rows share one protocol.** Internally
   consistent, but shifts the abstract's 14.18 → 14.37 and Table 1.

I lean to (1) for a manuscript at this stage, with the footnote doing the
honest work. (2) is the stronger choice if a reviewer is likely to re-run the
code, since the repository will produce 80.05.

**Status: investigation closed. Cause identified (mixed evaluator provenance),
not an unresolved deviation.**
