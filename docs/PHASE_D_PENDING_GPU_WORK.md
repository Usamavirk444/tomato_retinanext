# Pending GPU work — exact commands

Everything here needs the checkpoint archive and a CUDA machine. None of it
has been run. Results go into `reports/`; nothing under `reference_results/`
is overwritten.

Run them in this order — item 1 is the open scientific question.

---

## 1. S3 seed 456 — the open question

Two checkpoints exist for this cell. The **July** one is provisionally
canonical; see `S3_SEED456_INVESTIGATION.md`.

```bash
# confirm you have the right weights BEFORE evaluating
sha256sum outputs/checkpoints/retinanet_convnext_fpn_cbam/seed_456/*.pth
#   want 0e9f3dcfcb53c816895a83aecb249d03d3ec6ddc136f9c353c16a7a0b80efc07
#   NOT  52ee4088a89fb3364aad8079c22c1d0b2e85f57ddbaa9841091712158d0ce741
```

`evaluate.py --model retinanet_convnext_fpn_cbam --seed 456` resolves to
`outputs/checkpoints/` and will silently load whichever file is there. Put
the July checkpoint in place first, or evaluate by explicit path.

**Expected: 94.68 % mAP@0.5.**

- reproduces → S3 = 94.61 ± 0.10 confirmed; the abstract's claim that a
  configuration omitting BiFPN scores marginally higher in-domain holds.
- does not reproduce → Table 2, §4.4 and the abstract all need revisiting.

---

## 2. Main results, three seeds each

```bash
for m in ssdlite retinanet retinanet_l1 retinanet_hybrid; do
    python evaluate.py --model $m
done
python scripts/eval/eval_yolov8n_torchmetrics.py
```

Expected, mean over seeds 42/123/456:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| SSDLite-MobileNetV3 | 80.24 | 65.03 |
| YOLOv8n | 87.65 | 70.85 |
| RetinaNet-R50 | 89.16 | 77.03 |
| RetinaNeXt | 94.42 | 81.98 |

---

## 3. Ablation, in-domain

```bash
python ablation_convnext_baseline/evaluate_convnext.py   # S2  94.19
python ablation_fpn_cbam/evaluate_fpn_cbam.py            # S3  94.61
python ablation_bifpn_only/evaluate_bifpn_only.py        # S4  94.18
```

---

## 4. External benchmark, 805 images

```bash
for m in ssdlite yolov8n retinanet retinanet_l1 retinanet_hybrid; do
  for s in 42 123 456; do
    python cross_eval/cross_dataset_3_eval.py --model $m --seed $s
  done
done
python scripts/eval/cross_dataset_ablation.py --model all
python scripts/eval/recompute_external_clean.py
```

Expected: RetinaNeXt 64.67 ± 2.24, 805 images, 7,307 boxes, 8 classes.

Confirm first that the external set is the filtered one:

```bash
python scripts/data/filter_external_evaluable.py
```

---

## 5. Overlap-excluded subset

```bash
python evaluate.py --model retinanet_hybrid \
    --config configs/experiment_cleantest.yaml     # expect 92.82
```

---

## 6. Confusion matrices and FPS

```bash
python scripts/eval/confusion_matrix.py --model retinanet_hybrid \
    --seed 42 --conf-thresh 0.25        # 0.25 is NOT the function default
python scripts/eval/benchmark_fps.py
```

FPS is hardware-bound; the paper's figures are from an A100-SXM4-40GB and
will not reproduce on other hardware. Treat the ordering, not the absolute
values, as the result.

---

## 7. Protocol checks

```bash
python scripts/verify/verify_box_loss.py
python scripts/verify/verify_conf_threshold.py --model retinanet_hybrid --seed 42
python scripts/verify/compare_evaluators.py --model retinanet_hybrid --seed 42 --limit 100
```

Previously run on the VM: box losses as documented; conf 0.25 an exact
cell-for-cell match; evaluator difference 0.086 pp.

---

## 8. Promote the report to evidence level A

Collect the fresh numbers into JSON:

```json
{"retinanet_hybrid_seed42": {"mAP_50": 0.9438, "mAP_50_95": 0.8206}}
```

then:

```bash
python scripts/verify/reproducibility_report.py --fresh fresh_results.json \
    --out reports/
```

Rows backed by a fresh evaluation become **level A — independently
reproduced**. Until then every metric is level B, reconstructed from
archived CSVs, which confirms the manuscript transcribes its own results and
nothing more.

---

## 9. R figures

```bash
Rscript figures/R/run_all.R
python  scripts/figures/make_manuscript_figures.py
```

Needs R ≥ 4.2 with `tidyverse`, `patchwork`, `svglite`. Figure 1 also needs
the prepared dataset. Check each output for clipping and label collisions —
the circular per-class figure (manuscript Figure 4) is the one most prone to
overlapping labels.
