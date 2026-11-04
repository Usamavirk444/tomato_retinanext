# Reproducibility status

Every number in this repository is labelled with how it was obtained. Three
levels are used throughout, and nothing is described as reproduced unless it
was regenerated from a checkpoint on hardware other than the machine that
produced the original.

| Level | Meaning |
|---|---|
| **A** | Independently regenerated from a checkpoint in a fresh run and compared against the archived value |
| **B** | Reconstructed from archived per-class CSVs; no model was re-run |
| **C** | Statically verified — code read, manifests counted, hashes checked |

Fresh-run environment: Windows, NVIDIA A100-SXM4-40GB, Python/PyTorch
2.6.0+cu124, torchvision 0.21.0+cu124, torchmetrics 1.8.2, ultralytics 8.4.45,
numpy 2.2.6. Captured in `reports/environment_vm.txt` and
`reports/pip_freeze_vm.txt`.

---

## In-domain test set (2,487 images, 9 classes)

mAP@0.5, mean ± SD over seeds 42 / 123 / 456.

| Model | Fresh run | Published | Diff | Level |
|---|---|---|---|---|
| SSDLite-MobileNetV3-Large | 80.05 ± 0.16 | 80.24 | −0.19 | **A**, see note |
| YOLOv8n | 87.65 ± 0.29 | 87.65 | 0.00 | **A** |
| RetinaNet-R50-FPN-v2 | 89.16 ± 0.40 | 89.16 | 0.00 | **A** |
| RetinaNeXt | 94.42 ± 0.05 | 94.42 | 0.00 | **A** |

Nine of the twelve individual runs reproduced to four decimal places — every
seed of YOLOv8n, RetinaNet and RetinaNeXt is identical to the archived value,
not merely close.

**SSDLite note.** The published SSDLite figure was computed with the custom
per-class script, not TorchMetrics; no archived TorchMetrics result for
SSDLite on the full test set exists. The fresh run is therefore the first
same-protocol measurement rather than a failed reproduction. Full analysis:
`docs/SSDLITE_DEVIATION.md`.

Source: `outputs/phase_d_gpu/`.

## External benchmark (805 images, 7,307 boxes, 8 classes)

mAP@0.5 and mAP@0.5:0.95, mean ± SD over three seeds. Overlap-free subset,
no fine-tuning.

| Model | Fresh mAP@0.5 | Published | Fresh @0.5:0.95 | Published | Level |
|---|---|---|---|---|---|
| SSDLite | 39.21 ± 3.17 | 39.21 ± 3.17 | 27.34 ± 2.44 | 27.34 ± 2.44 | **A** |
| YOLOv8n | 38.25 ± 1.03 | 38.25 ± 1.03 | 26.01 ± 0.23 | 26.01 ± 0.23 | **A** |
| S1 RetinaNet-R50 | 39.31 ± 1.14 | 39.31 ± 1.14 | 28.82 ± 0.74 | 28.82 ± 0.74 | **A** |
| S1-L1 RetinaNet-R50 | 40.70 ± 1.38 | 40.70 ± 1.38 | 29.94 ± 0.95 | 29.94 ± 0.95 | **A** |
| S2 ConvNeXt-T + FPN | 62.01 ± 4.73 | 62.01 ± 4.73 | 46.49 ± 3.69 | 46.49 ± 3.69 | **A** |
| S3 + CBAM | 61.45 ± 2.97 | 61.45 ± 2.97 | 46.35 ± 2.61 | 46.35 ± 2.61 | **A** |
| S4 + BiFPN×3 | 61.89 ± 2.44 | 61.89 ± 2.44 | 46.82 ± 2.36 | 46.82 ± 2.36 | **A** |
| **S5 RetinaNeXt** | **64.67 ± 2.24** | 64.67 ± 2.24 | **49.70 ± 2.57** | 49.70 ± 2.57 | **A** |

Exact to every digit, standard deviations included. 88 internal consistency
checks passed, 0 failed (`outputs/cross_eval_3_clean_corrected/validation_report.txt`).

This was the manuscript's weakest evidence before Phase D.3 — level B for all
eight configurations — and is now its strongest.

Source: `outputs/cross_eval_3_clean/` (raw, 24 per-class CSVs) and
`outputs/cross_eval_3_clean_corrected/` (8-class re-average).

## Still level B or C

| Item | Level | Why |
|---|---|---|
| Per-class in-domain AP@0.5 (Table 1, Figure 4) | B | Fresh CSVs store AP@0.5:0.95 under the legacy `AP_` name; the fresh run used the pre-Fix-C evaluator |
| Confusion matrices (Figure 5) | B | Reproduced from archived matrices at conf 0.25 |
| Training curves (Figure 6) | B | Archived history CSVs; no retraining performed |
| FPS benchmark | B | Archived; hardware-dependent |
| Dataset composition (24,876 images, 67,595 boxes) | C | Manifest recounted against disk |
| Checkpoint integrity (24 required) | C | SHA-256 verified, `reports/preflight.json` |
| S3 seed 456 checkpoint selection | A | July epoch-68 file, hash-verified before load, reproduced 94.68 |

The per-class row is the one worth closing if you want Table 1 and Figure 4 at
level A: re-run the three in-domain models with the Fix C evaluator (the
`AP5095_` column plus corrected heading), which was not on the VM for this
run. Nothing else requires GPU time.

---

## External protocol, as actually implemented

The funnel is two independent reductions plus one class exclusion. All three
were verified by direct recount, not taken from the scripts' constants:

```
2,396  external images
−1,345  share a source photograph with training data
        (1,273 by source filename, 72 by flip/rotation-invariant
         thumbnail matching)
=1,051  overlap-free
  −246  retain no matched-class annotation after the class mapping
        (all 246 are Bacterial-Spot-only)
=  805  evaluated, 7,307 ground-truth boxes
        Mosaic Virus has 0 instances -> mAP averaged over 8 classes
```

The 246 are dropped at load time by `load_cross_dataset_3()`; they are never
inferred on and contribute neither ground truth nor false positives. The
class exclusion is separate and happens in `recompute_external_clean.py`.
This matches the manuscript's Section 4.6 exactly. Detail, including two
suggested wording footnotes: `docs/EXTERNAL_PROTOCOL_AUDIT.md`.

---

## Code corrections made during Phase D.3

Behaviour-preserving. Regression tests in `tests/test_phase_d3_fixes.py`.

- **Fix A** — `torch.mps.empty_cache()` was guarded by `hasattr`, which is
  true on CUDA builds because the module exists while the backend does not.
  It raised after the results CSV was written, aborting multi-seed loops
  after the first seed. Now guarded on real backend availability, with a CUDA
  branch.
- **Fix B** — `--checkpoint` was parsed and then ignored; the default path was
  always rebuilt. A reviewer asking for a specific file silently got another
  one. Now honoured, with a hard error on a missing file rather than a
  fallback.
- **Fix C** — per-class AP was printed and stored as "AP@0.50" when
  TorchMetrics `map_per_class` returns AP@0.50:0.95. An unambiguous
  `AP5095_<class>` column was added and the legacy `AP_<class>` alias kept so
  archived CSVs stay readable.
- **R figure scripts** (all 10) — resolved their own directory via
  `sys.frame(1)$ofile`, which R sets only under `source()`, never
  `Rscript`. The fallback was a hardcoded absolute path into
  `tomato_vm_clean/`, the read-only archive, which would have been written
  to. Now derived from `commandArgs()`.
- **`eval_external_clean.py`** — `import cross_dataset_ablation` failed
  because that file moved to `scripts/eval/` without the path being added.

No checkpoint, archived result, or manuscript file was modified at any point.
`tomato_vm_clean/` is unchanged.

---

## Not yet verified

- **Figures.** The six manuscript figures were regenerated in RStudio by the
  author. The SVG files have not been supplied to this repository and have
  **not** been inspected. `figures/R/figures/` is empty here.
- **Declarations block** — funding, author contributions, code availability
  statement. Awaiting author input.
