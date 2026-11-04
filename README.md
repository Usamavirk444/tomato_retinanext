# RetinaNeXt — multi-class tomato leaf disease detection

Reproduction code for *RetinaNeXt: A Single-Stage Detector for Multi-Class
Tomato Leaf Disease Detection with Improved Cross-Dataset Generalization*.

RetinaNeXt extends RetinaNet-ResNet50-FPN-v2 with a ConvNeXt-Tiny backbone, a
three-repeat BiFPN in place of the standard FPN, and CBAM at all five pyramid
levels. On a nine-class, 24,876-image benchmark it reaches **94.42 % ± 0.05
mAP@0.5**. On an independent, overlap-free external benchmark (805 images, no
fine-tuning) it reaches **64.67 % ± 2.24** — a drop of 29.75 pp, against
41.03 pp for SSDLite, 49.40 pp for YOLOv8n and 49.85 pp for RetinaNet-R50.

> **Before relying on any number here**, read
> [`docs/REPRODUCIBILITY_STATUS.md`](docs/REPRODUCIBILITY_STATUS.md). It
> states exactly what has been verified by execution, what by code inspection
> only, and what still needs a GPU.

Every external-benchmark figure above was independently regenerated from the
checkpoints on separate hardware (NVIDIA A100, torch 2.6.0+cu124) and matched
the archived values exactly, standard deviations included. Three of the four
in-domain figures reproduced to four decimal places on every seed; the
SSDLite row differs by 0.19 pp for a known reason documented in
[`docs/SSDLITE_DEVIATION.md`](docs/SSDLITE_DEVIATION.md).

---

## Contents

```
configs/            experiment configurations
src/                models, dataset, training, evaluation
ablation_*/         S2, S3, S4 ablation implementations
cross_eval/         external benchmark evaluator
scripts/
  data/             dataset preparation and filtering
  figures/          one command for all six manuscript figures
  eval/             evaluation, tables, FPS, confusion matrices
  verify/           protocol checks and validation against the paper
figures/
  R/                R figure scripts (00_setup … 08_speed, run_all)
  architecture/     editable architecture diagrams + generator
  source_csv/       data the R scripts read
manifests/          split membership, overlap sets, checkpoint inventory
reference_results/  the CSVs behind every published number (~1.8 MB)
docs/               protocols, attribution, limitations, audits
tests/              import and manifest checks
```

---

## 1. Environment

```bash
git clone <repository-url>
cd RetinaNeXt_Reproducible

conda create -n retinanext python=3.10 -y
conda activate retinanext

# CPU / Apple Silicon
pip install -r requirements.txt

# CUDA
pip install -r requirements_cuda.txt
```

Verify:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Both files include `ultralytics==8.4.45`, which the YOLOv8n baseline needs —
`train_all.py` imports it at module level, so it is required even for runs
that do not touch YOLOv8n. **Ultralytics is AGPL-3.0**; see
[`docs/LICENSING_AUDIT.md`](docs/LICENSING_AUDIT.md) before redistributing
anything built with it.

`requirements.txt` pins **torchvision 0.25.0**. This matters: the
`giou` / `l1` box-loss default of `retinanet_resnet50_fpn_v2` is
version-dependent and affects the S1 baseline. See
[`docs/EVALUATION_PROTOCOLS.md`](docs/EVALUATION_PROTOCOLS.md).

> **Which stack produced the published numbers.** The verified Phase D.3
> results came from `requirements_cuda.txt` — **torch 2.6.0+cu124,
> torchvision 0.21.0+cu124**, on an NVIDIA A100-SXM4-40GB. `requirements.txt`
> pins newer CPU builds (torch 2.10.0, torchvision 0.25.0) for convenience on
> machines without CUDA; results from that stack have not been verified
> against the manuscript. Use `requirements_cuda.txt` to reproduce the
> reported figures. Exact versions: `reports/environment_vm.txt` and
> `reports/pip_freeze_vm.txt`.

R figures additionally need R ≥ 4.2 with `tidyverse`, `patchwork`, `svglite`.

---

## 2. Datasets

Both are CC BY 4.0 and must be downloaded manually — see
[`docs/DATASET_ATTRIBUTION.md`](docs/DATASET_ATTRIBUTION.md) for full
provenance and citation requirements.

**Training** — Science Data Bank, DOI `10.57760/sciencedb.agriculture.00246`,
obtained via Roboflow `tdg-cikuh/tdg-vq3xp`. 27,297 images, 11 classes.

**External** — Roboflow `forfun-cyh93/tomatoes_diseases` v1. 2,396 images,
YOLOv8-OBB format.

### Prepare the nine-class benchmark

```bash
python scripts/data/prepare_nine_class.py \
    --source /path/to/downloaded/tomato_27297 \
    --dest   data/clean_yolo_dataset
```

This removes classes 9 (Bacterial Spot) and 10 (Iron Deficiency) and restores
the **exact** original split from
`manifests/split_manifest_clean_yolo_dataset.csv` — 17,419 / 4,970 / 2,487.
It does not re-randomise. If any file named in the manifest is missing it
reports every one and exits rather than guessing.

Expected: `24,876 images, 67,595 boxes, 9 classes`.

### Overlap-excluded test subset and external benchmark

```bash
python scripts/data/build_clean_testset.py       # 1,933-image subset
python scripts/data/build_clean_crossdataset.py  # 2,396 -> 1,051 kept
```

### Validate

```bash
python scripts/verify/validate_against_paper.py
```

Checks split sizes, box counts, class mapping, overlap sets, external
filtering, and all published metrics against the reference CSVs.

---

## 3. Checkpoints

**Not in this repository** — 7.86 GB across 24 files.
`manifests/checkpoint_inventory.csv` lists every one with model, seed, size,
SHA-256 and destination path. The external archive is not yet published; see
[`docs/REPRODUCIBILITY_STATUS.md`](docs/REPRODUCIBILITY_STATUS.md).

To evaluate without retraining, place them under
`outputs/checkpoints/<model>/seed_<n>/` as recorded in the manifest, then
follow [`QUICKSTART.md`](QUICKSTART.md).

---

## 4. Training

All experiments use seeds **42, 123, 456**. Omitting `--seed` runs all three.

```bash
# baselines
python train_all.py --model ssdlite
python train_all.py --model yolov8n
python train_all.py --model retinanet            # S1, GIoU box loss

# proposed
python train_all.py --model retinanet_hybrid     # S5 RetinaNeXt

# S1-L1: S1 with the L1 box loss the ConvNeXt stages use
python train_all.py --model retinanet_l1

# single seed
python train_all.py --model retinanet_hybrid --seed 42

# 2-epoch smoke test, separate outputs
python train_all.py --model retinanet_hybrid --smoke-test --seed 42
```

Ablation stages have their own scripts:

```bash
python ablation_convnext_baseline/train_convnext_baseline.py --seed 42  # S2
python ablation_fpn_cbam/train_fpn_cbam.py --seed 42                    # S3
python ablation_bifpn_only/train_bifpn_only.py --seed 42                # S4
```

Outputs go to `outputs/{checkpoints,logs}/<model>/seed_<n>/`.

Roughly **30 h per model for three seeds on an A100**.

### What the training actually does

AdamW, lr 1e-4, weight decay 1e-4, batch 8, 70 epochs, 5 warm-up epochs,
cosine annealing to **`eta_min=1e-7`**, EMA (decay 0.9999), early stopping
patience 20, input 640×640.

The manuscript's §3.3 states a final LR of 1×10⁻⁶; the code uses 1e-7 for
every torchvision model, and only YOLOv8n reaches 1e-6. The code is
authoritative — see
[`docs/LEARNING_RATE_DISCREPANCY.md`](docs/LEARNING_RATE_DISCREPANCY.md).

---

## 5. Evaluation

```bash
# full in-domain test set (2,487 images), TorchMetrics
python evaluate.py --model retinanet_hybrid

# overlap-excluded subset (1,933 images)
python evaluate.py --model retinanet_hybrid --config configs/experiment_cleantest.yaml

# YOLOv8n under TorchMetrics, for comparability with the others
python scripts/eval/eval_yolov8n_torchmetrics.py

# external benchmark (805 images, 8 classes)
python cross_eval/cross_dataset_3_eval.py --model retinanet_hybrid --seed 42
python scripts/eval/recompute_external_clean.py     # aggregate all 8 configs

# ablation stages on the external benchmark
python scripts/eval/cross_dataset_ablation.py --model all

# per-class AP, confusion matrices, PR curves
python scripts/eval/evaluate_per_class.py
python scripts/eval/confusion_matrix.py --model retinanet_hybrid --seed 42 --conf-thresh 0.25
python scripts/eval/make_real_pr_curves.py

# speed and parameters
python scripts/eval/benchmark_fps.py

# Table 1
python scripts/eval/build_results_table.py
```

**Pass `--conf-thresh 0.25` explicitly** for confusion matrices. The function
default is 0.05 but the published matrices used the CLI default of 0.25,
verified by `scripts/verify/verify_conf_threshold.py`.

Three distinct evaluation protocols are in use and are deliberately not
merged. [`docs/EVALUATION_PROTOCOLS.md`](docs/EVALUATION_PROTOCOLS.md)
documents each.

### Protocol checks

```bash
python scripts/verify/verify_box_loss.py          # GIoU vs L1 per model
python scripts/verify/verify_conf_threshold.py --model retinanet_hybrid --seed 42
python scripts/verify/compare_evaluators.py --model retinanet_hybrid --seed 42 --limit 100
```

---

## 6. Figures and tables

```bash
Rscript figures/R/run_all.R
python  figures/architecture/make_fig03_svg.py   # Figure 2
```

Generating scripts use **historical** figure numbers. The mapping to final
manuscript numbering is in
[`docs/FIGURE_TABLE_MAP.md`](docs/FIGURE_TABLE_MAP.md) — renaming is manual.

| Manuscript | Script output |
|---|---|
| Figure 1 | `Fig01_dataset_analysis.svg` |
| Figure 2 | `Fig03_RetinaNeXt_architecture.svg` |
| Figure 3 | `Fig04_main_results.svg` |
| Figure 4 | `Fig08_per_class_ap_circular.svg` |
| Figure 5 | `Fig09_confusion_matrices.svg` |
| Figure 6 | `FigS1_training_curves.svg` |

Figure 1 needs the prepared dataset; the rest read `figures/source_csv/`.

---

## 7. Limitations

[`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) covers all of them.
The ones that most affect interpretation:

- **Source-image overlap** — the dataset is pre-augmented, so 22.3 % of test
  images share a source photograph with training. Both full and
  overlap-excluded results are reported.
- **YOLOv8n trains differently** — Ultralytics applies augmentation the
  torchvision models do not.
- **S3 seed 456 is unresolved** — two checkpoints give 94.68 and 91.23. The
  July one is provisionally canonical, pending independent evaluation.
- **S1 → S2 changes backbone and box loss together**; S1-L1 isolates it, but
  a head-normalisation difference remains uncontrolled.

---

## Citation

```bibtex
@article{retinanext2026,
  title   = {RetinaNeXt: A Single-Stage Detector for Multi-Class Tomato Leaf
             Disease Detection with Improved Cross-Dataset Generalization},
  note    = {Manuscript},
  year    = {2026}
}
```

Please also cite the datasets — see
[`docs/DATASET_ATTRIBUTION.md`](docs/DATASET_ATTRIBUTION.md).

## License

**The original source code in this repository is licensed under the GNU
Affero General Public License v3.0** — see [`LICENSE`](LICENSE) for the full
text.

AGPL-3.0 was chosen because the repository depends on
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0):
`train_all.py` imports it at module level, so the distributed work is a
combined work. No Ultralytics or torchvision source is copied or vendored
here — the exposure is dependency and runtime integration only. Note
AGPL-3.0 §13: if this code is ever deployed so that users interact with it
over a network, they must be offered the corresponding source.

**This licence covers the code in this repository and nothing else.** Four
other things are governed separately, and AGPL-3.0 grants no rights over any
of them:

| | Governed by |
|---|---|
| **Third-party dependencies** (torch, torchvision, ultralytics, …) | their own licences; installed from their own sources, not redistributed here |
| **Pretrained weights** used for initialization (torchvision COCO / ImageNet checkpoints) | **terms unstated upstream.** torchvision's documentation says its pretrained models "may have their own licenses or terms and conditions derived from the dataset used for training" |
| **Datasets** | CC BY 4.0, belonging to their publishers. This repository contains no dataset images or labels — only filename manifests, which are factual records |
| **Trained checkpoints** (the Zenodo deposit) | licensed separately; the deposit documents the fine-tuning as the authors' contribution and discloses the unresolved upstream status rather than asserting rights over it |

Licensing this repository AGPL-3.0 **does not** grant anyone redistribution
rights over the pretrained weights it was built from, nor over the trained
checkpoints. Full analysis, including the unresolved items:
[`docs/LICENSING_AUDIT.md`](docs/LICENSING_AUDIT.md).

---

## Verification status of every documented command

| Command | Status |
|---|---|
| `python scripts/verify/validate_against_paper.py` | **EXECUTED** — 50 PASS, 0 FAIL, 2 SKIP, 3 DISCREPANCY |
| `python scripts/verify/validate_against_paper.py --suite {dataset,results,protocol,manuscript}` | **EXECUTED** |
| `python tests/test_repo_integrity.py` | **EXECUTED** — 0 failures |
| `python tests/test_dataset_preparation.py` | **EXECUTED** — 0 failures |
| `python scripts/data/filter_external_evaluable.py` | **EXECUTED** — 1,051 → 805 / 7,307 / 8 classes |
| `python scripts/eval/build_tables.py` | **EXECUTED** — Tables 1–3 reproduce |
| `python scripts/figures/make_manuscript_figures.py --check` | **EXECUTED** |
| `python scripts/figures/make_manuscript_figures.py --skip-r` | **EXECUTED** — Figure 2 produced |
| `RETINANEXT_LOGS_DIR=reference_results/logs python scripts/eval/build_results_table.py` | **SMOKE-TESTED** — runs; reports *validation* mAP, not Table 1 |
| `python scripts/data/prepare_nine_class.py` | **SMOKE-TESTED** — missing-file path verified with fixtures; full run needs the 27,297-image source |
| `Rscript figures/R/run_all.R` | **INSPECTED ONLY** — R not available in the build environment |
| `python train_all.py --model …` | **GPU REQUIRED** |
| `python evaluate.py --model …` | **GPU REQUIRED** — also needs the checkpoint archive |
| `python cross_eval/cross_dataset_3_eval.py` | **GPU REQUIRED** |
| `python scripts/eval/benchmark_fps.py` | **GPU REQUIRED** |
| `python scripts/verify/verify_box_loss.py` | **GPU-FREE but needs torch** — previously run on the VM, all PASS |
| `python scripts/verify/verify_conf_threshold.py` | **needs torch + checkpoint** — previously run on the VM: conf 0.25 exact match |
| `python scripts/verify/compare_evaluators.py` | **needs torch + checkpoint** — previously run: 0.086 pp difference |

Full detail in [`docs/REPRODUCIBILITY_STATUS.md`](docs/REPRODUCIBILITY_STATUS.md).

### Generating everything from the shipped reference data

No GPU, no dataset, no checkpoints:

```bash
python scripts/verify/validate_against_paper.py   # all published metrics
python scripts/eval/build_tables.py --out tables/ # Tables 1-3
python scripts/figures/make_manuscript_figures.py # Figures 1-6 (needs R)
```
