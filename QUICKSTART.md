# Quick start — evaluate existing checkpoints, no retraining

For reproducing the paper's numbers from released checkpoints. Full
instructions are in [`README.md`](README.md); retraining takes ~30 h per model.

## 1. Environment (5 min)

```bash
conda create -n retinanext python=3.10 -y && conda activate retinanext
pip install -r requirements_cuda.txt      # or requirements.txt for CPU
```

## 2. Validate the repository (10 s, no data or GPU needed)

```bash
python scripts/verify/validate_against_paper.py
```

Expect **50 PASS, 1 FAIL, 2 SKIP**. The FAIL is deliberate — it flags the
known §3.3 learning-rate discrepancy. The SKIPs are the disputed S3 seed-456
result and the on-disk dataset if you have not built it.

This alone confirms every published metric against the reference CSVs,
without downloading anything.

## 3. Datasets (~1 h, mostly download)

```bash
# after downloading both sources - see docs/DATASET_ATTRIBUTION.md
python scripts/data/prepare_nine_class.py --source /path/to/tomato_27297 \
                                          --dest data/clean_yolo_dataset
python scripts/data/build_clean_testset.py
python scripts/data/build_clean_crossdataset.py
python scripts/verify/validate_against_paper.py      # now checks on-disk too
```

## 4. Checkpoints

Not yet published. `manifests/checkpoint_inventory.csv` lists all 24 required
files with SHA-256. Place them at:

```
outputs/checkpoints/<model>/seed_<n>/<name>_best.pth
```

Verify after download:

```bash
sha256sum -c manifests/checkpoint_sha256.txt
```

**S3 seed 456**: use the July checkpoint `0e9f3dcfcb53c816…`, not the
September one. See [`docs/S3_SEED456_INVESTIGATION.md`](docs/S3_SEED456_INVESTIGATION.md).

## 5. Evaluate (minutes per model on a GPU)

```bash
python evaluate.py --model retinanet_hybrid                 # 94.42 expected
python evaluate.py --model retinanet                        # 89.16
python evaluate.py --model ssdlite                          # 80.24
python scripts/eval/eval_yolov8n_torchmetrics.py            # 87.65

# external benchmark
python scripts/eval/recompute_external_clean.py             # RetinaNeXt 64.67

# Table 1
python scripts/eval/build_results_table.py
```

## 6. Figures

```bash
Rscript figures/R/run_all.R                   # needs R + tidyverse
python  figures/architecture/make_fig03_svg.py
```

Rename per [`docs/FIGURE_TABLE_MAP.md`](docs/FIGURE_TABLE_MAP.md) — scripts
use historical numbering.

---

## Expected headline numbers

| | in-domain mAP@0.5 | external mAP@0.5 |
|---|---|---|
| SSDLite-MobileNetV3 | 80.24 ± 0.23 | 39.21 ± 3.17 |
| YOLOv8n | 87.65 ± 0.29 | 38.25 ± 1.03 |
| RetinaNet-R50 | 89.16 ± 0.40 | 39.31 ± 1.14 |
| **RetinaNeXt** | **94.42 ± 0.05** | **64.67 ± 2.24** |

If your numbers differ, check torchvision is 0.25.0 and that you used the
July S3 checkpoint where relevant.
