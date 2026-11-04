# Manuscript figure / table → script → source → output

The generating scripts use HISTORICAL figure numbers that do not match the
final manuscript. The mapping below is authoritative; the "script output"
column is the filename the script writes, and the "manuscript" column is what
it becomes in the paper.

## Figures

| Manuscript | Script | Source data | Script output |
|---|---|---|---|
| **Figure 1** Dataset composition and annotation geometry | `figures/R/06_dataset_eda.R` | raw YOLO labels in `data/clean_yolo_dataset/` | `Fig01_dataset_analysis.svg` |
| **Figure 2** RetinaNeXt architecture | `figures/architecture/make_fig03_svg.py` | derived from the model code, no CSV | `Fig03_RetinaNeXt_architecture.svg` |
| **Figure 3** Four detectors, five metrics | `figures/R/01_main_results.R` | `source_csv/results/main_results_table.csv` + confusion matrices | `Fig04_main_results.svg` |
| **Figure 4** Per-class AP@0.50 | `figures/R/02_per_class.R` | `source_csv/results/*/per_class_test.csv` | `Fig08_per_class_ap_circular.svg` |
| **Figure 5** Row-normalised confusion matrices | `figures/R/07_confusion_and_errors.R` | `source_csv/results/*/confusion_matrix_test.csv` | `Fig09_confusion_matrices.svg` |
| **Figure 6** Training curves, S1–S5 | `figures/R/05_training_and_pr.R` | `source_csv/logs/*/seed_*/\*history.csv` | `FigS1_training_curves.svg` |

Renaming is NOT yet automated. After `Rscript figures/R/run_all.R`, map by hand:

```
Fig01_dataset_analysis.svg          -> Figure1.svg
Fig03_RetinaNeXt_architecture.svg   -> Figure2.svg
Fig04_main_results.svg              -> Figure3.svg
Fig08_per_class_ap_circular.svg     -> Figure4.svg
Fig09_confusion_matrices.svg        -> Figure5.svg
FigS1_training_curves.svg           -> Figure6.svg
```

## Tables

| Manuscript | Script | Source data |
|---|---|---|
| **Table 1** mAP@0.5 / mAP@0.5:0.95, 4 models | `scripts/eval/build_results_table.py` | `reference_results/in_domain/main_results_table.csv` |
| **Table 2** Ablation S1–S5 + S1-L1, in-domain and external | in-domain CSVs + `reference_results/external/summary_cross_mean_std.csv` | both |
| **Table 3** External per-class AP@0.5, 8 classes | `scripts/eval/recompute_external_clean.py` | `reference_results/external/per_class_cross.csv` |

## Scripts producing figures NOT in the final manuscript

Retained because they are part of the analysis trail and are referenced by
`run_all.R`, but their outputs do not appear in the paper:

| Script | Output | Status |
|---|---|---|
| `03_ablation.R` | `Fig_ablation_indomain_vs_cross.svg` | supplementary |
| `04_cross_dataset.R` | `Fig14_cross_dataset.svg` | superseded by Table 3 |
| `05_training_and_pr.R` | `Fig07_pr_curves.svg` (second output) | supplementary |
| `08_speed.R` | `Fig13_accuracy_vs_speed.svg` | superseded by §4.5 text |

## Important: main_results_table.csv

Two versions exist in the original project and they disagree:

| File | YOLOv8n mAP@0.5 | Status |
|---|---|---|
| `outputs/results/main_results_table.csv` (May 2026) | 90.17 | **SUPERSEDED** |
| `R_diagrams/source_csv/results/main_results_table.csv` (Oct 2026) | 87.65 | **authoritative** |

This repository ships the October version as
`reference_results/in_domain/main_results_table.csv`. The May version is kept
as `main_results_table_SUPERSEDED_2026-05.csv` so the discrepancy stays
visible. The manuscript uses 87.65.
