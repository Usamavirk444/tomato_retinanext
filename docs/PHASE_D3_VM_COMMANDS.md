# Phase D.3 — remaining work, exact commands

Items 1 (code fixes) is complete. Items 2–5 need the GPU VM or an R
install and have **not** been run.

Copy these changed files to the VM first:

```
evaluate.py                       (fixes A and B)
src/evaluation/evaluator.py       (fix C)
tests/test_phase_d3_fixes.py      (new)
verify_all.sh                     (runs the new tests)
figures/R/*.R                     (path fix, all 10 scripts - see item 4)
```

Then confirm the fixes landed:

```powershell
python tests\test_phase_d3_fixes.py
```

Expect `0 failure(s)`. On the VM, where torch is installed, the
`B: runtime behaviour` test will also run instead of skipping — that is the
one that proves `--checkpoint` loads the file you name and errors on a
missing one.

---

## Item 2 — remaining in-domain evaluations

Fix A means multi-seed loops now work, so `--seed` can be omitted.

Run all three from the repository root:

```powershell
python evaluate.py --model ssdlite   --config configs\experiment_phase_d_gpu.yaml
python evaluate.py --model retinanet --config configs\experiment_phase_d_gpu.yaml
python scripts\eval\eval_yolov8n_torchmetrics.py
```

RetinaNet seed 42 already ran. Re-running it costs minutes and gives a
clean provenance record for all three seeds from one invocation, which is
worth more than the time saved by skipping it. If you would rather not,
retrieve the existing CSV from
`outputs\phase_d_gpu\retinanet\retinanet_seed_42_eval_results.csv`.

Expected, mean over the three seeds:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| SSDLite-MobileNetV3-Large | 80.24 | 65.03 |
| YOLOv8n | 87.65 | 70.85 |
| RetinaNet-ResNet50-FPN-v2 | 89.16 | 77.03 |

`eval_yolov8n_torchmetrics.py` takes no `--config`; it defaults to all three
seeds and writes to `outputs\results\yolov8n_torchmetrics\seed_<N>\`, which
already holds the published run and which it will **overwrite without
warning**. Redirect it:

```powershell
python scripts\eval\eval_yolov8n_torchmetrics.py `
    --out-root outputs\phase_d_gpu\yolov8n_torchmetrics
```

The flag is `--out-root`, not `--out-dir`.

---

## Item 3 — external evaluation, the scientific priority

This is a three-step chain, and the steps must run in this order — the
middle one is the only script that writes what the last one reads. Do **not**
call `cross_eval\cross_dataset_3_eval.py` directly: it writes to
`outputs\cross_eval_3\`, which is the unfiltered set that still contains the
source-image overlap, and `recompute_external_clean.py` will not see those
files at all.

```powershell
python scripts\data\build_clean_crossdataset.py
python scripts\eval\eval_external_clean.py
python scripts\eval\recompute_external_clean.py
```

- **Step 1** builds `data\cross_dataset_3_clean\` — expect **1,051 images**.
- **Step 2** runs all 8 stages × 3 seeds in one invocation (it imports
  `cross_dataset_3_eval.py` and `cross_dataset_ablation.py` rather than
  modifying them) and writes 24 `per_class_cross.csv` files to
  `outputs\cross_eval_3_clean\`. It handles the S3 seed-456 checkpoint
  correctly on its own: it uses the recovered July epoch-68 file and verifies
  its SHA-256 before loading. This is the long step — budget hours, not
  minutes.
- **Step 3** re-averages over the 8 classes that actually have ground truth,
  excluding Mosaic Virus (n_gt = 0), and writes
  `outputs\cross_eval_3_clean_corrected\`.

Step 3 refuses to overwrite a non-empty output folder. If it stops, **move**
the existing `outputs\cross_eval_3_clean_corrected\` aside — do not delete
it; it is the published result.

Optionally, to document the evaluable subset:

```powershell
python scripts\data\filter_external_evaluable.py
```

Expect `805 images, 7,307 boxes, 8 classes`, all four checks PASS. This is a
verification artefact only — step 3 does not read it, so skipping it changes
no number.

Expected:

| Model | External mAP@0.5 |
|---|---|
| SSDLite | 39.21 ± 3.17 |
| YOLOv8n | 38.25 ± 1.03 |
| RetinaNet-R50 | 39.31 ± 1.14 |
| RetinaNeXt | 64.67 ± 2.24 |

This is the paper's key claim and currently its weakest evidence — level B
for every model.

---

## Item 4 — the six figures

Any machine with R ≥ 4.2 and the prepared dataset.

```powershell
Rscript -e "install.packages(c('ggplot2','dplyr','tidyr','readr','scales','patchwork','svglite'), repos='https://cloud.r-project.org')"
```

```powershell
Rscript figures\R\run_all.R
python scripts\figures\make_manuscript_figures.py
```

Figures land in `figures\R\figures\`, not `figures\final\`.

A path bug in these scripts was fixed as part of this phase, so if you are
working from an older copy, take the new `figures/R/` with you. All ten
scripts resolved their own directory via `sys.frame(1)$ofile`, which R sets
only when a file is `source()`d — never under `Rscript`. The fallback was a
hardcoded absolute path to `tomato_vm_clean/R_diagrams`, so running any of
them with `Rscript` on this Mac would have read from, and written figures
into, the read-only research archive; on the VM or any other machine it would
simply have failed. The fallback now derives the directory from
`commandArgs()`, and the `source()`d path is unchanged.

Check each output for clipping and overlapping labels; the circular per-class
plot (Figure 4) is the one most prone to label collisions. Figure 1 needs the
prepared dataset present.

---

## Item 5 — capture the environment

Run on the VM and save the output:

```powershell
python -c "import platform,torch,torchvision;print(platform.platform());print('python',platform.python_version());print('torch',torch.__version__);print('torchvision',torchvision.__version__);print('cuda',torch.version.cuda);print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" > reports\environment_vm.txt
python -c "import torchmetrics,ultralytics,numpy;print('torchmetrics',torchmetrics.__version__);print('ultralytics',ultralytics.__version__);print('numpy',numpy.__version__)" >> reports\environment_vm.txt
nvidia-smi >> reports\environment_vm.txt
pip freeze > reports\pip_freeze_vm.txt
```

This is a real gap in the current evidence chain: nothing on record says
which software stack produced the fresh numbers.

---

## What to copy back

```
outputs\phase_d_gpu\                          fresh in-domain CSVs (ssdlite, retinanet)
outputs\phase_d_gpu\yolov8n_torchmetrics\     fresh YOLOv8n CSVs
outputs\cross_eval_3_clean\                   the 24 per_class_cross.csv files
outputs\cross_eval_3_clean_corrected\         re-averaged external results
reports\environment_vm.txt
reports\pip_freeze_vm.txt
reports\preflight.json
figures\R\figures\                            the figure SVGs
```

Bring `cross_eval_3_clean\` back as well as the corrected folder — it is the
raw per-class evidence, and without it the corrected summary cannot be
re-derived or checked.

Then here:

```bash
python scripts/verify/reproducibility_report.py --fresh fresh_results.json --out reports/
```

Rows backed by a fresh evaluation become **evidence level A**.
