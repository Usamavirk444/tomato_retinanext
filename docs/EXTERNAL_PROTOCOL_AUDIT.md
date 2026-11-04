# External evaluation protocol — read-only audit

Phase D.3, pre-execution. No GPU work was run. No file in `tomato_vm_clean/`
was modified; the archive was read only to recount label files.

---

## 1. Can `eval_external_clean.py` overwrite existing results?

**Partly — and the unprotected files are exactly the two that matter for the
headline numbers.**

Per-seed result directories **are** protected. Before loading any model
(lines 120–124) the script checks every `<model>/seed_<n>/` it is about to
write and calls `sys.exit` if one is non-empty:

```python
for m in models:
    for s in seeds:
        d = os.path.join(OUT_ROOT, m, f"seed_{s}")
        if os.path.isdir(d) and os.listdir(d):
            sys.exit(f"ERROR: {d} already has results - refusing to overwrite.")
```

This runs as a complete pass before any compute, so it aborts cleanly rather
than part-way through.

The two **aggregate** CSVs are **not** protected. Lines 144 and 155 open them
with mode `"w"`:

```
outputs/cross_eval_3_clean/summary_per_run.csv
outputs/cross_eval_3_clean/summary_aggregate.csv
```

Both exist in the archive (`tomato_vm_clean/outputs/cross_eval_3_clean/`) and
would be truncated and rewritten without any warning. There is no `--out-root`
flag; `OUT_ROOT` is a module-level constant at line 55.

Note the filename interaction at line 143: a full run writes
`summary_per_run.csv`, while a partial run (`--model` or `--seed` given) writes
`summary_per_run_partial.csv`. So a *partial* re-run cannot clobber the
archived aggregates, but a full one can.

### Safe procedure — no code change needed

The clean way is to never put the archived results where the script writes.
`outputs/cross_eval_3_clean/` is **not** present in
`RetinaNeXt_Reproducible/`; it exists only in the archive and in
`reference_results/external/`. So:

1. **Do not copy `outputs/cross_eval_3_clean/` to the VM.** With the folder
   absent, the per-seed guard passes, the fresh run writes all 24 CSVs plus
   both aggregates into empty space, and nothing archived is reachable.
2. The archived copies stay here, in two places already:
   `reference_results/external/` (in the repo) and
   `tomato_vm_clean/outputs/cross_eval_3_clean/` (the archive).
3. Bring the fresh results back into a **new** folder and compare, rather than
   merging into either archived copy.

If a `cross_eval_3_clean/` folder does turn out to be on the VM already, rename
it — `Rename-Item outputs\cross_eval_3_clean outputs\cross_eval_3_clean_ARCHIVED`
— rather than deleting it, then run.

---

## 2. Protocol trace, and the 246 Bacterial-Spot-only images

**The 246 images are genuinely excluded from the AP calculation.** They are not
evaluated and then discounted — they never enter the evaluation at all. The
exclusion happens one stage earlier than your question supposes, and it is not
`recompute_external_clean.py` that does it.

### The three stages

**Stage 1 — `build_clean_crossdataset.py`: 2,396 → 1,051 images.**
Copies only the files named in `kept_external_images.txt`. Removes images
sharing a source photograph with the training data. Asserts the keep-list has
exactly 1,051 entries and refuses to overwrite an existing destination.

**Stage 2 — `eval_external_clean.py` → `load_cross_dataset_3()`: 1,051 → 805
images.** This is where the 246 go. Two filters compose:

`parse_obb_label_file()` (cross_dataset_3_eval.py:146) drops every *box* whose
class is not one of the nine matched classes:

```python
if cls not in OBB_INDEX_TO_OURS:
    continue
```

`OBB_INDEX_TO_OURS` omits index 0 (Bacterial Spot) and index 3 (Iron
Deficiency). Then `load_cross_dataset_3()` (line 180) drops every *image* left
with no boxes:

```python
normalised_boxes = parse_obb_label_file(lbl_path)
if not normalised_boxes:
    skipped += 1
    continue
```

An image annotated only with Bacterial Spot therefore yields an empty box list
and is skipped entirely — no inference is run on it, and it contributes neither
ground truth nor false positives.

**Stage 3 — `recompute_external_clean.py`: 9 → 8 classes in the average.**
This stage does *not* touch images. It re-averages the stored per-class APs
over the classes with `n_gt > 0`, excluding Mosaic Virus. The original
`per_class_metrics()` assigned `ap50 = 0.0` when `n_gt == 0` and `run_one()`
averaged all nine classes, so a meaningless zero was dragged into every mAP.

### Verified by recount

I re-implemented stages 2's two filters exactly, without torch, and ran them
over the archive's `data/cross_dataset_3_clean/` (read-only):

```
evaluated: 805 images
skipped  : 246 images
GT boxes : 7307
classes with GT: 8

why images were skipped (their only label classes):
    246  Bacterial Spot
```

Per-class ground truth: Early Blight 632, Healthy 4,353, Late Blight 3,
Leaf Miners 60, Leaf Mold 93, **Mosaic Virus 0**, Septoria 18,
Spider Mites 1,749, Yellow Leaf Curl Virus 399.

All 246 skipped images were Bacterial-Spot-only; no image was lost to Iron
Deficiency or to the degenerate-box filter.

So the answer to your question is the first option, not the second: the
procedure evaluates **805** images, and separately excludes Mosaic Virus from
class averaging. Both reductions are real and they are independent.

---

## 3. Do the metrics correspond to the manuscript's 805-image benchmark?

**Yes.** Section 4.6 states:

> The original external dataset contained 2,396 images labelled with 11
> classes, nine of which matched the training dataset. After removing 1,345
> images sharing source photographs with the training data and 246 images
> containing only Bacterial Spot annotations, which were excluded from the
> training label set, the external evaluation comprised 805 images with 7,307
> ground-truth boxes across eight matched classes. Mosaic Virus had no
> remaining ground-truth instances and was excluded.

Every number checks out: 2,396 − 1,345 = 1,051; 1,051 − 246 = 805; 7,307
boxes; 8 classes with ground truth. The archived
`summary_cross_mean_std.csv` records `n_images=805, n_gt_boxes=7307,
classes_in_map=8, excluded="Mosaic Virus (0 GT)"` on every row, and
`recompute_external_clean.py` hardcodes `EXPECTED_IMAGES = 805` and
`EXPECTED_GT = 7307` as assertions that fail the run if the recount disagrees.

**No protocol change is needed, and I have made none.** The historical protocol
and the manuscript describe the same evaluation.

### Two wording imprecisions worth a footnote

Neither affects a number. Both are places where a reviewer re-running the code
would see a mechanism slightly different from the prose.

**(a) The 246 is an emergent count, not a targeted filter.** The manuscript
reads as though Bacterial-Spot-only images were sought out and removed. The
code never names Bacterial Spot as an exclusion: it keeps boxes of the nine
matched classes and discards images left empty. That all 246 casualties happen
to be Bacterial-Spot-only is a *result* of the filter, which I confirmed by
recount above, not an input to it. The `EXPECTED_IMAGES = 805` constant is a
post-hoc assertion, not a selector. Suggested footnote:

> Images retaining no matched-class annotations after the class mapping were
> excluded from evaluation; all 246 such images carried Bacterial Spot
> annotations only.

**(b) "sharing source photographs" covers two distinct mechanisms.** The 1,345
removals are 1,273 detected by source file name (the Roboflow name with the
`.rf.<hash>` suffix stripped) plus 72 renamed copies detected perceptually — a
16×16 greyscale thumbnail within one grey level of one of ours, including flips
and 90° rotations. The manuscript collapses both into one phrase. Suggested
footnote:

> Overlap was detected by source file name (1,273 images) and, for renamed
> copies, by near-duplicate thumbnail matching invariant to flips and 90°
> rotations (72 images).

### How I would report the protocol

State the funnel explicitly as 2,396 → 1,051 → 805 and label what each arrow
removes, since the two reductions have different causes and the middle number
appears in the scripts' own output. The class exclusion is a third, separate
thing: 805 images, 8 classes in the mAP, Mosaic Virus absent from the ground
truth rather than scoring zero. The manuscript already says all of this; the
footnotes above only close the gap between the prose and what the code
literally does.

Two caveats already in the manuscript that should stay: Late Blight (3 boxes)
and Septoria (18 boxes) carry per-class APs too sparse to interpret, and the
benchmark is a single external dataset.

---

## 4. One blocker for the VM run

`build_clean_crossdataset.py` reads its keep-list from

```
outputs/audit_external_overlap/kept_external_images.txt
```

That folder **does not exist in `RetinaNeXt_Reproducible/`**, so stage 1 will
exit with `ERROR: keep-list not found` on a fresh clone. The same 1,051 lines
are in the repo as `manifests/external_kept_1051.txt` — I diffed them, the
content is identical. Place the file where the script expects it (command in
§5); this is a file placement, not a code change, and I have not altered
either script.

---

## 5. What to take to the VM, what to run, what to bring back

### Take to the VM

Into the repository root on the GPU machine:

| From | To | Size |
|---|---|---|
| the four D.3 changed files (`evaluate.py`, `src/evaluation/evaluator.py`, `tests/test_phase_d3_fixes.py`, `verify_all.sh`) | same paths | small |
| `figures/R/*.R` — all 10, path fix | `figures\R\` | small |
| `tomato_vm_clean/data/cross_dataset_3/` | `data\cross_dataset_3\` | 200 MB |
| `tomato_vm_clean/outputs/audit_external_overlap/` | `outputs\audit_external_overlap\` | 1.1 MB |

Checkpoints are already on the VM (28 files, preflight verified 24/24 required).

**Do NOT take:** `outputs/cross_eval_3_clean/`, `outputs/cross_eval_3_clean_corrected/`,
`reference_results/`, or `data/cross_dataset_3_clean/`. The first two are the
archived results the fresh run must not be able to overwrite (§1); the last is
rebuilt by stage 1 and the builder refuses to run if it already exists.

If you would rather not copy `audit_external_overlap/`, place the keep-list
from the repo instead — identical content:

```powershell
New-Item -ItemType Directory -Force outputs\audit_external_overlap
Copy-Item manifests\external_kept_1051.txt `
          outputs\audit_external_overlap\kept_external_images.txt
```

### Run on the VM, in this order

```powershell
# 0. confirm the D.3 fixes arrived (the torch-dependent test now runs too)
python tests\test_phase_d3_fixes.py
python scripts\verify\preflight_checkpoints.py --json reports\preflight.json

# 1. environment capture - do this FIRST, it is 10 seconds and currently missing
python -c "import platform,torch,torchvision;print(platform.platform());print('python',platform.python_version());print('torch',torch.__version__);print('torchvision',torchvision.__version__);print('cuda',torch.version.cuda);print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" > reports\environment_vm.txt
python -c "import torchmetrics,ultralytics,numpy;print('torchmetrics',torchmetrics.__version__);print('ultralytics',ultralytics.__version__);print('numpy',numpy.__version__)" >> reports\environment_vm.txt
nvidia-smi >> reports\environment_vm.txt
pip freeze > reports\pip_freeze_vm.txt

# 2. in-domain, the gaps
python evaluate.py --model ssdlite   --config configs\experiment_phase_d_gpu.yaml
python evaluate.py --model retinanet --config configs\experiment_phase_d_gpu.yaml
python scripts\eval\eval_yolov8n_torchmetrics.py --out-root outputs\phase_d_gpu\yolov8n_torchmetrics

# 3. external - the scientific priority. Three steps, in order.
python scripts\data\build_clean_crossdataset.py      # expect: kept 1051 images
python scripts\eval\eval_external_clean.py           # expect: "Loaded 805 images, 7307 GT boxes (skipped 246)"
python scripts\eval\recompute_external_clean.py      # expect: all checks PASS, 805 / 7307 / 8 classes
```

Step 3's middle command is the long one — 8 stages × 3 seeds = 24 model loads.
Budget hours. The line to watch is `Loaded 805 images, 7307 GT boxes (skipped
246)`: if those three numbers print, the protocol in §2 is confirmed on the VM
too, before any AP is computed.

### Bring back here

One folder plus two files:

```
outputs\phase_d_gpu\          fresh in-domain CSVs, incl. yolov8n_torchmetrics\
outputs\cross_eval_3_clean\             the 24 per_class_cross.csv + 2 aggregates
outputs\cross_eval_3_clean_corrected\   the re-averaged summary + validation_report
reports\environment_vm.txt
reports\pip_freeze_vm.txt
reports\preflight.json
```

Bring `cross_eval_3_clean/` as well as the corrected folder — it is the raw
per-class evidence, and without it the corrected summary cannot be re-derived
or independently checked.

Drop them here under a new name — `Phase_D3_VM_results/` — not on top of
`reference_results/`. Then:

```bash
python scripts/verify/reproducibility_report.py --fresh fresh_results.json --out reports/
```

Rows backed by a fresh evaluation move from evidence level B to **A**. The
comparison target is `reference_results/external/summary_cross_mean_std.csv`:
RetinaNeXt 64.67 ± 2.24, S2 62.01 ± 4.73, S4 61.89 ± 2.44, S3 61.45 ± 2.97,
S1-L1 40.70 ± 1.38, S1 39.31 ± 1.14, SSDLite 39.21 ± 3.17, YOLOv8n 38.25 ± 1.03.
