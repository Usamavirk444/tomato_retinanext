# Known limitations

Issues a reader should know about before relying on these results. Each is
stated here rather than left for a reviewer to find.

## Dataset

**Source-image overlap between splits.** The dataset is pre-augmented: 24,876
images derive from 20,918 unique source photographs, so augmented variants of
one photograph can appear in different splits. 554 of the 2,487 test images
(22.3 %) share a source photograph with training. The paper reports results on
both the full test set and the 1,933-image overlap-excluded subset; RetinaNeXt
drops 94.42 → 92.82 (−1.60 pp), and the ranking of all four models is
unchanged.

**Resolution.** 7,867 of 24,876 images (31.6 %) were captured at 512×512 and
upsampled to 640×640. Upsampling may affect fine lesion detail.

**Spatial annotation structure.** 19.4 % of box centres fall within 0.06 of
the image centre and a further 21.0 % cluster at the four quadrant midpoints,
consistent with tiled augmentation in the source. This may reduce sensitivity
to lesions near image borders.

**Overlap detection may not be bit-reproducible.** 72 of the 1,345 removed
external images were caught by pixel comparison (`pixel diff < 1`), which can
differ across platforms and image libraries. The manifests
(`manifests/external_*`) are therefore authoritative, not the detection code.

## Training

**YOLOv8n is not trained like the other three.** It runs through Ultralytics,
which applies mosaic, copy-paste and mixup augmentation that the torchvision
models do not, uses its own iteration-based warm-up, and sets a doubled
classification loss gain (`cls=2.0`). The optimizer, learning rate, schedule
type and seed are matched, but the training conditions are not identical.

**Minimum learning rate differs from the manuscript.** Code uses
`eta_min=1e-7` for all torchvision models; §3.3 states 1×10⁻⁶. Only YOLOv8n
reaches 1e-6. See `LEARNING_RATE_DISCREPANCY.md`.

**Warm-up starts at 10 % of base LR**, not zero.

**Latency measurements have high variance.** RetinaNet ±9.71 ms on 30.91
(31 %) and RetinaNeXt ±10.30 on 38.26 (27 %), against YOLOv8n's ±0.24 on 11.98
(2 %). Consistent with warm-up iterations not being discarded for the
torchvision models. FPS figures should be read as indicative.

## Evaluation

**Three different evaluators.** See `EVALUATION_PROTOCOLS.md`. The in-domain /
external protocol difference was measured at 0.086 pp and is negligible
relative to the reported drops, but the confusion-matrix protocol differs
substantially from both (confidence 0.25, IoU-ordered matching).

**Confusion-matrix threshold is 0.25, not 0.05** as stated in the manuscript
and in `figures/R/00_setup.R`. Verified by rebuilding the matrices at both
values.

**Precision/recall/F1 are operating-point metrics.** Computed at IoU ≥ 0.50
and confidence ≥ 0.05 from the confusion matrices, seed 42 only. Unlike mAP
they are not threshold-free, and a higher confidence threshold would raise
precision and lower recall for every model.

## Ablation

**S1 → S2 changes two things.** The published S1 uses GIoU box regression
(set by torchvision's `retinanet_resnet50_fpn_v2` factory) while every
ConvNeXt stage uses L1 (the class default). The S1-L1 retrain isolates this:
the loss accounts for +0.36 pp in-domain and −0.49 pp externally, leaving
+4.67 pp and +21.05 pp to the backbone. Comparisons among S2–S5 are unaffected.

**A residual head difference remains.** `retinanet_resnet50_fpn_v2` builds its
head with GroupNorm(32); the ConvNeXt stages use the default head without
normalisation. Unquantified. Confined to the head, and implausible as an
explanation for 21 pp of cross-domain difference, but not controlled.

**S3 seed 456 is unresolved.** Two checkpoints exist, giving 94.68 and 91.23.
The July one is provisionally canonical on documented evidence but has not
been independently re-evaluated. See `S3_SEED456_INVESTIGATION.md`. This
matters because S3 vs S5 decides whether "omitting BiFPN scores marginally
higher in-domain" holds.

## Repository

**Checkpoints are not included** — 7.86 GB across 24 files. See
`manifests/checkpoint_inventory.csv` for names, sizes and SHA-256 hashes. The
archive is not yet published.

**Figure renaming is manual.** Generating scripts use historical numbers; see
`FIGURE_TABLE_MAP.md`.

**Dataset download is manual.** Both sources require accepting terms on the
publisher's site.
