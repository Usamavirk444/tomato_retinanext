# Licensing audit — code and model weights

Prepared for the public release. Not legal advice; this records what the
code actually does and what the upstream terms actually say, and separates
both from what is merely common practice.

**Headline: the repository has a hard licensing constraint that is not about
the weights.** Seven files import Ultralytics, one of them at module level,
and `train_all.py` monkey-patches an Ultralytics internal class. Under
Ultralytics' stated reading of AGPL-3.0 the repository is a combined work and
must be AGPL-3.0 to be distributed.

> **Decision taken.** The author has selected **AGPL-3.0** for the original
> repository source code. The official GNU AGPL v3.0 text is now at
> [`../LICENSE`](../LICENSE) (661 lines, 34,523 bytes, verbatim and
> unmodified). `README.md` and `DATASET_ATTRIBUTION.md` have been updated to
> point at it.
>
> This licence applies to **this repository's source code only**. It grants
> no rights over third-party dependencies, the torchvision pretrained weights
> used for initialization, the datasets, or the trained checkpoints. Those
> are addressed in §2 and §3 and several remain unresolved upstream.

---

## 1. Third-party code the repository depends on

| Component | Licence | How it is used | Reach |
|---|---|---|---|
| **ultralytics 8.4.45** | **AGPL-3.0** | YOLOv8n training and evaluation; internal class patched | **repository-wide** |
| torch, torchvision | BSD-3-Clause | models, training, evaluation | permissive |
| torchmetrics | Apache-2.0 | mAP computation | permissive |
| pycocotools | BSD-2-Clause | COCO-style evaluation | permissive |
| numpy, pandas, matplotlib, seaborn, Pillow, PyYAML, tqdm, opencv-python | BSD / Apache-2.0 / MIT / MPL-2.0 (opencv: Apache-2.0) | utilities, plotting | permissive |
| R: ggplot2, dplyr, tidyr, readr, scales, patchwork, svglite | MIT / GPL-2+ depending on package | figure generation only, not imported by the Python code | figures only |

Everything except ultralytics is permissive and imposes no copyleft on this
repository.

### Where ultralytics reaches into the code

```
train_all.py                              MODULE-LEVEL import  <-- deepest
cross_eval/cross_dataset_3_eval.py        lazy import
evaluate.py                               lazy import
scripts/eval/benchmark_fps.py             lazy import
scripts/eval/confusion_matrix.py          lazy import
scripts/eval/eval_yolov8n_torchmetrics.py lazy import
scripts/eval/evaluate_per_class.py        lazy import
```

### No third-party source code is copied into this repository

Audited file by file. **Six of the seven touch only the public API** —
`from ultralytics import YOLO`, then `YOLO(path)` and its `.val()` /
`.predict()` methods. No Ultralytics source is copied, vendored, or
redistributed anywhere in the repository.

`train_all.py` is the single exception, and even it copies nothing. At lines
61–81 it imports an internal class and wraps one method at runtime:

```python
from ultralytics.utils.tal import TaskAlignedAssigner
_orig_tal_forward = TaskAlignedAssigner.forward
...
TaskAlignedAssigner.forward = _patched_tal_forward   # MPS workaround
```

`_patched_tal_forward` is original code: it moves tensors to CPU, calls the
**original** method it saved, and moves the result back. None of Ultralytics'
assignment algorithm is reproduced. So this is a runtime modification of
AGPL-licensed software rather than a copy of it — the distinction matters,
because what is distributed here is a patch script, not a fork.

The lazy imports elsewhere delay loading but do not change the analysis: the
distributed work still requires ultralytics to perform its documented
function.

**Torchvision**, by contrast, is used purely through its public API — the
hybrid and ablation models import `RetinaNet`, `AnchorGenerator` and
`LastLevelP6P7` and compose them. `bifpn.py` and `cbam.py` contain no
torchvision imports at all; they are original implementations of published
architectures (BiFPN from the EfficientDet paper, CBAM from Woo et al.).
A scan for torchvision's distinctive internals and for upstream copyright
headers found nothing copied.

**Bottom line for §5 of the release checklist: the repository contains no
copied or modified third-party source. Its AGPL exposure comes from
dependency and runtime integration only.**

(Incidentally, that patch is an Apple-Silicon workaround and is inert on the
CUDA machine used for the final runs. It has not been altered.)

---

## 2. Model weights — upstream initialization

Confirmed by reading the source, not assumed:

| Checkpoints | Initialized from | Upstream training data | Source file |
|---|---|---|---|
| B1 SSDLite | `SSDLite320_MobileNet_V3_Large_Weights.DEFAULT` | **COCO** | `src/models/ssdlite.py:30` |
| S1, S1-L1 | `RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT` | **COCO** | `src/models/retinanet.py:41` |
| S2–S5 | `ConvNeXt_Tiny_Weights.DEFAULT` | **ImageNet-1k** | `src/models/hybrid/convnext_backbone.py:70` |
| B2 YOLOv8n | `yolov8n.pt` (Ultralytics) | COCO, **AGPL-3.0 asserted** | `train_all.py:257` |

Note that B1, S1 and S1-L1 start from **COCO-pretrained detection weights**,
not merely an ImageNet backbone. That is a stronger provenance claim than
"ImageNet-pretrained" and should be described accurately in the manuscript.

### torchvision weights: BSD covers the code, not the weight files

torchvision's library code is BSD-3-Clause. The downloadable weight files
carry no separate licence, and torchvision's documentation states:

> "The pre-trained models provided in this library may have their own licenses
> or terms and conditions derived from the dataset used for training. It is
> your responsibility to determine whether you have permission to use the
> models for your use case."
> — https://docs.pytorch.org/vision/stable/models.html

So BSD-3-Clause cannot be read across to the weights, nor to derivatives of
them. Upstream does not resolve it either: COCO annotations are CC BY 4.0 but
the images are Flickr photographs under individual terms; ImageNet's access
terms are framed around non-commercial research and ImageNet does not own its
images.

Whether a fine-tuned weight file is a derivative work of the weights it
started from, or of the data behind those weights, is unsettled generally.
Publishing fine-tuned torchvision derivatives is near-universal in the
literature and no enforcement is known, but **common practice is not a
permission grant** and is not recorded here as one.

---

## 3. Recommendation

Four different things are at stake and they do not get one answer.

| Layer | What it covers | Status | Outcome |
|---|---|---|---|
| **Repository source code** | Everything in this repo — all original work, no copied third-party source | **Resolved** | **AGPL-3.0**, `LICENSE` in place. Forced by the ultralytics dependency, not by anything copied |
| **Third-party dependencies** | ultralytics (AGPL-3.0); torch, torchvision, torchmetrics, pycocotools and the rest (permissive) | Not redistributed — installed by pip from their own sources | Nothing to license. Declared, not vendored |
| **Pretrained weights** used for initialization | torchvision COCO / ImageNet checkpoints | **Unresolved upstream** — no explicit licence on the weight files; see §2 | Disclose provenance. Assert nothing |
| **Trained checkpoints** (Zenodo) | 21 public + 3 YOLOv8n held back | Inherits the row above | License **your contribution** (CC BY 4.0) and disclose upstream status. **Do not assert a licence over the upstream weights** |

**AGPL-3.0 on this repository does not propagate to the weights.** A software
licence covers the work it is applied to; applying AGPL-3.0 to code the
author wrote says nothing about weight files the author did not create and
whose upstream terms are unstated. Nobody gains redistribution rights over
torchvision's pretrained checkpoints because this repository is AGPL-3.0, and
the deposit documentation must not be read as implying otherwise.

The third row is the one to be careful about. Nothing in this audit
establishes that torchvision's pretrained weights, or derivatives of them,
are yours to relicense. Stating "these checkpoints are CC BY 4.0" without
qualification would claim a permission nobody has granted. Stating "my
fine-tuning, configuration and documentation are CC BY 4.0; the weights were
initialized from torchvision checkpoints whose own terms are unstated
upstream" is accurate and is what the deposit README says.

Note also that the repository's licence and the checkpoints' licence are
separate instruments: an AGPL-3.0 repository does not make the weights
AGPL-3.0, and a CC BY 4.0 deposit does not loosen AGPL obligations on the
code.

### Licence: AGPL-3.0 for the repository

Given the module-level import and the runtime patch of an Ultralytics
internal, AGPL-3.0 is the defensible choice. The alternatives are weaker:

| Option | Assessment |
|---|---|
| **AGPL-3.0 throughout** | **Recommended.** Clean compliance, no per-file carve-outs, costs an academic project essentially nothing. |
| MIT/Apache-2.0 with AGPL carve-out for YOLOv8n files | Mixed licensing is messy and the carve-out is hard to draw when `train_all.py` — the main entry point — is itself the deepest integration. |
| MIT/Apache-2.0 throughout | Not defensible while ultralytics is imported and patched. |
| Remove YOLOv8n support, then MIT/Apache-2.0 | Clean, but YOLOv8n is a manuscript baseline; removing it from the public code weakens reproducibility more than AGPL-3.0 costs. |

### A consequence worth noticing

If the repository is AGPL-3.0 and publicly released, the main obstacle to
depositing the three YOLOv8n checkpoints largely falls away: AGPL compliance
on Ultralytics' own reading means publishing the corresponding source, which
is exactly what the public repository does. Licensing the repository
AGPL-3.0 and then including the YOLOv8n weights — labelled AGPL-3.0 in the
Zenodo record — is a coherent position. Holding them back is still the more
cautious one, and costs only 17.9 MB of a 7.86 GB deposit.

This is a genuine decision point rather than a formality, and the two routes
differ mainly in how much residual risk you want to carry.

### Unresolved obligations, listed honestly

1. ~~**No `LICENSE` file exists**~~ — **resolved.** The official GNU AGPL
   v3.0 text is at `LICENSE`, and the references in `README.md` and
   `docs/DATASET_ATTRIBUTION.md` now resolve to it.
2. **AGPL-3.0 §13 (network use)** obliges source provision to users
   interacting over a network. Not triggered today — this is offline research
   code — but relevant if a demo or API is ever deployed. Note that
   `requirements.txt` pins `fastapi`, `uvicorn` and `starlette`, so a web
   component may be intended; if so, §13 becomes live.
3. **Weight redistribution** carries the residual uncertainty in §2 for all
   21 non-YOLO checkpoints. Mitigation is disclosure, not resolution.
4. **CC BY 4.0 attribution** for the tomato dataset requires crediting the
   source actually used. `docs/DATASET_ATTRIBUTION.md` already notes that the
   manuscript cites Science Data Bank [36] but not the Roboflow TDG project
   the files came from. Both should be cited.
5. **An institutional check is warranted.** Items 1–3 are the sort of thing a
   university research office or technology-transfer contact answers quickly,
   and that answer is worth more than this audit.

---

## 4. Two packaging defects found during the audit — both now fixed

**`requirements_cuda.txt` was UTF-16LE with a BOM**, from `pip freeze > file`
in PowerShell. Converted to UTF-8 with LF endings; all 57 dependency
specifiers are byte-identical afterwards.

A correction to an earlier claim of mine: I said pip could not parse the
UTF-16 file. **That was wrong.** Tested against pip 25.3, which decodes the
BOM and reads through to the final line — verified by planting an invalid
requirement at the end of a UTF-16 copy and confirming pip reported it. The
conversion is still worth doing, because UTF-16 breaks `grep`, meaningful
`git diff`, many editors, and any tooling that assumes UTF-8 (GitHub's
dependency graph among them), and because it is inconsistent with every other
text file in the repository. But it was not blocking installation, and it
should not be described as if it were.

**`requirements.txt` omitted `ultralytics`.** YOLOv8n training and evaluation
could not run from that file alone — `train_all.py` fails at its module-level
import. Added `ultralytics==8.4.45` and `ultralytics-thop==2.0.19`, the
versions the Phase D.3 results were produced with.

Note the licensing consequence: `requirements.txt` previously pulled in no
AGPL dependency, and now it does. That is the honest reflection of what the
code requires — `train_all.py` cannot import without it — but it means both
install paths now bring in AGPL-3.0 software, which reinforces the
recommendation in §3.
