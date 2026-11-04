# Dataset attribution and provenance

## Training dataset — two-step provenance

The nine-class benchmark derives from a Science Data Bank deposit, obtained
in practice through a Roboflow re-publication. Both must be credited.

### Upstream — Science Data Bank

- **Title**: Tomato Leaf Disease Dataset (V2)
- **Depositor**: zsq
- **Year**: 2025
- **DOI**: `10.57760/sciencedb.agriculture.00246`
- **URL**: https://www.scidb.cn/en/detail?dataSetId=a380b0183f5441dc944e2eaf45cb3552
- Manuscript reference **[36]**

**Verified against a local copy**: 27,297 images, 27,297 label files, 0 empty
labels, 11 class indices (0–10), 71,178 boxes.

### As obtained — Roboflow Universe

- **Project**: `tdg-cikuh/tdg-vq3xp` ("TDG")
- **Version**: 2024-07-07 21:53
- **URL**: https://universe.roboflow.com/tdg-cikuh/tdg-vq3xp
- **License**: **CC BY 4.0**
- Recorded in the original project at `data/README.dataset.txt`

Roboflow applied preprocessing and augmentation before export, visible in the
`.rf.<hash>` filename suffixes. **This is the origin of the source-image
overlap documented in §2**: 24,876 images derive from only **20,918 unique
source photographs**, so augmented variants of one photograph appear in more
than one split. The 1,933-image overlap-excluded test subset exists to measure
the effect.

> **Attribution note.** CC BY 4.0 requires crediting the source you obtained
> the work from. The manuscript cites [36] (Science Data Bank) but not the
> Roboflow TDG project the files actually came from. Citing both is the
> correct form.

### Class removal

Two of the eleven classes were dropped:

| Source index | Boxes | Class | Reason (§2) |
|---|---|---|---|
| 9 | 2,965 | Bacterial Spot | lesion morphology overlaps several fungal classes |
| 10 | 618 | Iron Deficiency | too sparsely represented |

71,178 − 2,965 − 618 = **67,595**, matching the manuscript. The nine retained
classes already occupy indices 0–8, so no remapping is needed;
`scripts/data/prepare_nine_class.py` asserts this rather than assuming it.

## External benchmark

- **Project**: `forfun-cyh93/tomatoes_diseases`
- **Version**: v1, 2024-02-09 19:31; exported 21 May 2026
- **URL**: https://universe.roboflow.com/forfun-cyh93/tomatoes_diseases
- **License**: **CC BY 4.0**
- Manuscript reference **[37]**
- Format: YOLOv8 Oriented Object Detection
- Preprocessing: EXIF-stripped auto-orientation, resize 640×640 stretch
- Augmentation: 3 versions per source image, salt-and-pepper on 1.96 % of pixels
- **2,396 images** — stated in the export's own `README.roboflow.txt`

Filtering (all counts verified against files):

```
2,396  original
-1,345  removed: share a source photograph with the training set
 1,051  kept
  -246  excluded: contain only Bacterial Spot, which is not a trained class
   805  evaluated, 7,307 ground-truth boxes, 8 classes
```

Mosaic Virus has zero ground-truth instances after filtering and is excluded
from the external mAP.

Removal reasons, from `manifests/external_removed_1345.csv`:

| Reason | Count |
|---|---|
| same source file name + pixel-identical | 989 |
| same source file name | 284 |
| renamed copy (pixel diff < 1) | 72 |

## A third dataset appears in the original project — not used

`data/README.md` in the original project describes **"Tomato-Village"**
(Gehlot, Saxena & Gandhi, *Multimedia Systems*, 2023,
DOI 10.1007/s00530-023-01158-y). It is neither of the datasets above, is not
cited in the manuscript, and its classes do not match. It appears to be a
leftover from an earlier trial and is **excluded** from this repository.

## Licensing of this repository

Dataset licences are CC BY 4.0 and belong to their publishers. This repository
contains **no dataset images or labels** — only manifests of filenames, which
are factual records. Code is licensed separately; see `LICENSE`.
