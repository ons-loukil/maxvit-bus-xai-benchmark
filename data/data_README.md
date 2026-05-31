# Data: Unified Breast Ultrasound Corpus

This folder documents the data used to train the MaxViT classifier and to run the XAI
benchmark. The corpus unifies three public breast-ultrasound (BUS) datasets into a single
three-class problem: **normal**, **benign**, and **malignant**.

> The image and mask files are **not** redistributed in this repository. Download each
> source dataset from its original location (links below) and respect its individual
> license and terms of use. This page documents how the unified corpus is assembled so the
> result is reproducible from the public sources.

---

## Source datasets

| Dataset | Classes contributed | Lesion masks | Source (Kaggle) | Reference |
|---|---|---|---|---|
| **BUSI** | normal, benign, malignant | benign, malignant | `aryashah2k/breast-ultrasound-images-dataset` | Al-Dhabyani et al., *Data Brief* 28:104863 (2020) |
| **BUS-UCLM** | normal, benign, malignant | benign, malignant | `orvile/bus-uclm-breast-ultrasound-dataset` | Vallez et al., *Sci. Data* 12(1):242 (2025) |
| **BUS-UC** | benign, malignant | benign, malignant | `orvile/bus-uc-breast-ultrasound` | Iqbal & Sharif, *Eng. Appl. Artif. Intell.* 127:107292 (2024) |

**Class harmonisation.** Each source dataset's labels are mapped onto the common label set
`{normal, benign, malignant}`. The `normal` class contains lesion-free scans and therefore
has **no lesion mask**; masks exist only for the `benign` and `malignant` classes.

---

## Directory layout

After assembly, the data are organised as one folder per class for the images, and a
parallel folder per lesion class for the masks:

```
data/
├── normal/        *.png        # images only (no lesion, no mask)
├── benign/        *.png
└── malignant/     *.png

masks/
├── benign/        *.png        # lesion masks; filename matches the image
└── malignant/     *.png        # (normal has no masks)
```

**Image/mask pairing.** Each mask filename is aligned to its image filename so the two can
be paired by name. An image with more than one lesion keeps several masks distinguished by
a numeric suffix (for example `name_1.png`, `name_2.png`); the XAI notebook merges all
masks of one image into a single binary mask via a logical OR.

---

## How the unified corpus is assembled

### 1. Images

The classification images from each source dataset are collected, per class, into
`data/normal/`, `data/benign/`, and `data/malignant/`. Any mask files that ship alongside
the images in a source folder are removed from these class folders, so the `data/` folders
contain **images only**.

### 2. Lesion masks (benign and malignant only)

Masks from the three sources are normalised so that every mask filename matches the
corresponding image filename, then placed under `masks/benign/` and `masks/malignant/`:

- **BUSI** — ground-truth masks carry a `_mask` suffix (and `_mask_1`, `_mask_2`, ... for
  images with multiple lesions). The `_mask` token is stripped from the filename so each
  mask matches its image; multi-lesion variants are preserved.
- **BUS-UCLM** — the segmentation masks are binarised (any non-zero pixel set to 255) and
  routed to the benign or malignant folder using the labels in the dataset's `INFO.csv`.
- **BUS-UC** — provides per-class `Benign` / `Malignant` masks, which are copied in
  directly.
- **Normal** — no masks are produced, since these scans contain no lesion. Localisation
  metrics (Dice, EBPG) are consequently undefined for the normal class.

---

## Mask format expected by the XAI notebook

The benchmark notebook (`02_xai_benchmark.ipynb`) reads masks from a root folder with
`benign/` and `malignant/` subfolders:

```
MASK_ROOT/
├── benign/
└── malignant/
```

Set `MASK_ROOT` in the Configuration cell to point at this folder. Masks are read as
grayscale, with any non-zero pixel treated as lesion foreground, resized to the model input
size, and (when an image has several masks) merged with a logical OR before the localisation
metrics are computed.

---

## Licensing

The unified corpus is derived from third-party datasets, each governed by its own license.
This repository provides only the assembly logic, not the image or mask files. Please obtain
the data from the original sources above and comply with their terms, and cite the three
source datasets in any work that uses this corpus.
