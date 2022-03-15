# TCGA-LUSC and TCGA-LUAD 
Probability estimation on 5-year survival of non-small cell lung cancer patients

## Introduction
Histopathology aims to identify tumor cells, cancer subtypes, and
the stage and level of differentiation of cancer. Hematoxylin and Eosin (H&E)-stained slides are
the most common type of histopathology data used for clinical decision making. Treatments assigned to patients after diagnosis are not personalized and their
impact on cancer trajectory is complex, so the survival status of a patient is not deterministic. In
this work, we use the H&E slides of non-small cell lung cancers from The Cancer Genome Atlas
Program ([TCGA](portal.gdc.cancer.gov)) to estimate the the 5-year survival probability of cancer patients. 

The dataset has 1512 whole slide images
from 1009 patients, and 352 of them died in 5-years. We split the samples by patients and source
institutions into training, validation, and test set, which has 1203, 151, and 158 samples respectively.
The whole slide images contain numerous pixels, so we cropped the slides into tiles at 20x magnification at 1024 × 1024 with 1/4 overlapping, resized them to 299 × 299 with bicubic interpolation, and filter out the tiles with more than 85% area covered by the background. The representations of each tile
are trained with self-supervised momentum contrastive learning (MoCo) (Chen et al., 2020), and
the slide-level prediction is obtained from a multiple-instance learning network (Ilse et al., 2018)
trained with the binary label of survival in 5 years.

## Datasets

Download the extracted embeddings [here](https://drive.google.com/drive/folders/1d2mnR7esOvCUTSDgwCmihLiFpSAwlnM3?usp=sharing)

## Dataloader
If you are using PyTorch, we provide the dataloader [here](https://github.com/jackzhu727/deep-probability-estimation/blob/main/datasets/cancer_survival.py).

