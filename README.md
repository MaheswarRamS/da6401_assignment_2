# DA6401 Assignment 2 — Visual Perception Pipeline on Oxford-IIIT Pet

## Overview

This repository implements a complete multi-task visual perception pipeline on the Oxford-IIIT Pet dataset, covering three tasks:

- **Task 1 — Classification:** Predict the breed of a pet (37 classes) using a VGG11-based classifier
- **Task 2 — Localization:** Predict a bounding box `(x_center, y_center, width, height)` in pixel space using a VGG11-based regressor
- **Task 3 — Segmentation:** Predict a trimap mask (foreground / background / boundary) using a VGG11 U-Net

All three models share the same VGG11 encoder backbone and are unified in `MultiTaskPerceptionModel`.

---

## Project Structure

```
.
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py          # Oxford-IIIT Pet dataset loader
├── losses/
│   ├── __init__.py
│   └── iou_loss.py              # Custom IoU loss (mean / sum reduction)
├── models/
│   ├── __init__.py
│   ├── layers.py                # CustomDropout
│   ├── vgg11.py                 # VGG11 encoder (with BatchNorm)
│   ├── classification.py        # VGG11Classifier
│   ├── localization.py          # VGG11Localizer
│   ├── segmentation.py          # VGG11UNet (U-Net decoder)
│   └── multitask.py             # MultiTaskPerceptionModel
├── train.py                     # Training script for all three tasks
├── analysis.py                  # W&B experiment scripts (sections 2.1–2.8)
├── inference.py                 # Inference and evaluation on test set
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Oxford-IIIT Pet dataset

```
data/pets/
    images/          ← .jpg images
    annotations/
        trainval.txt
        test.txt
        trimaps/     ← .png trimap masks
        xmls/        ← Pascal VOC bounding box annotations
```

---

## Training

```bash
# Classification only
python train.py -t classification -d ./data/pets -ep 50 -bs 16 -lr 0.0001

# Localization only
python train.py -t localization -d ./data/pets -ep 50 -bs 16 -lr 0.0001

# Segmentation only
python train.py -t segmentation -d ./data/pets -ep 50 -bs 16 -lr 0.0001

# All three tasks sequentially
python train.py -t all -d ./data/pets -ep 50 -bs 16 -lr 0.0001
```

### Training Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--data_dir` | `-d` | `./data/pets` | Path to dataset root |
| `--task` | `-t` | `all` | `all` / `classification` / `localization` / `segmentation` |
| `--epochs` | `-ep` | `50` | Number of training epochs |
| `--batch_size` | `-bs` | `16` | Batch size |
| `--lr` | `-lr` | `1e-3` | Learning rate |
| `--weight_decay` | `-wd` | `1e-4` | L2 regularisation |
| `--dropout_p` | `-dp` | `0.5` | Dropout probability |
| `--val_fraction` | `-vf` | `0.1` | Fraction of trainval used for validation |
| `--num_workers` | `-nm` | `0` | DataLoader worker processes |
| `--num_breeds` | `-nb` | `37` | Number of classification classes |
| `--seg_classes` | `-sc` | `3` | Number of segmentation classes |
| `--cls_ckpt` | `-cck` | `checkpoints/classifier.pth` | Classification checkpoint path |
| `--loc_ckpt` | `-lck` | `checkpoints/localizer.pth` | Localization checkpoint path |
| `--seg_ckpt` | `-sck` | `checkpoints/segmentation.pth` | Segmentation checkpoint path |
| `--wandb_project` | `-wp` | `DA6401_Assignment_2` | W&B project name |
| `--wandb_entity` | `-we` | `None` | W&B entity name |

Checkpoints are saved to `checkpoints/` and updated only when validation loss improves.

---

## Inference

Runs `MultiTaskPerceptionModel` on the test split and reports all metrics:

```bash
python inference.py -d ./data/pets
```

### Inference Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--data_dir` | `-d` |`./data/pets` | Path to dataset root |
| `--batch_size` | `-bs` | `16` | Batch size |
| `--num_workers` | `-nm` | `0` | DataLoader worker processes |
| `--num_breeds` | `-nb` | `37` | Number of classification classes |
| `--seg_classes` | `-sc` | `3` | Number of segmentation classes |
| `--cls_ckpt` | `-cck` | `checkpoints/classifier.pth` | Classification checkpoint |
| `--loc_ckpt` | `-lck` | `checkpoints/localizer.pth` | Localization checkpoint |
| `--seg_ckpt` | `-sck` | `checkpoints/segmentation.pth` | Segmentation checkpoint |
| `--wandb_project` | `-wp` | `DA6401_Assignment_2` | W&B project name |
| `--wandb_entity` | `-we` | `None` | W&B entity name |

### Metrics Reported

| Task | Metrics |
|---|---|
| Classification | Accuracy, F1 (macro), Precision (macro), Recall (macro) |
| Localization | Mean IoU |
| Segmentation | Pixel Accuracy, Mean IoU |

---

## Model Architecture

### VGG11 Encoder (`models/vgg11.py`)

Follows the official VGG11 paper with BatchNorm injected after each convolution:

```
Block 1: Conv(64)        → BN → ReLU → MaxPool  →  112×112
Block 2: Conv(128)       → BN → ReLU → MaxPool  →   56×56
Block 3: Conv(256) × 2  → BN → ReLU → MaxPool  →   28×28
Block 4: Conv(512) × 2  → BN → ReLU → MaxPool  →   14×14
Block 5: Conv(512) × 2  → BN → ReLU → MaxPool  →    7×7
Output : [B, 512, 7, 7] for 224×224 input
```

## Links

Wandb Link: 
https://wandb.ai/bt25d030-indian-institute-of-technology-madras/DA6401_Assign2_Analysis/reports/DA6401-Assignment-2--VmlldzoxNjQ5MDc1NA?accessToken=qaojmegw1cgmqadmxe4gefgm13c1wiqg6s67v9sleb5xk81zcctio6aoinkeoml9

GitHub Repo:
https://github.com/MaheswarRamS/da6401_assignment_2.git
