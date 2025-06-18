# Deepfake Detection with Vision-Language Models

This project implements deepfake detection using CLIP-based models with parameter-efficient fine-tuning strategies like LoRA. It supports frame-level and video-level evaluation, ROC curve plotting, and visualization of model errors.

## Project Structure

```
deepfake-detector/
├── datasets/               # Real and fake frames
├── utils/                  # Helpers: metrics, transforms, etc.
├── model_baseline.py       # CLIP baseline model
├── model_lora.py           # CLIP + LoRA model
├── train_base.py           # Training script for baseline
├── train_lora.py           # Training script for LoRA
├── evaluate.py             # Evaluation script
├── vis_mistakes.py         # Visualize misclassified samples
├── environment.yml         # Conda environment
├── requirements.txt        # pip requirements
├── README.md               # This file
```

---
## Quick Start

```bash
bash run.sh
```


## Installation

```bash
conda env create -f environment.yml
conda activate deepfake-detector
```

---

## Dataset

### FaceForensics++ (C40)

Download: [Dropbox Link](https://www.dropbox.com/t/2Amyu4D5TulaIofv)

Organize the dataset as:

```
datasets/
├── Real_youtube/
├── FaceSwap/
├── NeuralTextures/
```
### Dataset Unpack Instructions

```bash
unzip path/to/faceforensics.zip -d datasets/FaceSwap
unzip path/to/real_youtube.zip -d datasets/Real_youtube
unzip path/to/neuraltextures.zip -d datasets/NeuralTextures
```

---

## Start

### Train baseline model:

```bash
python train_base.py
```

### Train LoRA model:

```bash
python train_lora.py
```

### Evaluate a model:

```bash
python evaluate.py \
  --model_type base \
  --ckpt clip_base.pth
```

Other options:

* `--model_type lora` with `--ckpt clip_lora.pth`


### Expected Run-time

- Training (LoRA, 5 epochs): ~5 minutes on a single RTX 3090
- Evaluation: ~1 minute
- Score generation (all modes): ~2 minutes


---

## Visualizations

### ROC Curve (automatically saved during evaluation)

* Frame-level & video-level ROC
* Saved to `figures/roc_curve_<model_type>.png`

### Visualize model mistakes:

```bash
python vis_mistakes.py
```

This script will:

* Randomly sample misclassified images
* Show their prediction vs. ground truth
* Save the images in a `mistakes/` folder

---

## Results

| Model | Level       | AUC   | ACC   | F1    |
|--------|-------------|-------|-------|-------|
| Base  | Frame-level | 0.634 | 0.793 | 0.881 |
| Base  | Video-level | 0.651 | 0.782 | 0.874 |
| LoRA  | Frame-level | 0.803 | 0.884 | 0.936 |
| LoRA  | Video-level | 0.806 | 0.891 | 0.940 |

---

## Pretrained Weights

You can download the pretrained models and score files from the [latest release](https://github.com/EllaChang1011/deepfake-detector/releases/tag/v1.0).

- `clip_base.pth`
- `clip_lora.pth`

--- 

## 📝 Report

For details, refer to the "report.pdf".

