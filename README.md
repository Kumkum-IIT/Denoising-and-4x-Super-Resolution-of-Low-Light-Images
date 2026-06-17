# Real-ESRGAN Fine-Tuning for Low-Light Image Super-Resolution

## Overview

This project fine-tunes a pretrained **Real-ESRGAN x4+** model on the **DLP Jan 2026 NPPE-3** image super-resolution competition dataset. The goal is to reconstruct high-resolution images from degraded low-resolution inputs while optimizing for the competition's custom RMSE evaluation metric.

The pipeline includes:

* Loading official Real-ESRGAN pretrained weights
* Low-light image normalization via automatic brightness stretching
* Patch-based fine-tuning with paired augmentations
* Exponential Moving Average (EMA) model tracking
* Full-image tiled validation
* Test-Time Augmentation (TTA)
* Tiled inference for memory-efficient super-resolution
* Automated Kaggle submission generation

---

## Model Architecture

**Generator:** Real-ESRGAN x4+

| Parameter        | Value   |
| ---------------- | ------- |
| Network          | RRDBNet |
| Input Channels   | 3       |
| Output Channels  | 3       |
| Feature Channels | 64      |
| RRDB Blocks      | 23      |
| Growth Channels  | 32      |
| Upscale Factor   | 4×      |

Pretrained weights:

* `RealESRGAN_x4plus.pth`
* Source: Official Real-ESRGAN release

---

## Dataset Structure

Expected directory layout:

```text
train/
├── input/
│   ├── image_001.jpg
│   └── ...
└── ground_truth/
    ├── image_001.jpg
    └── ...

test/
└── input/
    ├── image_001.jpg
    └── ...
```

Training set:

* 1200 LR-HR image pairs

Testing set:

* 300 low-resolution images

---

## Preprocessing

### Automatic Brightness Normalization

Many competition images are extremely dark. Before training and inference, each LR image undergoes percentile-based contrast stretching:

* Lower percentile: 2%
* Upper percentile: 98%
* Adaptive fallback for low-contrast images

This improves compatibility with the pretrained Real-ESRGAN feature distribution.

---

## Data Augmentation

Random augmentations applied during training:

### Geometric

* Horizontal Flip
* Vertical Flip
* 90° Rotations

### Photometric

* Gamma Correction: `0.6 – 1.8`
* Brightness Scaling: `0.8 – 1.2`

---

## Training Configuration

| Parameter     | Value        |
| ------------- | ------------ |
| Patch Size    | 64           |
| Batch Size    | 16           |
| Epochs        | 40           |
| Learning Rate | 2e-4         |
| Minimum LR    | 1e-7         |
| Optimizer     | AdamW        |
| Weight Decay  | 1e-4         |
| EMA Decay     | 0.999        |
| GPUs          | 2 × Tesla T4 |

### Learning Rate Schedule

Cosine Annealing LR:

```text
Epoch 1 → 20
↓
LR Restart
↓
Epoch 21 → 40
```

---

## Loss Function

Custom fine-tuning loss:

### Charbonnier Loss

Robust L1-style reconstruction loss:

```math
L_char = sqrt((prediction-target)^2 + ε^2)
```

### Color Consistency Loss

Preserves image color distribution:

```math
L_color = |mean(pred)-mean(target)|
```

### Final Loss

```math
Loss = L_char + 0.1 × L_color
```

---

## Validation Strategy

Unlike patch-based validation, evaluation is performed on full images.

### Tiled Validation

* Tile Size: 128
* Overlap: 16

Predictions are merged back into a complete image before computing RMSE.

### Competition Metric

Images are converted to grayscale:

```text
Gray = 0.299R + 0.587G + 0.114B
```

100 uniformly spaced pixels are sampled and RMSE is computed exactly as required by the competition.

---

## Training Results

### Resume Checkpoint

Training resumed from:

```text
Epoch 14
Best RMSE = 31.189
```

### Final Performance

| Epoch | RMSE   |
| ----- | ------ |
| 28    | 31.153 |
| 32    | 29.506 |
| 36    | 28.401 |
| 39    | 27.896 |
| 40    | 27.778 |

### Best Validation RMSE

```text
27.778
```

---

## Inference Pipeline

### Tiled Inference

Large images are processed in overlapping patches.

Configuration:

```text
Tile Size: 128
Overlap: 16
```

### Hann Window Blending

Smoothly blends overlapping regions to eliminate seam artifacts.

### Test-Time Augmentation (TTA)

8-way augmentation:

* Original
* Horizontal Flip
* Rotations (0°, 90°, 180°, 270°)
* Rotated + Flipped variants

Final prediction:

```text
Average of all 8 outputs
```

---

## Output Resolution

Competition target resolution:

```text
312 × 312
↓
1250 × 1250
```

Predictions are resized if necessary using bilinear interpolation.

---

## Checkpoints

Saved after every epoch:

```text
checkpoints/
├── last.pth
├── best.pth
├── epoch_001.pth
├── epoch_002.pth
└── ...
```

Checkpoint contents:

* Model weights
* EMA weights
* Optimizer state
* Best RMSE
* Current epoch

---

## Submission Generation

The notebook automatically:

1. Loads all generated predictions
2. Converts images to grayscale
3. Samples 100 evenly spaced pixels
4. Creates submission format

Output:

```text
submission.csv
```

Shape:

```text
(300, 101)
```

Columns:

```text
ID, pixel_0, pixel_1, ..., pixel_99
```

---

## Hardware

Training Environment:

* Kaggle Notebooks
* 2 × NVIDIA Tesla T4 GPUs
* Mixed Precision Training (AMP)

---

## Dependencies

```bash
pip install realesrgan
pip install basicsr
pip install facexlib
pip install gfpgan
```

Core libraries:

* PyTorch
* torchvision
* Real-ESRGAN
* BasicSR
* NumPy
* Pandas
* Pillow
* tqdm

---

## Key Features

✅ Official Real-ESRGAN pretrained initialization

✅ Low-light image enhancement

✅ EMA model averaging

✅ Full-image validation

✅ Cosine annealing with LR restart

✅ 8× Test-Time Augmentation

✅ Seam-free tiled inference

✅ Competition-specific RMSE evaluation

✅ Automated submission generation

---

## Final Result

**Best Validation RMSE: 27.778**

The fine-tuned Real-ESRGAN model significantly improved over the pretrained baseline and produced high-quality super-resolved outputs suitable for submission to the DLP Jan 2026 NPPE-3 competition.
