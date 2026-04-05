# UAD-Net: Unit Attention DeepLabv3+ for Road Segmentation

This repository contains a custom PyTorch implementation of a semantic segmentation network designed for complex, unstructured driving environments (trained on the IDD20k II dataset). 

## Architecture Highlights
* **Backbone:** MobileNetV2 for lightweight, edge-efficient feature extraction.
* **Context Module:** Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context.
* **Attention Mechanism:** A custom Unit Attention Module (UAM) combining Position and Channel Attention to refine spatial dependencies.

## Performance Metrics
The model achieved the following results on the validation set (1,055 images):
* **Mean Pixel Accuracy (MPA):** 92.93%
* **Mean IoU (MIoU):** 80.18%
  * Non-drivable IoU: 93.37%
  * Road IoU: 85.94%
  * Drivable Fallback IoU: 61.24%

## Visual Results
*(Upload your `prediction_uad_idd20kII.png` image to your repo, then link it here)*
![Segmentation Results](prediction_uad_idd20kII.png)

## Usage
Run the training and evaluation pipeline:
`python uad_idd20II_imp.py`