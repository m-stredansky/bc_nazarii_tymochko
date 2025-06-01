# 📦 Digital Twin for Sinter Quality Evaluation using SAM

This repository contains a segmentation pipeline designed for evaluating the quality of iron ore sinter. It uses Meta’s Segment Anything Model (SAM) to detect particles on a conveyor belt image and calculate critical quality metrics like the Sinter Quality Index (SQI). The project is part of a bachelor thesis at the Technical University of Košice.

---

## 🚀 Features

- High-quality segmentation using **SAM ViT-H**
- Fine-tuned **LoRA + SAM ViT-B** variant for improved domain performance
- Calculation of:
  - Equivalent Circle Diameter
  - Bounding box & rotated minimum rectangle
  - Area, orientation, and axis lengths
- Visualizations: overlay, bounding boxes, histograms
- Sinter Quality Index (SQI) metric

---

## 📁 Repository Structure

bc_nazarii_tymochko/
├── system_files/
│ ├── SAM/ # Contains SAM model weights and segmentation scripts
│ ├── img1.jpg # Test image
│ └── segment-anything/ # Cloned Meta SAM repo
├── Sam_LoRA/ # Fine-tuned model using LoRA
├── output_sam/ # Outputs from ViT-H model
├── output_lora/ # Outputs from LoRA model
├── sam.py # Segmentation with ViT-H
├── lora_segm.py # Segmentation with LoRA model
└── README.md



---

## ⚙️ Installation Instructions

Follow these steps to install and configure the environment for running the segmentation system.

### 1. Clone the Repository

```bash
git clone https://github.com/Nazar1119/digital_twin.git
cd digital_twin

### 2. Create a virtual evironment
```
python3 -m venv venv
source venv/bin/activate

