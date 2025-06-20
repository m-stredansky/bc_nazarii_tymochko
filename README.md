# 📦 Digital Twin for Sinter Quality Evaluation using SAM

This repository contains a segmentation pipeline designed for evaluating the quality of iron ore sinter. It uses Meta’s **Segment Anything Model (SAM)** to detect particles on conveyor belt images and calculate critical quality metrics such as the **Sinter Quality Index (SQI)**. The project is part of a bachelor thesis at the Technical University of Košice.

---

## 📁 Repository Structure & Key Components

- **`outputs/`**  
  This directory contains the **best segmentation results** generated during experimentation. These include visual outputs of sinter particles segmented using different model configurations.

- **`Sam_LoRA/dataset/`**  
  Contains the dataset used for **adapter-based fine-tuning** of the SAM model (LoRA).  
  - Organized into `train/` and `test/` folders  
  - Includes annotated masks and corresponding images for adaptive tuning

- **`system_files/SAM/`**  
  Includes segmentation scripts for running two different setups:  
  - `lora_segm.py`: Uses **LoRA-tuned SAM** for enhanced segmentation performance  
  - `sam.py`: Uses the **default Segment Anything Model** without any fine-tuning

---

## ⚙️ Installation Instructions

Follow these steps to install and configure the environment for running the segmentation system.

### 1. Clone the Repository

```bash
git clone https://github.com/Nazar1119/digital_twin.git
cd digital_twin
```
### 2. Create a virtual evironment
```
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Python Dependencies
```
pip install -r requirements.txt
```

### 4. Download SAM Model Weights

```
cd Sam_LoRA
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

cd ../system_files/SAM
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 5. Clone the Official Segment Anything Repository

```
cd ..
git clone https://github.com/facebookresearch/segment-anything.git
```

### 6. Prepare Output Folders

```
cd system_files
mkdir output_sam
mkdir output_lora
```
### 7. And then you also need to go to folder Sam_LoRA, then run this commands:

```
poetry config virtualenvs.in-project false
poetry install --all-extras
poetry run pip install --upgrade gradio safetensors monai
```

### To start programm you need run this:
Run with ViT-H model:

```
python system_files/SAM/sam.py
```
Run with LoRA model:

```
python system_files/SAM/lora_segm.py
```
Then output will be saved into output_sam and output_lora folders
