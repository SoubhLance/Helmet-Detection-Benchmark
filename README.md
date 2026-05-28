# Helmet Detection using YOLOv8 with Advanced Attention Mechanisms

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> **Real-time helmet and non-helmet detection using YOLOv8 with 6 different attention mechanisms including CBAM, SimAM, PA-CBAM, Scale-Adaptive CBAM, and Hybrid approaches.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Qualitative Analysis](#qualitative-analysis)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This repository presents a comprehensive study of attention mechanisms integrated with YOLOv8 for helmet detection in surveillance and construction safety applications. We evaluate six different attention modules:

- **Baseline YOLOv8n**: Standard YOLOv8 nano without attention
- **CBAM (Convolutional Block Attention Module)**: Sequential channel and spatial attention
- **PA-CBAM (Position-Aware CBAM)**: CBAM with coordinate encoding
- **Scale-Adaptive CBAM**: Dynamic reduction ratio based on channel depth
- **SimAM (Simple Parameter-Free Attention Module)**: Zero-parameter attention
- **CBAM+SimAM Hybrid**: Combined attention approach

All models are trained on a diverse helmet detection dataset and evaluated on metrics including mAP50, mAP50-95, precision, and recall.

## Features

- 🚀 **6 Different Attention Mechanisms** - Compare and analyze various attention approaches
- 📊 **Comprehensive Evaluation** - Detailed metrics and visualization for each model
- 🎨 **Architecture Diagrams** - Clear visualization of each attention module
- 🖼️ **Qualitative Results** - Side-by-side comparison of detection outputs
- 💻 **Local Training Ready** - Optimized for RTX 4060 GPU with CUDA 12.1
- 🔄 **Easy Model Switching** - Modular design for quick experimentation
- 📈 **Training Logs** - Automatic saving of metrics, graphs, and checkpoints

## Models Implemented

| Model | Attention Type | Parameters | Key Features | Best mAP50 (Dataset B) |
|-------|---------------|------------|--------------|------------|
| **Baseline** | None | 3.01M | Standard YOLOv8n | 0.9781 |
| **CBAM** | Channel + Spatial | 3.03M | Reduction ratio 16 | 0.9786 |
| **PA-CBAM** | Position-Aware | 3.03M | Coordinate encoding | 0.9792 |
| **Scale-Adaptive CBAM** | Dynamic Reduction | 3.03M | Adaptive ratios (4/8/12/16) | 0.9785 |
| **SimAM** | Parameter-Free | 3.01M | Zero learnable params | 0.9788 |
| **CBAM+SimAM** | Hybrid | 3.03M | Combined approach | **0.9793** |

## Architecture

### Attention Mechanism Visualizations

#### CBAM (Convolutional Block Attention Module)
![CBAM Architecture](figures/cbam.png)
*CBAM sequentially applies channel and spatial attention*

#### SimAM (Simple Parameter-Free Attention Module)
![SimAM Architecture](figures/simam.png)
*Energy-based neuron-level attention with zero parameters*

#### Overall Model Architecture
![Overall Architecture](figures/architecture.png)
*YOLOv8 backbone with integrated attention modules*

## Dataset

We use the **Helmet Detection Dataset** from Kaggle, containing:
- **17,248** training images
- **2,438** validation images
- **2** classes: `helmet` and `non-helmet`
- Diverse scenarios including construction sites, factories, and roads

[Download Dataset](https://www.kaggle.com/datasets/octophobia/helmet-detection-dataset)

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (RTX 4060 or higher recommended)
- 16GB+ RAM
- 50GB+ storage space

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Helmet-Detection-YOLOv8.git
cd Helmet-Detection-YOLOv8
```

### Step 2: Create Virtual Environment
```bash
python3.10 -m venv yolov8_env
source yolov8_env/bin/activate  # On Windows: yolov8_env\Scripts\activate
```

### Step 3: Install PyTorch with CUDA Support
```bash
# For RTX 4060 (CUDA 12.1)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt 
```

### Step 5: Verify Installation
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')" 
```

## Usage
### Directory Structure Setup
```bash
# Update dataset path in each notebook
DATASET = "path/to/your/helmet-detection-dataset/archive7_extracted"
```

## Training Models

### 1. Baseline YOLOv8n

```bash
jupyter notebook models/baseline.ipynb
```

### 2. CBAM with 3 Layers

```bash
jupyter notebook models/cbam_3layer.ipynb
```

### 3. Position-Aware CBAM (PA-CBAM)

```bash
jupyter notebook models/pacbam.ipynb
```

### 4. Scale-Adaptive CBAM

```bash
jupyter notebook models/scale_adaptive_cbam.ipynb
```

### 5. SimAM (3 Layers)

```bash
jupyter notebook models/simam_3layer.ipynb
```

### 6. CBAM + SimAM Hybrid

```bash
jupyter notebook models/cbam_simam.ipynb
```

---

# Training Configuration

All models use the following training parameters:

* Epochs: 100
* Batch Size: 16
* Image Size: 640×640
* Optimizer: SGD (`lr=0.01`, `momentum=0.937`)
* AMP: True (Mixed Precision)
* Patience: 50 (early stopping)
* Workers: 2

---

# Output Structure

Each training run saves results to:

```text
runs/{model_name}/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
├── labels.jpg
└── args.yaml
```

* `best.pt` — Best model checkpoint
* `last.pt` — Final epoch checkpoint
* `results.png` — Training curves
* `confusion_matrix.png` — Confusion matrix
* `labels.jpg` — Dataset labels visualization
* `args.yaml` — Training arguments

---

# Results

> Tests on both datasets were performed at a confidence threshold of 0.001 and an IoU threshold of 0.5.

## Quantitative Results — Dataset B (1,766 images, Benchmark)

| Model               | Precision | Recall | mAP50  | mAP50-95 | Inference Time (ms) |
| ------------------- | --------- | ------ | ------ | -------- | ------------------- |
| Baseline            | 0.9528    | 0.9507 | 0.9781 | 0.6773   | 31.65               |
| CBAM                | 0.9463    | 0.9487 | 0.9786 | 0.6753   | 31.59               |
| PA-CBAM             | 0.9515    | 0.9467 | 0.9792 | 0.6763   | 7.36                |
| Scale-Adaptive CBAM | **0.9560**| 0.9395 | 0.9785 | 0.6744   | **5.22**            |
| SimAM               | 0.9496    | 0.9450 | 0.9788 | 0.6745   | 29.60               |
| CBAM+SimAM          | 0.9479    | **0.9497** | **0.9793** | 0.6742 | 7.93          |

## Quantitative Results — Dataset C (132 images, Self-Acquired)

| Model               | Precision | Recall | mAP50  | mAP50-95 | Inference Time (ms) |
| ------------------- | --------- | ------ | ------ | -------- | ------------------- |
| Baseline            | **0.8028** | 0.5804 | **0.6754** | 0.2814 | 5.48              |
| CBAM                | 0.7865    | 0.5826 | 0.6653 | 0.2796   | 6.25                |
| PA-CBAM             | 0.7857    | **0.5929** | 0.6748 | **0.2854** | 6.28            |
| Scale-Adaptive CBAM | 0.7801    | 0.5792 | 0.6597 | 0.2780   | 7.40                |
| SimAM               | 0.7883    | 0.5910 | 0.6737 | 0.2809   | **4.99**            |
| CBAM+SimAM          | 0.7946    | 0.5914 | 0.6698 | 0.2773   | 6.54                |

**Key takeaways:**
- On the benchmark dataset (B), **CBAM+SimAM** achieves the highest mAP50 (0.9793), while **Scale-Adaptive CBAM** delivers the fastest inference (5.22 ms).
- On the self-acquired dataset (C), the **Baseline** leads in mAP50 and precision; **PA-CBAM** achieves the best recall and mAP50-95.
- All models achieve sub-100 ms latency, suitable for real-time deployment at ~25–30 FPS on edge GPUs.

---

# Qualitative Results

```text
figures/qualitative_results.png
```

Detection results comparison across different attention mechanisms. Attention-based variants show improved detection in crowded and partially occluded scenes, with cleaner bounding box placement and more consistent confidence scores compared to the baseline.

---

# Citation

If you use this work in your research, please cite:

```bibtex
@software{helmet_detection_yolov8,
  author = {Soubhik SaDHU, Debjit Babu, Soumyajit Das},
  title = {Helmet Detection using YOLOv8 with Advanced Attention Mechanisms},
  year = {2026},
  url = {https://github.com/SoubhLance/Helmet-Detection-Benchmark},
  version = {1.0.0}
}
```

---

# License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# Acknowledgments

* Ultralytics YOLOv8 — Object detection framework
* CBAM Paper — Convolutional Block Attention Module
* SimAM Paper — Simple Parameter-Free Attention Module
* Kaggle Helmet Dataset — Dataset provider

---

# Contact

For questions or collaborations, please open an issue or contact:

* Email: [your.email@example.com](mailto:your.email@example.com)
* GitHub: @SoubhLance, devlance492, Soumyajit-here

---

If you find this repository useful, please consider giving it a star ⭐
