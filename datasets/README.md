# Dataset Description

This project uses three datasets: one large-scale training dataset and two independent testing datasets for evaluating robustness and generalization under real-world conditions.

---

# Dataset-A: Training Dataset

Dataset-A is a binary helmet detection dataset sourced from Kaggle:

https://www.kaggle.com/datasets/octophobia/helmet-detection-dataset

The dataset is derived from the HardHat-Vest Dataset V3 by Muhammet Zahit Aydın. The original four-class annotation scheme was remapped into two classes:

- `helmet` (label 0) — retained from the original Helmet class
- `non-helmet` (label 1) — relabeled from the original Head class

The original `Vest` and `Person` classes were discarded.

All images were standardized to `640×640` resolution and converted to YOLO-format `.txt` annotations.

## Dataset Split

| Split | Images |
|---|---|
| Train | 17,248 |
| Validation | 2,438 |
| Test | 2,455 |

## Annotation Statistics

| Class | Annotations |
|---|---|
| Helmet | 57,860 |
| Non-Helmet | 124,826 |
| Total | 181,581 |

## Characteristics

The dataset contains diverse outdoor construction scenarios including:

- Scaffolding environments
- Multi-worker scenes
- Inter-object occlusions
- Small helmet instances due to elevated camera viewpoints
- Dense construction-site activity

These variations provide broad coverage of realistic training conditions for helmet detection.

---

# Dataset-B: Benchmark Testing Dataset

Dataset-B is an external benchmark testing dataset consisting of `1,767` images from the Hard Hat Workers collection:

https://www.kaggle.com/datasets/ammarnassanalhajali/hard-hat-workers

The dataset was assembled by researchers at Northeastern University, China.

## Dataset Statistics

- Total Images: 1,767
- Approximate Annotations: 27,039
- Average Annotations per Image: ~3.8

The dataset contains annotations for:

- Helmet
- Head
- Person

## Characteristics

Scenes primarily include:

- Industrial yards
- Manufacturing environments
- Ground-level worker viewpoints
- Slightly elevated viewpoints
- Dense multi-worker frames
- Structural background clutter

Foreground–background ambiguity introduced by machinery and structural elements makes Dataset-B challenging for evaluating cross-distribution robustness.

This dataset is used exclusively for evaluation and is not included during training.

---

# Dataset-C: Self-Acquired Testing Dataset

Dataset-C is a self-acquired testing dataset containing `132` images captured using a mobile camera across real-world environments not represented in the publicly sourced datasets.

## Characteristics

The dataset includes:

- Workers at varying distances
- Handheld camera viewpoints
- Mild motion blur
- Perspective distortion
- Partial helmet occlusions
- Small worker groups
- Real-world deployment variability

This dataset is used to evaluate out-of-distribution generalization under practical deployment conditions where camera characteristics and environmental conditions differ from the curated training distribution.

---

# Notes

- All experiments were conducted using YOLO-format annotations.
- Images were resized to `640×640` during preprocessing.
- Dataset-C is not publicly released due to collection and privacy considerations.
