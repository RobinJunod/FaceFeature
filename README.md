# FaceFeature

![Python](https://img.shields.io/badge/python-3.9–3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **FaceFeature** took the challenge to build a robust facial feature detection model using only vanilla data. The images used for this are labelled and available on kaggle. it consists of 96x96 grey image of centred and clear faces. The main tool - data augmeentation


<div align="center">
  <img src="results/demo.gif" width="70%" alt="Webcam demo of FaceFeature"/>
  <p><em>Live webcam demo (runs at 20 FPS on an ThinkpadX1)</em></p>
</div>

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Evaluation & Benchmarks](#evaluation--benchmarks)

---

## Quick Start

### 1. Clone & install

```bash
# Clone
git clone https://github.com/RobinJunod/FaceFeature.git

cd FaceFeature

# Create local env (optional)
python -m venv .venv 

source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the webcam inference

```bash
python webcam.py
```


### 3. Train from scratch

Change the training dataset path and the ouput path to you will.

```bash
python training.py
```

## Dataset
The original dataset consist of grey scale low res images. The faces are well centered which make the problem easely solvable with a simple network. Even the statistacal mean is a good model on this vanilla dataset. However the goal is to make it the more robust possible. To achieve this, we need heavy data augmentation and a good network.
<div align="center">
  <img src="results/baseline.png" width="50%" alt="Webcam demo of FaceFeature"/>
  <p><em>Original data from the csv file</em></p>
</div>

<div align="center">
  <img src="results/data_aug_0.png" width="30%" alt="Aug 0"/>
  <img src="results/data_aug_1.png" width="30%" alt="Aug 1"/>
  <img src="results/data_aug_2.png" width="30%" alt="Aug 2"/>
  <br/>
  <img src="results/data_aug_3.png" width="30%" alt="Aug 3"/>
  <img src="results/data_aug_4.png" width="30%" alt="Aug 3"/>
  <p><em>Data Augmented Samples</em></p>
</div>


## Model

- 13.25M parameters 

- Input: (1, 96, 96)

- Layers: Conv + ResBlock + FC Head

- Output: 30 (x,y) keypoint coordinates




## Evaluation & Benchmarks

| Model        | RMSE (%) ↓ | FPS (CPU) ↑ | Size MB |
| ------------ | ---------- | ----------- | ------ |
| KeypointNetM | 0.004       | 25         | 50 M  |

> **Note** Benchmarks computed on the original 96x96 images test split.

---






