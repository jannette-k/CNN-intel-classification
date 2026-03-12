# 🧠 Custom CNN with TensorFlow — Intel Image Classification

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-83.75%25-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Intel%20Image%20Classification-0077B5?style=for-the-badge)

A custom Convolutional Neural Network (CNN) built from scratch using TensorFlow and Keras, trained on the Intel Image Classification dataset. This project demonstrates a complete deep learning pipeline — from dataset preparation and intensive preprocessing to model training, evaluation, and result visualization.

---

## 🏆 Result

| Metric | Value |
|---|---|
| ✅ Final Validation Accuracy | **83.75%** |
| 📁 Number of Classes | 6 |
| 🖼️ Training Images | ~14,000 |
| 🏗️ Model | Custom CNN (3 Conv Blocks) |

---

## 📁 Project Structure

```
CNN_Intel_Classification/
├── CNN_Intel.ipynb        # Main Google Colab notebook
└── README.md              # Project documentation
```

---

## 📊 Dataset

**Intel Image Classification**
- 🔗 Source: [Kaggle — Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- 🖼️ Total Images: ~14,000
- 📁 Classes: **6**

| # | Class | Description |
|---|---|---|
| 1 | 🏢 Buildings | Urban building structures |
| 2 | 🌲 Forest | Dense forest and trees |
| 3 | 🧊 Glacier | Ice and glacier landscapes |
| 4 | ⛰️ Mountain | Mountain scenery |
| 5 | 🌊 Sea | Ocean and sea views |
| 6 | 🛣️ Street | Street and road scenes |

### Dataset Folder Structure
```
intel_dataset/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_test/
    └── seg_test/
        ├── buildings/
        ├── forest/
        ├── glacier/
        ├── mountain/
        ├── sea/
        └── street/
```

---

## ⚙️ Setup & Requirements

This project runs entirely on **Google Colab** — no local installation needed.

### Prerequisites
- Google Account (for Colab + Drive)
- Kaggle Account (to download dataset)
- GPU Runtime enabled *(Runtime → Change runtime type → T4 GPU)*

### Libraries Used
```python
tensorflow >= 2.19.0
numpy
matplotlib
pathlib
```

---

## 🔄 Pipeline Overview

### Block 1 — Import Libraries
All required libraries are imported and GPU availability is confirmed.

### Block 2 — Mount Google Drive & Load Dataset
The dataset is accessed directly from Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

train_dir = pathlib.Path("/content/drive/MyDrive/intel_dataset/seg_train/seg_train")
test_dir  = pathlib.Path("/content/drive/MyDrive/intel_dataset/seg_test/seg_test")
```

### Block 3 — Intensive Preprocessing
A full preprocessing and augmentation pipeline was applied:

| Technique | Value | Purpose |
|---|---|---|
| Image Resizing | 64×64 px | Uniform input size |
| Normalization | `/255.0` | Scale pixels to [0, 1] |
| Random Horizontal Flip | Enabled | Increase data diversity |
| Random Rotation | 10% | Simulate different camera angles |
| Random Zoom | 10% | Simulate distance variation |
| Train / Val Split | 80% / 20% | Evaluate generalization |
| Batch Size | 16 | Memory-efficient training |
| `.shuffle().prefetch()` | buffer=500 | Optimized pipeline speed |

### Block 4 — Build Custom CNN
A CNN built from scratch with **3 convolutional blocks**:

```
Input (64×64×3)
      ↓
Conv Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
      ↓
Conv Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
      ↓
Conv Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.4)
      ↓
GlobalAveragePooling2D
      ↓
Dense(128, ReLU) → Dropout(0.5)
      ↓
Output: Dense(6, Softmax)
```

### Block 5 — Compile & Train

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Batch Size | 16 |
| Max Epochs | 30 |
| Early Stopping | Patience = 7 |
| LR Reduction | Factor = 0.5, Patience = 3 |
| Model Checkpoint | Saves best model to Drive |

### Block 6 — Plot Results
Training and validation accuracy and loss curves plotted across all epochs to visualize model performance and detect overfitting.

### Block 7 — Final Evaluation
```python
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Final Validation Accuracy: {acc*100:.2f}%")
# ✅ Final Validation Accuracy: 83.75%
```

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
2. Upload to Google Drive in a folder named `intel_dataset`
3. Open `CNN_Intel.ipynb` in Google Colab
4. Enable GPU: **Runtime → Change runtime type → T4 GPU**
5. Run all blocks from top to bottom one at a time

---

## 🔑 Key Concepts Demonstrated

- Building a CNN from scratch with TensorFlow/Keras
- Uploading and managing datasets via Google Drive in Colab
- Intensive image preprocessing and augmentation pipeline
- Batch Normalization for stable and faster training
- Dropout regularization to prevent overfitting
- GlobalAveragePooling for better generalization
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Plotting training curves to evaluate model performance

---

## 👤 Author

Made with ❤️ using TensorFlow and Google Colab
