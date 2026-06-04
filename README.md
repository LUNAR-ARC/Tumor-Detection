# 🔬 Tumor Detection

> Deep learning system for detecting and classifying brain tumors from MRI scans, with GradCAM visualization and a Flask diagnostic dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)
![MRI](https://img.shields.io/badge/Input-Brain%20MRI-teal?style=flat-square)

---

## 📌 Overview

Tumor Detection is a CNN-based medical imaging system that classifies brain MRI scans into tumor categories (glioma, meningioma, pituitary tumor) or healthy (no tumor). It provides confidence scores, GradCAM heatmap overlays for interpretability, and a clean Flask web interface for uploading and reviewing scans.

---

## 🧠 Architecture

```
Brain MRI Upload
        │
        ▼
  Image Preprocessing
  (Grayscale → Resize 224×224 → Normalize)
        │
        ▼
  CNN Classifier
  (ResNet-50 / EfficientNet, fine-tuned)
        │
        ▼
  Tumor Classification
  ┌─────────────────────────────────────┐
  │  No Tumor   → Healthy               │
  │  Glioma     → High-grade brain tumor│
  │  Meningioma → Meningeal tumor       │
  │  Pituitary  → Pituitary gland tumor │
  └─────────────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
  GradCAM Heatmap            Confidence Score
  (Tumor region highlighted) + Scan Record
        │                          │
        └──────────┬───────────────┘
                   ▼
          Flask Dashboard
```

---

## 🩻 Tumor Classes

| Class | Description |
|---|---|
| `No Tumor` | Healthy brain — no detectable abnormality |
| `Glioma` | Tumor arising from glial cells — most common malignant brain tumor |
| `Meningioma` | Tumor of the meninges — usually benign but space-occupying |
| `Pituitary Tumor` | Tumor of the pituitary gland — can affect hormones |

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is **not a substitute for professional radiological or neurological diagnosis**.

---

## 🗂️ Project Structure

```
Tumor-Detection/
├── backend/
│   ├── app.py                    # Flask entry point
│   ├── model.py                  # CNN inference + GradCAM
│   ├── preprocess.py             # MRI normalization pipeline
│   └── database.py               # SQLite scan history
├── frontend/
│   ├── templates/
│   │   └── index.html            # Upload + result dashboard
│   └── static/
│       ├── style.css
│       └── app.js
├── model/
│   ├── train.py                  # Fine-tuning script
│   ├── tumor_detection_model.pth # Trained weights
│   └── class_map.json            # 4-class label mapping
├── database/
│   └── scans.db
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- GPU recommended (CUDA compatible)

### 1. Clone the Repository

```bash
git clone https://github.com/LUNAR-ARC/Tumor-Detection.git
cd Tumor-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

```
flask
torch
torchvision
opencv-python
numpy
Pillow
flask-cors
```

### 4. Download Dataset (training only)

[Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Place in `data/brain_tumor_mri/` with subdirectories: `glioma/`, `meningioma/`, `notumor/`, `pituitary/`.

---

## 🤖 Model Training

```bash
python model/train.py \
  --data_dir data/brain_tumor_mri \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.0001 \
  --output model/tumor_detection_model.pth
```

Training details:
- Architecture: ResNet-50 (pretrained ImageNet → fine-tuned)
- Augmentations: random flip, rotation, brightness/contrast jitter
- Loss: CrossEntropyLoss with class weights
- Optimizer: Adam + ReduceLROnPlateau

---

## 🚀 Running the Application

```bash
python backend/app.py
```

Navigate to `http://localhost:5000`. Upload an MRI scan image (JPG/PNG) to get a tumor classification result with heatmap.

---

## 🔬 GradCAM Visualization

GradCAM generates a heatmap indicating which regions of the MRI scan most influenced the classification. This is critical for medical AI interpretability — ensuring the model focuses on neurologically relevant structures rather than artifacts or scanner noise.

The heatmap is overlaid on the original MRI and returned as a base64-encoded PNG.

---

## 📡 API Reference

### `POST /predict`

**Request (multipart/form-data):**
```
file: <MRI scan image (JPG/PNG)>
```

**Response:**
```json
{
  "prediction": "Glioma",
  "confidence": 0.91,
  "heatmap": "<base64_png>",
  "scan_id": "scan_001",
  "timestamp": "2025-04-12T08:45:00Z"
}
```

### `GET /history`

Returns all previous scans with predictions, confidence, and timestamps.

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Model | ResNet-50 (PyTorch, fine-tuned) |
| Dataset | Brain Tumor MRI Dataset (Kaggle) |
| Explainability | GradCAM |
| Backend | Flask + Flask-CORS |
| Frontend | HTML/CSS/JS |
| Database | SQLite (scan history) |
| Image Processing | OpenCV + Pillow |

---

## 📄 License

MIT License. See `LICENSE` for details.
