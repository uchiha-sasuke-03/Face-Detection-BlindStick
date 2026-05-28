# Smart Blind Stick — AI Object Detection & Navigation System

An advanced, real-time AI computer vision system designed to act as the **"eyes"** for a smart blind stick. Powered by **YOLOv8 Nano**, **OpenCV DNN SSD**, **MobileFaceNet (ONNX)**, and **Tesseract OCR**, this system runs locally on a camera feed to detect obstacles, recognize known faces by name, read text via OCR, and analyze traffic lights — all while providing seamless **audio feedback** without blocking the video stream.

Optimized for **Raspberry Pi 4** and CPU deployment — ultra-lightweight with high FPS.

---

## 🚀 Features

### 🔍 Object Detection & Spatial Awareness
Uses a pre-trained **YOLOv8 Nano (yolov8n)** model for high-speed CPU object detection. 
- **Distance Estimation**: Uses an advanced maximum dimension ratio (height/width relative to the frame) to consistently identify objects as "close", "medium", or "far" regardless of their shape (tall vs. wide).
- **Location**: "left", "straight ahead", or "right".
- **Frame Skipping**: Analyzes every second frame while reusing bounding boxes, massively boosting rendering and apparent FPS.

> **Smart Alert Mode**: Only objects at **"close" distance** trigger voice alerts. Medium and far objects are shown on-screen but remain silent — reducing noise and prioritizing immediate hazards.

### 👤 Face Recognition (MobileFaceNet ONNX)
When YOLO detects a **"person"**, the system automatically identifies them:
- **OpenCV DNN SSD**: Highly accurate face detection, optimized for low-light conditions.
- **MobileFaceNet (ONNX)**: Extracts **512-dimensional face embeddings** with a ~14MB model footprint. No PyTorch or TensorFlow required!
- **Instant Name Announcements**: Recognized faces bypass the standard generic obstacle delays and are announced immediately with a dedicated "I see [Name]" voice prompt.
- **Aggressive Training Augmentation**: Each image generates **19 variants** (gamma correction, Gaussian blur, sensor noise, cropping, slight rotations, aggressive CLAHE) for extreme robustness across varying lighting conditions and angles.

### 🚦 Continuous Traffic Light Detection
When a traffic light is detected, the AI zooms in, analyzes the dominant colors using **OpenCV HSV masking**, and announces:
> *"Traffic light is Red / Yellow / Green"*

### 🛑 Automatic Road Sign Reading
Detects road signs (e.g., Stop Signs) and automatically triggers the **OCR** engine to read the text written on the sign aloud.

### 📖 On-Demand Text Reading (Tesseract OCR)
Point the camera at any book, document, or sign and press **`r`** to scan and read all visible text using **Tesseract OCR** with OpenCV preprocessing (Grayscale + CLAHE + Otsu Threshold).

### 🔊 Asynchronous Audio Engine (TTS)
Uses native **Windows SAPI** (`win32com`) inside an isolated, non-blocking background thread. The video feed will *never* lag, stutter, or freeze while the computer is speaking.

---

## 💻 Prerequisites & Requirements

### Software Requirements
- **Python 3.10–3.12**
- **Git**
- **Tesseract OCR Engine**:
  - Windows: [Download from UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
  - Linux/Raspberry Pi: `sudo apt install tesseract-ocr`

---

## ⚙️ Installation Guide

### Step 1: Clone the Repository
```cmd
git clone https://github.com/uchiha-sasuke-03/Face-Detection-BlindStick.git
cd Face-Detection-BlindStick
```

### Step 2: Create a Virtual Environment (Recommended)
```cmd
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```cmd
pip install -r requirements.txt
```

> **Note:** The first run of the training or main script will automatically download pre-trained weights for YOLOv8, OpenCV SSD, and MobileFaceNet ONNX into the `models/` directory.

### Step 4: Setup Facial Recognition
Place photos of individuals inside the `known_faces/` folder. You can use individual headshots (e.g., `Anupam.jpg`) or subfolders for multiple photos.

### Step 5: Train the Face Recognition Model
```cmd
python train_faces.py
```

This will run the lightweight MobileFaceNet pipeline, expanding your dataset with 19 augmentation variants per image, and saving the embeddings to `face_embeddings_v7.pkl`.

---

## 🏃 How to Run

```cmd
python main.py
```

### Controls While Running
| Key | Action |
|-----|--------|
| **`q`** | Quit the application and safely release the camera |
| **`r`** | Manually trigger OCR to read visible text aloud |

---

## 🛠️ Project Structure

```
├── main.py                  # Core application loop (YOLOv8 + OCR + TTS)
├── train_faces.py           # Face embedding training pipeline (MobileFaceNet)
├── yolov8n.pt               # YOLOv8 Nano model weights
├── known_faces/             # Face recognition image database
│   ├── PersonName.jpg       # Individual headshot photos
│   └── PersonName/          # Subfolder with multiple photos
│       └── *.jpg
├── models/                  # Auto-downloaded system weights
│   ├── deploy.prototxt      # OpenCV SSD detector architecture
│   ├── res10_300x300...     # OpenCV SSD detector weights
│   └── mobilefacenet.onnx   # MobileFaceNet 512-dim embedding model
└── modules/
    ├── audio_tts.py         # Async Windows SAPI voice engine (non-blocking)
    ├── face_recognizer.py   # OpenCV SSD + MobileFaceNet matching engine
    ├── ocr_reader.py        # Tesseract OCR text reading module
    ├── spatial_grid.py      # 3×3 grid positioning & distance estimation
    ├── road_signs.py        # Threaded road sign detection + OCR
    └── traffic_analyzer.py  # OpenCV HSV color analysis for traffic lights
```

---

## 🧠 Face Recognition Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING (Offline)                     │
│                                                          │
│  known_faces/  ──►  OpenCV DNN SSD  ──►  MobileFaceNet   │
│  (images)            Detect               (ONNX Runtime) │
│       │                                     │            │
│       ▼                                     ▼            │
│  19 augmented                       face_embeddings_v7   │
│  variants each                      (512-dim per-person) │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    RUNTIME (Live)                         │
│                                                          │
│  YOLO detects   ──►  Crop person  ──►  OpenCV SSD Face   │
│  "person"            region            Detection         │
│                                           │              │
│                                           ▼              │
│                                    MobileFaceNet (ONNX)  │
│                                    (512-dim embedding)   │
│                                           │              │
│                                           ▼              │
│                                    Cosine Similarity     │
│                                    vs. trained DB        │
│                                           │              │
│                                           ▼              │
│                                 Match ≥ 0.30 ?           │
│                                   YES → "I see Anupam"   │
│                                   NO  → "person close"   │
└──────────────────────────────────────────────────────────┘
```

---

## 👥 Team

Built with ❤️ by the AMC Institutions team.
