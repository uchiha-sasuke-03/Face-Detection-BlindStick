# Smart Blind Stick — AI Object Detection & Navigation System

An advanced, real-time AI computer vision system designed to act as the **"eyes"** for a smart blind stick. Powered by **YOLOv8**, **DeepFace**, and **Facenet512**, this system runs locally on a camera feed to detect obstacles, recognize known faces by name, read text via OCR, and analyze traffic lights — all while providing seamless **audio feedback** without blocking the video stream.

Optimized for **Raspberry Pi 4** deployment — no PyTorch required.

---

## 🚀 Features

### 🔍 Object Detection & Spatial Awareness
Uses a pre-trained **YOLOv8s** model to detect common objects (people, vehicles, furniture, etc.). The camera feed is divided into a **3×3 grid** to estimate:
- **Distance**: "close", "medium distance", or "far" (based on bounding box area ratio)
- **Location**: "left", "straight ahead", or "right"

> **Smart Alert Mode**: Only objects at **"close" distance** trigger voice alerts. Medium and far objects are shown on-screen but remain silent — reducing noise and prioritizing immediate hazards.

### 👤 Face Recognition (DeepFace + Facenet512)
When YOLO detects a **"person"**, the system automatically tries to identify them:
- **OpenCV Haar Cascade**: Lightweight, fast face detection optimized for ARM/Raspberry Pi hardware.
- **Facenet512**: Extracts **512-dimensional face embeddings** via DeepFace — high accuracy, low resource usage.
- **Cosine Similarity Matching**: Compares live embeddings against a trained database using weighted scoring (60% best individual + 40% average centroid).
- **YOLO Integration**: Replaces the generic "person" label with the recognized name — speaks **"Anupam close, straight ahead"** instead of "person close, straight ahead".
- **Low-Light Robustness**: Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) as a fallback when face detection initially fails.
- **Training Augmentation**: Each image generates 9 variants (gamma correction, brightness shifts, CLAHE, horizontal flip) for robust matching across lighting conditions.

### 🚦 Continuous Traffic Light Detection
When a traffic light is detected, the AI zooms in, analyzes the dominant colors using **OpenCV HSV masking**, and announces:
> *"Traffic light is Red / Yellow / Green"*

### 🛑 Automatic Road Sign Reading
Detects road signs (e.g., Stop Signs) and automatically triggers the **OCR** engine to read the text written on the sign aloud.

### 📖 On-Demand Text Reading (OCR)
Point the camera at any book, document, or sign and press **`r`** to scan and read all visible text using **EasyOCR**.

### 🔊 Asynchronous Audio Engine (TTS)
Uses native **Windows SAPI** (`win32com`) inside an isolated, non-blocking background thread. The video feed will *never* lag, stutter, or freeze while the computer is speaking.

### 🛡️ Smart Anti-Spam Muzzle
Prevents auditory overload by enforcing a **3-second cooldown** on repeated obstacle announcements in the same grid location, while giving high-priority exceptions to rapidly changing hazards like traffic lights.

---

## 💻 Prerequisites & Requirements

This software is built to run on both **Windows** (for development) and **Raspberry Pi 4** (for deployment).

### 1. Hardware Requirements

#### For PC/Laptop (Development)
- A Windows PC/Laptop (a dedicated GPU is recommended for higher framerates, but CPU-only works).
- A standard USB Webcam.

#### For Raspberry Pi (Deployment)
- **Raspberry Pi 4 Model B (4GB or 8GB RAM)**: 4GB is the minimum required to run YOLOv8 and DeepFace concurrently without freezing.
- **Camera**: Raspberry Pi Camera Module (v2 or v3) or a compatible USB Webcam.
- **Storage**: MicroSD Card (At least 32GB, Class 10/A1) for OS, libraries, and models.
- **Power Supply**: Official 15W USB-C power supply to prevent under-voltage throttling.
- **Cooling**: A fan or heatsink is highly recommended to prevent thermal throttling during continuous AI inference.
- **Audio Output**: Headphones or a small portable speaker connected via the 3.5mm audio jack or Bluetooth.

### 2. Software Requirements

#### Operating System
- **Windows 10/11** (for development)
- **Raspberry Pi OS (64-bit)** (for deployment). *Note: The 64-bit OS is required for modern AI libraries.*

#### Core Dependencies
- **Python 3.10–3.12** (Python 3.12 is recommended)
- **Git** (to clone the repository)
- **C++ Build Tools**: Visual Studio Build Tools (Windows) or `build-essential` (Linux/Pi) for compiling certain Python packages.

#### System-level Libraries (Raspberry Pi / Linux Only)
If you are deploying on a Raspberry Pi, you need to install standard system dependencies for OpenCV and GUI before running `pip install`:
```bash
sudo apt update
sudo apt install libgl1-mesa-glx libglib2.0-0
```

---

## ⚙️ Installation Guide

### Step 1: Clone the Repository
```cmd
git clone https://github.com/uchiha-sasuke-03/Obstacle-Detection-for-Smart-BlindStick.git
cd Obstacle-Detection-for-Smart-BlindStick
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

> **Note:** The first run will automatically download pre-trained weights for YOLOv8 and Facenet512 into your system cache. This may take a few minutes.

### Step 4: Setup Facial Recognition

**Option A: Individual Photos (Simplest)**
1. Place clear headshot images directly inside the `known_faces/` folder.
2. Name each file with the person's name (e.g., `John.jpg`, `Mom.png`).

**Option B: Multiple Photos per Person (Best Accuracy)**
1. Create a subfolder inside `known_faces/` named after the person.
2. Place multiple photos of that person inside the subfolder.

```
known_faces/
├── Anupam.jpg              ← Individual headshot
├── Harsha.jpg
├── Nandeeshwar Sir/        ← Subfolder = person name
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
```

### Step 5: Train the Face Recognition Model
```cmd
python train_faces.py
```

This will:
- Scan all images in `known_faces/` (including subfolders)
- Detect faces using **OpenCV Haar Cascade** (fast, lightweight)
- Extract **512-dimensional Facenet512 embeddings** per face via **DeepFace**
- Generate **9 augmented variants** per image (gamma, brightness, CLAHE, flip)
- Save the database to `known_faces/face_embeddings_v6.pkl`

**Example output:**
```
==============================================================
  SMART BLIND STICK - FACE TRAINING v6 (DeepFace Facenet512)
==============================================================
  Model         : Facenet512 (512-dim embeddings)
  Detector      : OpenCV Haar Cascade
  Backend       : DeepFace (TF-Keras)
  Augmentation  : flip + gamma + brightness + CLAHE (9x)
==============================================================

  [ 1/14] Adithya/Adithya.jpg
           OK - 9/9 variants embedded
  [ 2/14] Anupam/Anupam.jpg
           OK - 9/9 variants embedded
  ...

==============================================================
  TRAINING COMPLETE
==============================================================
  People registered : 5
  Total embeddings  : 125
  Source images OK  : 14
  Skipped images    : 0
```

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
├── main.py                  # Core application loop (YOLOv8 + face integration)
├── train_faces.py           # Face embedding training pipeline (DeepFace + Facenet512)
├── yolov8s.pt               # YOLOv8 Small model weights
├── known_faces/             # Face recognition image database
│   ├── PersonName.jpg       # Individual headshot photos
│   └── PersonName/          # Subfolder with multiple photos
│       └── *.jpg
└── modules/
    ├── audio_tts.py         # Async Windows SAPI voice engine (non-blocking)
    ├── face_recognizer.py   # DeepFace Facenet512 face matching engine
    ├── ocr_reader.py        # EasyOCR text reading module
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
│  known_faces/  ──►  OpenCV Haar   ──►  Facenet512        │
│  (images)          Cascade Detect     (512-dim vector)   │
│       │                                     │            │
│       ▼                                     ▼            │
│  9 augmented                       face_embeddings.pkl   │
│  variants each                     (per-person DB with   │
│  (gamma, flip,                      individual + avg     │
│   brightness,                       embeddings)          │
│   CLAHE)                                                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    RUNTIME (Live)                         │
│                                                          │
│  YOLO detects   ──►  Crop person  ──►  OpenCV detects    │
│  "person"            region            face in crop      │
│                                           │              │
│                                           ▼              │
│                                    Facenet512 via        │
│                                    DeepFace              │
│                                    (512-dim embedding)   │
│                                           │              │
│                                           ▼              │
│                                    Cosine Similarity     │
│                                    vs. trained DB        │
│                                           │              │
│                                           ▼              │
│                                 Match ≥ 0.50 ?           │
│                                   YES → "Anupam close,   │
│                                          straight ahead" │
│                                   NO  → "person close,   │
│                                          straight ahead" │
└──────────────────────────────────────────────────────────┘
```

---

## 🚧 Future Roadmap

- **Indian Road Sign Detection**: Train a custom YOLOv8 model on an Indian Road Sign dataset to recognize iconography-based signs (which don't rely on text).
- **Raspberry Pi Deployment**: Deploy on Raspberry Pi 4 with camera module (DeepFace is already optimized for this).
- **Multi-Language OCR**: Extend EasyOCR to support Hindi and other regional languages.
- **Distance Estimation**: Use monocular depth estimation for more accurate obstacle distance measurement.
- **Cross-Platform TTS**: Migrate from Windows SAPI to pyttsx3 for Linux/Raspberry Pi support.

---

## 👥 Team

Built with ❤️ by the AMC Institutions team.
