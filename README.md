# 🛡️ Aegis Threat Guard
### Real-Time Multi-Camera Weapon Detection & Threat Verification System

> A YOLOv8-powered security platform with Digital Image Processing (DIP) enhancement, SAHI sliced inference, multi-camera RTSP support, and a live web dashboard — built for real-world surveillance environments.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Detection Pipeline](#-detection-pipeline)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Web Dashboard](#-web-dashboard)
- [Alert System](#-alert-system)
- [Performance Optimizations](#-performance-optimizations)
- [Models](#-models)

---

## 🔍 Overview

Aegis Threat Guard is a real-time weapon detection system designed for security surveillance. It uses **YOLOv8** object detection with a multi-stage verification pipeline to minimize false alarms while maintaining high sensitivity to genuine threats.

The system ingests live camera feeds (USB, IP/RTSP), runs person detection to isolate human crops, applies a full **Digital Image Processing (DIP)** enhancement pipeline, then performs sliced weapon inference using **SAHI** — all served through a live Flask web dashboard with camera management, alert history, and runtime-configurable settings.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **YOLOv8 Dual-Model** | Separate person (`yolov8n`) and weapon (`best.pt`) models for precision targeting |
| 🔬 **SAHI Sliced Inference** | 640px tiled inference with 25% overlap — catches small weapons missed by full-frame detection |
| 🖼️ **DIP Enhancement** | Wavelet (db2) subband boost → CLAHE → Unsharp Mask pipeline improves low-light/blurry frames |
| 🕐 **Temporal Verification** | Frame accumulation (N consecutive frames) before confirming a threat — eliminates transient false positives |
| 📷 **Multi-Camera Support** | Unlimited cameras via USB index or RTSP URL, each with independent detection workers |
| 🌐 **Live Web Dashboard** | Real-time MJPEG streams, threat feed, camera management, and settings UI |
| 📡 **MQTT Alerts** | Publishes structured JSON payloads to any MQTT broker on threat confirmation |
| 🔊 **Audio Buzzer** | Plays system alert sound (macOS) or terminal bell on confirmed detection |
| 📸 **Snapshot Logging** | Optionally saves timestamped evidence images to `results/` |
| ⚡ **Motion-Based Frame Skip** | Skips AI inference on static scenes — saves 30–50% CPU on idle cameras |
| 🔥 **Model Warm-up** | Runs dummy inference at startup — eliminates first-frame JIT latency spike |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flask Web Dashboard                      │
│         (Camera Streams · Settings · Alert History)         │
└───────────────────────┬─────────────────────────────────────┘
                        │  per-camera thread
              ┌─────────▼──────────┐
              │  CameraStreamWorker │  (web/app.py)
              │  ┌───────────────┐  │
              │  │ Motion Detect │  │ ← Skip inference if static
              │  └──────┬────────┘  │
              │         │           │
              │  ┌──────▼────────┐  │
              │  │ WeaponDetector│  │  (src/detection/detector.py)
              │  └──────┬────────┘  │
              └─────────┼───────────┘
                        │
        ┌───────────────▼─────────────────────┐
        │           Detection Pipeline         │
        │                                      │
        │  1. Person Detection (YOLOv8n)       │
        │     └─ Crop + 35%/25% margin         │
        │                                      │
        │  2. SAHI Tile Slicing (640px, 25%)   │
        │     └─ Per-tile DIP Enhancement      │
        │        ├─ Wavelet db2 boost          │
        │        ├─ CLAHE (LAB space)          │
        │        └─ Unsharp Mask               │
        │                                      │
        │  3. Weapon YOLO Inference            │
        │     └─ NMS deduplication             │
        │                                      │
        │  4. ThreatVerifier                   │
        │     ├─ Confidence check (>= 0.6)     │
        │     ├─ Frame accumulation (N=3)      │
        │     └─ Miss-streak tolerance (2fr)   │
        │                                      │
        │  5. Alert Dispatch                   │
        │     ├─ MQTT publish                  │
        │     ├─ Audio buzzer                  │
        │     └─ Snapshot save (optional)      │
        └──────────────────────────────────────┘
```

---

## 🔬 Detection Pipeline

### Stage 1 — Person Detection
- Uses `yolov8n.pt` to locate humans in the frame (class 0, conf >= 0.45)
- Expands each person bounding box by **+35% horizontal** / **+25% vertical** margin
- This captures extended arms, raised weapons, and partially visible threats

### Stage 2 — SAHI Sliced Inference
- Divides the human crop into 640x640 tiles with **25% overlap**
- Each tile is independently enhanced and inferred
- Results from all tiles are merged then **NMS-deduplicated** (IoU 0.45) to remove cross-tile duplicates

### Stage 3 — DIP Enhancement (per tile)
Applied in `src/dip/enhancement.py`:

| Step | Method | Effect |
|---|---|---|
| **Wavelet Boost** | 2D DWT (Daubechies-2), HF subband x 1.25 | Sharpens weapon edges in luminance channel |
| **CLAHE** | LAB color space, clipLimit=2.0, grid=8x8 | Adaptive contrast — fixes low-light cameras |
| **Unsharp Mask** | Gaussian sigma=1.5, strength=1.4 | Restores fine edge definition after CLAHE |

### Stage 4 — Threat Verification
`ThreatVerifier` applies a **temporal accumulation gate**:
1. Confidence must be >= `CONF_MIN` (0.6) — low-confidence detections are discarded
2. Must appear in `N` consecutive frames (default: 3) — prevents transient false positives
3. **Miss-streak tolerance**: 1 missed frame does not immediately reset the counter; 2 consecutive misses are needed — handles motion blur and frame drops
4. On confirmation → fires Alert pipeline

---

## 📁 Project Structure

```
Digital_Final/
├── main.py                      # Entry point — starts Flask server
├── requirements.txt             # Python dependencies
│
├── config/
│   ├── settings.py              # All tunable constants + save/load helpers
│   ├── cameras.json             # Persisted camera list (auto-managed)
│   └── runtime_settings.json   # Live runtime overrides from dashboard
│
├── models/
│   ├── best.pt                  # Custom-trained YOLOv8 weapon model
│   └── yolov8n.pt               # YOLOv8 nano — person detection
│
├── src/
│   ├── detection/
│   │   ├── detector.py          # WeaponDetector: SAHI + YOLO + NMS
│   │   └── verifier.py          # ThreatVerifier: accumulation + alert gate
│   ├── dip/
│   │   └── enhancement.py       # DIP pipeline: Wavelet -> CLAHE -> Unsharp
│   ├── alerts/
│   │   └── alert_manager.py     # AlertManager: MQTT + Buzzer + Snapshots
│   └── ui/
│       └── visualizer.py        # Bounding box / overlay rendering
│
├── web/
│   ├── app.py                   # Flask app: routes, camera workers, streams
│   ├── templates/               # Jinja2 HTML templates
│   └── static/                  # CSS, JS, assets
│
└── results/                     # Incident snapshots (if enabled)
```

---

## 📦 Requirements

| Package | Min Version | Purpose |
|---|---|---|
| `ultralytics` | >= 8.0.0 | YOLOv8 inference |
| `opencv-python` | >= 4.8.0 | Video capture, image processing, NMS |
| `PyWavelets` | >= 1.4.0 | 2D DWT for DIP enhancement |
| `numpy` | >= 1.24.0 | Array operations |
| `Flask` | >= 3.0.0 | Web dashboard server |
| `paho-mqtt` | >= 2.0.0 | MQTT alert publishing |
| `torch` | >= 2.0.0 | PyTorch backend for YOLO |
| `torchvision` | >= 0.15.0 | Torchvision transforms |
| `Pillow` | >= 10.0.0 | Image I/O |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Digital_Final
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place model files
```
models/
├── best.pt        <- Custom weapon detection model (YOLOv8)
└── yolov8n.pt     <- Person detection (download from ultralytics)
```

> Download `yolov8n.pt` automatically on first run via `from ultralytics import YOLO; YOLO("yolov8n.pt")` if not present.

---

## 🚀 Usage

### Start the web dashboard
```bash
python main.py
```

The server starts at **http://localhost:5001** by default.

### Custom host/port
```bash
python main.py --host 0.0.0.0 --port 8080
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Network interface to bind |
| `--port` | `5001` | Port to serve the dashboard |

---

## 🔧 Configuration

All core constants live in `config/settings.py`. Key settings:

### Detection Thresholds
```python
CONF_MIN = 0.6                # Minimum weapon confidence to pass verification
FRAME_ACCUMULATION_MIN = 3    # Consecutive frames required to confirm a threat
TARGET_CLASSES = ['Pistol']   # Which weapons to actively detect
```

### DIP Pipeline
```python
USE_DIP = True          # Enable/disable full DIP enhancement
USE_WAVELET = True      # Enable/disable wavelet subband boost
USE_SAHI = True         # Enable/disable SAHI tile slicing
USE_HUMAN_ZOOM = True   # Enable/disable person-crop focus
```

### SAHI Settings
```python
SAHI_TILE_SIZE = 640    # Tile size in pixels
SAHI_OVERLAP = 0.25     # Overlap ratio between tiles (25%)
SAHI_NMS_IOU = 0.45     # IoU threshold for cross-tile NMS deduplication
HUMAN_MARGIN_X = 0.35   # Horizontal margin expansion around person crop
HUMAN_MARGIN_Y = 0.25   # Vertical margin expansion around person crop
```

### Alert Settings
```python
MQTT_ENABLED = True
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "security/threats"
BUZZER_ENABLED = True
SAVE_SNAPSHOTS = False   # Set True to save evidence images to results/
```

> Most settings can also be changed live from the **Settings page** in the dashboard without restarting.

---

## 🌐 Web Dashboard

| Page | Route | Description |
|---|---|---|
| **Camera View** | `/` | Live MJPEG streams for all active cameras |
| **Settings** | `/settings` | Runtime config: confidence, model, DIP toggles |
| **Dashboard** | `/dashboard` | Threat event history and system status |

### Camera Management
- **Add Camera** — enter a USB index (e.g. `0`) or RTSP URL (e.g. `rtsp://192.168.1.100/stream`)
- **Delete Camera** — removes the camera and stops its detection worker
- Camera list is automatically persisted to `config/cameras.json`

---

## 🚨 Alert System

On confirmed threat detection, `AlertManager` dispatches simultaneously:

### 1. MQTT Payload
```json
{
  "event": "WEAPON_DETECTED_CONFIRMED",
  "threat": "Pistol",
  "verification_score": 0.847,
  "timestamp": "20261017_193045"
}
```
Published to `security/threats` topic. Subscribe with any MQTT client or integrate with Home Assistant, Node-RED, etc.

### 2. Audio Buzzer
Plays macOS system alert (`/System/Library/Sounds/Ping.aiff`) with a 2-second cooldown between alerts. Falls back to terminal bell on non-macOS systems.

### 3. Snapshot Logging *(optional)*
Saves a timestamped JPEG to `results/ALERT_Pistol_20261017_193045.jpg` when `SAVE_SNAPSHOTS = True`.

---

## ⚡ Performance Optimizations

### Motion-Based Frame Skip
The detection worker computes frame-to-frame pixel difference before running YOLO. If the scene is static (mean diff < 1.5), inference is skipped entirely for up to 8 consecutive frames.

- **Idle camera**: ~2–5 YOLO inferences/second instead of ~45
- **Active scene**: Full-speed inference, no delay
- Active threats always override the skip (`has_threat=True`)

### Module-Level CLAHE
`_CLAHE` is created **once** at module import, not per-frame/per-tile. With SAHI generating dozens of tiles per frame, this eliminates significant allocation overhead.

### Model Warm-up
Both YOLO models run a dummy 640x640 inference immediately after loading, triggering PyTorch JIT compilation before any real camera frame arrives.

### Daubechies-2 Wavelet
Upgraded from basic Haar to `db2` wavelet, which captures smoother weapon contour edges while producing fewer reconstruction artifacts on forward/inverse DWT.

---

## 🤖 Models

| Model File | Task | Architecture | Notes |
|---|---|---|---|
| `best.pt` | Weapon detection | YOLOv8 (custom) | Trained on pistol/knife/threat dataset |
| `yolov8n.pt` | Person detection | YOLOv8 Nano | Ultralytics pretrained COCO |

### Supported Weapon Classes
```python
ALL_CLASSES = ["Pistol", "Knife", "Smartphone", "Purse", "Bill", "Card"]
TARGET_CLASSES = ["Pistol"]   # Active detection classes (configurable in dashboard)
```

---

## 📄 License

This project was developed as part of a Digital Image Processing (DIP) final project.

---

*Built with [YOLOv8](https://github.com/ultralytics/ultralytics) · [Flask](https://flask.palletsprojects.com/) · [PyWavelets](https://pywavelets.readthedocs.io/) · [OpenCV](https://opencv.org/)*
