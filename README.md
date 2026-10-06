# 🛡️ AI Weapon Detection & Verification System (DIP Pipeline)

An end-to-end weapon detection, false-positive rejection, and real-time security surveillance system. The system combines deep learning ([Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)) with classical **Digital Image Processing (DIP)** algorithms (Wavelet DWT, CLAHE, Unsharp Masking, Canny edge density, Aspect Ratio, and Contour Solidity) to minimize false alarms and provide verified security alerts.

---

## 📑 Table of Contents
1. [System Architecture & Directory Structure](#-system-architecture--directory-structure)
2. [How the Pipeline Works (Step-by-Step Flowchart)](#-how-the-pipeline-works-step-by-step-flowchart)
3. [Comprehensive Function & Module Reference](#-comprehensive-function--module-reference)
   - [1. DIP Enhancement (`src/dip/enhancement.py`)](#1-dip-enhancement-srcdipenhancementpy)
   - [2. DIP Shape Analysis (`src/dip/shape_analysis.py`)](#2-dip-shape-analysis-srcdipshape_analysispy)
   - [3. Weapon Detection & Human-Zoom (`src/detection/detector.py`)](#3-weapon-detection--human-zoom-srcdetectiondetectorpy)
   - [4. Threat Verification & Rejection Logic (`src/detection/verifier.py`)](#4-threat-verification--rejection-logic-srcdetectionverifierpy)
   - [5. Alert Dispatcher & Logger (`src/alerts/alert_manager.py`)](#5-alert-dispatcher--logger-srcalertsalert_managerpy)
   - [6. Visualizer & HUD Rendering (`src/ui/visualizer.py`)](#6-visualizer--hud-rendering-srcuivisualizerpy)
   - [7. Core CLI & Main Orchestrator (`main.py`)](#7-core-cli--main-orchestrator-mainpy)
   - [8. Web Dashboard & Multi-Threaded Streaming (`web/app.py`)](#8-web-dashboard--multi-threaded-streaming-webapppy)
   - [9. Legacy & Experimentation Scripts](#9-legacy--experimentation-scripts)
4. [Configuration Reference (`config/settings.py`)](#-configuration-reference-configsettingspy)
5. [Installation & Setup](#-installation--setup)
6. [How to Run](#-how-to-run)
7. [Live Keyboard Controls](#-live-keyboard-controls)

---

## 📁 System Architecture & Directory Structure

```
Digital_Final/
├── config/
│   ├── __init__.py
│   ├── cameras.json          # Multi-camera configurations (local USB/FaceTime & IP RTSP)
│   └── settings.py          # Centralized configuration (thresholds, weights, camera, MQTT)
├── src/
│   ├── dip/
│   │   ├── __init__.py
│   │   ├── enhancement.py   # 2D Wavelet DWT, LAB CLAHE, and Unsharp Masking
│   │   └── shape_analysis.py# ROI aspect ratio, Canny edge density & contour solidity
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py      # Human-Zoom ROI, Bicubic super-sampling, YOLO inference
│   │   └── verifier.py      # Flowchart verification (Confidence, Weighted Scoring, Frame Accumulation)
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── alert_manager.py # macOS audio buzzer, MQTT publishing, auto-snapshot logger
│   └── ui/
│       ├── __init__.py
│       └── visualizer.py    # Overlay rendering, alarm banner, Picture-in-Picture ROI, side-by-side
├── web/
│   ├── app.py               # Flask web server with decoupled capture & detection threads
│   ├── static/              # CSS styles and JavaScript (dashboard, camera grid, settings)
│   └── templates/           # HTML templates (camera monitoring, multi-cam grid, configuration)
├── models/
│   ├── best.pt              # Fine-tuned YOLO weapon detection model weights
│   └── yolov8n.pt           # YOLOv8 nano person detection model
├── data/
│   └── images/              # Benchmark test image dataset (image1.jpg - image14.jpg)
├── results/                 # Verified detections, comparison outputs, and alert incident snapshots
├── main.py                  # Primary CLI application entry point (Camera, Dual-cam, Compare)
├── detecting-images.py      # Backward-compatible script interface
└── preprocessing-images.py  # Image preprocessing & Wavelet experiments
```

---

## ⚙️ How the Pipeline Works (Step-by-Step Flowchart)

The system is designed with a **False-Positive Rejection Architecture** to ensure harmless elongated items (e.g., phones, pens, wallets, remotes) are not misclassified as weapons:

```
[Raw Camera Frame]
        │
        ▼
[Stage 1: Human-Guided Targeted SAHI]
   ├── Detects humans using yolov8n.pt (conf >= 0.35)
   ├── Expands bounding box by +35% horizontal / +25% vertical margin (capturing outstretched aiming arms & waist)
   ├── Adaptive Slicing: Slices human ROI into overlapping high-res tiles (640px) at 1:1 native camera resolution
   ├── Upscales small crops using Bicubic Super-sampling (min dimension >= 640px)
   └── (Fallback: Full frame processed directly if no human is present)
        │
        ▼
[Stage 2: DIP Enhancement]
   ├── 2D Discrete Wavelet Transform (Haar): boosts high-frequency subbands (LH, HL, HH x 1.25)
   ├── LAB Color Space CLAHE: equalizes luminance channel without color distortion
   └── Unsharp Masking: restores crisp edges via Gaussian subtraction
        │
        ▼
[Stage 3: YOLO Weapon Inference]
   └── Runs weapon model (best.pt) at high-resolution (imgsz=1280)
        │
   ├─ No Detections ──────────────────────────────────────────────► [Skip Frame]
   └─ Weapon Candidate Detected (conf, bbox, ROI crop)
        │
        ▼
[Stage 4: Confidence Check & DIP Shape Verification]
   ├─ AI Confidence < 60%  ───────────────────────────────────────► [Skip Frame] (Noise)
   │
   ├─ AI Confidence > 90% (High Confidence Direct Pass) ─────────┐
   │                                                              │
   └─ AI Confidence 60% – 90% (Moderate Confidence Verification)  │
           │                                                      │
           ▼                                                      │
     [ROI Extraction & Classical DIP Shape Analysis]              │
     • Aspect Ratio / Elongation (Weapons: 1.2 – 4.0)             │
     • Canny Edge Density (Metallic edges vs soft textures)       │
     • Contour Solidity (Area / Convex Hull for grips/triggers)   │
           │                                                      │
           ▼                                                      │
     [Weighted Composite Scoring]                                 │
     Score = (0.70 × AI_Confidence) + (0.30 × Shape_Score)        │
           ├─ Score < 70% ────────────────────────────────────────┴► [Skip Frame]
           └─ Score ≥ 70%
                   │
                   ├──────────────────────────────────────────────┘
                   ▼
[Stage 5: Temporal Frame Accumulation]
   ├── Decays counter by -1 if no candidate is detected
   ├── Increments counter by +1 on consecutive verified detections
   ├─ Count < 4 frames ───────────────────────────────────────────► [Hold & Accumulate Status]
   └─ Count ≥ 4 consecutive frames (Spurious flicker rejected)
           │
           ▼
[Stage 6: Dashboard Check & Confirmed Threat Dispatch]
   ├─ Monitoring Armed = Inactive ────────────────────────────────► [Skip Frame]
   └─ Monitoring Armed = Active ──► [CONFIRMED THREAT]
           │
           ├── 🔊 Audio Alert: macOS system buzzer chime
           ├── 📡 MQTT Publish: JSON event to security/threats
           ├── 🚨 Visual Banner: Red flashing warning banner & HUD
           └── 📸 Incident Snapshot: Timestamped image saved to results/
```

---

## 🔍 Comprehensive Function & Module Reference

### 1. DIP Enhancement ([`src/dip/enhancement.py`](file:///Users/matsakonz/Downloads/Digital_Final/src/dip/enhancement.py))

Provides classical image processing transforms that accentuate fine metallic borders, firearm contours, and blade highlights.

#### `apply_wavelet_subband_boost(l_channel, boost_factor=1.25)`
* **Description**: Applies 2D Discrete Wavelet Transform (DWT) using the Haar wavelet to the Luminance ($L$) channel.
* **How It Works**:
  1. Decomposes the single-channel image into 4 frequency subbands: $LL$ (low-frequency approximation), $LH$ (horizontal details), $HL$ (vertical details), and $HH$ (diagonal details).
  2. Multiplies the detail subbands ($LH, HL, HH$) by `boost_factor` ($1.25\times$).
  3. Reconstructs the channel using the Inverse Discrete Wavelet Transform (IDWT).
  4. Clips pixel values to $[0, 255]$ and converts back to `uint8`.
* **Inputs**:
  * `l_channel` (`numpy.ndarray`): Luminance channel from LAB color space.
  * `boost_factor` (`float`, default `1.25`): Multiplier for high-frequency subbands.
* **Returns**: `numpy.ndarray` — Edge-boosted luminance channel.

#### `apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))`
* **Description**: Contrast Limited Adaptive Histogram Equalization.
* **How It Works**: Converts the image from BGR to LAB color space, extracts the $L$ (lightness) channel, applies CLAHE to prevent over-amplifying noise while bringing out shadowed weapons, and converts back to BGR.
* **Inputs**:
  * `image` (`numpy.ndarray`): BGR image.
  * `clip_limit` (`float`, default `2.0`): Threshold for contrast limiting.
  * `tile_grid_size` (`tuple`, default `(8, 8)`): Grid size for local histogram equalization.
* **Returns**: `numpy.ndarray` — Contrast-enhanced BGR image.

#### `apply_unsharp_mask(image, sigma=1.5, strength=1.4)`
* **Description**: High-frequency edge restoration filter.
* **How It Works**: Computes a blurred image via Gaussian blur with standard deviation `sigma`. Subtracts the blurred image from the original according to the formula:
  $$\text{Sharpened} = \text{image} \cdot \text{strength} - \text{blurred} \cdot (\text{strength} - 1.0)$$
* **Inputs**:
  * `image` (`numpy.ndarray`): BGR image.
  * `sigma` (`float`, default `1.5`): Gaussian kernel spread.
  * `strength` (`float`, default `1.4`): Edge boost weight.
* **Returns**: `numpy.ndarray` — Sharpened image.

#### `apply_dip_enhancement(image, use_wavelet=True)`
* **Description**: Main pipeline function integrating Wavelet DWT, CLAHE, and Unsharp Masking.
* **How It Works**:
  1. Splits input image into LAB channels.
  2. Optionally boosts high frequencies on the $L$ channel with `apply_wavelet_subband_boost`.
  3. Applies CLAHE on the enhanced $L$ channel.
  4. Merges LAB channels and converts back to BGR.
  5. Applies `apply_unsharp_mask` on the output.
* **Inputs**:
  * `image` (`numpy.ndarray`): Input BGR image or ROI.
  * `use_wavelet` (`bool`, default `True`): Whether to include Wavelet DWT subband boosting.
* **Returns**: `numpy.ndarray` — Fully enhanced image.

---

### 2. DIP Shape Analysis ([`src/dip/shape_analysis.py`](file:///Users/matsakonz/Downloads/Digital_Final/src/dip/shape_analysis.py))

Evaluates geometric properties on candidate weapon crops to distinguish actual weapons from everyday items.

#### `compute_shape_similarity(roi_crop)`
* **Description**: Computes a normalized shape score ($0.0$ to $1.0$) based on three classical DIP metrics:
  1. **Aspect Ratio / Elongation**: Handguns, rifles, and knives exhibit characteristic elongation ratios between $1.2$ and $4.0$.
     * Aspect in $[1.2, 4.0] \rightarrow 0.95$ score.
     * Aspect in $[1.0, 1.2) \rightarrow 0.70$ score.
     * Outside range $\rightarrow 0.55$ score.
  2. **Canny Edge Density**: Canny edge detection ($50, 150$) computes edge density:
     $$\text{Density} = \frac{\sum \text{edge\_pixels}}{w \times h}$$
     Weapons have sharp metallic boundaries, scoring $\text{Density} \times 4.0$ clipped between $0.40$ and $1.0$.
  3. **Contour Solidity**: Finds the largest external contour $C$ and its convex hull $H$:
     $$\text{Solidity} = \frac{\text{Area}(C)}{\text{Area}(H)}$$
     Weapons feature non-convex geometries (trigger guards, handles, barrels), typical solidity falling in $[0.30, 0.88] \rightarrow 0.90$ score. Convex blobs score $0.60$.
* **Composite Formula**:
  $$\text{Shape\_Score} = 0.40 \cdot \text{Aspect\_Score} + 0.35 \cdot \text{Density\_Score} + 0.25 \cdot \text{Solidity\_Score}$$
* **Inputs**: `roi_crop` (`numpy.ndarray`): Bounding box crop of the detected weapon.
* **Returns**: `float` — Normalized shape confidence ($0.00$ to $1.00$).

---

### 3. Weapon Detection & Targeted SAHI ([`src/detection/detector.py`](file:///Users/matsakonz/Downloads/Digital_Final/src/detection/detector.py))

Combines person detection, generous safety margins, multi-tile slicing (SAHI), and high-resolution weapon inference.

#### Key Functions & Class `WeaponDetector`
* **`generate_slices(img_w, img_h, tile_size=640, overlap_ratio=0.25)`**:
  * Calculates sliding window tile coordinates across the expanded human crop with 25% overlap to ensure weapons straddling slice borders are never cut in half.
* **`apply_nms(candidates, iou_threshold=0.45)`**:
  * Merges duplicate weapon candidate detections produced by overlapping tiles using OpenCV NMS (`cv2.dnn.NMSBoxes`).
* **`WeaponDetector.__init__(weapon_model_path, person_model_path)`**:
  * Loads YOLO weapon model (`best.pt`) and person detection model (`yolov8n.pt`).
* **`WeaponDetector.detect(frame, conf_threshold=0.50, imgsz=1280, use_zoom=True, use_sahi=True, ...)`**:
  * **How It Works**:
    1. Runs person detector (`yolov8n.pt`, `conf=0.35`) across the full frame.
    2. **Generous Safety Margins**: Expands person bounding boxes by **+35% horizontal margin** and **+25% vertical margin**, guaranteeing extended arms aiming guns, waist holsters, and dropped weapons near feet are captured.
    3. **Targeted SAHI Slicing**:
       * If the crop is large (tall or close person), slices into overlapping 640px tiles at **1:1 native camera resolution**, preserving microscopic pixel details of handguns and knives without downsampling.
       * If the crop is small/distant ($<640\text{px}$), upscales using **Bicubic Super-Sampling** (`cv2.INTER_CUBIC`).
    4. Applies DIP enhancement (`apply_dip_enhancement`) per tile.
    5. Runs weapon model on each tile.
    6. **Coordinate Translation & Sliced NMS**: Remaps tile detections to crop coordinates, then to full 1080p frame coordinates, and deduplicates overlap artifacts via NMS.
    7. **Fallback**: Runs inference on the full frame if no humans are detected or if zoom is disabled.
  * **Returns**:
    * `raw_candidates` (`list[dict]`): List of candidate dictionaries with keys `box` $(x_1, y_1, x_2, y_2)$, `cls`, `label`, `conf`, and `roi` image.
    * `person_boxes` (`list[tuple]`): List of $(px_1, py_1, px_2, py_2, p\_conf)$.

---

### 4. Threat Verification & Rejection Logic ([`src/detection/verifier.py`](file:///Users/matsakonz/Downloads/Digital_Final/src/detection/verifier.py))

Enforces the verification pipeline to eliminate false positives.

#### Class `ThreatVerifier`
* **`__init__(alert_manager=None)`**:
  * Configures thresholds (`CONF_MIN=0.60`, `CONF_HIGH=0.80`, `WEIGHT_AI=0.70`, `WEIGHT_SHAPE=0.30`, `WEIGHTED_SCORE_MIN=0.70`, `FRAME_ACCUMULATION_MIN=4`).
  * Initializes state tracking (`consecutive_frames=0`, `dashboard_active=True`).
* **`verify_candidates(candidates, frame)`**:
  * **How It Works**:
    1. **Confidence Filtering**: If candidate AI confidence $< 60\%$, discards as noise.
    2. **Branching**:
       * If AI confidence $\ge 80\%$, grants a direct pass (`shape_score = 1.0`, `weighted_score = ai_conf`).
       * If AI confidence is between $60\%$ and $80\%$, computes `compute_shape_similarity(roi)` and evaluates:
         $$\text{Weighted\_Score} = (0.70 \cdot \text{AI\_Conf}) + (0.30 \cdot \text{Shape\_Score})$$
         Requires $\text{Weighted\_Score} \ge 0.70$ to advance.
    3. **Temporal Frame Accumulation**:
       * If a candidate passes, increments `consecutive_frames`.
       * If no candidate passes in this frame, decrements `consecutive_frames` towards 0.
       * Requires `consecutive_frames >= 4` to confirm threat.
    4. **Dashboard Armed Check & Alerting**:
       * If confirmed and `dashboard_active` is `True`, triggers `alert_manager.on_detection_confirm`.
  * **Returns**:
    * `verified_threats` (`list[dict]`): Threat metadata including `status` (`"ACCUMULATING (N/4)"` or `"CONFIRMED"`), `weighted_score`, `ai_conf`, `shape_score`.
    * `is_confirmed` (`bool`): `True` when $\ge 4$ consecutive frames pass and armed status is active.

---

### 5. Alert Dispatcher & Logger ([`src/alerts/alert_manager.py`](file:///Users/matsakonz/Downloads/Digital_Final/src/alerts/alert_manager.py))

Dispatches audio alarms, MQTT telemetry, and evidence files.

#### Class `AlertManager`
* **`__init__()`**: Initializes alert cooldown timer (2.0s) and attempts MQTT connection.
* **`_init_mqtt()`**: Connects asynchronously to MQTT broker (`localhost:1883`, topic `security/threats`) via `paho-mqtt` if installed.
* **`trigger_buzzer()`**: Plays macOS audio chime (`afplay /System/Library/Sounds/Ping.aiff`) in a background process, rate-limited by cooldown. Falls back to terminal bell (`\a`).
* **`publish_mqtt(threat_data)`**: Publishes JSON-serialized threat telemetry payload to the configured MQTT broker and topic.
* **`save_incident_snapshot(frame, threat_label)`**: Saves a timestamped full-resolution JPEG (`ALERT_<label>_<YYYYMMDD_HHMMSS>.jpg`) into `results/`.
* **`on_detection_confirm(frame, threat_label, score)`**: Orchestrates alert sequence: sounds buzzer, publishes MQTT payload, and saves incident snapshot.
* **`close()`**: Stops the background MQTT loop and disconnects client cleanly.

---

### 6. Visualizer & HUD Rendering ([`src/ui/visualizer.py`](file:///Users/matsakonz/Downloads/Digital_Final/src/ui/visualizer.py))

Handles graphic rendering, bounding boxes, alarm banners, and PiP overlays.

#### Functions & Class `Visualizer`
* **`draw_label(img, text, pos, ...)`**: Helper that draws text with an auto-sized dark background pill for high contrast.
* **`render_overlay(frame, person_boxes, verified_threats, is_confirmed, hud_info=None)`**:
  * Draws person bounding boxes in orange.
  * Draws threat boxes:
    * **Accumulating**: Orange/Yellow box with count label `VERIFYING (N/4)`.
    * **Confirmed**: Flashing red/red-orange box with label `CONFIRMED: <label> (X%)`.
  * Renders a flashing top red alarm banner across the monitor when `is_confirmed` is active.
  * Renders HUD telemetry at the bottom.
* **`render_detect_image(frame, person_boxes, verified_threats, is_confirmed, hud_info=None, use_dip=True, use_wavelet=True)`**:
  * Transforms background image with full DIP enhancement.
  * Draws overlays using `render_overlay`.
  * **Picture-in-Picture (PiP) Inset**: Crops the verified threat weapon ROI, upscales it with bicubic interpolation, sharpens it, and renders an inset window with a bezel border in the top-right corner.
* **`draw_timestamp(img, timestamp_str=None, pos=(20, 38))`**: Draws CCTV-style `REC YYYY-MM-DD HH:MM:SS` watermark.
* **`draw_mode_badge(img, mode="ORIGINAL CAM", is_threat=False)`**: Renders top-right badge indicating current video mode (`ORIGINAL CAM` vs `DETECT IMAGE`).
* **`create_side_by_side(original, detected)`**: Horizontally concatenates raw camera view and annotated detection view.

---

### 7. Core CLI & Main Orchestrator ([`main.py`](file:///Users/matsakonz/Downloads/Digital_Final/main.py))

The main desktop runtime supporting single-camera, dual-camera grid, and batch test image verification.

#### `scan_available_cameras(max_to_check=4)`
* **Description**: Probes video capture devices index `0` through `max_to_check - 1` and returns a list of actively readable indices.

#### `open_cap(camera_idx, resolution=(1920, 1080))`
* **Description**: Initializes an OpenCV `VideoCapture`, configuring requested width and height.

#### `run_camera(...)`
* **Description**: Live camera detection loop.
* **How It Works**:
  1. Initializes `AlertManager`, `WeaponDetector`, `ThreatVerifier`, and `Visualizer`.
  2. Reads primary camera feed and processes detection/verification.
  3. Supports secondary camera feed when in dual-camera mode (`'g'` key), displaying both streams side-by-side (960x540 each).
  4. Listens for live hotkeys (`'k'`, `'0'`, `'1'`, `'g'`, `'d'`, `'w'`, `'z'`, `'c'`, `'s'`, `'q'`).

#### `run_image_comparison(images_dir, output_dir, ...)`
* **Description**: Batch processes all test images in `data/images/`, generates side-by-side comparison images, and saves them to `results/verified_<image_name>`.

---

### 8. Web Dashboard & Multi-Threaded Streaming ([`web/app.py`](file:///Users/matsakonz/Downloads/Digital_Final/web/app.py))

Flask-based web interface featuring decoupled camera ingestion, multi-camera grid viewing, real-time configuration tuning, and smart auto-switching.

#### Class `CameraStreamWorker`
* **`__init__(cam_id, source, name)`**: Initializes dedicated capture and detection threads for a camera.
* **`_capture_loop()`**: Independent thread reading frames at 30+ FPS directly from the hardware or RTSP source into a double buffer.
* **`_detection_loop()`**: Dedicated worker thread executing YOLO and DIP verification in the background without blocking the live video stream.
* **`get_frame()` & `get_detection_state()`**: Thread-safe accessors for latest frame and threat telemetry.
* **`stop()`**: Releases camera resource.

#### Class `CameraHub`
* **`load_cameras()` & `save_cameras(cameras)`**: Reads and writes camera registry to `config/cameras.json`.
* **`get_worker(cam_id)`**: Retrieves or spawns a `CameraStreamWorker` for the given camera ID.
* **`release_cam(cam_id)`**: Shuts down worker and releases video capture.

#### Helper & Generator Functions
* **`create_offline_frame(camera_name)`**: Generates a placeholder JPEG when a camera is offline or disconnected.
* **`generate_video_stream(cam_id)`**: Multipart MJPEG generator.
  * **Smart Auto-Switch**: Streams clean 30+ FPS `ORIGINAL CAM` when safe; automatically switches to `DETECT IMAGE` with bounding boxes and zoomed PiP inset whenever a weapon is verified.
* **`start_server(host="0.0.0.0", port=5001)`**: Boots Flask server, automatically trying fallback ports (`5001` - `5010`) if port is in use.

#### Web & REST API Routes
| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main live camera surveillance dashboard ([`camera.html`](file:///Users/matsakonz/Downloads/Digital_Final/web/templates/camera.html)) |
| `/settings` | `GET` | System settings & camera management UI ([`settings.html`](file:///Users/matsakonz/Downloads/Digital_Final/web/templates/settings.html)) |
| `/video_feed/<cam_id>` | `GET` | High-performance MJPEG live stream for specified camera |
| `/api/cameras` | `GET` | Returns list of configured cameras from `cameras.json` |
| `/api/cameras` | `POST` | Registers a new local USB or IP camera |
| `/api/cameras/<cam_id>` | `DELETE` | Removes camera and releases stream worker |
| `/api/status` | `GET` | Polling endpoint for FPS, threat counts, person count, and recent event logs |
| `/api/view_mode/<mode>` | `POST` | Manually sets stream view mode (`'auto'`, `'detect'`, `'original'`) |
| `/api/settings` | `GET` | Fetches current verification thresholds and weights |
| `/api/settings` | `POST` | Updates thresholds (`CONF_MIN`, `CONF_HIGH`, weights, etc.) in real time |
| `/api/toggle/<feature>` | `POST` | Toggles pipeline feature (`'zoom'`, `'dip'`, `'wavelet'`) |
| `/api/snapshot` | `POST` | Captures and saves manual evidence snapshot to `results/` |

---

### 9. Legacy & Experimentation Scripts

* **[`detecting-images.py`](file:///Users/matsakonz/Downloads/Digital_Final/detecting-images.py)**: Backward-compatible wrapper exposing:
  * `open_camera(...)`: Wraps `main.run_camera`.
  * `compare_all_images(...)`: Wraps `main.run_image_comparison`.
  * `detect_from_camera`: Function alias for `open_camera`.
* **[`preprocessing-images.py`](file:///Users/matsakonz/Downloads/Digital_Final/preprocessing-images.py)**: Wavelet experimental testbed:
  * `apply_symlet_transform(image)`: Symlet 2 DWT returning horizontal detail subband ($LH$).
  * `apply_daubechies_transform(image)`: Daubechies 2 DWT returning horizontal detail subband ($LH$).
  * `apply_haar_transform(image)`: Haar DWT returning horizontal detail subband ($LH$).
  * `enhance_contrast(image)`: Min-max normalization contrast stretcher.
  * `process_images_in_folder(input_folder, output_folder)`: Batch wavelet testbed runner.

---

## ⚙️ Configuration Reference ([`config/settings.py`](file:///Users/matsakonz/Downloads/Digital_Final/config/settings.py))

| Parameter | Default Value | Description |
|---|---|---|
| `CONF_MIN` | `0.60` | Minimum AI confidence threshold. Detections $< 60\%$ are discarded. |
| `CONF_HIGH` | `0.80` | High-confidence threshold. Detections $\ge 80\%$ bypass shape analysis. |
| `WEIGHT_AI` | `0.70` | Weight assigned to AI confidence in moderate bracket verification ($70\%$). |
| `WEIGHT_SHAPE` | `0.30` | Weight assigned to DIP shape similarity score ($30\%$). |
| `WEIGHTED_SCORE_MIN` | `0.70` | Minimum composite score required to pass verification ($\ge 70\%$). |
| `FRAME_ACCUMULATION_MIN` | `4` | Number of consecutive positive frames required to confirm a threat. |
| `DASHBOARD_ACTIVE` | `False` | Default monitoring armed status in CLI mode. |
| `IMGSZ` | `1280` | YOLO inference input image resolution. |
| `USE_HUMAN_ZOOM` | `True` | Enables person detection crop and ROI zooming. |
| `USE_SAHI` | `True` | Enables Targeted SAHI (Multi-tile Slicing) on human crops. |
| `HUMAN_MARGIN_X` | `0.35` | +35% horizontal safety margin around person (captures extended arms & aiming). |
| `HUMAN_MARGIN_Y` | `0.25` | +25% vertical safety margin around person (captures raised hands & dropped guns). |
| `SAHI_TILE_SIZE` | `640` | Tile resolution for sliced inference (native 1:1 camera pixel scale). |
| `SAHI_OVERLAP` | `0.25` | 25% overlap between adjacent tiles to prevent seam clipping. |
| `SAHI_NMS_IOU` | `0.45` | IoU threshold for OpenCV NMS to deduplicate overlapping slice detections. |
| `USE_DIP` | `True` | Enables CLAHE and Unsharp Masking pipeline. |
| `USE_WAVELET` | `True` | Enables 2D Wavelet DWT luminance subband boosting. |
| `CAMERA_INDEX` | `0` | Default hardware camera device index. |
| `CAMERA_RESOLUTION` | `(1920, 1080)` | Hardware capture resolution request. |
| `WINDOW_SIZE` | `(1600, 900)` | Default GUI window display dimensions. |
| `BUZZER_ENABLED` | `True` | Enables audio alert chime on confirmed threat. |
| `MQTT_ENABLED` | `True` | Enables MQTT telemetry publishing. |
| `MQTT_BROKER` | `"localhost"` | MQTT broker host. |
| `MQTT_PORT` | `1883` | MQTT broker port. |
| `MQTT_TOPIC` | `"security/threats"` | MQTT topic destination. |

---

## 💻 Installation & Setup

### 1. Prerequisites
* Python 3.9 – 3.12
* macOS, Linux, or Windows (Audio buzzer configured for macOS `afplay` with terminal bell fallback)

### 2. Virtual Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install ultralytics opencv-python pywavelets numpy flask paho-mqtt
```

---

## 🚀 How to Run

### Option A: Web Surveillance Dashboard (Recommended)
Launch the multi-threaded web application with live camera grid, smart auto-switch, and settings editor:
```bash
python web/app.py
```
Open your browser at: **`http://localhost:5001`**

### Option B: Desktop CLI Live Camera
Run single-camera detection with full HUD overlays:
```bash
python main.py --mode camera --camera 0
```

### Option C: Desktop Dual-Camera Grid Mode
Simultaneously stream and detect from two connected cameras (e.g., FaceTime camera + external USB/iPhone camera):
```bash
python main.py --mode dual
```

### Option D: Batch Test Dataset Verification
Evaluate the verification pipeline across all benchmark test images in `data/images/`:
```bash
python main.py --mode compare
```
Outputs side-by-side comparison images into the `results/` folder.

---

## 🎮 Live Keyboard Controls (Desktop Mode)

When running `main.py` in desktop camera mode, use the following interactive hotkeys:

| Key | Action |
|---|---|
| **`k`** or **`TAB`** | **Switch Camera**: Cycles to the next connected camera on the fly |
| **`0`** / **`1`** / **`2`** | **Direct Switch**: Switches directly to camera index 0, 1, or 2 |
| **`g`** | **Dual-Camera Grid**: Toggles split-screen streaming for two cameras |
| **`t`** | **Toggle Targeted SAHI**: Toggles multi-tile slicing on human crop ON / OFF |
| **`d`** | **Toggle DIP**: Toggles CLAHE + Unsharp Masking ON / OFF |
| **`w`** | **Toggle Wavelet**: Toggles 2D Wavelet DWT edge boosting ON / OFF |
| **`z`** | **Toggle Human-Zoom**: Toggles person ROI cropping & super-sampling ON / OFF |
| **`c`** | **Toggle Side-by-Side**: Toggles side-by-side original vs detected view |
| **`s`** | **Manual Snapshot**: Saves high-resolution snapshot to `./results/` |
| **`q`** or **`ESC`** | **Quit**: Exits the application and releases video captures |
