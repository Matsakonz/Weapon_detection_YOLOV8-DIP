#!/usr/bin/env python3
"""
Single / Batch Image Weapon Detector & Frame Boundary Exporter
---------------------------------------------------------------
Loads an uploaded picture, detects weapons using the exact same multi-stage
pipeline as `web/app.py` (Human Zoom, SAHI, DIP with Wavelet DWT), draws
precise frame boundaries (bounding boxes, confidence tags, PIP crop), and
exports the resulting annotated picture.

Features:
  1. Upload/Select:
     - GUI File Dialog: Pops up automatically if run with no args (or --gui).
     - CLI Argument: `--image path/to/photo.jpg` (or `--input path/to/folder`).
     - Web UI: Run `--web` to launch an interactive drag-and-drop web uploader.
  2. Frame Boundaries:
     - Detected Weapons (Threat boxes with high-contrast label pills & confidence %).
     - Detected Person (Human Zoom ROI boundary).
     - Picture-in-Picture (PIP) zoomed weapon crop overlay.
     - Optional Side-by-Side comparison export (Original vs. Detected).
  3. Export:
     - Saves annotated image to `results/exported_detections/` (or `--output`).
     - Saves companion detection metadata JSON (boxes, confidences, classes).
"""

import os
import sys
import time
import argparse
import json
import glob
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    resolve_model_path,
    MODEL_PERSON,
    IMGSZ,
    CONF_MIN,
    TARGET_CLASSES,
    USE_HUMAN_ZOOM,
    USE_SAHI,
    USE_DIP,
    USE_WAVELET,
    load_runtime_settings
)
from src.detection.detector import WeaponDetector
from src.dip.enhancement import apply_dip_enhancement


def select_file_via_dialog():
    """Opens a native system file picker dialog to select an image file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select Picture for Weapon Detection",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"[Warning] GUI File Dialog error: {e}")
        return ""


def draw_label_pill(img, text, x1, y1, font_scale=0.6,
                    text_color=(255, 255, 255), bg_color=(220, 20, 60), thickness=2):
    """Draws a modern rounded label tag above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position box above bbox, or inside if too close to image top
    box_y2 = y1 - 4 if y1 - th - 10 > 0 else y1 + th + 10
    box_y1 = box_y2 - th - 8
    box_x1 = max(0, x1)
    box_x2 = min(img.shape[1], box_x1 + tw + 12)

    # Background rectangle
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    # Text
    cv2.putText(img, text, (box_x1 + 6, box_y2 - 5),
                font, font_scale, text_color, thickness, cv2.LINE_AA)


def annotate_frame_boundaries(
    frame, raw_candidates, person_boxes,
    draw_person_boxes=True, draw_pip=True,
    use_dip=True, use_wavelet=True
):
    """
    Draws frame boundaries, bounding boxes, labels, and PIP zoom crop on frame.
    Returns the annotated image canvas.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Person Frame Boundaries (Cyan/Blue boundary)
    if draw_person_boxes and person_boxes:
        for (px1, py1, px2, py2, p_conf) in person_boxes:
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 180, 0), 2)
            # Corner accents
            c_len = min(25, int((px2 - px1) * 0.15))
            cv2.line(canvas, (px1, py1), (px1 + c_len, py1), (255, 220, 50), 3)
            cv2.line(canvas, (px1, py1), (px1, py1 + c_len), (255, 220, 50), 3)
            cv2.line(canvas, (px2, py2), (px2 - c_len, py2), (255, 220, 50), 3)
            cv2.line(canvas, (px2, py2), (px2, py2 - c_len), (255, 220, 50), 3)

            p_tag = f"Human ROI ({int(p_conf * 100)}%)"
            draw_label_pill(canvas, p_tag, px1, py1, font_scale=0.55,
                            text_color=(20, 20, 20), bg_color=(255, 180, 0), thickness=1)

    # 2. Draw Weapon Frame Boundaries (Red/Crimson bounding box)
    threat_rois = []
    for cand in raw_candidates:
        x1, y1, x2, y2 = cand['box']
        label = cand.get('label', 'Weapon')
        conf = cand.get('conf', 0.0)
        roi = cand.get('roi')

        # Weapon bounding box
        box_color = (0, 30, 235)  # Bright Red in BGR
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

        # High-visibility corner brackets
        corner_len = min(20, max(6, int((x2 - x1) * 0.2)))
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3)
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 3)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 3)
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 3)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 3)
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 3)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 3)

        # Label tag pill
        tag_text = f"THREAT: {label.upper()} {int(conf * 100)}%"
        draw_label_pill(canvas, tag_text, x1, y1, font_scale=0.6,
                        text_color=(255, 255, 255), bg_color=box_color, thickness=2)

        threat_rois.append((roi, label, conf, (x1, y1, x2, y2)))

    # 3. Picture-in-Picture (PIP) Zoomed Weapon Crop
    if draw_pip and threat_rois:
        best_roi, b_label, b_conf, b_box = max(threat_rois, key=lambda t: t[2])
        if best_roi is None or best_roi.size == 0:
            bx1, by1, bx2, by2 = b_box
            best_roi = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]

        if best_roi is not None and best_roi.size > 0:
            # Scaled PIP box proportional to frame dimensions
            pip_w = min(320, int(w * 0.28))
            pip_h = min(220, int(h * 0.28))
            pip_x = w - pip_w - 25
            pip_y = 25

            zoomed = cv2.resize(best_roi, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
            if use_dip:
                zoomed = apply_dip_enhancement(zoomed, use_wavelet=False)

            # Outer border & bezel
            cv2.rectangle(canvas, (pip_x - 4, pip_y - 28), (pip_x + pip_w + 4, pip_y + pip_h + 4), (18, 18, 18), -1)
            cv2.rectangle(canvas, (pip_x - 4, pip_y - 28), (pip_x + pip_w + 4, pip_y + pip_h + 4), (0, 0, 230), 2)

            # Paste crop
            canvas[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = zoomed

            # PIP Header Title
            title = f"ZOOMED THREAT ROI: {b_label.upper()} ({int(b_conf * 100)}%)"
            cv2.putText(canvas, title, (pip_x + 6, pip_y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Status Bar Header
    num_threats = len(raw_candidates)
    num_persons = len(person_boxes)
    status_bar_h = 38
    status_bar = canvas.copy()
    cv2.rectangle(status_bar, (0, 0), (w, status_bar_h), (20, 20, 20), -1)
    cv2.addWeighted(status_bar, 0.85, canvas, 0.15, 0, canvas)

    status_str = f"WEAPON DETECTION | Threats Found: {num_threats} | Persons: {num_persons}"
    cv2.putText(canvas, status_str, (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def create_side_by_side(original, annotated):
    """Generates a side-by-side comparison image."""
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    target_h = min(h1, h2)
    
    vis_orig = original if h1 == target_h else cv2.resize(original, (int(w1 * (target_h / h1)), target_h))
    vis_anno = annotated if h2 == target_h else cv2.resize(annotated, (int(w2 * (target_h / h2)), target_h))

    # Add label on original
    orig_copy = vis_orig.copy()
    cv2.rectangle(orig_copy, (0, 0), (orig_copy.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(orig_copy, "INPUT ORIGINAL PICTURE", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.hstack([orig_copy, vis_anno])
    # Vertical white separator line
    sep_x = orig_copy.shape[1]
    cv2.line(combined, (sep_x, 0), (sep_x, target_h), (255, 255, 255), 3)
    return combined


def process_and_export_image(image_path, detector, args, output_dir):
    """
    Runs the complete detection pipeline on image_path, draws frame boundaries,
    and exports the result to output_dir.
    """
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        return None

    print(f"\n[Processing] Loading: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not read image at: {image_path}")
        return None

    h, w = frame.shape[:2]
    print(f"  • Resolution: {w}x{h} px")

    t0 = time.time()
    # Detection pipeline matching app.py
    raw_candidates, person_boxes = detector.detect(
        frame,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        use_zoom=not args.no_zoom,
        use_sahi=not args.no_sahi,
        target_classes=args.target_classes,
        use_dip=not args.no_dip,
        use_wavelet=not args.no_wavelet
    )
    inference_time = round(time.time() - t0, 3)

    print(f"  • Completed in: {inference_time}s")
    print(f"  • Detected Persons: {len(person_boxes)}")
    print(f"  • Detected Threats: {len(raw_candidates)}")
    for i, c in enumerate(raw_candidates):
        print(f"    - Threat #{i+1}: {c['label']} | Conf: {c['conf']:.3f} | Box: {c['box']}")

    # Render frame boundary overlays
    annotated = annotate_frame_boundaries(
        frame, raw_candidates, person_boxes,
        draw_person_boxes=not args.no_person_box,
        draw_pip=not args.no_pip,
        use_dip=not args.no_dip,
        use_wavelet=not args.no_wavelet
    )

    if args.side_by_side:
        annotated = create_side_by_side(frame, annotated)

    # Export paths
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_img_name = f"{base_name}_detected.jpg"
    out_img_path = os.path.join(output_dir, out_img_name)
    
    cv2.imwrite(out_img_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✓ Exported picture: {out_img_path}")

    # Optional: Export isolated weapon crop ROI
    roi_paths = []
    if args.export_rois and raw_candidates:
        for idx, cand in enumerate(raw_candidates):
            roi = cand.get('roi')
            if roi is not None and roi.size > 0:
                roi_name = f"{base_name}_threat_{idx+1}_{cand['label']}.jpg"
                roi_path = os.path.join(output_dir, roi_name)
                cv2.imwrite(roi_path, roi)
                roi_paths.append(roi_path)
                print(f"✓ Exported Threat Crop: {roi_path}")

    # Export metadata JSON
    meta_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    meta_info = {
        "source_image": os.path.abspath(image_path),
        "exported_image": os.path.abspath(out_img_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": {"width": w, "height": h},
        "inference_time_sec": inference_time,
        "parameters": {
            "model": args.model,
            "conf_threshold": args.conf,
            "imgsz": args.imgsz,
            "use_zoom": not args.no_zoom,
            "use_sahi": not args.no_sahi,
            "use_dip": not args.no_dip,
            "use_wavelet": not args.no_wavelet,
            "target_classes": args.target_classes
        },
        "threat_count": len(raw_candidates),
        "person_count": len(person_boxes),
        "threats": [
            {
                "id": i + 1,
                "label": c["label"],
                "confidence": round(float(c["conf"]), 4),
                "box_xyxy": list(c["box"])
            }
            for i, c in enumerate(raw_candidates)
        ],
        "persons": [
            {
                "id": i + 1,
                "box_xyxy": list(p[:4]),
                "confidence": round(float(p[4]), 4)
            }
            for i, p in enumerate(person_boxes)
        ]
    }
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    print(f"✓ Exported Metadata: {meta_path}")

    return {
        "annotated_path": out_img_path,
        "metadata_path": meta_path,
        "roi_paths": roi_paths,
        "threat_count": len(raw_candidates),
        "person_count": len(person_boxes),
        "annotated_image": annotated
    }


def run_web_uploader(detector, args):
    """
    Launches a modern, responsive web application for interactive image uploading,
    real-time detection boundary rendering, and one-click export.
    """
    from flask import Flask, request, jsonify, render_template_string, send_file
    import base64

    web_app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Weapon Detection & Frame Boundary Exporter</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
      <style>
        :root {
          --bg-primary: #0f172a;
          --bg-secondary: #1e293b;
          --card-bg: rgba(30, 41, 59, 0.7);
          --accent: #ef4444;
          --accent-hover: #dc2626;
          --text: #f8fafc;
          --text-muted: #94a3b8;
          --border: #334155;
          --border-glow: rgba(239, 68, 68, 0.3);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: 'Inter', sans-serif;
          background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
          color: var(--text);
          min-height: 100vh;
          padding: 30px 20px;
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        .container {
          max-width: 1200px;
          width: 100%;
        }
        header {
          text-align: center;
          margin-bottom: 25px;
        }
        h1 {
          font-size: 2.2rem;
          font-weight: 700;
          background: linear-gradient(135deg, #f87171, #ef4444, #f59e0b);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 8px;
        }
        p.subtitle {
          color: var(--text-muted);
          font-size: 1.05rem;
        }
        .pipeline-badges {
          display: flex;
          justify-content: center;
          gap: 10px;
          margin-top: 12px;
          flex-wrap: wrap;
        }
        .badge {
          background: rgba(239, 68, 68, 0.15);
          color: #fca5a5;
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 20px;
          padding: 4px 12px;
          font-size: 0.8rem;
          font-weight: 600;
        }
        .main-card {
          background: var(--card-bg);
          backdrop-filter: blur(12px);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 24px;
          box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        .drop-zone {
          border: 2px dashed #475569;
          border-radius: 12px;
          padding: 40px 20px;
          text-align: center;
          cursor: pointer;
          transition: all 0.25s ease;
          background: rgba(15, 23, 42, 0.4);
        }
        .drop-zone:hover, .drop-zone.dragover {
          border-color: var(--accent);
          background: rgba(239, 68, 68, 0.05);
          box-shadow: 0 0 20px var(--border-glow);
        }
        .drop-icon {
          font-size: 48px;
          margin-bottom: 12px;
        }
        .btn-upload {
          background: var(--accent);
          color: white;
          border: none;
          padding: 10px 24px;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          margin-top: 15px;
          font-size: 0.95rem;
          transition: all 0.2s;
        }
        .btn-upload:hover {
          background: var(--accent-hover);
          transform: translateY(-1px);
        }
        #result-section {
          margin-top: 25px;
          display: none;
        }
        .stats-bar {
          display: flex;
          justify-content: space-between;
          background: #111827;
          border-radius: 10px;
          padding: 12px 18px;
          margin-bottom: 16px;
          flex-wrap: wrap;
          gap: 12px;
          font-size: 0.9rem;
        }
        .stat-item span {
          font-weight: 700;
          color: #38bdf8;
        }
        .stat-item.threat span {
          color: #ef4444;
        }
        .preview-box {
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid var(--border);
          background: #000;
          display: flex;
          justify-content: center;
          align-items: center;
        }
        .preview-box img {
          max-width: 100%;
          height: auto;
          display: block;
        }
        .actions-bar {
          display: flex;
          justify-content: flex-end;
          gap: 12px;
          margin-top: 18px;
        }
        .btn-download {
          background: #10b981;
          color: white;
          text-decoration: none;
          padding: 10px 20px;
          border-radius: 8px;
          font-weight: 600;
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }
        .btn-download:hover { background: #059669; }
        .spinner {
          display: none;
          margin: 20px auto;
          width: 40px;
          height: 40px;
          border: 4px solid rgba(255,255,255,0.1);
          border-top-color: var(--accent);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      </style>
    </head>
    <body>
      <div class="container">
        <header>
          <h1>Weapon Detection & Frame Boundary Exporter</h1>
          <p class="subtitle">Upload a picture to run Human Zoom + SAHI + Wavelet DIP and export frame boundaries</p>
          <div class="pipeline-badges">
            <span class="badge">Model: {{ model }}</span>
            <span class="badge">Human Zoom: ON</span>
            <span class="badge">SAHI Slicing: ON</span>
            <span class="badge">Wavelet DIP: ON</span>
            <span class="badge">Conf Threshold: {{ conf }}</span>
          </div>
        </header>

        <div class="main-card">
          <div class="drop-zone" id="dropZone">
            <div class="drop-icon">📷</div>
            <h3>Drag & Drop your picture here</h3>
            <p style="color: var(--text-muted); margin-top: 6px; font-size: 0.9rem;">Supports JPG, JPEG, PNG, WEBP</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            <button class="btn-upload" onclick="document.getElementById('fileInput').click()">Browse Picture</button>
          </div>

          <div class="spinner" id="spinner"></div>

          <div id="result-section">
            <div class="stats-bar">
              <div class="stat-item threat">Threats Detected: <span id="threatCount">0</span></div>
              <div class="stat-item">Persons Detected: <span id="personCount">0</span></div>
              <div class="stat-item">Inference Time: <span id="timeSec">0.0s</span></div>
              <div class="stat-item">Saved To: <span id="savedPath" style="color: #a7f3d0; font-family: monospace;">-</span></div>
            </div>

            <div class="preview-box">
              <img id="resultImage" src="" alt="Detection with Frame Boundaries">
            </div>

            <div class="actions-bar">
              <a id="downloadLink" class="btn-download" download="detected_picture.jpg" href="#">
                📥 Download Exported Picture
              </a>
            </div>
          </div>
        </div>
      </div>

      <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const spinner = document.getElementById('spinner');
        const resultSection = document.getElementById('result-section');
        const resultImage = document.getElementById('resultImage');
        const downloadLink = document.getElementById('downloadLink');

        ['dragenter', 'dragover'].forEach(eventName => {
          dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
          }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
          dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
          }, false);
        });

        dropZone.addEventListener('drop', (e) => {
          const files = e.dataTransfer.files;
          if (files.length > 0) uploadFile(files[0]);
        });

        fileInput.addEventListener('change', () => {
          if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
        });

        function uploadFile(file) {
          spinner.style.display = 'block';
          resultSection.style.display = 'none';

          const formData = new FormData();
          formData.append('image', file);

          fetch('/api/detect_and_export', {
            method: 'POST',
            body: formData
          })
          .then(res => res.json())
          .then(data => {
            spinner.style.display = 'none';
            if (data.success) {
              resultSection.style.display = 'block';
              resultImage.src = 'data:image/jpeg;base64,' + data.image_base64;
              downloadLink.href = 'data:image/jpeg;base64,' + data.image_base64;
              downloadLink.download = data.filename;
              document.getElementById('threatCount').innerText = data.threat_count;
              document.getElementById('personCount').innerText = data.person_count;
              document.getElementById('timeSec').innerText = data.inference_time_sec + 's';
              document.getElementById('savedPath').innerText = data.saved_file;
            } else {
              alert('Detection error: ' + (data.error || 'Unknown error'));
            }
          })
          .catch(err => {
            spinner.style.display = 'none';
            alert('Upload failed: ' + err);
          });
        }
      </script>
    </body>
    </html>
    """

    @web_app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, model=args.model, conf=args.conf)

    @web_app.route('/api/detect_and_export', methods=['POST'])
    def detect_api():
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        # Save to temp path
        temp_dir = os.path.join(args.output_dir, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        # Process and export
        result = process_and_export_image(temp_path, detector, args, args.output_dir)
        if result is None:
            return jsonify({'success': False, 'error': 'Failed to process image'}), 500

        # Encode JPEG to base64 for direct browser rendering
        ret, buf = cv2.imencode('.jpg', result['annotated_image'], [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64_str = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'success': True,
            'threat_count': result['threat_count'],
            'person_count': result['person_count'],
            'saved_file': result['annotated_path'],
            'filename': os.path.basename(result['annotated_path']),
            'image_base64': b64_str
        })

    port = 5055
    print("\n" + "=" * 70)
    print(f" 🌐 Web Uploader Active! Open in browser: http://localhost:{port}")
    print("=" * 70 + "\n")
    web_app.run(host='0.0.0.0', port=port, debug=False)


def parse_arguments():
    runtime_cfg = load_runtime_settings() if callable(load_runtime_settings) else {}
    default_model = runtime_cfg.get("model_weapon", "best.pt")
    default_conf = runtime_cfg.get("conf_min", CONF_MIN)
    default_targets = runtime_cfg.get("target_classes", TARGET_CLASSES)

    parser = argparse.ArgumentParser(
        description="Upload a picture, detect weapons with SAHI+DIP+Zoom, set frame boundaries, and export."
    )
    parser.add_argument(
        "--image", "-i", type=str, default="",
        help="Path to input picture file (if empty and not --web, opens file dialog)"
    )
    parser.add_argument(
        "--input", type=str, default="",
        help="Path to folder of images for batch processing"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=os.path.join(PROJECT_ROOT, "results", "exported_detections"),
        help="Directory to save exported annotated pictures (default: results/exported_detections)"
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Start interactive web uploader interface at http://localhost:5055"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Force open file picker dialog to choose image"
    )
    parser.add_argument(
        "--side-by-side", action="store_true",
        help="Export side-by-side comparison (Original | Annotated with Frame Boundaries)"
    )
    parser.add_argument(
        "--export-rois", action="store_true", default=True,
        help="Export cropped weapon ROIs as separate image files"
    )
    parser.add_argument(
        "--no-pip", action="store_true",
        help="Disable Picture-in-Picture zoomed weapon ROI corner overlay"
    )
    parser.add_argument(
        "--no-person-box", action="store_true",
        help="Do not draw the detected human boundary box"
    )
    parser.add_argument(
        "--model", type=str, default=default_model,
        help=f"Model filename in models/ (default: {default_model})"
    )
    parser.add_argument(
        "--conf", type=float, default=default_conf,
        help=f"Confidence threshold (default: {default_conf})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMGSZ,
        help=f"Inference resolution (default: {IMGSZ})"
    )
    parser.add_argument(
        "--no-zoom", action="store_true", default=not runtime_cfg.get("use_zoom", USE_HUMAN_ZOOM),
        help="Disable Human Zoom"
    )
    parser.add_argument(
        "--no-sahi", action="store_true", default=not runtime_cfg.get("use_sahi", USE_SAHI),
        help="Disable SAHI sliced inference"
    )
    parser.add_argument(
        "--no-dip", action="store_true", default=not runtime_cfg.get("use_dip", USE_DIP),
        help="Disable DIP enhancement"
    )
    parser.add_argument(
        "--no-wavelet", action="store_true", default=not runtime_cfg.get("use_wavelet", USE_WAVELET),
        help="Disable Wavelet DWT boost"
    )
    parser.add_argument(
        "--target-classes", nargs="+", default=default_targets,
        help=f"Target threat classes (default: {default_targets})"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show exported image in OpenCV window upon completion"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Model resolution
    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        sys.exit(1)

    print("=" * 70)
    print(" 🎯 WEAPON DETECTION & FRAME BOUNDARY EXPORTER")
    print(f" Model: {args.model} | Person Model: {MODEL_PERSON}")
    print(f" Conf: {args.conf:.2f} | Zoom: {not args.no_zoom} | SAHI: {not args.no_sahi} | DIP: {not args.no_dip}")
    print("=" * 70)

    # Initialize Detector
    detector = WeaponDetector(model_path, MODEL_PERSON)

    # Web Mode
    if args.web:
        run_web_uploader(detector, args)
        return

    # Image Target Resolution
    target_images = []

    if args.input and os.path.isdir(args.input):
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
        for ext in valid_exts:
            target_images.extend(glob.glob(os.path.join(args.input, ext)))
        target_images = sorted(list(set(target_images)))
        print(f"[Batch Mode] Found {len(target_images)} images in folder: {args.input}")

    elif args.image:
        target_images = [args.image]

    else:
        # No image specified -> Open File Selection Dialog
        print("[Upload] Opening file selection dialog to choose picture...")
        selected = select_file_via_dialog()
        if not selected:
            print("[Upload] No picture selected. (You can also specify --image <path> or --web)")
            sys.exit(0)
        target_images = [selected]

    # Process all selected images
    exported_results = []
    for img_p in target_images:
        res = process_and_export_image(img_p, detector, args, args.output_dir)
        if res:
            exported_results.append(res)

    print("\n" + "=" * 70)
    print(f" 🎉 Completed! Exported {len(exported_results)} image(s) to: {args.output_dir}")
    print("=" * 70 + "\n")

    if args.show and exported_results:
        first_img = exported_results[0]["annotated_image"]
        # Resize if image exceeds display size
        dh, dw = first_img.shape[:2]
        max_dim = 1200
        if dw > max_dim or dh > max_dim:
            scale = min(max_dim / dw, max_dim / dh)
            first_img = cv2.resize(first_img, (int(dw * scale), int(dh * scale)))
        cv2.imshow("Exported Picture with Frame Boundaries", first_img)
        print("Press any key to close the preview window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
