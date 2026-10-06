import cv2
import numpy as np
from ultralytics import YOLO
from src.dip.enhancement import apply_dip_enhancement
from config.settings import IMGSZ, USE_HUMAN_ZOOM, USE_DIP, USE_WAVELET

class WeaponDetector:
    """
    Two-stage weapon detector:
    1. Human-Zoom: Detects humans first, crops/zooms with 20% margin around hands/arms.
    2. Bicubic Super-Sampling: Upscales small crops to >= 640px.
    3. DIP Enhancement: CLAHE + Unsharp Masking + Wavelet edge boosting.
    4. YOLO Weapon Inference: High-resolution evaluation (imgsz=1280).
    5. Coordinate Translation: Remaps detections back to full frame.
    6. Fallback: Full-frame detection if no humans are present.
    """
    def __init__(self, weapon_model_path, person_model_path):
        print(f"[Detector] Loading Weapon Model: {weapon_model_path}")
        self.weapon_model = YOLO(weapon_model_path)
        print(f"[Detector] Loading Person Model: {person_model_path}")
        self.person_model = YOLO(person_model_path)

    def detect(self, frame, conf_threshold=0.50, imgsz=IMGSZ,
               use_zoom=USE_HUMAN_ZOOM, use_dip=USE_DIP, use_wavelet=USE_WAVELET):
        """
        Executes detection on frame.
        
        Returns raw candidate detections:
          raw_candidates: list of dicts:
            {
              'box': (x1, y1, x2, y2),
              'cls': int,
              'label': str,
              'conf': float,
              'roi': numpy_array (crop of the weapon)
            }
          person_boxes: list of (px1, py1, px2, py2, p_conf)
        """
        h, w, _ = frame.shape
        raw_candidates = []
        person_boxes = []

        if use_zoom and self.person_model is not None:
            p_results = self.person_model(frame, classes=[0], conf=0.35, verbose=False)
            boxes = p_results[0].boxes

            if len(boxes) > 0:
                for p_box in boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                    p_conf = float(p_box.conf[0])
                    person_boxes.append((px1, py1, px2, py2, p_conf))

                    # 20% margin around person to include hands / arms holding weapons
                    pad_x = int((px2 - px1) * 0.20)
                    pad_y = int((py2 - py1) * 0.15)
                    x1 = max(0, px1 - pad_x)
                    y1 = max(0, py1 - pad_y)
                    x2 = min(w, px2 + pad_x)
                    y2 = min(h, py2 + pad_y)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Upscale small crops with Bicubic interpolation for high pixel density
                    min_dim = 640
                    ch, cw = crop.shape[:2]
                    if cw < min_dim or ch < min_dim:
                        scale = max(min_dim / max(1, cw), min_dim / max(1, ch))
                        new_w, new_h = int(cw * scale), int(ch * scale)
                        crop_scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    else:
                        scale = 1.0
                        crop_scaled = crop

                    # Apply DIP Enhancement
                    inference_crop = apply_dip_enhancement(crop_scaled, use_wavelet=use_wavelet) if use_dip else crop_scaled

                    # Run weapon model on enhanced crop
                    w_results = self.weapon_model(inference_crop, conf=conf_threshold, imgsz=imgsz, verbose=False)
                    for wb in w_results[0].boxes:
                        scx1, scy1, scx2, scy2 = map(int, wb.xyxy[0])
                        # Map back through bicubic scale
                        cx1 = int(scx1 / scale)
                        cy1 = int(scy1 / scale)
                        cx2 = int(scx2 / scale)
                        cy2 = int(scy2 / scale)

                        # Translate crop coordinates back to full frame
                        gx1 = max(0, min(w, x1 + cx1))
                        gy1 = max(0, min(h, y1 + cy1))
                        gx2 = max(0, min(w, x1 + cx2))
                        gy2 = max(0, min(h, y1 + cy2))

                        w_cls = int(wb.cls[0])
                        w_conf = float(wb.conf[0])
                        w_label = self.weapon_model.names[w_cls]
                        roi_crop = frame[gy1:gy2, gx1:gx2].copy()

                        raw_candidates.append({
                            'box': (gx1, gy1, gx2, gy2),
                            'cls': w_cls,
                            'label': w_label,
                            'conf': w_conf,
                            'roi': roi_crop
                        })

                return raw_candidates, person_boxes

        # Fallback or zoom=False: run weapon detection on full frame
        inference_frame = apply_dip_enhancement(frame, use_wavelet=use_wavelet) if use_dip else frame
        w_results = self.weapon_model(inference_frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
        for wb in w_results[0].boxes:
            gx1, gy1, gx2, gy2 = map(int, wb.xyxy[0])
            w_cls = int(wb.cls[0])
            w_conf = float(wb.conf[0])
            w_label = self.weapon_model.names[w_cls]
            roi_crop = frame[gy1:gy2, gx1:gx2].copy()

            raw_candidates.append({
                'box': (gx1, gy1, gx2, gy2),
                'cls': w_cls,
                'label': w_label,
                'conf': w_conf,
                'roi': roi_crop
            })

        return raw_candidates, person_boxes
