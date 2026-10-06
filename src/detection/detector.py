import cv2
import numpy as np
from ultralytics import YOLO
from src.dip.enhancement import apply_dip_enhancement
from config.settings import (
    IMGSZ,
    USE_HUMAN_ZOOM,
    USE_SAHI,
    TARGET_CLASSES,
    HUMAN_MARGIN_X,
    HUMAN_MARGIN_Y,
    SAHI_TILE_SIZE,
    SAHI_OVERLAP,
    SAHI_NMS_IOU,
    USE_DIP,
    USE_WAVELET
)

def generate_slices(img_w, img_h, tile_size=640, overlap_ratio=0.25):
    """
    Generates overlapping tile coordinates (x1, y1, x2, y2) covering the image.
    Ensures 100% area coverage with overlap along edges.
    """
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]

    step_x = max(1, int(tile_size * (1.0 - overlap_ratio)))
    step_y = max(1, int(tile_size * (1.0 - overlap_ratio)))

    x_starts = []
    x = 0
    while x + tile_size < img_w:
        x_starts.append(x)
        x += step_x
    x_starts.append(max(0, img_w - tile_size))
    x_starts = sorted(list(set(x_starts)))

    y_starts = []
    y = 0
    while y + tile_size < img_h:
        y_starts.append(y)
        y += step_y
    y_starts.append(max(0, img_h - tile_size))
    y_starts = sorted(list(set(y_starts)))

    slices = []
    for ys in y_starts:
        for xs in x_starts:
            slices.append((xs, ys, min(img_w, xs + tile_size), min(img_h, ys + tile_size)))
    return slices

def apply_nms(candidates, iou_threshold=0.45):
    """
    Removes duplicate overlapping candidate detections using OpenCV NMS.
    """
    if len(candidates) <= 1:
        return candidates

    boxes = [[c['box'][0], c['box'][1], c['box'][2] - c['box'][0], c['box'][3] - c['box'][1]] for c in candidates]
    scores = [c['conf'] for c in candidates]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.1, nms_threshold=iou_threshold)

    if len(indices) == 0:
        return []

    kept_indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
    return [candidates[i] for i in kept_indices]

class WeaponDetector:
    """
    Multi-stage Targeted SAHI Weapon Detector:
    1. Human-Guided Region Detection: Detects persons first via YOLOv8-Nano.
    2. Generous Safety Margins (+35% X, +25% Y): Guarantees extended arms, aiming guns,
       holsters, and dropped weapons near feet are fully captured in the ROI.
    3. Targeted SAHI Slicing: Slices the human ROI into overlapping high-res tiles (640px)
       at 1:1 native camera pixel resolution, preventing small weapon downsampling.
    4. Bicubic Super-Sampling: Upscales small/distant crops (<640px) with INTER_CUBIC.
    5. DIP Enhancement: CLAHE + Unsharp Masking + Wavelet DWT edge boosting per tile.
    6. YOLO Weapon Inference: High-resolution inference on each slice.
    7. Sliced NMS & Coordinate Translation: Maps tile boxes back to full 1080p frame.
    8. Full-Frame Fallback: Runs inference on full frame if no humans are detected.
    """
    def __init__(self, weapon_model_path, person_model_path):
        print(f"[Detector] Loading Weapon Model: {weapon_model_path}")
        self.weapon_model = YOLO(weapon_model_path)
        print(f"[Detector] Loading Person Model: {person_model_path}")
        self.person_model = YOLO(person_model_path)

    def detect(self, frame, conf_threshold=0.50, imgsz=IMGSZ,
               use_zoom=USE_HUMAN_ZOOM, use_sahi=USE_SAHI,
               target_classes=TARGET_CLASSES,
               margin_x=HUMAN_MARGIN_X, margin_y=HUMAN_MARGIN_Y,
               tile_size=SAHI_TILE_SIZE, overlap=SAHI_OVERLAP, nms_iou=SAHI_NMS_IOU,
               use_dip=USE_DIP, use_wavelet=USE_WAVELET):
        """
        Executes detection on frame using Targeted SAHI and DIP pipeline.

        Returns:
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

                    # Generous expanded margin (+35% X, +25% Y) for extended arms / dropped weapons
                    pad_x = int((px2 - px1) * margin_x)
                    pad_y = int((py2 - py1) * margin_y)
                    x1 = max(0, px1 - pad_x)
                    y1 = max(0, py1 - pad_y)
                    x2 = min(w, px2 + pad_x)
                    y2 = min(h, py2 + pad_y)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    ch, cw = crop.shape[:2]

                    # --- TARGETED SAHI SLICING PIPELINE ---
                    if use_sahi:
                        # If crop is too small, upscale to minimum tile dimension first
                        if cw < tile_size and ch < tile_size:
                            scale = max(tile_size / max(1, cw), tile_size / max(1, ch))
                            scaled_w, scaled_h = int(cw * scale), int(ch * scale)
                            crop_scaled = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
                        else:
                            scale = 1.0
                            crop_scaled = crop

                        sc_h, sc_w = crop_scaled.shape[:2]
                        tiles = generate_slices(sc_w, sc_h, tile_size=tile_size, overlap_ratio=overlap)

                        person_candidates = []
                        for (tx1, ty1, tx2, ty2) in tiles:
                            tile_img = crop_scaled[ty1:ty2, tx1:tx2]
                            if tile_img.size == 0:
                                continue

                            # Apply DIP Enhancement on tile
                            inference_tile = apply_dip_enhancement(tile_img, use_wavelet=use_wavelet) if use_dip else tile_img

                            # Run weapon model on slice
                            w_results = self.weapon_model(inference_tile, conf=conf_threshold, imgsz=tile_size, verbose=False)

                            for wb in w_results[0].boxes:
                                bx1, by1, bx2, by2 = map(int, wb.xyxy[0])
                                # Map back to scaled crop
                                crop_bx1 = tx1 + bx1
                                crop_by1 = ty1 + by1
                                crop_bx2 = tx1 + bx2
                                crop_by2 = ty1 + by2

                                # Map back across bicubic scale if applied
                                orig_cx1 = int(crop_bx1 / scale)
                                orig_cy1 = int(crop_by1 / scale)
                                orig_cx2 = int(crop_bx2 / scale)
                                orig_cy2 = int(crop_by2 / scale)

                                # Translate to global frame coordinates
                                gx1 = max(0, min(w, x1 + orig_cx1))
                                gy1 = max(0, min(h, y1 + orig_cy1))
                                gx2 = max(0, min(w, x1 + orig_cx2))
                                gy2 = max(0, min(h, y1 + orig_cy2))

                                if gx2 <= gx1 or gy2 <= gy1:
                                    continue

                                w_cls = int(wb.cls[0])
                                w_conf = float(wb.conf[0])
                                w_label = self.weapon_model.names[w_cls]

                                # Filter by target threat classes (e.g. Pistol, Knife)
                                if target_classes and w_label.lower() not in [c.lower() for c in target_classes]:
                                    continue

                                roi_crop = frame[gy1:gy2, gx1:gx2].copy()

                                person_candidates.append({
                                    'box': (gx1, gy1, gx2, gy2),
                                    'cls': w_cls,
                                    'label': w_label,
                                    'conf': w_conf,
                                    'roi': roi_crop
                                })

                        # Deduplicate overlapping slice detections with NMS
                        deduped = apply_nms(person_candidates, iou_threshold=nms_iou)
                        raw_candidates.extend(deduped)

                    else:
                        # Single-crop zoom mode without SAHI
                        min_dim = 640
                        if cw < min_dim or ch < min_dim:
                            scale = max(min_dim / max(1, cw), min_dim / max(1, ch))
                            crop_scaled = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
                        else:
                            scale = 1.0
                            crop_scaled = crop

                        inference_crop = apply_dip_enhancement(crop_scaled, use_wavelet=use_wavelet) if use_dip else crop_scaled
                        w_results = self.weapon_model(inference_crop, conf=conf_threshold, imgsz=imgsz, verbose=False)

                        for wb in w_results[0].boxes:
                            scx1, scy1, scx2, scy2 = map(int, wb.xyxy[0])
                            cx1, cy1 = int(scx1 / scale), int(scy1 / scale)
                            cx2, cy2 = int(scx2 / scale), int(scy2 / scale)

                            gx1 = max(0, min(w, x1 + cx1))
                            gy1 = max(0, min(h, y1 + cy1))
                            gx2 = max(0, min(w, x1 + cx2))
                            gy2 = max(0, min(h, y1 + cy2))

                            if gx2 <= gx1 or gy2 <= gy1:
                                continue

                            w_cls = int(wb.cls[0])
                            w_conf = float(wb.conf[0])
                            w_label = self.weapon_model.names[w_cls]

                            # Filter by target threat classes
                            if target_classes and w_label.lower() not in [c.lower() for c in target_classes]:
                                continue

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
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            w_cls = int(wb.cls[0])
            w_conf = float(wb.conf[0])
            w_label = self.weapon_model.names[w_cls]

            # Filter by target threat classes
            if target_classes and w_label.lower() not in [c.lower() for c in target_classes]:
                continue

            roi_crop = frame[gy1:gy2, gx1:gx2].copy()

            raw_candidates.append({
                'box': (gx1, gy1, gx2, gy2),
                'cls': w_cls,
                'label': w_label,
                'conf': w_conf,
                'roi': roi_crop
            })

        return raw_candidates, person_boxes
