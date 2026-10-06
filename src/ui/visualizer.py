import cv2
import numpy as np

def draw_label(img, text, pos, font_scale=0.7, text_color=(255, 255, 255), bg_color=(30, 30, 30), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    y = max(h + 10, y)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 10, y + baseline + 4), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

class Visualizer:
    """
    Renders bounding boxes, alert banners, and HUD overlays.
    """
    def __init__(self):
        self.flash_state = False

    def render_overlay(self, frame, person_boxes, verified_threats, is_confirmed, hud_info=None):
        annotated = frame.copy()
        h, w, _ = frame.shape
        self.flash_state = not self.flash_state

        # 1. Draw detected persons
        for (px1, py1, px2, py2, p_conf) in person_boxes:
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 180, 0), 1)
            cv2.putText(annotated, f"Person {p_conf:.2f}", (px1, max(20, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)

        # 2. Draw verified threats
        for threat in verified_threats:
            (x1, y1, x2, y2) = threat['box']
            status = threat.get('status', '')
            score = threat.get('weighted_score', threat.get('ai_conf', 0.0))
            label = threat['label']

            if status == "CONFIRMED":
                # High alert: Red flashing box
                color = (0, 0, 255) if self.flash_state else (0, 60, 255)
                box_thickness = 4
                tag = f"CONFIRMED: {label} ({int(score * 100)}%)"
                tag_bg = (0, 0, 200)
            elif "ACCUMULATING" in status:
                # Yellow/Orange box while gathering frames
                color = (0, 165, 255)
                box_thickness = 2
                accum = threat.get('accum_count', 1)
                tag = f"VERIFYING ({accum}/4): {label} ({int(score * 100)}%)"
                tag_bg = (0, 120, 180)
            else:
                color = (0, 255, 255)
                box_thickness = 2
                tag = f"{label} ({int(score * 100)}%)"
                tag_bg = (40, 40, 40)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
            draw_label(annotated, tag, (x1, max(25, y1 - 10)), font_scale=0.65, bg_color=tag_bg)

        # 3. Flashing Threat Banner when Confirmed
        if is_confirmed and self.flash_state:
            # Draw semi-transparent top red alarm banner
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 220), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
            cv2.putText(annotated, "WARNING: THREAT CONFIRMED (MQTT / BUZZER ACTIVE)",
                        (w // 2 - 380, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. HUD Status
        if hud_info:
            draw_label(annotated, hud_info, (20, h - 25), font_scale=0.6, bg_color=(20, 20, 20))

        return annotated

    def render_detect_image(self, frame, person_boxes, verified_threats, is_confirmed,
                            hud_info=None, use_dip=True, use_wavelet=True):
        """
        Renders the true DETECT IMAGE:
        1. Transforms background with full DIP enhancement (CLAHE + Wavelet + Sharpening).
        2. Overlays threat detection bounding boxes and verification metadata.
        3. Renders a high-resolution Zoomed Threat ROI (Picture-in-Picture) window.
        4. Flashing warning banner if confirmed.
        """
        from src.dip.enhancement import apply_dip_enhancement

        h, w, _ = frame.shape
        # 1. Transform frame with DIP enhancement
        detect_base = apply_dip_enhancement(frame, use_wavelet=use_wavelet) if use_dip else frame.copy()

        # 2. Render overlays
        annotated = self.render_overlay(detect_base, person_boxes, verified_threats, is_confirmed, hud_info)

        # 3. Add Picture-in-Picture Zoomed Weapon Crop
        if len(verified_threats) > 0:
            best_threat = verified_threats[0]
            roi = best_threat.get('roi')
            (bx1, by1, bx2, by2) = best_threat['box']
            if roi is None or roi.size == 0:
                roi = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]

            if roi is not None and roi.size > 0:
                pip_w, pip_h = 280, 200
                pip_x = w - pip_w - 20
                pip_y = 65

                zoomed_roi = cv2.resize(roi, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
                if use_dip:
                    zoomed_roi = apply_dip_enhancement(zoomed_roi, use_wavelet=False)

                # Draw dark bezel background and border
                cv2.rectangle(annotated, (pip_x - 3, pip_y - 24), (pip_x + pip_w + 3, pip_y + pip_h + 3), (12, 12, 12), -1)
                border_color = (0, 0, 255) if is_confirmed else (0, 180, 255)
                cv2.rectangle(annotated, (pip_x - 3, pip_y - 24), (pip_x + pip_w + 3, pip_y + pip_h + 3), border_color, 2)

                # Paste zoomed weapon crop
                annotated[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = zoomed_roi

                # Label on bezel
                score_pct = int(best_threat.get('weighted_score', best_threat.get('ai_conf', 0.8)) * 100)
                tag_label = f"ZOOMED THREAT ROI: {best_threat['label'].upper()} ({score_pct}%)"
                cv2.putText(annotated, tag_label, (pip_x + 6, pip_y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def draw_timestamp(self, img, timestamp_str=None, pos=(20, 38)):
        """Renders CCTV style timestamp on monitor feed."""
        if not timestamp_str:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw_label(img, f"REC  {timestamp_str}", pos, font_scale=0.6,
                   text_color=(255, 255, 255), bg_color=(15, 15, 15), thickness=1)

    def draw_mode_badge(self, img, mode="ORIGINAL CAM", is_threat=False):
        """Renders top-right camera mode badge (ORIGINAL vs DETECT CAM)."""
        h, w, _ = img.shape
        pos = (w - 200, 38)
        if is_threat:
            draw_label(img, f"[ {mode} ]", pos, font_scale=0.6,
                       text_color=(255, 255, 255), bg_color=(0, 0, 180), thickness=2)
        else:
            draw_label(img, f"[ {mode} ]", pos, font_scale=0.6,
                       text_color=(210, 210, 210), bg_color=(25, 25, 25), thickness=1)

    def create_side_by_side(self, original, detected):
        """Creates side-by-side comparison image."""
        orig_view = original.copy()
        draw_label(orig_view, "CAMERA FEED (ORIGINAL)", (20, 35), font_scale=0.7, bg_color=(30, 30, 30))
        return np.hstack([orig_view, detected])
