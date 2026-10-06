import os
import sys
import time
import argparse
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    MODEL_WEAPON,
    MODEL_PERSON,
    DATA_IMAGES_DIR,
    RESULTS_DIR,
    CAMERA_INDEX,
    CAMERA_RESOLUTION,
    WINDOW_SIZE,
    CONF_MIN,
    IMGSZ,
    USE_HUMAN_ZOOM,
    USE_SAHI,
    TARGET_CLASSES,
    USE_DIP,
    USE_WAVELET
)
from src.detection.detector import WeaponDetector
from src.detection.verifier import ThreatVerifier
from src.alerts.alert_manager import AlertManager
from src.ui.visualizer import Visualizer, draw_label

def scan_available_cameras(max_to_check=4):
    """Detects all actively connected camera indices."""
    found = []
    for idx in range(max_to_check):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(idx)
            cap.release()
    return found if found else [0]

def open_cap(camera_idx, resolution=CAMERA_RESOLUTION):
    """Helper to initialize and configure a camera capture device."""
    cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return cap

def run_camera(camera_index=CAMERA_INDEX, conf_threshold=CONF_MIN, imgsz=IMGSZ,
               use_zoom=USE_HUMAN_ZOOM, use_sahi=USE_SAHI, target_classes=TARGET_CLASSES,
               use_dip=USE_DIP, use_wavelet=USE_WAVELET,
               start_dual_mode=False):
    """
    Executes live detection with Multi-Camera support:
    - Dynamic switching between cameras with 'k' or '0', '1'...
    - Dual-Camera split-screen grid view with 'g'
    - Targeted SAHI Multi-Tile Slicing on human crop with 't'
    - Threat Class Filtering (Defaults to detecting Pistol & Knife)
    - Full Flowchart Verification Pipeline (DIP, Zoom, Confidence, Shape, Accumulation, Alerts)
    """
    available_cams = scan_available_cameras()
    print(f"[Camera Scanner] Detected {len(available_cams)} camera(s): {available_cams}")

    if camera_index not in available_cams:
        camera_index = available_cams[0]

    alert_mgr = AlertManager()
    detector = WeaponDetector(MODEL_WEAPON, MODEL_PERSON)
    verifier = ThreatVerifier(alert_manager=alert_mgr)
    visualizer = Visualizer()

    dual_mode = start_dual_mode and len(available_cams) >= 2
    active_cam_idx = camera_index
    cap_primary = open_cap(active_cam_idx)
    cap_secondary = open_cap(available_cams[1]) if dual_mode else None

    if not cap_primary.isOpened():
        print(f"[Error] Could not open camera {active_cam_idx}.")
        return

    actual_w = int(cap_primary.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap_primary.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Active Camera {active_cam_idx}: {actual_w}x{actual_h} | Base Conf: {conf_threshold} | Classes: {target_classes}")

    window_name = "Threat Detection System ('k':cam | 'g':dual | 't':SAHI | 'd':DIP | 'w':wavelet | 'z':zoom | 's':snap | 'q':quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_SIZE[0], WINDOW_SIZE[1])

    prev_time = time.time()
    side_by_side = False

    try:
        while True:
            ret1, frame1 = cap_primary.read()
            if not ret1:
                print(f"[Camera] Feed from Camera {active_cam_idx} lost.")
                break

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 1. Detection on Primary Camera
            raw_cands1, person_boxes1 = detector.detect(
                frame1, conf_threshold=conf_threshold, imgsz=imgsz,
                use_zoom=use_zoom, use_sahi=use_sahi, target_classes=target_classes,
                use_dip=use_dip, use_wavelet=use_wavelet
            )
            verified1, is_confirmed1 = verifier.verify_candidates(raw_cands1, frame1)

            cam_tag = f"CAM {active_cam_idx}"
            hud_info1 = (
                f"[{cam_tag}] FPS: {fps:.1f} | Threats: {len(verified1)} | Persons: {len(person_boxes1)} | "
                f"SAHI: {'ON' if use_sahi else 'OFF'} | Zoom: {'ON' if use_zoom else 'OFF'} | DIP: {'ON' if use_dip else 'OFF'}"
            )
            annotated1 = visualizer.render_overlay(frame1, person_boxes1, verified1, is_confirmed1, hud_info1)

            # 2. Dual-Camera Mode Handling
            if dual_mode and cap_secondary is not None:
                ret2, frame2 = cap_secondary.read()
                if ret2:
                    raw_cands2, person_boxes2 = detector.detect(
                        frame2, conf_threshold=conf_threshold, imgsz=imgsz,
                        use_zoom=use_zoom, use_sahi=use_sahi, target_classes=target_classes,
                        use_dip=use_dip, use_wavelet=use_wavelet
                    )
                    verified2, is_confirmed2 = verifier.verify_candidates(raw_cands2, frame2)
                    sec_idx = available_cams[1] if active_cam_idx == available_cams[0] else available_cams[0]
                    hud_info2 = f"[CAM {sec_idx}] Threats: {len(verified2)} | Persons: {len(person_boxes2)}"
                    annotated2 = visualizer.render_overlay(frame2, person_boxes2, verified2, is_confirmed2, hud_info2)

                    # Resize both to 960x540 for combined side-by-side dual screen
                    v1_small = cv2.resize(annotated1, (960, 540))
                    v2_small = cv2.resize(annotated2, (960, 540))
                    draw_label(v1_small, f"CAMERA {active_cam_idx} (PRIMARY)", (20, 35), font_scale=0.7, bg_color=(20, 20, 20))
                    draw_label(v2_small, f"CAMERA {sec_idx} (SECONDARY)", (20, 35), font_scale=0.7, bg_color=(20, 20, 20))
                    display_img = np.hstack([v1_small, v2_small])
                else:
                    display_img = annotated1
            else:
                display_img = visualizer.create_side_by_side(frame1, annotated1) if side_by_side else annotated1

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[System] Shutting down.")
                break
            elif key == ord('k') or key == 9:  # 'k' or TAB: Switch to next available camera
                if len(available_cams) > 1:
                    curr_pos = available_cams.index(active_cam_idx) if active_cam_idx in available_cams else 0
                    next_idx = available_cams[(curr_pos + 1) % len(available_cams)]
                    print(f"[Camera Switch] Switching from Camera {active_cam_idx} -> Camera {next_idx}...")
                    cap_primary.release()
                    active_cam_idx = next_idx
                    cap_primary = open_cap(active_cam_idx)
            elif key in [ord('0'), ord('1'), ord('2')]:  # Direct number key switch
                target_idx = int(chr(key))
                if target_idx in available_cams and target_idx != active_cam_idx:
                    print(f"[Camera Switch] Switching directly to Camera {target_idx}...")
                    cap_primary.release()
                    active_cam_idx = target_idx
                    cap_primary = open_cap(active_cam_idx)
            elif key == ord('g'):  # Toggle Dual-Camera Grid
                if len(available_cams) >= 2:
                    dual_mode = not dual_mode
                    if dual_mode:
                        sec_idx = available_cams[1] if active_cam_idx == available_cams[0] else available_cams[0]
                        cap_secondary = open_cap(sec_idx)
                        print(f"[Dual Mode] ON: Streaming Camera {active_cam_idx} + Camera {sec_idx}")
                    else:
                        if cap_secondary is not None:
                            cap_secondary.release()
                            cap_secondary = None
                        print("[Dual Mode] OFF: Returned to single camera view")
                else:
                    print("[Dual Mode] Requires at least 2 connected cameras.")
            elif key == ord('d'):
                use_dip = not use_dip
                print(f"[DIP] Toggled: {'ON' if use_dip else 'OFF'}")
            elif key == ord('w'):
                use_wavelet = not use_wavelet
                print(f"[Wavelet] Toggled: {'ON' if use_wavelet else 'OFF'}")
            elif key == ord('z'):
                use_zoom = not use_zoom
                print(f"[Human-Zoom] Toggled: {'ON' if use_zoom else 'OFF'}")
            elif key == ord('t'):
                use_sahi = not use_sahi
                print(f"[Targeted SAHI] Toggled: {'ON' if use_sahi else 'OFF'}")
            elif key == ord('c'):
                side_by_side = not side_by_side
                print(f"[Side-by-Side] Toggled: {'ON' if side_by_side else 'OFF'}")
            elif key == ord('s'):
                alert_mgr.save_incident_snapshot(display_img, f"CAM{active_cam_idx}_SNAPSHOT")

    finally:
        cap_primary.release()
        if cap_secondary is not None:
            cap_secondary.release()
        cv2.destroyAllWindows()
        alert_mgr.close()

def run_image_comparison(images_dir=DATA_IMAGES_DIR, output_dir=RESULTS_DIR,
                         conf_threshold=CONF_MIN, imgsz=IMGSZ,
                         use_zoom=USE_HUMAN_ZOOM, use_sahi=USE_SAHI,
                         target_classes=TARGET_CLASSES,
                         use_dip=USE_DIP, use_wavelet=USE_WAVELET):
    """Batch compares original vs verified detections on test images."""
    if not os.path.exists(images_dir):
        images_dir = os.path.join(os.path.dirname(__file__))

    detector = WeaponDetector(MODEL_WEAPON, MODEL_PERSON)
    verifier = ThreatVerifier()
    visualizer = Visualizer()

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[Batch] Processing {len(image_files)} test images from {images_dir} (SAHI: {'ON' if use_sahi else 'OFF'} | Classes: {target_classes})...")

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(images_dir, fname)
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        raw_cands, person_boxes = detector.detect(
            frame, conf_threshold=conf_threshold, imgsz=imgsz,
            use_zoom=use_zoom, use_sahi=use_sahi, target_classes=target_classes,
            use_dip=use_dip, use_wavelet=use_wavelet
        )
        verified_threats, is_confirmed = verifier.verify_candidates(raw_cands, frame)

        annotated = visualizer.render_overlay(frame, person_boxes, verified_threats, is_confirmed)
        comparison = visualizer.create_side_by_side(frame, annotated)

        out_path = os.path.join(output_dir, f"verified_{fname}")
        cv2.imwrite(out_path, comparison)
        print(f"[{idx}/{len(image_files)}] Saved {out_path} ({len(verified_threats)} threat(s) verified)")

    print(f"\n[Batch] Completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Camera Weapon Detection & Verification System")
    parser.add_argument("--mode", choices=["camera", "compare", "dual"], default="camera", help="Run mode (camera, compare, dual)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Default camera index")
    parser.add_argument("--conf", type=float, default=CONF_MIN, help="Confidence threshold")
    parser.add_argument("--sahi", action="store_true", default=USE_SAHI, help="Enable Targeted SAHI multi-tile slicing")
    parser.add_argument("--no-sahi", action="store_false", dest="sahi", help="Disable Targeted SAHI")
    parser.add_argument("--classes", nargs="+", default=TARGET_CLASSES, help="Target threat classes to detect (default: Pistol Knife)")
    args = parser.parse_args()

    if args.mode == "camera":
        run_camera(camera_index=args.camera, conf_threshold=args.conf, use_sahi=args.sahi, target_classes=args.classes)
    elif args.mode == "dual":
        run_camera(camera_index=args.camera, conf_threshold=args.conf, use_sahi=args.sahi, target_classes=args.classes, start_dual_mode=True)
    else:
        run_image_comparison(conf_threshold=args.conf, use_sahi=args.sahi, target_classes=args.classes)
