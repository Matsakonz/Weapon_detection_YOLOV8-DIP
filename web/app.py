import os
import sys
import json
import time
import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response, request, jsonify

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MODEL_WEAPON,
    MODEL_PERSON,
    RESULTS_DIR,
    CONF_MIN,
    CONF_HIGH,
    WEIGHT_AI,
    WEIGHT_SHAPE,
    WEIGHTED_SCORE_MIN,
    FRAME_ACCUMULATION_MIN,
    DASHBOARD_ACTIVE,
    IMGSZ,
    USE_HUMAN_ZOOM,
    USE_SAHI,
    USE_DIP,
    USE_WAVELET
)
from src.detection.detector import WeaponDetector
from src.detection.verifier import ThreatVerifier
from src.alerts.alert_manager import AlertManager
from src.ui.visualizer import Visualizer

CAMERAS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "cameras.json")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Shared State
state = {
    "conf_min": CONF_MIN,
    "conf_high": CONF_HIGH,
    "weight_ai": WEIGHT_AI,
    "weight_shape": WEIGHT_SHAPE,
    "weighted_score_min": WEIGHTED_SCORE_MIN,
    "frame_accum_min": FRAME_ACCUMULATION_MIN,
    "dashboard_active": True,
    "use_zoom": USE_HUMAN_ZOOM,
    "use_sahi": USE_SAHI,
    "use_dip": USE_DIP,
    "use_wavelet": USE_WAVELET,
    "active_camera_id": "cam_1",
    "fps": 0.0,
    "threat_count": 0,
    "person_count": 0,
    "is_confirmed": False,
    "threat_label": "",
    "view_mode": "auto",
    "recent_events": [],
    "last_frame": None
}

# Singletons
alert_mgr = AlertManager()
detector = WeaponDetector(MODEL_WEAPON, MODEL_PERSON)
verifier = ThreatVerifier(alert_manager=alert_mgr)
visualizer = Visualizer()
model_lock = threading.Lock()

from datetime import datetime

class CameraStreamWorker:
    """
    Dedicated background worker per camera:
    - Continuously reads latest camera frames without stream blocking.
    - Decoupled detection loop runs AI models in background.
    - Result: Ultra-smooth 30+ FPS original camera stream, automatically switching
      to detect cam whenever a weapon candidate is detected!
    """
    def __init__(self, cam_id, source, name):
        self.cam_id = cam_id
        self.source = source
        self.name = name
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Detection state
        self.det_lock = threading.Lock()
        self.person_boxes = []
        self.verified_threats = []
        self.is_confirmed = False
        self.has_threat = False
        self.last_threat_time = 0
        self.fps = 0.0

        # Start capture thread
        self.cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.cap_thread.start()

        # Start detection thread
        self.det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.det_thread.start()

    def _capture_loop(self):
        fps_prev = time.time()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                src = int(self.source) if str(self.source).isdigit() else self.source
                self.cap = cv2.VideoCapture(src)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    time.sleep(1.0)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.04)
                continue

            now = time.time()
            dt = now - fps_prev
            fps_prev = now
            if dt > 0:
                self.fps = round(1.0 / dt, 1)

            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(0.005)

    def _detection_loop(self):
        while self.running:
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is None:
                time.sleep(0.05)
                continue

            # Run AI detection + Flowchart verification
            with model_lock:
                raw_cands, person_boxes = detector.detect(
                    frame_to_process, conf_threshold=state["conf_min"], imgsz=IMGSZ,
                    use_zoom=state["use_zoom"], use_sahi=state["use_sahi"],
                    use_dip=state["use_dip"], use_wavelet=state["use_wavelet"]
                )

                verifier.dashboard_active = state["dashboard_active"]
                verifier.conf_min = state["conf_min"]
                verifier.conf_high = state["conf_high"]
                verifier.weight_ai = state["weight_ai"]
                verifier.weight_shape = state["weight_shape"]
                verifier.weighted_score_min = state["weighted_score_min"]
                verifier.frame_accum_min = state["frame_accum_min"]
                verified_threats, is_confirmed = verifier.verify_candidates(raw_cands, frame_to_process)

            now = time.time()
            has_threat_now = (len(verified_threats) > 0 or is_confirmed)

            with self.det_lock:
                self.person_boxes = person_boxes
                self.verified_threats = verified_threats
                self.is_confirmed = is_confirmed

                if has_threat_now:
                    self.has_threat = True
                    self.last_threat_time = now
                    # Record event
                    best_t = verified_threats[0]
                    state["threat_label"] = best_t['label']
                    evt = {
                        "camera": self.name,
                        "label": best_t['label'],
                        "score": round(best_t.get('weighted_score', best_t.get('ai_conf', 0.8)), 2),
                        "time": time.strftime("%H:%M:%S")
                    }
                    if not state["recent_events"] or state["recent_events"][0]["time"] != evt["time"]:
                        state["recent_events"].insert(0, evt)
                        if len(state["recent_events"]) > 20:
                            state["recent_events"].pop()
                else:
                    # Hold threat display for 1.2s to prevent rapid flickering
                    if (now - self.last_threat_time) > 1.2:
                        self.has_threat = False

            # Update shared state if active camera
            if state.get("active_camera_id") == self.cam_id:
                state["fps"] = self.fps
                state["threat_count"] = len(verified_threats)
                state["person_count"] = len(person_boxes)
                state["is_confirmed"] = is_confirmed
                state["has_threat"] = self.has_threat

            time.sleep(0.02)

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_detection_state(self):
        with self.det_lock:
            return {
                "persons": list(self.person_boxes),
                "threats": list(self.verified_threats),
                "is_confirmed": self.is_confirmed,
                "has_threat": self.has_threat
            }

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

class CameraHub:
    """Manages active CameraStreamWorkers for local and IP cameras."""
    def __init__(self):
        self.workers = {}
        self.lock = threading.Lock()

    def get_source(self, cam_id):
        cams = self.load_cameras()
        for c in cams:
            if c['id'] == cam_id:
                src = c['source']
                return int(src) if str(src).isdigit() else src
        return 1

    def get_name(self, cam_id):
        cams = self.load_cameras()
        for c in cams:
            if c['id'] == cam_id:
                return c.get('name', cam_id)
        return cam_id

    def load_cameras(self):
        if os.path.exists(CAMERAS_CONFIG_PATH):
            with open(CAMERAS_CONFIG_PATH, 'r') as f:
                return json.load(f)
        return []

    def save_cameras(self, cameras):
        with open(CAMERAS_CONFIG_PATH, 'w') as f:
            json.dump(cameras, f, indent=2)

    def get_worker(self, cam_id):
        with self.lock:
            if cam_id in self.workers and self.workers[cam_id].running:
                return self.workers[cam_id]

            cams = self.load_cameras()
            cam_info = next((c for c in cams if c['id'] == cam_id), None)
            if not cam_info:
                # Default to index 1 if not found
                cam_info = {"id": cam_id, "source": "1", "name": cam_id}

            worker = CameraStreamWorker(cam_id, cam_info['source'], cam_info.get('name', cam_id))
            self.workers[cam_id] = worker
            return worker

    def release_cam(self, cam_id):
        with self.lock:
            if cam_id in self.workers:
                self.workers[cam_id].stop()
                del self.workers[cam_id]

cam_hub = CameraHub()

def create_offline_frame(camera_name="CAMERA"):
    """Creates a sleek minimalist black & white offline placeholder frame."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (16, 16), (1264, 704), (35, 35, 35), 1)
    title = f"[ {camera_name.upper()} ]"
    sub = "CONNECTING / NO SIGNAL"
    cv2.putText(frame, title, (480, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, sub, (490, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    ret_enc, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    return buffer.tobytes() if ret_enc else b''

def generate_video_stream(cam_id):
    """
    Generates high-FPS stream:
    - When NO weapon is detected: Stream clean ORIGINAL CAM frame at full smooth 30+ FPS!
    - When weapon is detected: Automatically switch to DETECT CAM with bounding boxes & HUD!
    - Monitor always displays real-time CCTV timestamp.
    """
    state["active_camera_id"] = cam_id
    cam_name = cam_hub.get_name(cam_id)
    prev_time = time.time()

    while True:
        worker = cam_hub.get_worker(cam_id)
        if worker is None:
            offline_bytes = create_offline_frame(cam_name)
            if offline_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + offline_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        frame = worker.get_frame()
        if frame is None:
            time.sleep(0.04)
            continue

        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 30.0
        state["fps"] = round(fps, 1)
        state["last_frame"] = frame.copy()

        det = worker.get_detection_state()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Smart Mode Switch:
        view_mode = state.get("view_mode", "auto")
        show_detect_image = (view_mode == "detect") or (view_mode == "auto" and det.get("has_threat", False))

        if show_detect_image:
            out_frame = visualizer.render_detect_image(
                frame, det["persons"], det["threats"], det["is_confirmed"],
                hud_info=f"{cam_name} | FPS: {fps:.1f} | DETECT IMAGE",
                use_dip=state["use_dip"], use_wavelet=state["use_wavelet"]
            )
            visualizer.draw_mode_badge(out_frame, mode="DETECT IMAGE", is_threat=det.get("has_threat", False))
            visualizer.draw_timestamp(out_frame, timestamp_str)
        else:
            out_frame = frame.copy()
            visualizer.draw_mode_badge(out_frame, mode="ORIGINAL CAM", is_threat=False)
            visualizer.draw_timestamp(out_frame, timestamp_str)

        ret_enc, buffer = cv2.imencode('.jpg', out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret_enc:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.02)

# Web Page Routes
@app.route('/')
def index():
    return render_template('camera.html', active_page='camera')

@app.route('/settings')
def settings_page():
    return render_template('settings.html', active_page='settings')

@app.route('/video_feed')
@app.route('/video_feed/<cam_id>')
def video_feed(cam_id=None):
    if not cam_id:
        cam_id = state.get("active_camera_id", "cam_1")
    return Response(generate_video_stream(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# REST API Endpoints
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    return jsonify(cam_hub.load_cameras())

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    data = request.json or {}
    name = data.get('name', '').strip()
    cam_type = data.get('type', 'ip')
    source = data.get('source', '').strip()

    if not name or not source:
        return jsonify({"success": False, "message": "Name and Source are required."}), 400

    cameras = cam_hub.load_cameras()
    new_id = f"cam_{int(time.time())}"
    cameras.append({
        "id": new_id,
        "name": name,
        "source": source,
        "type": cam_type,
        "enabled": True
    })
    cam_hub.save_cameras(cameras)
    return jsonify({"success": True, "camera_id": new_id})

@app.route('/api/cameras/<cam_id>', methods=['DELETE'])
def delete_camera(cam_id):
    cameras = cam_hub.load_cameras()
    filtered = [c for c in cameras if c['id'] != cam_id]
    if len(filtered) == len(cameras):
        return jsonify({"success": False, "message": "Camera not found."}), 404

    cam_hub.save_cameras(filtered)
    # Release worker if running
    cam_hub.release_cam(cam_id)
    return jsonify({"success": True})

@app.route('/api/status', methods=['GET'])
def get_status():
    active_id = state.get("active_camera_id", "cam_1")
    worker = cam_hub.get_worker(active_id)
    det = worker.get_detection_state() if worker else {}
    return jsonify({
        "fps": state["fps"],
        "threat_count": state["threat_count"],
        "person_count": state["person_count"],
        "is_confirmed": state["is_confirmed"],
        "has_threat": det.get("has_threat", False),
        "threat_label": state["threat_label"],
        "view_mode": state.get("view_mode", "auto"),
        "use_sahi": state["use_sahi"],
        "use_zoom": state["use_zoom"],
        "use_dip": state["use_dip"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recent_events": state["recent_events"][:10]
    })

@app.route('/api/view_mode/<mode>', methods=['POST'])
def set_view_mode(mode):
    if mode in ['auto', 'detect', 'original']:
        state['view_mode'] = mode
        return jsonify({"success": True, "view_mode": mode})
    return jsonify({"error": "Invalid view mode"}), 400

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify({
        "CONF_MIN": state["conf_min"],
        "CONF_HIGH": state["conf_high"],
        "WEIGHT_AI": state["weight_ai"],
        "WEIGHT_SHAPE": state["weight_shape"],
        "WEIGHTED_SCORE_MIN": state["weighted_score_min"],
        "FRAME_ACCUMULATION_MIN": state["frame_accum_min"],
        "DASHBOARD_ACTIVE": state["dashboard_active"],
        "USE_SAHI": state["use_sahi"]
    })

@app.route('/api/settings', methods=['POST'])
def save_settings():
    data = request.json or {}
    if "CONF_MIN" in data: state["conf_min"] = float(data["CONF_MIN"])
    if "CONF_HIGH" in data: state["conf_high"] = float(data["CONF_HIGH"])
    if "WEIGHT_AI" in data: state["weight_ai"] = float(data["WEIGHT_AI"])
    if "WEIGHT_SHAPE" in data: state["weight_shape"] = float(data["WEIGHT_SHAPE"])
    if "WEIGHTED_SCORE_MIN" in data: state["weighted_score_min"] = float(data["WEIGHTED_SCORE_MIN"])
    if "FRAME_ACCUMULATION_MIN" in data: state["frame_accum_min"] = int(data["FRAME_ACCUMULATION_MIN"])
    if "USE_SAHI" in data: state["use_sahi"] = bool(data["USE_SAHI"])
    return jsonify({"success": True})

@app.route('/api/toggle/<feature>', methods=['POST'])
def toggle_feature(feature):
    if feature == 'zoom':
        state['use_zoom'] = not state['use_zoom']
        return jsonify({"feature": "zoom", "state": state['use_zoom']})
    elif feature == 'sahi':
        state['use_sahi'] = not state['use_sahi']
        return jsonify({"feature": "sahi", "state": state['use_sahi']})
    elif feature == 'dip':
        state['use_dip'] = not state['use_dip']
        return jsonify({"feature": "dip", "state": state['use_dip']})
    elif feature == 'wavelet':
        state['use_wavelet'] = not state['use_wavelet']
        return jsonify({"feature": "wavelet", "state": state['use_wavelet']})
    return jsonify({"error": "Unknown feature"}), 400

@app.route('/api/snapshot', methods=['POST'])
def manual_snapshot():
    frame = state.get("last_frame")
    if frame is not None:
        path = alert_mgr.save_incident_snapshot(frame, "MANUAL_WEB_SNAPSHOT")
        return jsonify({"success": True, "path": path})
    return jsonify({"success": False, "message": "No active frame"}), 400

def start_server(host="0.0.0.0", port=5001):
    for p in range(port, port + 10):
        try:
            print(f"\n=======================================================")
            print(f" 🛡️  Aegis Threat Guard Dashboard Online!")
            print(f" 🌐  Open in browser: http://localhost:{p}")
            print(f"=======================================================\n")
            app.run(host=host, port=p, debug=False, threaded=True)
            break
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 48:
                print(f"[Notice] Port {p} busy, trying port {p+1}...")
                continue
            raise e

if __name__ == '__main__':
    start_server(port=5001)
