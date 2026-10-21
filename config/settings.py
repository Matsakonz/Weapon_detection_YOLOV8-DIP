import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model directories & helper
MODELS_DIR = os.path.join(BASE_DIR, "models")

def get_available_models():
    """Lists all .pt model files found in the models/ folder."""
    if not os.path.exists(MODELS_DIR):
        return []
    return sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") and not f.startswith(".")])

def resolve_model_path(model_filename=None):
    """Resolves absolute path for a model file in models/ or project root."""
    if model_filename:
        path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(path):
            return path
        fallback = os.path.join(BASE_DIR, model_filename)
        if os.path.exists(fallback):
            return fallback

    available = get_available_models()
    if available:
        for preferred in ("best4.pt", "best.pt", "best2.pt", "best3.pt"):
            if preferred in available:
                return os.path.join(MODELS_DIR, preferred)
        return os.path.join(MODELS_DIR, available[0])
    return os.path.join(MODELS_DIR, "best4.pt")

# Model paths
MODEL_WEAPON = resolve_model_path()

MODEL_PERSON = os.path.join(MODELS_DIR, "yolov8n.pt")
if not os.path.exists(MODEL_PERSON):
    MODEL_PERSON = os.path.join(BASE_DIR, "yolov8n.pt")

# Directory paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Real-Time Threat Verification Thresholds
CONF_MIN = 0.6
FRAME_ACCUMULATION_MIN = 3
DASHBOARD_ACTIVE = True

# Threat Class Filtering (Default detects only Pistol, Knife excluded)
ALL_CLASSES = ["Pistol", "Knife", "Smartphone", "Purse", "Bill", "Card"]
TARGET_CLASSES = ['Pistol']

# DIP & Camera Settings
IMGSZ = 1280
USE_HUMAN_ZOOM = True
USE_SAHI = True
HUMAN_MARGIN_X = 0.35          # +35% margin around person to capture extended arms/aiming
HUMAN_MARGIN_Y = 0.25          # +25% vertical margin for raised hands & dropped weapons
SAHI_TILE_SIZE = 640           # Tile resolution for sliced inference
SAHI_OVERLAP = 0.25            # 25% overlap between adjacent tiles
SAHI_NMS_IOU = 0.45            # IoU threshold to deduplicate detections across tiles
USE_DIP = True
USE_WAVELET = True
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (1920, 1080)
WINDOW_SIZE = (1600, 900)

# Alert Settings (MQTT, Buzzer & Snapshot Logging)
BUZZER_ENABLED = True
BUZZER_SOUND_PATH = "/System/Library/Sounds/Ping.aiff"  # macOS native alert sound
SAVE_SNAPSHOTS = True
MQTT_ENABLED = True
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "security/threats"

# Runtime Shared Settings File (Synchronizes main.py and Web Dashboard)
import json
import re

RUNTIME_SETTINGS_FILE = os.path.join(BASE_DIR, "config", "runtime_settings.json")
SETTINGS_FILE = os.path.abspath(__file__)

def get_default_settings():
    available = get_available_models()
    default_model = "best4.pt" if "best4.pt" in available else (available[0] if available else "best.pt")
    return {
        "model_weapon": default_model,
        "conf_min": CONF_MIN,
        "frame_accum_min": FRAME_ACCUMULATION_MIN,
        "dashboard_active": DASHBOARD_ACTIVE,
        "save_snapshots": SAVE_SNAPSHOTS,
        "target_classes": list(TARGET_CLASSES),
        "all_classes": list(ALL_CLASSES),
        "use_zoom": USE_HUMAN_ZOOM,
        "use_sahi": USE_SAHI,
        "use_dip": USE_DIP,
        "use_wavelet": USE_WAVELET,
        "camera_index": CAMERA_INDEX
    }

def save_settings_to_file(updates):
    """
    Directly persists updated settings variables into config/settings.py.
    """
    if not os.path.exists(SETTINGS_FILE):
        return False
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        key_map = {
            "CONF_MIN": "CONF_MIN",
            "conf_min": "CONF_MIN",
            "FRAME_ACCUMULATION_MIN": "FRAME_ACCUMULATION_MIN",
            "frame_accum_min": "FRAME_ACCUMULATION_MIN",
            "DASHBOARD_ACTIVE": "DASHBOARD_ACTIVE",
            "dashboard_active": "DASHBOARD_ACTIVE",
            "SAVE_SNAPSHOTS": "SAVE_SNAPSHOTS",
            "save_snapshots": "SAVE_SNAPSHOTS",
            "TARGET_CLASSES": "TARGET_CLASSES",
            "target_classes": "TARGET_CLASSES",
            "USE_SAHI": "USE_SAHI",
            "use_sahi": "USE_SAHI",
            "USE_DIP": "USE_DIP",
            "use_dip": "USE_DIP",
            "USE_WAVELET": "USE_WAVELET",
            "use_wavelet": "USE_WAVELET",
            "USE_HUMAN_ZOOM": "USE_HUMAN_ZOOM",
            "use_zoom": "USE_HUMAN_ZOOM",
        }

        for update_key, target_var in key_map.items():
            if update_key in updates:
                val = updates[update_key]
                val_repr = repr(val)
                pattern = rf'^{target_var}\s*=\s*.*$'
                if re.search(pattern, content, flags=re.MULTILINE):
                    content = re.sub(pattern, f"{target_var} = {val_repr}", content, flags=re.MULTILINE)
                else:
                    content += f"\n{target_var} = {val_repr}\n"

        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"[Settings] Error writing to settings.py: {e}")
        return False

def load_runtime_settings():
    """Loads shared settings, merging defaults with runtime_settings.json if present."""
    settings = get_default_settings()
    if os.path.exists(RUNTIME_SETTINGS_FILE):
        try:
            with open(RUNTIME_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    settings.update(saved)
        except Exception as e:
            print(f"[Settings] Warning reading runtime settings: {e}")
    return settings

def save_runtime_settings(updates):
    """Saves updated settings directly to settings.py and runtime_settings.json."""
    # 1. Update config/settings.py directly
    save_settings_to_file(updates)

    # 2. Update runtime_settings.json safely filtering non-serializable objects
    try:
        current = load_runtime_settings()
        for k, v in updates.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                try:
                    json.dumps(v)
                    current[k] = v
                except (TypeError, OverflowError):
                    pass
        with open(RUNTIME_SETTINGS_FILE, "w") as f:
            json.dump(current, f, indent=2)
        return True
    except Exception as e:
        print(f"[Settings] Error saving runtime settings: {e}")
        return False

