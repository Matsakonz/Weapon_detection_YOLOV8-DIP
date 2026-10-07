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
DATA_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Flowchart Verification Thresholds
CONF_MIN = 0.60              # Below 60% -> Skip Frame
CONF_HIGH = 0.80             # Above 90% -> Direct pass to Frame Accumulation
WEIGHT_AI = 0.70             # AI confidence weight in 60-90% verification
WEIGHT_SHAPE = 0.30          # Shape similarity weight in 60-90% verification
WEIGHTED_SCORE_MIN = 0.70    # > 74% (>= 0.75) required to pass
FRAME_ACCUMULATION_MIN = 4   # > 3 Frames (>= 4 consecutive detections)
DASHBOARD_ACTIVE = False      # Dashboard / monitoring armed status

# Threat Class Filtering (Default detects only actual weapons: Pistol & Knife)
ALL_CLASSES = ["Pistol", "Knife", "Smartphone", "Purse", "Bill", "Card"]
TARGET_CLASSES = ["Pistol", "Knife"]

# DIP & Camera Settings
IMGSZ = 1280
USE_HUMAN_ZOOM = True
USE_SAHI = True                # Targeted Human SAHI (Multi-tile Slicing)
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

# Alert Settings (MQTT & Buzzer)
BUZZER_ENABLED = True
BUZZER_SOUND_PATH = "/System/Library/Sounds/Ping.aiff"  # macOS native alert sound
MQTT_ENABLED = True
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "security/threats"

# Runtime Shared Settings File (Synchronizes main.py and Web Dashboard)
import json

RUNTIME_SETTINGS_FILE = os.path.join(BASE_DIR, "config", "runtime_settings.json")

def get_default_settings():
    available = get_available_models()
    default_model = "best4.pt" if "best4.pt" in available else (available[0] if available else "best.pt")
    return {
        "model_weapon": default_model,
        "conf_min": CONF_MIN,
        "conf_high": CONF_HIGH,
        "weight_ai": WEIGHT_AI,
        "weight_shape": WEIGHT_SHAPE,
        "weighted_score_min": WEIGHTED_SCORE_MIN,
        "frame_accum_min": FRAME_ACCUMULATION_MIN,
        "dashboard_active": DASHBOARD_ACTIVE,
        "target_classes": list(TARGET_CLASSES),
        "all_classes": list(ALL_CLASSES),
        "use_zoom": USE_HUMAN_ZOOM,
        "use_sahi": USE_SAHI,
        "use_dip": USE_DIP,
        "use_wavelet": USE_WAVELET,
        "camera_index": CAMERA_INDEX
    }

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
    """Saves updated settings to runtime_settings.json for sharing across web and CLI."""
    try:
        current = load_runtime_settings()
        for k, v in updates.items():
            current[k] = v
        with open(RUNTIME_SETTINGS_FILE, "w") as f:
            json.dump(current, f, indent=2)
        return True
    except Exception as e:
        print(f"[Settings] Error saving runtime settings: {e}")
        return False

