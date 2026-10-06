import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths (fallback to root if not found in models/)
MODEL_WEAPON = os.path.join(BASE_DIR, "models", "best.pt")
if not os.path.exists(MODEL_WEAPON):
    MODEL_WEAPON = os.path.join(BASE_DIR, "best.pt")

MODEL_PERSON = os.path.join(BASE_DIR, "models", "yolov8n.pt")
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
