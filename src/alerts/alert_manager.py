import os
import re
import json
import time
import cv2

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from config.settings import (
    BUZZER_ENABLED,
    BUZZER_SOUND_PATH,
    MQTT_ENABLED,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    RESULTS_DIR
)

class AlertManager:
    """
    Handles alert dispatching:
    1. Audio Buzzer alert (system sound)
    2. MQTT threat message publishing
    3. Incident snapshot logging
    """
    def __init__(self):
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Min seconds between audio alerts
        self.mqtt_client = None
        self._init_mqtt()

    def _init_mqtt(self):
        if not MQTT_ENABLED or not PAHO_AVAILABLE:
            return
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT} (Topic: {MQTT_TOPIC})")
        except Exception as e:
            print(f"[MQTT] Warning: Could not connect to broker ({e}). Alerts will run locally.")
            self.mqtt_client = None

    def trigger_buzzer(self):
        """Triggers local macOS buzzer / alert chime."""
        if not BUZZER_ENABLED:
            return
        curr = time.time()
        if curr - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = curr
            if os.path.exists(BUZZER_SOUND_PATH):
                os.system(f"afplay {BUZZER_SOUND_PATH} &")
            else:
                print("\a", end="", flush=True)  # Terminal bell fallback

    def publish_mqtt(self, threat_data):
        """Publishes structured JSON detection payload over MQTT."""
        if self.mqtt_client is not None:
            try:
                payload = json.dumps(threat_data)
                self.mqtt_client.publish(MQTT_TOPIC, payload)
            except Exception as e:
                print(f"[MQTT] Publish error: {e}")

    def save_incident_snapshot(self, frame, threat_label):
        """Saves timestamped incident evidence image to results/."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clean_label = re.sub(r'[^a-zA-Z0-9]', '_', threat_label)
        filename = f"ALERT_{clean_label}_{timestamp}.jpg"
        filepath = os.path.join(RESULTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"[ALERT] Snapshot saved to: {filepath}")
        return filepath

    def on_detection_confirm(self, frame, threat_label, score):
        """Dispatches buzzer, MQTT, and incident logging on confirmed threat."""
        self.trigger_buzzer()
        threat_info = {
            "event": "WEAPON_DETECTED_CONFIRMED",
            "threat": threat_label,
            "verification_score": round(float(score), 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.publish_mqtt(threat_info)
        return self.save_incident_snapshot(frame, threat_label)

    def close(self):
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
