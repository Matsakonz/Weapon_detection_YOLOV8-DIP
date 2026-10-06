"""
Weapon Detection & Verification System (Backward Compatibility Interface)
Delegates to modular architecture in src/ and config/.
"""
import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_camera, run_image_comparison
from config.settings import CAMERA_INDEX, CONF_MIN, IMGSZ

# Backward-compatible function aliases
def open_camera(camera_index=CAMERA_INDEX, conf_threshold=CONF_MIN, imgsz=IMGSZ,
                use_human_zoom=True, use_dip=True, use_wavelet=True, **kwargs):
    run_camera(camera_index=camera_index, conf_threshold=conf_threshold, imgsz=imgsz,
               use_zoom=use_human_zoom, use_dip=use_dip, use_wavelet=use_wavelet)

def compare_all_images(conf_threshold=CONF_MIN, imgsz=IMGSZ, use_human_zoom=True, use_dip=True, use_wavelet=True, **kwargs):
    run_image_comparison(conf_threshold=conf_threshold, imgsz=imgsz,
                         use_zoom=use_human_zoom, use_dip=use_dip, use_wavelet=use_wavelet)

detect_from_camera = open_camera

if __name__ == "__main__":
    # Runs the full flowchart verification pipeline on camera
    open_camera(camera_index=0, conf_threshold=0.60)