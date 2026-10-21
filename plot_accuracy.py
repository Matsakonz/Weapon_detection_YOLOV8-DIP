#!/usr/bin/env python3
"""
Performance Accuracy Benchmark and Plotting Tool
------------------------------------------------
Evaluates weapon detection accuracy across different:
  1. Distances (3m, 6m, 9m, 12m)
  2. Weapon Occlusion / Cover Displacements (0cm, 5cm, 10cm, 15cm)
  3. Brightness Levels (100%, 75%, 50%, 25%)

Uses the same detection pipeline as web/app.py:
  - WeaponDetector (YOLOv8 best.pt + YOLOv8n person detector)
  - Human Zoom (Targeted ROI)
  - SAHI (Sliced Slicing Sliced Inference)
  - DIP Enhancement (CLAHE + Wavelet DWT edge boost + Unsharp)

Generates:
  1. 3D Graph (Z: Accuracy, X: Distance 3m-12m, Y: Weapon Cover 0cm-15cm)
  2. 2D Graph (Y: Accuracy, X: Distance 3m-12m) + 4 individual 2D graphs
     (Y: Accuracy, X: Weapon Cover 0cm-15cm for 3m, 6m, 9m, 12m)
  3. 3D Graph for Brightness (Z: Accuracy, X: Distance at 0cm cover, Y: Brightness 100%, 75%, 50%, 25%)
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    resolve_model_path,
    MODEL_PERSON,
    IMGSZ,
    CONF_MIN,
    TARGET_CLASSES,
    USE_HUMAN_ZOOM,
    USE_SAHI,
    USE_DIP,
    USE_WAVELET,
    load_runtime_settings
)
from src.detection.detector import WeaponDetector


def str2bool(v):
    """Helper to parse boolean command-line arguments (e.g. true/false, 1/0, yes/no)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected (true/false), got '{v}'")


def parse_arguments():
    runtime_cfg = load_runtime_settings() if callable(load_runtime_settings) else {}
    default_model = runtime_cfg.get("model_weapon", "best.pt")
    default_conf = runtime_cfg.get("conf_min", CONF_MIN)
    default_targets = runtime_cfg.get("target_classes", TARGET_CLASSES)

    parser = argparse.ArgumentParser(
        description="Benchmark weapon detection accuracy on the Performance dataset and generate 3D/2D graphs."
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "Performance"),
        help="Path to Performance directory containing distance folders (default: Performance)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save generated plots and benchmark JSON (default: results/performance_plots or results/performance_plots_pure_ai)"
    )
    parser.add_argument(
        "--model", type=str, default=default_model,
        help=f"Weapon detection model filename in models/ (default: {default_model})"
    )
    parser.add_argument(
        "--conf", type=float, default=default_conf,
        help=f"Confidence threshold for detection (default: {default_conf})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMGSZ,
        help=f"Inference image size (default: {IMGSZ})"
    )
    
    # Pure AI baseline shortcut
    parser.add_argument(
        "--pure-ai", "--raw-ai", action="store_true",
        help="Pure AI Mode: Disables Human Zoom, SAHI, DIP, and Wavelet (evaluates raw YOLO model performance only)"
    )

    # Boolean arguments (support true/false, e.g. --zoom false --dip false)
    parser.add_argument("--zoom", type=str2bool, nargs="?", const=True, default=None,
                        help="Enable/disable Human Zoom (true/false)")
    parser.add_argument("--sahi", type=str2bool, nargs="?", const=True, default=None,
                        help="Enable/disable SAHI Sliced Inference (true/false)")
    parser.add_argument("--dip", type=str2bool, nargs="?", const=True, default=None,
                        help="Enable/disable DIP Enhancement (true/false)")
    parser.add_argument("--wavelet", type=str2bool, nargs="?", const=True, default=None,
                        help="Enable/disable Wavelet DWT boost in DIP (true/false)")

    # Legacy flag arguments
    parser.add_argument(
        "--no-zoom", action="store_true", default=False,
        help="Disable Human Zoom ROI crop"
    )
    parser.add_argument(
        "--no-sahi", action="store_true", default=False,
        help="Disable SAHI sliced inference"
    )
    parser.add_argument(
        "--no-dip", action="store_true", default=False,
        help="Disable DIP enhancement"
    )
    parser.add_argument(
        "--no-wavelet", action="store_true", default=False,
        help="Disable Wavelet DWT boost in DIP"
    )

    parser.add_argument(
        "--target-classes", nargs="+", default=default_targets,
        help=f"Target threat classes to detect (default: {default_targets})"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-running AI detection even if cached benchmark results exist"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display interactive plot windows upon completion"
    )
    
    parsed = parser.parse_args()

    # Resolve final boolean flags
    if parsed.pure_ai:
        parsed.use_zoom = False
        parsed.use_sahi = False
        parsed.use_dip = False
        parsed.use_wavelet = False
    else:
        # Check explicit --zoom, --sahi, --dip, --wavelet or fall back to --no-* flags and runtime config
        cfg_zoom = runtime_cfg.get("use_zoom", USE_HUMAN_ZOOM)
        cfg_sahi = runtime_cfg.get("use_sahi", USE_SAHI)
        cfg_dip = runtime_cfg.get("use_dip", USE_DIP)
        cfg_wavelet = runtime_cfg.get("use_wavelet", USE_WAVELET)

        parsed.use_zoom = parsed.zoom if parsed.zoom is not None else (False if parsed.no_zoom else cfg_zoom)
        parsed.use_sahi = parsed.sahi if parsed.sahi is not None else (False if parsed.no_sahi else cfg_sahi)
        parsed.use_dip = parsed.dip if parsed.dip is not None else (False if parsed.no_dip else cfg_dip)
        parsed.use_wavelet = parsed.wavelet if parsed.wavelet is not None else (False if parsed.no_wavelet else cfg_wavelet)

    is_pure_ai = not (parsed.use_zoom or parsed.use_sahi or parsed.use_dip or parsed.use_wavelet)
    parsed.is_pure_ai = is_pure_ai
    if parsed.output_dir is None:
        folder_name = "performance_plots_pure_ai" if is_pure_ai else "performance_plots"
        parsed.output_dir = os.path.join(PROJECT_ROOT, "results", folder_name)

    return parsed


def load_image_paths(data_dir):
    """
    Scans the Performance folder and returns structured dictionary:
    data[distance][cover] = list of 5 image paths
    """
    distances = ["3 meter", "6 meter", "9 meter", "12 meter"]
    covers = ["0 cm", "5 cm", "10 cm", "15 cm"]
    
    dataset = {}
    valid_exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")

    for dist in distances:
        dataset[dist] = {}
        for cov in covers:
            folder_path = os.path.join(data_dir, dist, cov)
            if not os.path.isdir(folder_path):
                print(f"[Warning] Folder missing: {folder_path}")
                dataset[dist][cov] = []
                continue
            
            files = []
            for ext in valid_exts:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
            files = sorted(list(set(files)))
            dataset[dist][cov] = files

    return dataset


def evaluate_single_image(detector, img_path, conf_threshold, imgsz,
                          use_zoom, use_sahi, target_classes,
                          use_dip, use_wavelet, brightness_alpha=1.0):
    """
    Loads an image, optionally scales brightness, and runs detector.detect().
    Returns (detected_bool, max_conf, candidate_count, elapsed_sec)
    """
    t0 = time.time()
    frame = cv2.imread(img_path)
    if frame is None:
        return False, 0.0, 0, 0.0

    if brightness_alpha != 1.0:
        # Scale brightness: alpha in [0.25, 0.50, 0.75, 1.0]
        frame = cv2.convertScaleAbs(frame, alpha=brightness_alpha, beta=0)

    raw_cands, _ = detector.detect(
        frame,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        use_zoom=use_zoom,
        use_sahi=use_sahi,
        target_classes=target_classes,
        use_dip=use_dip,
        use_wavelet=use_wavelet
    )
    elapsed = time.time() - t0

    if raw_cands and len(raw_cands) > 0:
        confs = [c['conf'] for c in raw_cands]
        return True, float(max(confs)), len(raw_cands), elapsed
    return False, 0.0, 0, elapsed


def check_cache_validity(cache_file, args):
    """
    Returns (is_valid, cached_data). Checks if cache exists and matches current benchmark parameters.
    """
    if args.no_cache or not os.path.exists(cache_file):
        return False, None
    try:
        with open(cache_file, "r") as f:
            cached = json.load(f)
        c_meta = cached.get("metadata", {})
        if (c_meta.get("model") == args.model and
            abs(c_meta.get("conf", 0) - args.conf) < 1e-4 and
            c_meta.get("use_zoom") == args.use_zoom and
            c_meta.get("use_sahi") == args.use_sahi and
            c_meta.get("use_dip") == args.use_dip and
            c_meta.get("use_wavelet") == args.use_wavelet):
            return True, cached
    except Exception as e:
        print(f"[Warning] Failed reading cache {cache_file}: {e}")
    return False, None


def run_benchmark(dataset, detector, args, cache_file):
    """
    Executes the complete benchmark across:
      1. Baseline distance x cover grid (80 images)
      2. Brightness variation on 0cm cover (4 distances x 4 brightness levels x 5 images)
    Returns structured results dict.
    """
    is_valid, cached = check_cache_validity(cache_file, args)
    if is_valid:
        print(f"[Benchmark] Loaded cached results from {cache_file}")
        return cached

    # If cache not used or invalid, guarantee detector is initialized
    if detector is None:
        model_path = resolve_model_path(args.model)
        person_path = MODEL_PERSON if args.use_zoom else None
        print(f"[Init] Initializing WeaponDetector with {model_path} (Person detector: {'Enabled' if person_path else 'Disabled'})...")
        detector = WeaponDetector(model_path, person_path)

    is_pure_ai = not (args.use_zoom or args.use_sahi or args.use_dip or args.use_wavelet)
    mode_str = "PURE AI (Raw YOLO Inference)" if is_pure_ai else "CUSTOM / HYBRID PIPELINE"
    if args.use_zoom and args.use_sahi and args.use_dip and args.use_wavelet:
        mode_str = "FULL APP.PY PIPELINE (Zoom + SAHI + Wavelet DIP)"

    print("\n" + "=" * 70)
    print(" 🚀 RUNNING ACCURACY BENCHMARK ON PERFORMANCE DATASET")
    print(f" Mode: {mode_str}")
    print(f" Model: {args.model} | Conf: {args.conf:.2f} | Imgsz: {args.imgsz}")
    print(f" Zoom: {args.use_zoom} | SAHI: {args.use_sahi} | DIP: {args.use_dip} | Wavelet: {args.use_wavelet}")
    print("=" * 70)

    distances = ["3 meter", "6 meter", "9 meter", "12 meter"]
    covers = ["0 cm", "5 cm", "10 cm", "15 cm"]
    brightness_levels = [100, 75, 50, 25]

    results = {
        "metadata": {
            "model": args.model,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "use_zoom": args.use_zoom,
            "use_sahi": args.use_sahi,
            "use_dip": args.use_dip,
            "use_wavelet": args.use_wavelet,
            "pure_ai_mode": is_pure_ai,
            "target_classes": args.target_classes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "distance_cover": {},
        "brightness": {}
    }

    # 1. Evaluate standard distance x cover (100% brightness)
    print("\n[Step 1/2] Evaluating Distance vs Weapon Occlusion (80 images)...")
    for dist in distances:
        results["distance_cover"][dist] = {}
        for cov in covers:
            imgs = dataset.get(dist, {}).get(cov, [])
            total_pics = len(imgs)
            detected_pics = 0
            conf_list = []
            img_records = []

            for i, p in enumerate(imgs):
                det, conf, cnt, el = evaluate_single_image(
                    detector, p, args.conf, args.imgsz,
                    args.use_zoom, args.use_sahi, args.target_classes,
                    args.use_dip, args.use_wavelet, brightness_alpha=1.0
                )
                if det:
                    detected_pics += 1
                    conf_list.append(conf)
                img_records.append({
                    "file": os.path.basename(p),
                    "detected": det,
                    "conf": conf,
                    "count": cnt,
                    "time_sec": round(el, 3)
                })

            acc_pct = (detected_pics / total_pics * 100.0) if total_pics > 0 else 0.0
            avg_conf = (sum(conf_list) / len(conf_list)) if conf_list else 0.0

            results["distance_cover"][dist][cov] = {
                "detected": detected_pics,
                "total": total_pics,
                "accuracy_pct": round(acc_pct, 1),
                "avg_conf": round(avg_conf, 3),
                "images": img_records
            }
            print(f"  • {dist:>8} | {cov:>5} -> Detected: {detected_pics}/{total_pics} ({acc_pct:5.1f}%) | Avg Conf: {avg_conf:.3f}")

    # 2. Evaluate Brightness on 0 cm cover images
    print("\n[Step 2/2] Evaluating Brightness Levels (100%, 75%, 50%, 25%) on 0 cm cover...")
    for b in brightness_levels:
        b_key = f"{b}%"
        results["brightness"][b_key] = {}
        alpha = b / 100.0

        for dist in distances:
            # If 100%, reuse from Step 1
            if b == 100 and dist in results["distance_cover"] and "0 cm" in results["distance_cover"][dist]:
                step1_res = results["distance_cover"][dist]["0 cm"]
                results["brightness"][b_key][dist] = {
                    "detected": step1_res["detected"],
                    "total": step1_res["total"],
                    "accuracy_pct": step1_res["accuracy_pct"],
                    "avg_conf": step1_res["avg_conf"]
                }
                print(f"  • Brightness {b_key:>4} | {dist:>8} -> Detected: {step1_res['detected']}/{step1_res['total']} ({step1_res['accuracy_pct']:5.1f}%) [Reused]")
                continue

            imgs = dataset.get(dist, {}).get("0 cm", [])
            total_pics = len(imgs)
            detected_pics = 0
            conf_list = []

            for p in imgs:
                det, conf, cnt, el = evaluate_single_image(
                    detector, p, args.conf, args.imgsz,
                    args.use_zoom, args.use_sahi, args.target_classes,
                    args.use_dip, args.use_wavelet, brightness_alpha=alpha
                )
                if det:
                    detected_pics += 1
                    conf_list.append(conf)

            acc_pct = (detected_pics / total_pics * 100.0) if total_pics > 0 else 0.0
            avg_conf = (sum(conf_list) / len(conf_list)) if conf_list else 0.0

            results["brightness"][b_key][dist] = {
                "detected": detected_pics,
                "total": total_pics,
                "accuracy_pct": round(acc_pct, 1),
                "avg_conf": round(avg_conf, 3)
            }
            print(f"  • Brightness {b_key:>4} | {dist:>8} -> Detected: {detected_pics}/{total_pics} ({acc_pct:5.1f}%) | Avg Conf: {avg_conf:.3f}")

    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved benchmark data cache: {cache_file}")

    return results


# =========================================================================
# PLOTTING FUNCTIONS
# =========================================================================

def set_plot_style():
    """Applies modern clean typography and styling for all figures."""
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6


def plot_3d_accuracy_distance_cover(benchmark_data, output_dir):
    """
    1. 3D Graph:
       - Z-axis: Accuracy (how many times detected in 5 pics / %)
       - X-axis: Distance / Displacement (3m, 6m, 9m, 12m)
       - Y-axis: Space covering weapon (0cm, 5cm, 10cm, 15cm)
    Generates both:
      A. 3D Surface / Mesh plot with colormap and data points
      B. 3D Bar plot showing discrete bins
    """
    dist_labels = ["3m", "6m", "9m", "12m"]
    dist_keys = ["3 meter", "6 meter", "9 meter", "12 meter"]
    dist_vals = [3, 6, 9, 12]

    cover_labels = ["0cm", "5cm", "10cm", "15cm"]
    cover_keys = ["0 cm", "5 cm", "10 cm", "15 cm"]
    cover_vals = [0, 5, 10, 15]

    # Build Z grid: dimensions (len(cover), len(dist))
    # X corresponds to distance, Y corresponds to cover
    Z_count = np.zeros((len(cover_vals), len(dist_vals)))
    Z_pct = np.zeros((len(cover_vals), len(dist_vals)))

    dc_data = benchmark_data["distance_cover"]
    for j, d_key in enumerate(dist_keys):
        for i, c_key in enumerate(cover_keys):
            cell = dc_data.get(d_key, {}).get(c_key, {})
            det = cell.get("detected", 0)
            pct = cell.get("accuracy_pct", 0.0)
            Z_count[i, j] = det
            Z_pct[i, j] = pct

    X, Y = np.meshgrid(dist_vals, cover_vals)

    # ----------------- FIGURE 1A: 3D Surface Plot -----------------
    fig = plt.figure(figsize=(12, 8.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z_count,
        cmap=cm.viridis,
        edgecolor="#1f2d3d",
        linewidth=0.8,
        antialiased=True,
        alpha=0.85
    )

    # Overlay scatter points and text annotations on surface
    for i in range(len(cover_vals)):
        for j in range(len(dist_vals)):
            c_val = Z_count[i, j]
            p_val = Z_pct[i, j]
            ax.scatter([dist_vals[j]], [cover_vals[i]], [c_val],
                       # pyrefly: ignore[bad-keyword-argument]
                       color="#e74c3c", s=45, edgecolors="white", linewidth=1.2, zorder=10)
            ax.text(dist_vals[j], cover_vals[i], c_val + 0.15,
                    f"{int(c_val)}/5\n({p_val:.0f}%)",
                    fontsize=8.5, fontweight="bold", ha="center", va="bottom",
                    color="#1a252f")

    # Axes configuration
    ax.set_title("3D Weapon Detection Accuracy\nDistance vs. Occlusion Space",
                 fontsize=15, fontweight="bold", pad=20, color="#1a252f")
    ax.set_xlabel("Distance / Displacement (meters)", fontsize=11, fontweight="bold", labelpad=12)
    ax.set_ylabel("Space Covering Weapon (cm)", fontsize=11, fontweight="bold", labelpad=12)
    ax.set_zlabel("Accuracy (Detections out of 5 pics)", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_xticks(dist_vals)
    ax.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(cover_vals)
    ax.set_yticklabels(cover_labels, fontsize=10, fontweight="bold")
    ax.set_zlim(0, 5.5)
    ax.set_zticks([0, 1, 2, 3, 4, 5])
    ax.set_zticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=9)

    ax.view_init(elev=26, azim=-55)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
    cbar.set_label("Detection Count (out of 5)", fontsize=10, fontweight="bold")
    cbar.set_ticks([0, 1, 2, 3, 4, 5])

    plt.tight_layout()
    out_path_surf = os.path.join(output_dir, "1_3d_accuracy_surface.png")
    plt.savefig(out_path_surf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved 3D Surface Plot: {out_path_surf}")

    # ----------------- FIGURE 1B: 3D Bar Plot -----------------
    fig_bar = plt.figure(figsize=(12, 8.5), dpi=300)
    ax_bar = fig_bar.add_subplot(111, projection="3d")

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_bottom = np.zeros_like(x_flat)
    dz_flat = Z_count.ravel()

    dx = 1.0
    dy = 1.5

    # Color map mapping
    norm = plt.Normalize(0, 5)
    colors = cm.viridis(norm(dz_flat))

    ax_bar.bar3d(x_flat - dx/2, y_flat - dy/2, z_bottom, dx, dy, dz_flat,
                 color=colors, edgecolor="#2c3e50", shade=True, alpha=0.9)

    for k in range(len(x_flat)):
        if dz_flat[k] > 0:
            pct_k = Z_pct.ravel()[k]
            ax_bar.text(x_flat[k], y_flat[k], dz_flat[k] + 0.15,
                        f"{int(dz_flat[k])}/5\n({pct_k:.0f}%)",
                        fontsize=8, fontweight="bold", ha="center", va="bottom", color="#111111")

    ax_bar.set_title("3D Weapon Detection Accuracy (Bar View)\nDistance vs. Occlusion Space",
                     fontsize=15, fontweight="bold", pad=20, color="#1a252f")
    ax_bar.set_xlabel("Distance / Displacement (meters)", fontsize=11, fontweight="bold", labelpad=12)
    ax_bar.set_ylabel("Space Covering Weapon (cm)", fontsize=11, fontweight="bold", labelpad=12)
    ax_bar.set_zlabel("Accuracy (Detections out of 5 pics)", fontsize=11, fontweight="bold", labelpad=10)

    ax_bar.set_xticks(dist_vals)
    ax_bar.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax_bar.set_yticks(cover_vals)
    ax_bar.set_yticklabels(cover_labels, fontsize=10, fontweight="bold")
    ax_bar.set_zlim(0, 5.5)
    ax_bar.set_zticks([0, 1, 2, 3, 4, 5])
    ax_bar.set_zticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=9)

    ax_bar.view_init(elev=28, azim=-50)

    plt.tight_layout()
    out_path_bars = os.path.join(output_dir, "1_3d_accuracy_bars.png")
    plt.savefig(out_path_bars, dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"✓ Saved 3D Bar Plot: {out_path_bars}")


def plot_2d_accuracy_graphs(benchmark_data, output_dir):
    """
    2. 2D Graphs:
       A. Main 2D Graph:
          Y-axis: Accuracy (detections out of 5 pics / %)
          X-axis: Distance / Displacement (3m, 6m, 9m, 12m)
          Shows separate curves for 0cm, 5cm, 10cm, 15cm cover + Average curve.
       B. 4 Individual 2D Graphs (one for each distance: 3m, 6m, 9m, 12m):
          Y-axis: Accuracy (detections out of 5 pics / %)
          X-axis: Space covering weapon (0cm, 5cm, 10cm, 15cm)
       C. Combined 2x2 comparison grid of the 4 individual distance graphs.
    """
    dist_labels = ["3m", "6m", "9m", "12m"]
    dist_keys = ["3 meter", "6 meter", "9 meter", "12 meter"]
    dist_vals = [3, 6, 9, 12]

    cover_labels = ["0cm", "5cm", "10cm", "15cm"]
    cover_keys = ["0 cm", "5 cm", "10 cm", "15 cm"]
    cover_vals = [0, 5, 10, 15]

    cover_colors = {
        "0 cm": "#27ae60",   # Green (no cover - baseline)
        "5 cm": "#2980b9",   # Blue
        "10 cm": "#f39c12",  # Orange
        "15 cm": "#c0392b"   # Red (heavy cover)
    }
    cover_markers = {
        "0 cm": "o",
        "5 cm": "s",
        "10 cm": "^",
        "15 cm": "D"
    }

    dc_data = benchmark_data["distance_cover"]

    # ----------------- PART A: Main 2D Graph (Accuracy vs Distance) -----------------
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Plot line for each cover displacement
    avg_per_dist = []
    for c_key in cover_keys:
        acc_counts = [dc_data[d_key][c_key]["detected"] for d_key in dist_keys]
        acc_pcts = [dc_data[d_key][c_key]["accuracy_pct"] for d_key in dist_keys]

        line, = ax.plot(
            dist_vals, acc_counts,
            label=f"Cover {c_key}",
            color=cover_colors[c_key],
            marker=cover_markers[c_key],
            markersize=8,
            linewidth=2.5
        )

        # Annotate points
        for x, y, p in zip(dist_vals, acc_counts, acc_pcts):
            ax.annotate(
                f"{y}/5\n({p:.0f}%)",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color=cover_colors[c_key]
            )

    # Calculate and plot overall Average line
    for d_key in dist_keys:
        total_d = sum([dc_data[d_key][c]["detected"] for c in cover_keys])
        avg_per_dist.append(total_d / len(cover_keys))

    ax.plot(
        dist_vals, avg_per_dist,
        label="Average Across Covers",
        color="#34495e",
        linestyle="--",
        marker="*",
        markersize=11,
        linewidth=2.0
    )

    ax.set_title("Detection Accuracy vs. Distance / Displacement\n(Grouped by Weapon Occlusion Space)",
                 fontsize=14, fontweight="bold", pad=15, color="#1a252f")
    ax.set_xlabel("Displacement / Distance", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Accuracy (Detections out of 5 pics)", fontsize=12, fontweight="bold", labelpad=8)

    ax.set_xticks(dist_vals)
    ax.set_xticklabels(dist_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.3, 5.7)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=10)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    out_path_2d_main = os.path.join(output_dir, "2_2d_accuracy_vs_distance.png")
    plt.savefig(out_path_2d_main, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved 2D Accuracy vs Distance Graph: {out_path_2d_main}")

    # ----------------- PART B: 4 Individual 2D Graphs (Accuracy vs Cover for each Distance) -----------------
    dist_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]

    for idx, (d_key, d_lbl) in enumerate(zip(dist_keys, dist_labels)):
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5.5), dpi=300)

        counts = [dc_data[d_key][c_key]["detected"] for c_key in cover_keys]
        pcts = [dc_data[d_key][c_key]["accuracy_pct"] for c_key in cover_keys]
        color = dist_colors[idx]

        # Bar + Line combo
        bars = ax_ind.bar(
            cover_vals, counts,
            width=2.5,
            color=color,
            alpha=0.4,
            edgecolor=color,
            linewidth=1.5,
            label="Detection Count"
        )
        ax_ind.plot(
            cover_vals, counts,
            color=color,
            marker="o",
            markersize=9,
            linewidth=2.8,
            label="Trend"
        )

        for x, y, p in zip(cover_vals, counts, pcts):
            ax_ind.annotate(
                f"{y}/5 ({p:.0f}%)",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color="#1a252f"
            )

        ax_ind.set_title(f"Weapon Detection Accuracy at Distance: {d_lbl}\nEffect of Weapon Occlusion Cover",
                         fontsize=13, fontweight="bold", pad=12, color="#1a252f")
        ax_ind.set_xlabel("Space Covering the Weapon (cm)", fontsize=11, fontweight="bold", labelpad=8)
        ax_ind.set_ylabel("Accuracy (Detections out of 5 pics)", fontsize=11, fontweight="bold", labelpad=8)

        ax_ind.set_xticks(cover_vals)
        ax_ind.set_xticklabels(cover_labels, fontsize=10, fontweight="bold")
        ax_ind.set_ylim(0, 5.7)
        ax_ind.set_yticks([0, 1, 2, 3, 4, 5])
        ax_ind.set_yticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=9)

        ax_ind.grid(True, linestyle="--", alpha=0.5)
        ax_ind.legend(loc="upper right", fontsize=9.5)

        plt.tight_layout()
        clean_name = d_lbl.replace(" ", "")
        out_ind = os.path.join(output_dir, f"2_2d_accuracy_vs_cover_{clean_name}.png")
        plt.savefig(out_ind, dpi=300, bbox_inches="tight")
        plt.close(fig_ind)
        print(f"✓ Saved 2D Graph for {d_lbl}: {out_ind}")

    # ----------------- PART C: Combined 2x2 Grid of the 4 Distance Graphs -----------------
    fig_grid, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes_flat = axes.flatten()

    for idx, (d_key, d_lbl) in enumerate(zip(dist_keys, dist_labels)):
        ax_g = axes_flat[idx]
        counts = [dc_data[d_key][c_key]["detected"] for c_key in cover_keys]
        pcts = [dc_data[d_key][c_key]["accuracy_pct"] for c_key in cover_keys]
        color = dist_colors[idx]

        ax_g.bar(cover_vals, counts, width=2.5, color=color, alpha=0.35, edgecolor=color, linewidth=1.5)
        ax_g.plot(cover_vals, counts, color=color, marker="o", markersize=8, linewidth=2.5)

        for x, y, p in zip(cover_vals, counts, pcts):
            ax_g.annotate(
                f"{y}/5 ({p:.0f}%)",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1a252f"
            )

        ax_g.set_title(f"Distance: {d_lbl}", fontsize=12, fontweight="bold", pad=8, color="#1a252f")
        ax_g.set_xlabel("Space Covering Weapon", fontsize=10, fontweight="bold")
        ax_g.set_ylabel("Accuracy (Detections / 5)", fontsize=10, fontweight="bold")
        ax_g.set_xticks(cover_vals)
        ax_g.set_xticklabels(cover_labels, fontsize=9.5, fontweight="bold")
        ax_g.set_ylim(0, 5.7)
        ax_g.set_yticks([0, 1, 2, 3, 4, 5])
        ax_g.set_yticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=8.5)
        ax_g.grid(True, linestyle="--", alpha=0.5)

    fig_grid.suptitle("Accuracy vs. Space Covering Weapon Across All Distances (3m, 6m, 9m, 12m)",
                      fontsize=15, fontweight="bold", y=0.995, color="#1a252f")
    plt.tight_layout()
    out_grid = os.path.join(output_dir, "2_2d_accuracy_vs_cover_all_distances.png")
    plt.savefig(out_grid, dpi=300, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"✓ Saved Combined 2x2 Grid Graph: {out_grid}")


def plot_3d_accuracy_brightness(benchmark_data, output_dir):
    """
    3. 3D Graph for Brightness:
       - Brightness Percentages: 100%, 75%, 50%, 25%
       - Z-axis: Accuracy (detections out of 5 pics / %)
       - X-axis: Distance / Displacement (3m, 6m, 9m, 12m) using ONLY 0cm cover images
       - Y-axis: Brightness (100%, 75%, 50%, 25%)
    Generates:
      A. 3D Surface / Mesh plot of Accuracy
      B. 3D Bar plot of Accuracy
      C. 3D Surface plot of Average Confidence (shows nuanced degradation under dim lighting)
      D. 2D Dual-Panel Companion Plot (Left: Detection Count, Right: Confidence Score)
    """
    dist_labels = ["3m", "6m", "9m", "12m"]
    dist_keys = ["3 meter", "6 meter", "9 meter", "12 meter"]
    dist_vals = [3, 6, 9, 12]

    bright_labels = ["25%", "50%", "75%", "100%"]
    bright_keys = ["25%", "50%", "75%", "100%"]
    bright_vals = [25, 50, 75, 100]

    b_data = benchmark_data["brightness"]

    # Build Z grids
    Z_count = np.zeros((len(bright_vals), len(dist_vals)))
    Z_pct = np.zeros((len(bright_vals), len(dist_vals)))
    Z_conf = np.zeros((len(bright_vals), len(dist_vals)))

    for i, b_key in enumerate(bright_keys):
        for j, d_key in enumerate(dist_keys):
            cell = b_data.get(b_key, {}).get(d_key, {})
            det = cell.get("detected", 0)
            pct = cell.get("accuracy_pct", 0.0)
            conf = cell.get("avg_conf", 0.0)
            Z_count[i, j] = det
            Z_pct[i, j] = pct
            Z_conf[i, j] = conf

    X, Y = np.meshgrid(dist_vals, bright_vals)

    # ----------------- FIGURE 3A: 3D Surface Plot (Accuracy) -----------------
    fig = plt.figure(figsize=(12, 8.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z_count,
        cmap=cm.plasma,
        edgecolor="#1f2d3d",
        linewidth=0.8,
        antialiased=True,
        alpha=0.85
    )

    for i in range(len(bright_vals)):
        for j in range(len(dist_vals)):
            c_val = Z_count[i, j]
            p_val = Z_pct[i, j]
            ax.scatter([dist_vals[j]], [bright_vals[i]], [c_val],
                       # pyrefly: ignore[bad-keyword-argument]
                       color="#2ecc71", s=45, edgecolors="white", linewidth=1.2, zorder=10)
            ax.text(dist_vals[j], bright_vals[i], c_val + 0.15,
                    f"{int(c_val)}/5\n({p_val:.0f}%)",
                    fontsize=8.5, fontweight="bold", ha="center", va="bottom",
                    color="#1a252f")

    ax.set_title("3D Weapon Detection Accuracy Under Variable Brightness\n(Evaluated on Uncovered Weapon - 0 cm Space)",
                 fontsize=14, fontweight="bold", pad=20, color="#1a252f")
    ax.set_xlabel("Displacement / Distance (meters)", fontsize=11, fontweight="bold", labelpad=12)
    ax.set_ylabel("Lighting Brightness Level (%)", fontsize=11, fontweight="bold", labelpad=12)
    ax.set_zlabel("Accuracy (Detections out of 5 pics)", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_xticks(dist_vals)
    ax.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(bright_vals)
    ax.set_yticklabels(bright_labels, fontsize=10, fontweight="bold")
    ax.set_zlim(0, 5.5)
    ax.set_zticks([0, 1, 2, 3, 4, 5])
    ax.set_zticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=9)

    ax.view_init(elev=26, azim=-55)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
    cbar.set_label("Detection Count (out of 5)", fontsize=10, fontweight="bold")
    cbar.set_ticks([0, 1, 2, 3, 4, 5])

    fig.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08)
    out_path_3d_bright = os.path.join(output_dir, "3_3d_accuracy_brightness_surface.png")
    plt.savefig(out_path_3d_bright, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved 3D Brightness Surface Plot: {out_path_3d_bright}")

    # ----------------- FIGURE 3B: 3D Bar Plot for Brightness -----------------
    fig_bar = plt.figure(figsize=(12, 8.5), dpi=300)
    ax_bar = fig_bar.add_subplot(111, projection="3d")

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_bottom = np.zeros_like(x_flat)
    dz_flat = Z_count.ravel()

    dx = 1.0
    dy = 8.0

    norm = plt.Normalize(0, 5)
    colors = cm.plasma(norm(dz_flat))

    ax_bar.bar3d(x_flat - dx/2, y_flat - dy/2, z_bottom, dx, dy, dz_flat,
                 color=colors, edgecolor="#2c3e50", shade=True, alpha=0.9)

    for k in range(len(x_flat)):
        if dz_flat[k] > 0:
            pct_k = Z_pct.ravel()[k]
            ax_bar.text(x_flat[k], y_flat[k], dz_flat[k] + 0.15,
                        f"{int(dz_flat[k])}/5\n({pct_k:.0f}%)",
                        fontsize=8, fontweight="bold", ha="center", va="bottom", color="#111111")

    ax_bar.set_title("3D Weapon Detection Accuracy Under Variable Brightness (Bar View)\n(Evaluated on Uncovered Weapon - 0 cm Space)",
                     fontsize=14, fontweight="bold", pad=20, color="#1a252f")
    ax_bar.set_xlabel("Displacement / Distance (meters)", fontsize=11, fontweight="bold", labelpad=12)
    ax_bar.set_ylabel("Lighting Brightness Level (%)", fontsize=11, fontweight="bold", labelpad=12)
    ax_bar.set_zlabel("Accuracy (Detections out of 5 pics)", fontsize=11, fontweight="bold", labelpad=10)

    ax_bar.set_xticks(dist_vals)
    ax_bar.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax_bar.set_yticks(bright_vals)
    ax_bar.set_yticklabels(bright_labels, fontsize=10, fontweight="bold")
    ax_bar.set_zlim(0, 5.5)
    ax_bar.set_zticks([0, 1, 2, 3, 4, 5])
    ax_bar.set_zticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=9)

    ax_bar.view_init(elev=28, azim=-50)

    fig_bar.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08)
    out_path_3d_bars = os.path.join(output_dir, "3_3d_accuracy_brightness_bars.png")
    plt.savefig(out_path_3d_bars, dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"✓ Saved 3D Brightness Bar Plot: {out_path_3d_bars}")

    # ----------------- FIGURE 3C: 3D Surface Plot (Confidence Degradation) -----------------
    fig_conf = plt.figure(figsize=(12, 8.5), dpi=300)
    ax_c = fig_conf.add_subplot(111, projection="3d")

    surf_c = ax_c.plot_surface(
        X, Y, Z_conf,
        cmap=cm.cividis,
        edgecolor="#1f2d3d",
        linewidth=0.8,
        antialiased=True,
        alpha=0.85
    )

    for i in range(len(bright_vals)):
        for j in range(len(dist_vals)):
            cf_val = Z_conf[i, j]
            ax_c.scatter([dist_vals[j]], [bright_vals[i]], [cf_val],
                         # pyrefly: ignore[bad-keyword-argument]
                         color="#e67e22", s=45, edgecolors="white", linewidth=1.2, zorder=10)
            ax_c.text(dist_vals[j], bright_vals[i], cf_val + 0.015,
                      f"{cf_val:.3f}",
                      fontsize=8.5, fontweight="bold", ha="center", va="bottom",
                      color="#1a252f")

    ax_c.set_title("3D AI Detection Confidence Under Variable Brightness & Distance\n(Continuous Degradation Profile - 0 cm Space)",
                   fontsize=14, fontweight="bold", pad=20, color="#1a252f")
    ax_c.set_xlabel("Displacement / Distance (meters)", fontsize=11, fontweight="bold", labelpad=12)
    ax_c.set_ylabel("Lighting Brightness Level (%)", fontsize=11, fontweight="bold", labelpad=12)
    ax_c.set_zlabel("Average Confidence Score (0.0 - 1.0)", fontsize=11, fontweight="bold", labelpad=10)

    ax_c.set_xticks(dist_vals)
    ax_c.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax_c.set_yticks(bright_vals)
    ax_c.set_yticklabels(bright_labels, fontsize=10, fontweight="bold")
    ax_c.set_zlim(0.5, 1.0)

    ax_c.view_init(elev=26, azim=-55)

    cbar_c = fig_conf.colorbar(surf_c, ax=ax_c, shrink=0.55, aspect=14, pad=0.08)
    cbar_c.set_label("AI Confidence Score", fontsize=10, fontweight="bold")

    fig_conf.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08)
    out_path_3d_conf = os.path.join(output_dir, "3_3d_confidence_brightness_surface.png")
    plt.savefig(out_path_3d_conf, dpi=300, bbox_inches="tight")
    plt.close(fig_conf)
    print(f"✓ Saved 3D Brightness Confidence Surface Plot: {out_path_3d_conf}")

    # ----------------- FIGURE 3D: 2D Dual-Panel Companion Plot -----------------
    fig_2d, (ax_acc, ax_cnf) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    b_colors = {
        "100%": "#f39c12",
        "75%": "#e67e22",
        "50%": "#3498db",
        "25%": "#9b59b6"
    }
    b_markers = {
        "100%": "o",
        "75%": "s",
        "50%": "^",
        "25%": "D"
    }

    # Left Panel: Accuracy Rate (with slight visual offset so all 4 lines are clearly visible)
    for idx_b, b_key in enumerate(["100%", "75%", "50%", "25%"]):
        counts = [b_data[b_key][d_key]["detected"] for d_key in dist_keys]
        # Tiny visual vertical offset so overlapping 5/5 lines don't eclipse each other
        offset_y = (idx_b - 1.5) * 0.04
        vis_counts = [c + offset_y for c in counts]

        ax_acc.plot(
            dist_vals, vis_counts,
            label=f"Brightness {b_key}",
            color=b_colors[b_key],
            marker=b_markers[b_key],
            markersize=7.5,
            linewidth=2.2
        )

    ax_acc.set_title("Detection Accuracy (Rate)\nAcross Distances & Brightness",
                     fontsize=12, fontweight="bold", pad=10, color="#1a252f")
    ax_acc.set_xlabel("Displacement / Distance", fontsize=11, fontweight="bold", labelpad=8)
    ax_acc.set_ylabel("Accuracy (Detections out of 5 pics)", fontsize=11, fontweight="bold", labelpad=8)
    ax_acc.set_xticks(dist_vals)
    ax_acc.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax_acc.set_ylim(-0.2, 5.6)
    ax_acc.set_yticks([0, 1, 2, 3, 4, 5])
    ax_acc.set_yticklabels(["0 (0%)", "1 (20%)", "2 (40%)", "3 (60%)", "4 (80%)", "5 (100%)"], fontsize=9.5)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    # Right Panel: Confidence Score Degradation
    for b_key in ["100%", "75%", "50%", "25%"]:
        confs = [b_data[b_key][d_key]["avg_conf"] for d_key in dist_keys]

        ax_cnf.plot(
            dist_vals, confs,
            label=f"Brightness {b_key}",
            color=b_colors[b_key],
            marker=b_markers[b_key],
            markersize=7.5,
            linewidth=2.2
        )

        for x, c in zip(dist_vals, confs):
            ax_cnf.annotate(
                f"{c:.2f}",
                (x, c),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=b_colors[b_key]
            )

    ax_cnf.set_title("Average AI Detection Confidence Score\n(Shows subtle drop at lower lighting)",
                     fontsize=12, fontweight="bold", pad=10, color="#1a252f")
    ax_cnf.set_xlabel("Displacement / Distance", fontsize=11, fontweight="bold", labelpad=8)
    ax_cnf.set_ylabel("Average Confidence Score", fontsize=11, fontweight="bold", labelpad=8)
    ax_cnf.set_xticks(dist_vals)
    ax_cnf.set_xticklabels(dist_labels, fontsize=10, fontweight="bold")
    ax_cnf.set_ylim(0.5, 1.0)
    ax_cnf.grid(True, linestyle="--", alpha=0.5)
    ax_cnf.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    fig_2d.suptitle("Weapon Detection Performance Under Variable Brightness (0 cm Space)",
                    fontsize=14, fontweight="bold", y=0.99, color="#1a252f")
    plt.tight_layout()
    out_path_2d_bright = os.path.join(output_dir, "3_2d_accuracy_vs_brightness.png")
    plt.savefig(out_path_2d_bright, dpi=300, bbox_inches="tight")
    plt.close(fig_2d)
    print(f"✓ Saved 2D Brightness Dual-Panel Plot: {out_path_2d_bright}")


def print_summary_table(benchmark_data):
    """Prints a nicely formatted terminal table summarizing detection accuracy."""
    print("\n" + "=" * 78)
    print(" 📊 BENCHMARK ACCURACY SUMMARY REPORT")
    print("=" * 78)

    print("\n[Part 1] Accuracy Matrix (Detections out of 5 pics & %):")
    header = f"{'Distance':<12} | {'0 cm Cover':<14} | {'5 cm Cover':<14} | {'10 cm Cover':<14} | {'15 cm Cover':<14}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    dc = benchmark_data["distance_cover"]
    for dist in ["3 meter", "6 meter", "9 meter", "12 meter"]:
        row = f"{dist:<12} | "
        for cov in ["0 cm", "5 cm", "10 cm", "15 cm"]:
            cell = dc.get(dist, {}).get(cov, {})
            det = cell.get("detected", 0)
            pct = cell.get("accuracy_pct", 0.0)
            row += f"{det}/5 ({pct:4.0f}%)    | "
        print(row[:-2])
    print("-" * len(header))

    print("\n[Part 2] Brightness Accuracy Matrix (0 cm Cover):")
    b_header = f"{'Distance':<12} | {'100% Bright':<14} | {'75% Bright':<14} | {'50% Bright':<14} | {'25% Bright':<14}"
    print("-" * len(b_header))
    print(b_header)
    print("-" * len(b_header))

    b_data = benchmark_data["brightness"]
    for dist in ["3 meter", "6 meter", "9 meter", "12 meter"]:
        row = f"{dist:<12} | "
        for b in ["100%", "75%", "50%", "25%"]:
            cell = b_data.get(b, {}).get(dist, {})
            det = cell.get("detected", 0)
            pct = cell.get("accuracy_pct", 0.0)
            row += f"{det}/5 ({pct:4.0f}%)    | "
        print(row[:-2])
    print("-" * len(b_header))
    print("=" * 78 + "\n")


def main():
    args = parse_arguments()
    set_plot_style()

    # Resolve paths
    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"[Error] Performance data directory not found at: {args.data_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    cache_file = os.path.join(args.output_dir, "accuracy_results.json")

    # Load dataset
    print(f"[Init] Scanning Performance directory: {args.data_dir}")
    dataset = load_image_paths(args.data_dir)

    total_images = sum(len(imgs) for dist_dict in dataset.values() for imgs in dist_dict.values())
    print(f"[Init] Found {total_images} total evaluation images across all folders.")

    # Initialize Detector
    detector = None
    is_valid, _ = check_cache_validity(cache_file, args)
    if not is_valid:
        person_path = MODEL_PERSON if args.use_zoom else None
        print(f"[Init] Initializing WeaponDetector with {model_path} (Person detector: {'Enabled' if person_path else 'Disabled'})...")
        detector = WeaponDetector(model_path, person_path)

    # Run / Load Benchmark
    benchmark_data = run_benchmark(dataset, detector, args, cache_file)

    # Print summary table
    print_summary_table(benchmark_data)

    # Generate all requested plots
    print(f"\n[Plotting] Generating figures in: {args.output_dir}...")
    plot_3d_accuracy_distance_cover(benchmark_data, args.output_dir)
    plot_2d_accuracy_graphs(benchmark_data, args.output_dir)
    plot_3d_accuracy_brightness(benchmark_data, args.output_dir)

    print("\n" + "=" * 70)
    print(" 🎉 All Accuracy Plots Generated Successfully!")
    print(f" Output Directory: {args.output_dir}")
    print(" Generated Files:")
    print("   1. 1_3d_accuracy_surface.png")
    print("   2. 1_3d_accuracy_bars.png")
    print("   3. 2_2d_accuracy_vs_distance.png")
    print("   4. 2_2d_accuracy_vs_cover_3m.png")
    print("   5. 2_2d_accuracy_vs_cover_6m.png")
    print("   6. 2_2d_accuracy_vs_cover_9m.png")
    print("   7. 2_2d_accuracy_vs_cover_12m.png")
    print("   8. 2_2d_accuracy_vs_cover_all_distances.png")
    print("   9. 3_3d_accuracy_brightness_surface.png")
    print("  10. 3_3d_accuracy_brightness_bars.png")
    print("  11. 3_2d_accuracy_vs_brightness.png")
    print("  12. accuracy_results.json")
    print("=" * 70 + "\n")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
