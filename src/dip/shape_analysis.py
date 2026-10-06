import cv2
import numpy as np

def compute_shape_similarity(roi_crop):
    """
    Evaluates Shape Similarity on a weapon Region of Interest (ROI)
    using classical Digital Image Processing techniques:
    1. Aspect Ratio / Elongation analysis (weapons have directional length-to-width ratio)
    2. Edge Density analysis (detects metallic boundaries vs smooth organic shapes)
    3. Contour Solidity analysis (weapons have non-convex profiles like trigger guards/grips)
    
    Returns:
      shape_score (float): normalized value between 0.0 and 1.0 (0% - 100%)
    """
    if roi_crop is None or roi_crop.size == 0 or roi_crop.shape[0] < 8 or roi_crop.shape[1] < 8:
        return 0.50

    gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Aspect Ratio / Elongation (Guns & knives have characteristic aspect 1.2 - 4.5)
    aspect = max(w, h) / max(1, min(w, h))
    if 1.2 <= aspect <= 4.0:
        aspect_score = 0.95
    elif 1.0 <= aspect < 1.2:
        aspect_score = 0.70
    else:
        aspect_score = 0.55

    # 2. Canny Edge Density (detects distinct weapon contour edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(w * h)
    density_score = float(np.clip(edge_density * 4.0, 0.40, 1.0))

    # 3. Contour Solidity (measure of shape complexity and indentations)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / max(1.0, hull_area) if hull_area > 0 else 0.5
        solidity_score = 0.90 if (0.30 <= solidity <= 0.88) else 0.60
    else:
        solidity_score = 0.60

    # Weighted composite shape score
    shape_score = 0.40 * aspect_score + 0.35 * density_score + 0.25 * solidity_score
    return round(float(shape_score), 3)
