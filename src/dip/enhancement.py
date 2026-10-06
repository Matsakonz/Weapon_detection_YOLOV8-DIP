import cv2
import pywt
import numpy as np

def apply_wavelet_subband_boost(l_channel, boost_factor=1.25):
    """
    Applies 2D Discrete Wavelet Transform (Haar) and boosts
    high-frequency detail sub-bands (LH, HL, HH) to accentuate weapon edges.
    """
    try:
        coeffs = pywt.dwt2(l_channel.astype(np.float32), 'haar')
        LL, (LH, HL, HH) = coeffs

        LH *= boost_factor
        HL *= boost_factor
        HH *= boost_factor

        l_rec = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        l_rec = np.clip(l_rec, 0, 255).astype(np.uint8)

        if l_rec.shape != l_channel.shape:
            l_rec = cv2.resize(l_rec, (l_channel.shape[1], l_channel.shape[0]))
        return l_rec
    except Exception:
        return l_channel

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization in LAB color space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

def apply_unsharp_mask(image, sigma=1.5, strength=1.4):
    """
    Restores sharp high-frequency edge definition using Gaussian blur subtraction.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image, strength, blurred, -(strength - 1.0), 0)

def apply_dip_enhancement(image, use_wavelet=True):
    """
    Full Digital Image Processing (DIP) enhancement pipeline:
    1. Wavelet 2D DWT detail enhancement on Luminance channel
    2. LAB CLAHE adaptive contrast equalization
    3. Unsharp Masking high-frequency edge restoration
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if use_wavelet:
        l = apply_wavelet_subband_boost(l)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    return apply_unsharp_mask(enhanced, sigma=1.5, strength=1.4)
