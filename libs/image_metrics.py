"""Image quality metrics on a uint8 grey image (H, W): localized blur, hybrid sharpness,
noise, highlight/shadow clipping and entropy. The Image Quality Gate node scores them against
its thresholds. cv2 is imported inside the functions that use it.
"""

import numpy as np


def blur_detection(gray, block_size, center_weighted=False, var_threshold=30.0):
    """
    Localized Laplacian variance blur detection.

    In center-weighted mode, blocks are weighted by distance from center:
    - Center 30%: weight 3.0x (subject area)
    - Mid ring 30-60%: weight 1.0x
    - Outer 60%+: weight 0.3x (background)
    """
    import cv2

    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    weighted_blur = 0.0
    total_weight = 0.0

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            lap_var = cv2.Laplacian(block, cv2.CV_64F).var()
            is_blurry = 1.0 if lap_var < var_threshold else 0.0

            if center_weighted:
                bx = x + block_size / 2.0
                by = y + block_size / 2.0
                dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) / max_dist
                if dist < 0.3:
                    weight = 3.0
                elif dist < 0.6:
                    weight = 1.0
                else:
                    weight = 0.3
            else:
                weight = 1.0

            weighted_blur += is_blurry * weight
            total_weight += weight

    return weighted_blur / max(total_weight, 1.0)


def sharpness_hybrid(gray):
    """Hybrid sharpness: Laplacian variance + Tenengrad."""
    import cv2

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_score = lap.var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ten_score = np.mean(gx ** 2 + gy ** 2)
    return (lap_score + ten_score / 1000.0) / 2.0


def noise_estimation(gray):
    """Gaussian difference noise estimation."""
    import cv2

    blurred = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
    diff = np.abs(gray.astype(np.float64) - blurred)
    return np.mean(diff)


def clipping_analysis(gray, threshold=5):
    """Highlight/shadow clipping ratio."""
    total = gray.size
    shadows = np.sum(gray <= threshold)
    highlights = np.sum(gray >= 255 - threshold)
    return (shadows + highlights) / total


def entropy_analysis(gray):
    """Shannon entropy in bits (0-8)."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))
