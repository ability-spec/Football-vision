"""Stage 1: White-paint ridge extraction and baseline naive Canny+Hough comparison."""

from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np


def naive_canny_hough(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Baseline naive Canny + HoughLinesP on grayscale frame (to show why naive Hough fails)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    segs = cv2.HoughLinesP(edges, rho=1.0, theta=np.pi / 180, threshold=60, minLineLength=50, maxLineGap=15)
    if segs is None:
        return []
    return [tuple(map(int, s)) for s in segs[:, 0]]


def extract_white_paint_ridge(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 1: Extract 1-px horizontal white-ridge mask and white-paint mask inside green turf.

    Rejects:
      - Broadcast scorebugs (bottom 15% + top 5% margin)
      - Virtual blue line of scrimmage and yellow 1st-down line (via HSV saturation < 65)
      - Yellow telestrator arrows and colored team logos (via HSV saturation < 65)
      - Double-edge Canny splitting (via 1px horizontal morphological top-hat local maximum)
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    turf = cv2.inRange(hsv, (28, 25, 40), (88, 255, 255))
    turf_nearby = cv2.dilate(turf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0

    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_h)

    low_sat = hsv[:, :, 1] < 65
    bright = gray > 135

    mask = np.uint8((tophat_h > 22) & low_sat & bright & turf_nearby) * 255
    mask[int(0.85 * h):, :] = 0
    mask[:int(0.05 * h), :] = 0

    ridge = (tophat_h >= cv2.dilate(tophat_h, np.ones((1, 5), np.uint8))) & (mask > 0)
    ridge_u8 = np.uint8(ridge) * 255
    return ridge_u8, mask, turf
