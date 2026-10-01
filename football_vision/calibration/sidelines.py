"""Far sideline inner-edge (turf boundary) detection."""

from __future__ import annotations

from typing import Any, Dict, Optional
import cv2
import numpy as np


def detect_far_sideline(img: np.ndarray, turf: np.ndarray, y_pred_mid: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Detect the inner edge (turf boundary) of the thick white far sideline when visible in frame."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    y_min = 12
    y_max = int(0.38 * h)
    if y_pred_mid is not None:
        if y_pred_mid < 5 or y_pred_mid > 0.40 * h:
            return None
        y_min = max(10, int(y_pred_mid - 25))
        y_max = min(int(0.40 * h), int(y_pred_mid + 25))

    pts = []
    for x in range(20, w - 20, 4):
        for y in range(y_min, y_max):
            if gray[y - 3, x] > 175 and hsv[y - 3, x, 1] < 55:
                if gray[y - 3, x] - gray[y + 4, x] > 55:
                    if np.mean(turf[y + 4:min(h, y + 24), x] > 0) > 0.7:
                        pts.append((float(x), float(y)))
                        break
    if len(pts) < 35:
        return None
    pts_arr = np.array(pts)
    rng = np.random.default_rng(0)
    best_in = []
    for _ in range(300):
        i1, i2 = rng.choice(len(pts_arr), 2, replace=False)
        x1, y1 = pts_arr[i1]
        x2, y2 = pts_arr[i2]
        if abs(x2 - x1) < 150:
            continue
        s = (y2 - y1) / (x2 - x1)
        if abs(s) > 0.25:
            continue
        c = y1 - s * x1
        inl = np.where(np.abs(pts_arr[:, 1] - (s * pts_arr[:, 0] + c)) < 2.0)[0]
        if len(inl) > len(best_in):
            best_in = inl
    if len(best_in) < 35:
        return None
    s, c = np.polyfit(pts_arr[best_in, 0], pts_arr[best_in, 1], 1)
    y_mid = float(s * (w / 2.0) + c)
    if y_pred_mid is not None and abs(y_mid - y_pred_mid) > 18.0:
        return None
    return {"s": float(s), "c": float(c), "y_mid": y_mid, "inliers": int(len(best_in))}
