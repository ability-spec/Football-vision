"""Experimental CPU-only dark-blob player detector and K-Means team split.

Isolated from the production `football_vision` library (Rule #5 & Rule #8) and used strictly
for reproducing the Week-1 CPU-only plausibility benchmark when GPU YOLO weights are not loaded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans

from football_vision.field_spec import FIELD_WIDTH_YD
from football_vision.calibration.homography import (
    players_clustered,
    implied_player_height_ft,
)


def detect_and_project_players(
    img: np.ndarray,
    H: np.ndarray,
    x_range_yd: Tuple[float, float],
    y_max_px_frac: float = 0.82,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Detect players on turf (CPU fallback when YOLO/torch is absent), classify into 2 teams via
    jersey-color K-Means, and run Alex's plausibility checks.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    patch = hsv[int(0.20 * h):int(0.75 * h), int(0.10 * w):int(0.90 * w)]
    h_med = float(np.median(patch[:, :, 0]))

    # Players always contain dark regions (helmets, pads, cleats, skin, dark jerseys, or ground shadows)
    # that are significantly darker than both green turf (gray ~ 95..160) and white paint (>160).
    dark_thresh = 80 if np.median(gray) > 120 else 70
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    dark[:int(0.09 * h), :] = 0
    dark[int(y_max_px_frac * h):, :] = 0

    fg = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((11, 5), np.uint8))

    num_l, _, stats, _ = cv2.connectedComponentsWithStats(fg)
    candidate_boxes = []
    nom_h = int(round(0.092 * h))  # full head-to-cleat height ~ 46px on 506p, 75px on 814p
    nom_w = int(round(0.026 * w))
    for i in range(1, num_l):
        x, y, bw, bh, area = stats[i]
        if area < 38 or bw > int(0.12 * w):
            continue
        # If multiple linemen are stacked vertically in a tall blob, split into nominal player heights
        if bh > 1.45 * nom_h:
            n_split = min(4, max(2, int(round(bh / (0.85 * nom_h)))))
            sh = bh // n_split
            for s_i in range(n_split):
                sy = y + s_i * sh
                candidate_boxes.append((x, sy, max(bw, nom_w), max(sh, int(0.85 * nom_h)), float(area) / n_split))
        else:
            # Expand dark anchor to full player bounding box (helmet to cleats)
            target_h = max(bh, int(0.90 * nom_h))
            pad_top = (target_h - bh) // 2
            pad_bot = target_h - bh - pad_top
            ny = max(int(0.08 * h), y - pad_top)
            nh = min(int(y_max_px_frac * h) - ny, bh + pad_top + pad_bot)
            nw = max(bw, nom_w)
            nx = max(0, x - (nw - bw) // 2)
            candidate_boxes.append((nx, ny, nw, nh, float(area)))

    # Deduplicate overlapping/nearby boxes (ported from Alex's _dedupe in homography.py)
    min_dist = 0.026 * w
    kept_boxes = []
    for bx, by, bw, bh, score in sorted(candidate_boxes, key=lambda b: -b[4]):
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if all(np.hypot(cx - (kx + kw / 2.0), cy - (ky + kh / 2.0)) > min_dist for kx, ky, kw, kh in kept_boxes):
            kept_boxes.append((bx, by, bw, bh))

    raw_boxes = []
    features = []
    for x, y, bw, bh in kept_boxes:
        x = max(0, min(w - 8, int(x)))
        y = max(0, min(h - 8, int(y)))
        bw = max(6, min(w - x, int(bw)))
        bh = max(6, min(h - y, int(bh)))
        foot = np.float32([[[x + bw / 2.0, y + bh]]])
        fx, fy = cv2.perspectiveTransform(foot, H)[0, 0]
        # Exclude endzone lettering (fx <= 99.5 yd) and out-of-bounds sideline staff
        if x_range_yd[0] <= fx <= min(x_range_yd[1], 99.5) and 2.0 <= fy <= FIELD_WIDTH_YD - 2.0:
            ty0 = min(h - 2, y + int(0.15 * bh))
            ty1 = min(h, max(ty0 + 2, y + int(0.65 * bh)))
            tx0 = min(w - 2, x + int(0.15 * bw))
            tx1 = min(w, max(tx0 + 2, x + int(0.85 * bw)))
            torso_bgr = img[ty0:ty1, tx0:tx1]
            torso_hsv = hsv[ty0:ty1, tx0:tx1]
            if torso_bgr.size == 0:
                continue
            not_grass = np.abs(torso_hsv[:, :, 0].astype(float) - h_med) > 10
            if np.any(not_grass):
                mean_bgr = torso_bgr[not_grass].mean(axis=0)
            else:
                mean_bgr = torso_bgr.reshape(-1, 3).mean(axis=0)
            if not np.all(np.isfinite(mean_bgr)):
                continue
            raw_boxes.append((int(x), int(y), int(bw), int(bh), float(fx), float(fy)))
            features.append(mean_bgr)

    if len(raw_boxes) >= 2:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(np.array(features))
        x0 = np.median([b[4] for b, l in zip(raw_boxes, labels) if l == 0])
        x1 = np.median([b[4] for b, l in zip(raw_boxes, labels) if l == 1])
        if x0 > x1:
            labels = 1 - labels
    else:
        labels = np.zeros(len(raw_boxes), dtype=int)

    players = []
    for (x, y, bw, bh, fx, fy), lab in zip(raw_boxes, labels):
        players.append({
            "box_xywh": (x, y, bw, bh),
            "team": "team_a_player" if lab == 0 else "team_b_player",
            "field_xy_yd": (round(fx, 2), round(fy, 2)),
            "field_xy_ft": (round(fx * 3.0, 2), round(fy * 3.0, 2)),
        })

    xy_yd = np.array([p["field_xy_yd"] for p in players]) if players else np.zeros((0, 2))
    spread_yd = float(np.sqrt(np.mean(np.sum((xy_yd - xy_yd.mean(axis=0)) ** 2, axis=1)))) if len(xy_yd) else 0.0
    med_h_ft = implied_player_height_ft([p["box_xywh"] for p in players], H)

    checks = {
        "n_players": len(players),
        "rms_spread_yd": round(spread_yd, 2),
        "rms_spread_ft": round(spread_yd * 3.0, 2),
        "clustered_flag": players_clustered(xy_yd),
        "implied_median_height_ft": round(med_h_ft, 2) if med_h_ft is not None else None,
        "height_plausible": (3.0 <= med_h_ft <= 9.0) if med_h_ft is not None else False,
    }
    return players, checks
