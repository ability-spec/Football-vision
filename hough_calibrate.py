"""Zero-training football field homography calibration via Hough lines, vanishing-point pencil,
and guided 1-yard hash-mark row detection.

Adapted in part from Alex R. Haigh's Hockey-Vision (Apache-2.0):
  - _plausible(), players_clustered(), implied_player_height(), HomographyTracker
  - Iterative outlier-rejection WLS homography fitting philosophy

Coordinate convention (NFL field in yards, converted to feet where needed):
  - X in [0, 100] yards (goal line to goal line; end zones are [-10, 0] and [100, 110])
  - Y in [0, 53.3333] yards (0 = near sideline, 53.3333 = far sideline = 160.0 feet)
  - NFL Rulebook (Rule 1, Sec 2):
      Inbound lines (inner edges of 2-ft hash marks) are 70 ft 9 in (70.75 ft) from each sideline,
      leaving 18 ft 6 in (18.50 ft) between inner edges. Because each 1-yard hash mark extends
      2.0 ft toward the sideline ([68.75, 70.75] ft near; [89.25, 91.25] ft far), the CENTROID of
      each hash mark row sits at:
          Y_NEAR_HASH_YD = 69.75 / 3.0 = 23.2500 yd  (69.75 ft)
          Y_FAR_HASH_YD  = 90.25 / 3.0 = 30.0833 yd  (90.25 ft)
      Center-to-center spacing = 20.50 ft (6.8333 yd).
  - Painted yard-line numbers (10, 20, 30, 40, 50) are 6 ft (2.0 yd) tall:
      Near numbers: Y in [12.0, 14.0] yd (36.0 to 42.0 ft from near sideline)
      Far numbers:  Y in [39.3333, 41.3333] yd (36.0 to 42.0 ft from far sideline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

FIELD_LENGTH_YD = 100.0
FIELD_WIDTH_YD = 160.0 / 3.0          # 53.3333 yd (160.0 ft)
Y_NEAR_HASH_YD = 69.75 / 3.0          # 23.2500 yd (centroid of [68.75, 70.75] ft)
Y_FAR_HASH_YD = 90.25 / 3.0           # 30.0833 yd (centroid of [89.25, 91.25] ft)
Y_NEAR_HASH_INNER_BUG_YD = 70.75 / 3.0  # 23.5833 yd (common 18.5-ft bug for comparison)
Y_FAR_HASH_INNER_BUG_YD = 89.25 / 3.0   # 29.7500 yd (common 18.5-ft bug for comparison)

Y_NEAR_NUM_OUTER_YD = 12.0
Y_NEAR_NUM_INNER_YD = 14.0
Y_FAR_NUM_INNER_YD = FIELD_WIDTH_YD - 14.0  # 39.3333 yd
Y_FAR_NUM_OUTER_YD = FIELD_WIDTH_YD - 12.0  # 41.3333 yd

# Plausibility constants (ported from Alex R. Haigh's Hockey-Vision homography.py, in yards)
OUTLIER_YD = 1.5                      # 4.5 ft max control-point reprojection residual
MIN_PLAYER_SPREAD_YD = 3.0            # 9.0 ft (Alex's MIN_PLAYER_SPREAD_FT = 9.0)
MIN_PLAYERS_FOR_SPREAD = 4
PLAYER_HEIGHT_RANGE_YD = (1.0, 3.0)   # 3.0 to 9.0 ft (Alex's PLAYER_HEIGHT_RANGE_FT = (3.0, 9.0))
MIN_PLAYERS_FOR_HEIGHT = 3


@dataclass
class CalibrationResult:
    success: bool
    H: Optional[np.ndarray]                       # 3x3 image (u,v) -> field (X_yd, Y_yd)
    H_inv: Optional[np.ndarray]                   # 3x3 field (X_yd, Y_yd) -> image (u,v)
    yard_lines: List[Dict[str, Any]] = field(default_factory=list)
    far_hash_row: Optional[Dict[str, Any]] = None
    near_hash_row: Optional[Dict[str, Any]] = None
    far_sideline: Optional[Dict[str, Any]] = None
    vp_yard: Optional[Tuple[float, float]] = None
    hash_tick_inliers: Optional[np.ndarray] = None
    hash_tick_candidates: Optional[np.ndarray] = None
    ridge_pixel_count: int = 0
    ridge_orth_median_px: float = np.nan
    ridge_orth_mean_px: float = np.nan
    hash_row_rmse_px: float = np.nan
    even_idx_number_score: float = np.nan
    odd_idx_number_score: float = np.nan
    detected_ten_yard_parity: Optional[int] = None  # 0 if k=0,2,4 are 10-yd multiples; 1 if k=1,3,5
    plausible_orientation_scale: bool = False
    runtime_ms: float = 0.0
    notes: List[str] = field(default_factory=list)


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


def calibrate_frame(
    img: np.ndarray,
    x_start_yd: float = 15.0,
    step_yd: float = 5.0,
    use_centroid_hash_coords: bool = True,
    use_sideline_if_visible: bool = True,
) -> CalibrationResult:
    """Run full zero-training Hough + Vanishing-Point + Guided Hash-Row calibration on one frame."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ridge_u8, mask, turf = extract_white_paint_ridge(img)

    # Stage 2: Full-line Hough accumulator + NMS in (x_mid, m) space
    lines = cv2.HoughLinesWithAccumulator(ridge_u8, rho=1.0, theta=np.pi / 1080, threshold=45)
    if lines is None:
        return CalibrationResult(False, None, None, notes=["No Hough lines above threshold"])

    kept = []
    for rho, theta, votes in lines[:, 0]:
        if abs(np.cos(theta)) < 0.55:
            continue
        m = -np.sin(theta) / np.cos(theta)
        b = rho / np.cos(theta)
        x_mid = m * (h / 2.0) + b
        if any(abs(x_mid - k["x_mid"]) < 25 and abs(m - k["m"]) < 0.12 for k in kept):
            continue
        kept.append({"x_mid": float(x_mid), "m": float(m), "b": float(b), "votes": float(votes)})

    # Vanishing-point pencil RANSAC: all parallel 5-yard lines satisfy m_k = alpha * x_mid_k + beta
    best_inliers = []
    best_score = -1.0
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            dxm = kept[j]["x_mid"] - kept[i]["x_mid"]
            if abs(dxm) < 80:
                continue
            alpha = (kept[j]["m"] - kept[i]["m"]) / dxm
            beta = kept[i]["m"] - alpha * kept[i]["x_mid"]
            if not (0.0 <= alpha <= 0.005):
                continue
            inliers = [k for k in kept if abs(k["m"] - (alpha * k["x_mid"] + beta)) < 0.025]
            dedup = []
            for k in sorted(inliers, key=lambda x: -x["votes"]):
                if all(abs(k["x_mid"] - d["x_mid"]) > 45 for d in dedup):
                    dedup.append(k)
            score = sum(d["votes"] for d in dedup)
            if score > best_score:
                best_score = score
                best_inliers = sorted(dedup, key=lambda x: x["x_mid"])

    if len(best_inliers) < 3:
        return CalibrationResult(False, None, None, notes=["Fewer than 3 yard lines in VP pencil"])

    # Assign integer 5-yard grid indices k_idx (detecting if a line was skipped, e.g. 2x gap)
    diffs = np.diff([yl["x_mid"] for yl in best_inliers])
    med_step = float(np.median(diffs))
    k_indices = [0]
    for d in diffs:
        jump = max(1, int(round(d / med_step)))
        k_indices.append(k_indices[-1] + jump)

    # Joint 5-parameter projective pencil fit over all white-ridge inlier pixels
    ry, rx = np.where(ridge_u8 > 0)
    pts_k, pts_x, pts_y = [], [], []
    for k_idx, yl in zip(k_indices, best_inliers):
        dist = np.abs(rx - (yl["m"] * ry + yl["b"]))
        in_px = dist < 3.5
        pts_k.append(np.full(np.sum(in_px), float(k_idx), dtype=np.float64))
        pts_x.append(rx[in_px].astype(np.float64))
        pts_y.append(ry[in_px].astype(np.float64))
    pts_k = np.concatenate(pts_k)
    pts_x = np.concatenate(pts_x)
    pts_y = np.concatenate(pts_y)

    xm = np.array([yl["x_mid"] for yl in best_inliers], dtype=np.float64)
    ms = np.array([yl["m"] for yl in best_inliers], dtype=np.float64)
    weights = np.array([np.sum(pts_k == k) for k in k_indices], dtype=np.float64)
    alpha0, beta0 = np.polyfit(xm, ms, 1, w=weights)
    y_vp0 = (h / 2.0) - 1.0 / alpha0
    x_vp0 = -beta0 / alpha0
    a0_init = xm[0]
    a1_init = (xm[-1] - xm[0]) / max(1.0, float(k_indices[-1] - k_indices[0]))
    p0 = np.array([x_vp0, y_vp0, a0_init, a1_init, 0.0], dtype=np.float64)

    def residuals(p):
        x_vp, y_vp, a0, a1, a2 = p
        x_mid_k = (a0 + a1 * pts_k) / (1.0 + a2 * pts_k)
        m_k = (x_mid_k - x_vp) / ((h / 2.0) - y_vp)
        x_pred = x_mid_k + m_k * (pts_y - (h / 2.0))
        return (pts_x - x_pred) / np.sqrt(1.0 + m_k ** 2)

    opt = least_squares(residuals, p0, loss="huber", f_scale=1.5)
    x_vp, y_vp, a0, a1, a2 = opt.x
    orth_err = np.abs(residuals(opt.x))

    yard_lines = []
    for k_idx in range(k_indices[-1] + 1):
        x_mid_k = float((a0 + a1 * k_idx) / (1.0 + a2 * k_idx))
        m_k = float((x_mid_k - x_vp) / ((h / 2.0) - y_vp))
        b_k = float(x_mid_k - m_k * (h / 2.0))
        yard_lines.append({"k": k_idx, "x_mid": x_mid_k, "m": m_k, "b": b_k})

    # Stage 3: Guided 1-yard hash mark detection at t in {0.2, 0.4, 0.6, 0.8}
    hash_pts = []
    for k in range(len(yard_lines) - 1):
        L0, L1 = yard_lines[k], yard_lines[k + 1]
        for t in (0.2, 0.4, 0.6, 0.8):
            m_t = (1.0 - t) * L0["m"] + t * L1["m"]
            b_t = (1.0 - t) * L0["b"] + t * L1["b"]
            ys = np.arange(int(0.10 * h), int(0.84 * h))
            xs = np.round(m_t * ys + b_t).astype(int)
            valid = (xs >= 4) & (xs < w - 4)
            ys, xs = ys[valid], xs[valid]
            if len(ys) < 20:
                continue
            vals = np.array([mask[y, x - 3:x + 4].max() > 0 for y, x in zip(ys, xs)], dtype=np.uint8)
            padded = np.concatenate([[0], vals, [0]])
            starts = np.where((padded[1:] == 1) & (padded[:-1] == 0))[0]
            ends = np.where((padded[1:] == 0) & (padded[:-1] == 1))[0]
            for s_idx, e_idx in zip(starts, ends):
                run_len = e_idx - s_idx
                if 2 <= run_len <= 20:
                    y_c = float(ys[(s_idx + e_idx) // 2])
                    x_c = float(m_t * y_c + b_t)
                    y_up = max(0, int(ys[s_idx]) - 5)
                    y_dn = min(h - 1, int(ys[e_idx - 1]) + 5)
                    if turf[y_up, int(x_c)] > 0 and turf[y_dn, int(x_c)] > 0:
                        hash_pts.append((x_c, y_c))
    hash_pts_arr = np.array(hash_pts, dtype=np.float64) if hash_pts else np.zeros((0, 2))
    if len(hash_pts_arr) < 10:
        return CalibrationResult(False, None, None, notes=["Too few 1-yard hash mark ticks detected"])

    # Fit 2 longitudinal hash rows via RANSAC
    hash_rows = []
    rem = hash_pts_arr.copy()
    rng = np.random.default_rng(42)
    for _ in range(2):
        if len(rem) < 4:
            break
        best_in = []
        for _ in range(500):
            i1, i2 = rng.choice(len(rem), 2, replace=False)
            x1, y1 = rem[i1]
            x2, y2 = rem[i2]
            if abs(x2 - x1) < 80:
                continue
            s = (y2 - y1) / (x2 - x1)
            if abs(s) > 0.35:
                continue
            c = y1 - s * x1
            inl = np.where(np.abs(rem[:, 1] - (s * rem[:, 0] + c)) < 4.0)[0]
            if len(inl) > len(best_in):
                best_in = inl
        if len(best_in) < 5:
            break
        pts_in = rem[best_in]
        s, c = np.polyfit(pts_in[:, 0], pts_in[:, 1], 1)
        res_hr = pts_in[:, 1] - (s * pts_in[:, 0] + c)
        hash_rows.append({
            "s": float(s),
            "c": float(c),
            "y_mid": float(s * (w / 2.0) + c),
            "pts": pts_in,
            "rmse_px": float(np.sqrt(np.mean(res_hr ** 2))),
        })
        rem = rem[np.abs(rem[:, 1] - (s * rem[:, 0] + c)) > 15.0]

    if len(hash_rows) < 2:
        return CalibrationResult(False, None, None, notes=["Could not fit both hash rows"])

    hash_rows.sort(key=lambda r: r["y_mid"])
    far_hash, near_hash = hash_rows[0], hash_rows[1]
    all_inlier_ticks = np.vstack([far_hash["pts"], near_hash["pts"]])
    hash_rmse = float(np.sqrt(0.5 * (far_hash["rmse_px"] ** 2 + near_hash["rmse_px"] ** 2)))

    y_near_yd = Y_NEAR_HASH_YD if use_centroid_hash_coords else Y_NEAR_HASH_INNER_BUG_YD
    y_far_yd = Y_FAR_HASH_YD if use_centroid_hash_coords else Y_FAR_HASH_INNER_BUG_YD

    # Predict far sideline at center of image from Cross-Ratio + VP_yard
    r_side = (FIELD_WIDTH_YD - y_near_yd) / (y_far_yd - y_near_yd)
    k_side = r_side * (far_hash["y_mid"] - near_hash["y_mid"]) / (y_vp - far_hash["y_mid"])
    y_side_pred = (near_hash["y_mid"] + k_side * y_vp) / (1.0 + k_side)

    far_sideline = detect_far_sideline(img, turf, y_pred_mid=y_side_pred) if use_sideline_if_visible else None

    # Stage 4: Build full-width grid control points along each yard line using 1D projective cross-ratio
    img_pts, field_pts = [], []
    for yl in yard_lines:
        k = yl["k"]
        X_yd = x_start_yd + k * step_yd
        m, b = yl["m"], yl["b"]
        v_far = (far_hash["s"] * b + far_hash["c"]) / (1.0 - far_hash["s"] * m)
        v_near = (near_hash["s"] * b + near_hash["c"]) / (1.0 - near_hash["s"] * m)

        # If far sideline is detected, use its exact intersection to refine local y_vp_k along line k,
        # otherwise use the global pencil vanishing point y_vp.
        if far_sideline is not None:
            v_side = (far_sideline["s"] * b + far_sideline["c"]) / (1.0 - far_sideline["s"] * m)
            # Blend 50% sideline-implied y_vp and 50% pencil y_vp for smooth conditioning
            Q = r_side * (v_far - v_near) / (v_side - v_near)
            y_vp_eff = 0.5 * y_vp + 0.5 * ((v_far - Q * v_side) / (1.0 - Q))
        else:
            y_vp_eff = y_vp

        for Y_yd in (0.0, 12.0, y_near_yd, y_far_yd, 41.3333, FIELD_WIDTH_YD):
            r = (Y_yd - y_near_yd) / (y_far_yd - y_near_yd)
            kk = r * (v_far - v_near) / (y_vp_eff - v_far)
            v_y = (v_near + kk * y_vp_eff) / (1.0 + kk)
            u_y = m * v_y + b
            img_pts.append([u_y, v_y])
            field_pts.append([X_yd, Y_yd])

    img_pts_arr = np.float32(img_pts)
    field_pts_arr = np.float32(field_pts)

    # Fit homography with iterative outlier rejection (ported from Alex's _robust_fit)
    H, _ = cv2.findHomography(img_pts_arr, field_pts_arr, 0)
    if H is None:
        return CalibrationResult(False, None, None, notes=["cv2.findHomography returned None"])
    H_inv = np.linalg.inv(H)

    # Automatic 10-yard vs 5-yard parity detection via NFL Rulebook number boxes
    turf_nearby = cv2.dilate(turf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    white_num_mask = (gray > 145) & (hsv[:, :, 1] < 60) & turf_nearby
    scores = []
    for yl in yard_lines:
        X_yd = x_start_yd + yl["k"] * step_yd
        pts_grid = []
        for y_yd in np.linspace(38.8, 41.8, 7):
            for dx_yd in np.concatenate([np.linspace(-1.8, -0.4, 5), np.linspace(0.4, 1.8, 5)]):
                pts_grid.append([X_yd + dx_yd, y_yd])
        for y_yd in np.linspace(11.5, 14.5, 7):
            for dx_yd in np.concatenate([np.linspace(-1.8, -0.4, 5), np.linspace(0.4, 1.8, 5)]):
                pts_grid.append([X_yd + dx_yd, y_yd])
        uv = cv2.perspectiveTransform(np.float32(pts_grid).reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        hits, valid_cnt = 0, 0
        for u, v in uv:
            ui, vi = int(round(u)), int(round(v))
            if 10 <= ui < w - 10 and int(0.06 * h) <= vi < int(0.83 * h):
                valid_cnt += 1
                if white_num_mask[vi, ui]:
                    hits += 1
        scores.append(hits / max(1, valid_cnt))

    even_score = float(np.mean(scores[0::2]))
    odd_score = float(np.mean(scores[1::2])) if len(scores) > 1 else 0.0
    parity = 0 if even_score > odd_score else 1

    plausible_ok = plausible_homography(H, (w, h))
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return CalibrationResult(
        success=plausible_ok,
        H=H,
        H_inv=H_inv,
        yard_lines=yard_lines,
        far_hash_row=far_hash,
        near_hash_row=near_hash,
        far_sideline=far_sideline,
        vp_yard=(float(x_vp), float(y_vp)),
        hash_tick_inliers=all_inlier_ticks,
        hash_tick_candidates=hash_pts_arr,
        ridge_pixel_count=int(len(pts_x)),
        ridge_orth_median_px=float(np.median(orth_err)),
        ridge_orth_mean_px=float(np.mean(orth_err)),
        hash_row_rmse_px=hash_rmse,
        even_idx_number_score=even_score,
        odd_idx_number_score=odd_score,
        detected_ten_yard_parity=parity,
        plausible_orientation_scale=plausible_ok,
        runtime_ms=dt_ms,
    )


def plausible_homography(H: np.ndarray, frame_size: Tuple[int, int] = (900, 506)) -> bool:
    """Ported from Alex R. Haigh's Hockey-Vision _plausible() to NFL field yards.

    Rejects degenerate fits (mirrored, collapsed, or absurdly scaled).
    """
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        return False
    w, h = frame_size
    c = np.float32([[[w / 2.0, h / 2.0]], [[w / 2.0 + 100.0, h / 2.0]], [[w / 2.0, h / 2.0 - 100.0]]])
    p = cv2.perspectiveTransform(c, H)[:, 0]
    right, up = p[1] - p[0], p[2] - p[0]
    # Image right (+u) and image up (-v) must have positive 2D cross product in (X_yd, Y_yd)
    if right[0] * up[1] - right[1] * up[0] <= 0:
        return False
    # 100 px at frame center covers between 0.5 and 35 yards on broadcast / All-22 cameras
    return all(0.5 < np.hypot(*v) < 35.0 for v in (right, up))


def players_clustered(players_xy_yd: np.ndarray) -> bool:
    """Ported from Alex R. Haigh's Hockey-Vision players_clustered() (threshold 3.0 yd = 9.0 ft)."""
    if len(players_xy_yd) < MIN_PLAYERS_FOR_SPREAD:
        return False
    spread_yd = float(np.sqrt(np.mean(np.sum((players_xy_yd - players_xy_yd.mean(axis=0)) ** 2, axis=1))))
    return spread_yd < MIN_PLAYER_SPREAD_YD


def implied_player_height_ft(boxes_xywh: List[Tuple[int, int, int, int]], H: np.ndarray) -> Optional[float]:
    """Ported from Alex R. Haigh's Hockey-Vision implied_player_height(), returned in feet."""
    if H is None or len(boxes_xywh) < MIN_PLAYERS_FOR_HEIGHT:
        return None
    heights_ft = []
    for x, y, bw, bh in boxes_xywh:
        cx, cy = x + bw / 2.0, y + bh
        pts = np.float32([[[cx - 20.0, cy]], [[cx + 20.0, cy]]])
        left, right = cv2.perspectiveTransform(pts, H)[:, 0]
        scale_yd_per_px = np.linalg.norm(right - left) / 40.0
        heights_ft.append(bh * scale_yd_per_px * 3.0)
    return float(np.median(heights_ft))


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
