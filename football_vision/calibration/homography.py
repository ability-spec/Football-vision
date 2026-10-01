"""Stage 4: Projective cross-ratio control-point conditioning, homography fitting, parity detection,
and physical plausibility checks.

Adapted in part from Alex R. Haigh's Hockey-Vision (Apache-2.0):
  - plausible_homography(), players_clustered(), implied_player_height_ft()
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from football_vision.field_spec import (
    FIELD_WIDTH_YD,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    Y_NEAR_HASH_INNER_BUG_YD,
    Y_FAR_HASH_INNER_BUG_YD,
    MIN_PLAYER_SPREAD_YD,
    MIN_PLAYERS_FOR_SPREAD,
    MIN_PLAYERS_FOR_HEIGHT,
)
from football_vision.schema import CalibrationResult
from football_vision.calibration.white_ridge import extract_white_paint_ridge
from football_vision.calibration.yard_lines import detect_yard_lines_and_vp
from football_vision.calibration.hash_marks import detect_hash_rows_guided
from football_vision.calibration.sidelines import detect_far_sideline


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

    yard_lines, vp_yard, ridge_cnt, orth_med, orth_mean, err_yl = detect_yard_lines_and_vp(ridge_u8, h)
    if yard_lines is None or vp_yard is None:
        return CalibrationResult(
            False, None, None, notes=[err_yl or "Yard-line detection failed"],
            image_size=(w, h), failure_reason=err_yl,
        )
    x_vp, y_vp = vp_yard

    far_hash, near_hash, all_inlier_ticks, hash_pts_arr, hash_rmse, err_hr = detect_hash_rows_guided(
        yard_lines, mask, turf, h, w
    )
    if far_hash is None or near_hash is None or all_inlier_ticks is None:
        return CalibrationResult(
            False, None, None, notes=[err_hr or "Hash-row detection failed"],
            image_size=(w, h), failure_reason=err_hr,
        )

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

    H, _ = cv2.findHomography(img_pts_arr, field_pts_arr, 0)
    if H is None:
        return CalibrationResult(
            False, None, None, notes=["cv2.findHomography returned None"],
            image_size=(w, h), failure_reason="cv2.findHomography returned None",
        )
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
        ridge_pixel_count=ridge_cnt,
        ridge_orth_median_px=orth_med,
        ridge_orth_mean_px=orth_mean,
        hash_row_rmse_px=hash_rmse,
        even_idx_number_score=even_score,
        odd_idx_number_score=odd_score,
        detected_ten_yard_parity=parity,
        plausible_orientation_scale=plausible_ok,
        runtime_ms=dt_ms,
        image_size=(w, h),
        confidence=1.0 if plausible_ok else 0.0,
        x_coord_mode="relative_10yd" if plausible_ok else "uncalibrated",
        failure_reason=None if plausible_ok else "Failed orientation/scale plausibility check",
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
