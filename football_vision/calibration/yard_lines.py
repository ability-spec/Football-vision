"""Stage 2: Yard-line Hough detection, vanishing-point pencil RANSAC, and 5-parameter Huber refinement."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares


def detect_yard_lines_and_vp(
    ridge_u8: np.ndarray,
    h: int,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Tuple[float, float]], int, float, float, Optional[str]]:
    """Run full-line Hough accumulator, VP pencil RANSAC, and joint 5-parameter Huber refinement.

    Returns:
      (yard_lines, (x_vp, y_vp), ridge_pixel_count, orth_median_px, orth_mean_px, error_note)
    """
    lines = cv2.HoughLinesWithAccumulator(ridge_u8, rho=1.0, theta=np.pi / 1080, threshold=45)
    if lines is None:
        return None, None, 0, np.nan, np.nan, "No Hough lines above threshold"

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
        return None, None, 0, np.nan, np.nan, "Fewer than 3 yard lines in VP pencil"

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

    return (
        yard_lines,
        (float(x_vp), float(y_vp)),
        int(len(pts_x)),
        float(np.median(orth_err)),
        float(np.mean(orth_err)),
        None,
    )
