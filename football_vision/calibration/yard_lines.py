"""Stage 2: Yard-line Hough detection, vanishing-point pencil RANSAC, and 5-parameter Huber refinement."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares


_DEGENERATE_VP = "Degenerate vanishing point (yard lines are parallel or fit failed)"


def _suppress_near_duplicates(lines: np.ndarray, h: int) -> List[Dict[str, Any]]:
    """Greedy near-duplicate suppression of Hough lines, strongest first.

    Behaviour is identical to the original per-line Python loop (same scalar arithmetic, same
    comparisons, same order). Only the ``any(...)`` over previously kept lines is vectorised.
    At 1080p the fixed vote threshold yields ~10^5 Hough lines, and the nested Python loops were
    the dominant cost of calibration (9 s per frame versus 0.4 s in the Hough transform itself).
    """
    rows = np.asarray(lines).reshape(-1, 3)  # OpenCV 4: (N, 1, 3); OpenCV 5: (N, 3)
    kept: List[Dict[str, Any]] = []
    kept_x = kept_m = None
    n_kept = 0
    for rho, theta, votes in rows:
        cos_t = np.cos(theta)
        if abs(cos_t) < 0.55:
            continue
        m = -np.sin(theta) / np.cos(theta)
        b = rho / np.cos(theta)
        x_mid = m * (h / 2.0) + b
        if kept_x is None:
            # Match the scalar dtype (float32 on NumPy 2, float64 on NumPy 1) so every comparison
            # below is evaluated at exactly the precision the original scalar code used.
            kept_x = np.empty(len(rows), dtype=type(x_mid))
            kept_m = np.empty(len(rows), dtype=type(m))
        if n_kept and np.any((np.abs(x_mid - kept_x[:n_kept]) < 25)
                             & (np.abs(m - kept_m[:n_kept]) < 0.12)):
            continue
        kept_x[n_kept] = x_mid
        kept_m[n_kept] = m
        n_kept += 1
        kept.append({"x_mid": float(x_mid), "m": float(m), "b": float(b), "votes": float(votes)})
    return kept


def _pencil_ransac(kept: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Vanishing-point pencil RANSAC: all parallel 5-yard lines satisfy m_k = alpha*x_mid_k + beta.

    Exhaustive over line pairs in the original (i, j) order with the original strict ``>`` tie
    rule, so the winner is unchanged. The per-pair inlier test is evaluated for a whole row of
    pairs at once (float64, same operation order as the scalar code), and a pair is skipped only
    when even the sum of *all* its inlier votes cannot beat the current best score.
    """
    n = len(kept)
    best_inliers: List[Dict[str, Any]] = []
    best_score = -1.0
    if n < 2:
        return best_inliers
    kx = np.array([k["x_mid"] for k in kept], dtype=np.float64)
    km = np.array([k["m"] for k in kept], dtype=np.float64)
    kv = np.array([k["votes"] for k in kept], dtype=np.float64)
    for i in range(n - 1):
        dxm = kx[i + 1:] - kx[i]
        js = np.nonzero(np.abs(dxm) >= 80)[0]
        if js.size == 0:
            continue
        alpha = (km[i + 1:][js] - km[i]) / dxm[js]
        beta = km[i] - alpha * kx[i]
        ok = (alpha >= 0.0) & (alpha <= 0.005)
        if not ok.any():
            continue
        js, alpha, beta = js[ok], alpha[ok], beta[ok]
        inl = np.abs(km[None, :] - (alpha[:, None] * kx[None, :] + beta[:, None])) < 0.025
        vote_upper_bound = inl @ kv
        for r in range(js.size):
            # score <= sum of all inlier votes, so this can never skip a pair that would win
            # (the margin only guards float rounding when votes are not integers).
            if vote_upper_bound[r] * (1.0 + 1e-9) < best_score:
                continue
            idx = np.nonzero(inl[r])[0]
            order = idx[np.argsort(-kv[idx], kind="stable")]
            dedup: List[Dict[str, Any]] = []
            for t in order:
                cand = kept[t]
                if all(abs(cand["x_mid"] - d["x_mid"]) > 45 for d in dedup):
                    dedup.append(cand)
            score = sum(d["votes"] for d in dedup)
            if score > best_score:
                best_score = score
                best_inliers = sorted(dedup, key=lambda x: x["x_mid"])
    return best_inliers


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

    kept = _suppress_near_duplicates(lines, h)
    best_inliers = _pencil_ransac(kept)

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
    if not np.isfinite(alpha0) or alpha0 == 0.0:
        return None, None, 0, np.nan, np.nan, _DEGENERATE_VP   # zero slope: VP at infinity
    y_vp0 = (h / 2.0) - 1.0 / alpha0
    x_vp0 = -beta0 / alpha0
    a0_init = xm[0]
    a1_init = (xm[-1] - xm[0]) / max(1.0, float(k_indices[-1] - k_indices[0]))
    p0 = np.array([x_vp0, y_vp0, a0_init, a1_init, 0.0], dtype=np.float64)
    if not np.all(np.isfinite(p0)):
        # Exactly parallel yard lines (no perspective convergence, e.g. a straight-on view) give a
        # zero slope fit, so 1/alpha0 is infinite. The pencil model cannot represent that; refuse
        # the frame rather than crash a whole video run inside scipy.
        return None, None, 0, np.nan, np.nan, _DEGENERATE_VP

    def residuals(p):
        x_vp, y_vp, a0, a1, a2 = p
        x_mid_k = (a0 + a1 * pts_k) / (1.0 + a2 * pts_k)
        m_k = (x_mid_k - x_vp) / ((h / 2.0) - y_vp)
        x_pred = x_mid_k + m_k * (pts_y - (h / 2.0))
        return (pts_x - x_pred) / np.sqrt(1.0 + m_k ** 2)

    try:
        opt = least_squares(residuals, p0, loss="huber", f_scale=1.5)
    except ValueError:  # e.g. non-finite residuals from a pathological initial guess
        return None, None, 0, np.nan, np.nan, _DEGENERATE_VP
    x_vp, y_vp, a0, a1, a2 = opt.x
    if not np.all(np.isfinite(opt.x)) or abs((h / 2.0) - y_vp) < 1e-9:
        return None, None, 0, np.nan, np.nan, _DEGENERATE_VP
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
