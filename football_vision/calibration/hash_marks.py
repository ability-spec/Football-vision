"""Stage 3: Guided 1-yard hash-mark tick detection and longitudinal hash-row RANSAC fitting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def detect_hash_rows_guided(
    yard_lines: List[Dict[str, Any]],
    mask: np.ndarray,
    turf: np.ndarray,
    h: int,
    w: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[np.ndarray], np.ndarray, float, Optional[str]]:
    """Detect 1-yard hash mark ticks at t in {0.2, 0.4, 0.6, 0.8} and fit near/far hash rows via RANSAC.

    Returns:
      (far_hash, near_hash, all_inlier_ticks, hash_pts_arr, hash_rmse, error_note)
    """
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
        return None, None, None, hash_pts_arr, np.nan, "Too few 1-yard hash mark ticks detected"

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
        return None, None, None, hash_pts_arr, np.nan, "Could not fit both hash rows"

    hash_rows.sort(key=lambda r: r["y_mid"])
    far_hash, near_hash = hash_rows[0], hash_rows[1]
    all_inlier_ticks = np.vstack([far_hash["pts"], near_hash["pts"]])
    hash_rmse = float(np.sqrt(0.5 * (far_hash["rmse_px"] ** 2 + near_hash["rmse_px"] ** 2)))
    return far_hash, near_hash, all_inlier_ticks, hash_pts_arr, hash_rmse, None
