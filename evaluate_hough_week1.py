"""Week-1 De-Risking Benchmark: Zero-Training Hough + Vanishing-Point Calibration on Real NFL Frames.

Evaluates:
  1. Naive Canny + HoughLinesP failure modes
  2. Two-Stage Domain-Aware Hough + VP Pencil + Guided 1-Yard Hash-Row Calibrator
  3. Held-Out Ground-Truth Landmark Error (26 independent sideline & yard-number landmarks)
  4. Impact of NFL Rulebook Hash Centroid Geometry (20.50 ft center-to-center vs 18.50 ft inner-edge bug)
  5. Alex Haigh's 4 Plausibility Checks ported to NFL field coordinates
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np

from hough_calibrate import (
    FIELD_WIDTH_YD,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    Y_NEAR_NUM_OUTER_YD,
    Y_NEAR_NUM_INNER_YD,
    Y_FAR_NUM_INNER_YD,
    Y_FAR_NUM_OUTER_YD,
    naive_canny_hough,
    calibrate_frame,
    detect_and_project_players,
)

# ---------------------------------------------------------------------------
# Held-out Ground-Truth Landmarks (measured directly from image intensity edges,
# NEVER used by the Hash-Row + VP-Pencil homography fit when use_sideline_if_visible=False).
# Each entry is: (label, X_yd, Y_yd, true_v_px_along_yard_line)
# ---------------------------------------------------------------------------
HELD_OUT_LANDMARKS = {
    "Frame 1 (SEA vs SF — FOX Broadcast)": [
        # Far sideline (Y = 53.3333 yd) at X = 15, 20, 25, 30, 35 yd
        ("Sideline @ 15-yd", 15.0, FIELD_WIDTH_YD, 85.0),
        ("Sideline @ 20-yd", 20.0, FIELD_WIDTH_YD, 85.5),
        ("Sideline @ 25-yd", 25.0, FIELD_WIDTH_YD, 85.5),
        ("Sideline @ 30-yd", 30.0, FIELD_WIDTH_YD, 84.5),
        ("Sideline @ 35-yd", 35.0, FIELD_WIDTH_YD, 83.5),
        # Far numbers outer/inner edges (Y = 41.3333, 39.3333 yd) at 20 and 30 yd lines
        ("Far '20' outer (41.33 yd)", 20.0, Y_FAR_NUM_OUTER_YD, 161.0),
        ("Far '20' inner (39.33 yd)", 20.0, Y_FAR_NUM_INNER_YD, 174.0),
        ("Far '30' outer (41.33 yd)", 30.0, Y_FAR_NUM_OUTER_YD, 160.0),
        ("Far '30' inner (39.33 yd)", 30.0, Y_FAR_NUM_INNER_YD, 173.0),
    ],
    "Frame 2 (NYJ vs JAX — Midfield Logo, No Full Sideline)": [
        # Far sideline visible on left half of frame (X = 50, 55, 60 yd)
        ("Sideline @ 50-yd", 50.0, FIELD_WIDTH_YD, 16.5),
        ("Sideline @ 45-yd", 55.0, FIELD_WIDTH_YD, 16.0),
        ("Sideline @ 40-yd", 60.0, FIELD_WIDTH_YD, 15.0),
        # Far numbers '50', '40', '30' (X = 50, 60, 70 yd)
        ("Far '50' outer (41.33 yd)", 50.0, Y_FAR_NUM_OUTER_YD, 101.0),
        ("Far '50' inner (39.33 yd)", 50.0, Y_FAR_NUM_INNER_YD, 117.0),
        ("Far '40' outer (41.33 yd)", 60.0, Y_FAR_NUM_OUTER_YD, 94.0),
        ("Far '40' inner (39.33 yd)", 60.0, Y_FAR_NUM_INNER_YD, 110.0),
        ("Far '30' outer (41.33 yd)", 70.0, Y_FAR_NUM_OUTER_YD, 84.0),
        ("Far '30' inner (39.33 yd)", 70.0, Y_FAR_NUM_INNER_YD, 100.0),
        # Near numbers '50', '40', '30' (X = 50, 60, 70 yd)
        ("Near '50' inner (14.0 yd)", 50.0, Y_NEAR_NUM_INNER_YD, 367.0),
        ("Near '50' outer (12.0 yd)", 50.0, Y_NEAR_NUM_OUTER_YD, 397.0),
        ("Near '40' inner (14.0 yd)", 60.0, Y_NEAR_NUM_INNER_YD, 350.0),
        ("Near '40' outer (12.0 yd)", 60.0, Y_NEAR_NUM_OUTER_YD, 378.0),
        ("Near '30' inner (14.0 yd)", 70.0, Y_NEAR_NUM_INNER_YD, 340.0),
        ("Near '30' outer (12.0 yd)", 70.0, Y_NEAR_NUM_OUTER_YD, 365.0),
    ],
    "Frame 3 (NO vs CAR — All-22 Oblique + Telestrator)": [
        # Far '10' number outer/inner edges on the 10-yard line (X = 90 yd)
        ("Far '10' outer (41.33 yd)", 90.0, Y_FAR_NUM_OUTER_YD, 164.0),
        ("Far '10' inner (39.33 yd)", 90.0, Y_FAR_NUM_INNER_YD, 204.0),
    ],
}


def evaluate_held_out(cal, landmarks):
    """Evaluate pixel and yard error on held-out landmarks along their respective yard lines."""
    px_errs = []
    yd_errs = []
    details = []
    for label, x_yd, y_yd, v_true in landmarks:
        # Predicted image point (u_pred, v_pred) for field point (x_yd, y_yd)
        pt_img = cv2.perspectiveTransform(np.float32([[[x_yd, y_yd]]]), cal.H_inv)[0, 0]
        u_pred, v_pred = float(pt_img[0]), float(pt_img[1])

        # True image point lies on the yard line x = x_yd at row v = v_true:
        # Find the yard line for x_yd or interpolate its line equation from H_inv
        p_top = cv2.perspectiveTransform(np.float32([[[x_yd, FIELD_WIDTH_YD]]]), cal.H_inv)[0, 0]
        p_bot = cv2.perspectiveTransform(np.float32([[[x_yd, 0.0]]]), cal.H_inv)[0, 0]
        m_line = (p_bot[0] - p_top[0]) / (p_bot[1] - p_top[1])
        u_true = float(p_top[0] + m_line * (v_true - p_top[1]))

        err_px = abs(v_pred - v_true)
        # Project true image point (u_true, v_true) into field coordinates via H
        pt_field_est = cv2.perspectiveTransform(np.float32([[[u_true, v_true]]]), cal.H)[0, 0]
        err_yd = float(np.hypot(pt_field_est[0] - x_yd, pt_field_est[1] - y_yd))

        px_errs.append(err_px)
        yd_errs.append(err_yd)
        details.append({
            "label": label,
            "true_v_px": round(v_true, 1),
            "pred_v_px": round(v_pred, 1),
            "err_px": round(err_px, 2),
            "err_yd": round(err_yd, 3),
            "err_ft": round(err_yd * 3.0, 2),
        })
    return {
        "n_landmarks": len(landmarks),
        "mean_err_px": round(float(np.mean(px_errs)), 2),
        "median_err_px": round(float(np.median(px_errs)), 2),
        "max_err_px": round(float(np.max(px_errs)), 2),
        "mean_err_yd": round(float(np.mean(yd_errs)), 3),
        "median_err_yd": round(float(np.median(yd_errs)), 3),
        "mean_err_ft": round(float(np.mean(yd_errs) * 3.0), 2),
        "median_err_ft": round(float(np.median(yd_errs) * 3.0), 2),
        "max_err_yd": round(float(np.max(yd_errs)), 3),
        "details": details,
    }


def draw_top_down_radar(
    cal,
    players,
    x_start_yd: float,
    x_span_yd: Tuple[float, float],
    width_px: int = 560,
    height_px: int = 340,
) -> np.ndarray:
    """Render a top-down NFL field radar panel (analogous to Alex's draw_radar in Hockey-Vision)."""
    radar = np.full((height_px, width_px, 3), (38, 108, 46), dtype=np.uint8)
    margin_x, margin_y = 36, 28
    usable_w = width_px - 2 * margin_x
    usable_h = height_px - 2 * margin_y

    x_min, x_max = x_span_yd

    def f2r(x_yd: float, y_yd: float) -> Tuple[int, int]:
        u = margin_x + (x_yd - x_min) / (x_max - x_min) * usable_w
        # y_yd = 53.333 is far sideline (top of radar), y_yd = 0 is near sideline (bottom of radar)
        v = margin_y + (1.0 - y_yd / FIELD_WIDTH_YD) * usable_h
        return int(round(u)), int(round(v))

    # Draw alternating 5-yard turf mowing stripes
    for x0 in range(int(np.floor(x_min / 5.0) * 5), int(np.ceil(x_max / 5.0) * 5), 5):
        if (x0 // 5) % 2 == 0:
            p1 = f2r(max(x_min, float(x0)), FIELD_WIDTH_YD)
            p2 = f2r(min(x_max, float(x0 + 5)), 0.0)
            cv2.rectangle(radar, p1, p2, (44, 118, 52), -1)

    # Draw end zone if x_max > 100
    if x_max > 100.0:
        p1 = f2r(100.0, FIELD_WIDTH_YD)
        p2 = f2r(x_max, 0.0)
        cv2.rectangle(radar, p1, p2, (32, 75, 38), -1)

    # Draw 5-yard lines and yard numbers
    for x0 in range(int(np.ceil(x_min / 5.0) * 5), int(np.floor(x_max / 5.0) * 5) + 1, 5):
        p_top = f2r(float(x0), FIELD_WIDTH_YD)
        p_bot = f2r(float(x0), 0.0)
        thick = 2 if x0 in (0, 50, 100) else 1
        cv2.line(radar, p_top, p_bot, (235, 235, 235), thick, cv2.LINE_AA)
        if x0 % 10 == 0 and 0 < x0 < 100:
            disp_num = str(x0 if x0 <= 50 else 100 - x0)
            _, v_far_n = f2r(float(x0), 40.3)
            _, v_near_n = f2r(float(x0), 13.0)
            cv2.putText(radar, disp_num, (p_top[0] - 10, v_far_n + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(radar, disp_num, (p_top[0] - 10, v_near_n + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

    # Draw 1-yard hash rows
    for x1 in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
        if x1 > 100:
            continue
        for y_h in (Y_NEAR_HASH_YD, Y_FAR_HASH_YD):
            pa = f2r(float(x1), y_h - 0.4)
            pb = f2r(float(x1), y_h + 0.4)
            cv2.line(radar, pa, pb, (220, 220, 220), 1)

    # Draw outer sidelines
    cv2.rectangle(radar, f2r(x_min, FIELD_WIDTH_YD), f2r(x_max, 0.0), (255, 255, 255), 2)

    # Draw players (Team A = red/gold, Team B = blue/cyan)
    for p in players:
        px, py = f2r(*p["field_xy_yd"])
        col = (50, 75, 235) if p["team"] == "team_a_player" else (235, 165, 30)
        cv2.circle(radar, (px, py), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(radar, (px, py), 5, col, -1, cv2.LINE_AA)

    return radar


def main():
    out_dir = Path("/home/user/Football-Vision/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_cfg = [
        (
            "Frame 1 (SEA vs SF — FOX Broadcast)",
            "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-5.jpg",
            15.0,
            (10.0, 40.0),
        ),
        (
            "Frame 2 (NYJ vs JAX — Midfield Logo, No Full Sideline)",
            "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-4.jpg",
            50.0,
            (45.0, 80.0),
        ),
        (
            "Frame 3 (NO vs CAR — All-22 Oblique + Telestrator)",
            "/home/user/image-search/nfl-all-22-film-pre-snap-formation-offen-5.png",
            80.0,
            (75.0, 105.0),
        ),
    ]

    summary = {"frames": {}}
    panel_rows = []

    for title, path, x_start_yd, x_span_yd in frames_cfg:
        img = cv2.imread(path)
        h, w = img.shape[:2]

        # 1. Naive Canny + Hough baseline
        naive_segs = naive_canny_hough(img)
        vis_naive = img.copy()
        for x1, y1, x2, y2 in naive_segs:
            dx, dy = x2 - x1, y2 - y1
            # Color horizontal-ish segments red (showing scorebug/clutter) and vertical-ish segments cyan
            col = (0, 0, 255) if abs(dy) < 0.5 * abs(dx) else (255, 220, 0)
            cv2.line(vis_naive, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

        # 2. Calibrate with Hash+VP ONLY (use_sideline_if_visible=False) so sidelines & numbers are 100% held-out!
        cal_centroid = calibrate_frame(img, x_start_yd=x_start_yd, use_centroid_hash_coords=True, use_sideline_if_visible=False)
        cal_bug185 = calibrate_frame(img, x_start_yd=x_start_yd, use_centroid_hash_coords=False, use_sideline_if_visible=False)
        cal_with_side = calibrate_frame(img, x_start_yd=x_start_yd, use_centroid_hash_coords=True, use_sideline_if_visible=True)

        held_out_centroid = evaluate_held_out(cal_centroid, HELD_OUT_LANDMARKS[title])
        held_out_bug185 = evaluate_held_out(cal_bug185, HELD_OUT_LANDMARKS[title])
        held_out_with_side = evaluate_held_out(cal_with_side, HELD_OUT_LANDMARKS[title])

        # 3. Player detection + K-Means team assignment + Alex's plausibility checks
        players, checks = detect_and_project_players(img, cal_with_side.H, x_span_yd)

        # Render calibrated reprojection overlay + detected players
        vis_cal = img.copy()
        H_inv = cal_with_side.H_inv

        def f2i(pts):
            p = np.float32(pts).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(p, H_inv).reshape(-1, 2)

        x_min, x_max = x_span_yd
        for y_yd, col, thick in [
            (0.0, (0, 140, 255), 2),
            (FIELD_WIDTH_YD, (0, 140, 255), 2),
            (Y_NEAR_NUM_OUTER_YD, (255, 210, 0), 1),
            (Y_NEAR_NUM_INNER_YD, (255, 210, 0), 1),
            (Y_FAR_NUM_INNER_YD, (255, 210, 0), 1),
            (Y_FAR_NUM_OUTER_YD, (255, 210, 0), 1),
            (Y_NEAR_HASH_YD, (0, 255, 255), 2),
            (Y_FAR_HASH_YD, (0, 255, 255), 2),
        ]:
            p1, p2 = f2i([[x_min, y_yd], [x_max, y_yd]])
            cv2.line(vis_cal, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), col, thick, cv2.LINE_AA)

        for x_yd in np.arange(x_min, x_max + 0.1, 5.0):
            if x_yd > 100.0:
                continue
            p1, p2 = f2i([[x_yd, 0.0], [x_yd, FIELD_WIDTH_YD]])
            cv2.line(vis_cal, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), (0, 255, 0), 2, cv2.LINE_AA)

        for px, py in cal_with_side.hash_tick_inliers:
            cv2.circle(vis_cal, (int(round(px)), int(round(py))), 4, (255, 0, 255), -1, cv2.LINE_AA)

        for p in players:
            bx, by, bw, bh = p["box_xywh"]
            col = (50, 75, 235) if p["team"] == "team_a_player" else (235, 165, 30)
            cv2.rectangle(vis_cal, (bx, by), (bx + bw, by + bh), col, 2)

        radar = draw_top_down_radar(cal_with_side, players, x_start_yd, x_span_yd, width_px=560, height_px=340)

        # Resize all 3 panels to 560x340 and stack horizontally with clean titles
        p_naive = cv2.resize(vis_naive, (560, 340), interpolation=cv2.INTER_AREA)
        p_cal = cv2.resize(vis_cal, (560, 340), interpolation=cv2.INTER_AREA)

        for panel, header in [
            (p_naive, f"Naive Canny+Hough ({len(naive_segs)} noisy segments)"),
            (
                p_cal,
                f"Ours: {len(cal_with_side.yard_lines)} yd-lines + {len(cal_with_side.hash_tick_inliers)} hash ticks (Held-out err: {held_out_with_side['median_err_yd']:.2f} yd)",
            ),
            (
                radar,
                f"Top-Down Radar: {checks['n_players']} players | H_med={checks['implied_median_height_ft']}ft | Spread={checks['rms_spread_ft']}ft",
            ),
        ]:
            cv2.rectangle(panel, (0, 0), (560, 28), (20, 20, 20), -1)
            cv2.putText(panel, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        row_img = np.hstack([p_naive, p_cal, radar])
        panel_rows.append(row_img)

        summary["frames"][title] = {
            "resolution": f"{w}x{h}",
            "naive_hough_segments": len(naive_segs),
            "yard_lines_detected": len(cal_with_side.yard_lines),
            "hash_ticks_inliers": int(len(cal_with_side.hash_tick_inliers)),
            "hash_ticks_candidates": int(len(cal_with_side.hash_tick_candidates)),
            "far_sideline_detected": cal_with_side.far_sideline is not None,
            "vp_yard_px": [round(cal_with_side.vp_yard[0], 1), round(cal_with_side.vp_yard[1], 1)],
            "in_sample_ridge_pixels": cal_with_side.ridge_pixel_count,
            "in_sample_ridge_median_orth_px": round(cal_with_side.ridge_orth_median_px, 2),
            "in_sample_hash_row_rmse_px": round(cal_with_side.hash_row_rmse_px, 2),
            "ten_yard_parity_scores": {
                "even_idx_mean": round(cal_with_side.even_idx_number_score, 3),
                "odd_idx_mean": round(cal_with_side.odd_idx_number_score, 3),
                "detected_parity": cal_with_side.detected_ten_yard_parity,
            },
            "held_out_ground_truth_hash_plus_vp_only": held_out_centroid,
            "held_out_ground_truth_with_sideline": held_out_with_side,
            "held_out_ground_truth_18_5ft_rulebook_bug": held_out_bug185,
            "alex_plausibility_checks": checks,
            "runtime_ms_cpu": round(cal_with_side.runtime_ms, 1),
        }

    full_grid = np.vstack(panel_rows)
    out_img_path = out_dir / "hough_calibration_diagnostic.png"
    cv2.imwrite(str(out_img_path), full_grid)

    out_json_path = out_dir / "week1_hough_benchmark.json"
    out_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
