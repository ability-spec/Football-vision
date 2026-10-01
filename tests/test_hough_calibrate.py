"""Unit and integration tests for Football-Vision Week-1 Hough + Vanishing-Point Calibration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hough_calibrate import (
    FIELD_WIDTH_YD,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    extract_white_paint_ridge,
    calibrate_frame,
    plausible_homography,
    players_clustered,
    implied_player_height_ft,
)
from evaluate_hough_week1 import HELD_OUT_LANDMARKS, evaluate_held_out


class TestHoughCalibrate(unittest.TestCase):
    def test_nfl_rulebook_hash_centroid_spacing(self):
        """NFL inner edges are 18.5 ft apart, so 2-ft hash centroids are 20.5 ft (6.8333 yd) apart."""
        self.assertAlmostEqual(Y_NEAR_HASH_YD * 3.0, 69.75, places=4)
        self.assertAlmostEqual(Y_FAR_HASH_YD * 3.0, 90.25, places=4)
        self.assertAlmostEqual((Y_FAR_HASH_YD - Y_NEAR_HASH_YD) * 3.0, 20.50, places=4)
        self.assertAlmostEqual(FIELD_WIDTH_YD * 3.0, 160.0, places=4)

    def test_synthetic_virtual_lines_rejected_by_white_ridge_filter(self):
        """Yellow 1st-down line and blue LOS must be rejected while white yard line survives."""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        # Green turf background (BGR ~ (55, 115, 55))
        img[:, :] = (55, 115, 55)
        # White yard line at x = 100
        img[40:250, 99:103] = (235, 235, 235)
        # Yellow 1st-down line at x = 200 (BGR = (0, 230, 255))
        img[40:250, 198:203] = (0, 230, 255)
        # Blue LOS at x = 300 (BGR = (230, 70, 20))
        img[40:250, 298:303] = (230, 70, 20)

        ridge_u8, _, _ = extract_white_paint_ridge(img)
        self.assertGreater(np.sum(ridge_u8[50:240, 97:105] > 0), 150)
        self.assertEqual(np.sum(ridge_u8[50:240, 195:206] > 0), 0)
        self.assertEqual(np.sum(ridge_u8[50:240, 295:306] > 0), 0)

    def test_plausibility_checks_reject_mirrored_and_collapsed_fits(self):
        """Ported Alex Haigh plausibility checks must accept valid H and reject mirrored/singular H."""
        H_valid = np.array([
            [0.03, 0.0, 10.0],
            [0.0, -0.08, 45.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.assertTrue(plausible_homography(H_valid, (900, 506)))

        # Mirrored X axis -> negative cross product
        H_mirror = H_valid.copy()
        H_mirror[0, 0] = -0.03
        self.assertFalse(plausible_homography(H_mirror, (900, 506)))

        # Clustered players (< 3.0 yd RMS spread)
        clustered_pts = np.array([[25.0, 25.0], [25.5, 25.2], [24.8, 25.1], [25.1, 24.9]])
        spread_pts = np.array([[20.0, 15.0], [25.0, 25.0], [30.0, 35.0], [28.0, 18.0]])
        self.assertTrue(players_clustered(clustered_pts))
        self.assertFalse(players_clustered(spread_pts))

        # Implied player height (60 px box at 0.03 yd/px = 1.8 yd = 5.4 ft)
        boxes = [(400, 200, 20, 60), (450, 220, 20, 58), (500, 210, 20, 62)]
        h_ft = implied_player_height_ft(boxes, H_valid)
        self.assertAlmostEqual(h_ft, 5.4, places=1)

    def test_real_nfl_frames_calibration_and_held_out_accuracy(self):
        """All 3 real NFL frames must calibrate with < 0.40 yd held-out median error and correct 10-yd parity."""
        cases = [
            (
                "Frame 1 (SEA vs SF — FOX Broadcast)",
                "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-5.jpg",
                15.0,
                5,
                1,  # k=1 (20-yd) and k=3 (30-yd) have painted numbers
            ),
            (
                "Frame 2 (NYJ vs JAX — Midfield Logo, No Full Sideline)",
                "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-4.jpg",
                50.0,
                6,
                0,  # k=0 (50-yd), k=2 (40-yd), k=4 (30-yd) have painted numbers
            ),
            (
                "Frame 3 (NO vs CAR — All-22 Oblique + Telestrator)",
                "/home/user/image-search/nfl-all-22-film-pre-snap-formation-offen-5.png",
                80.0,
                5,
                0,  # k=2 (10-yd) has painted numbers
            ),
        ]
        for title, path, x0, exp_lines, exp_parity in cases:
            img = cv2.imread(path)
            self.assertIsNotNone(img, f"Missing test image {path}")
            cal = calibrate_frame(img, x_start_yd=x0, use_centroid_hash_coords=True, use_sideline_if_visible=True)
            self.assertTrue(cal.success, f"Calibration failed on {title}")
            self.assertEqual(len(cal.yard_lines), exp_lines)
            self.assertEqual(cal.detected_ten_yard_parity, exp_parity)
            self.assertLess(cal.ridge_orth_median_px, 1.5)
            self.assertLess(cal.hash_row_rmse_px, 1.5)

            held_out = evaluate_held_out(cal, HELD_OUT_LANDMARKS[title])
            self.assertLess(held_out["median_err_yd"], 0.40, f"Held-out error too high on {title}")


if __name__ == "__main__":
    unittest.main()
