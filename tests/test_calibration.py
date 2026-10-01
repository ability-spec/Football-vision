"""Unit and integration tests for Football-Vision Phase 0 calibration baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_vision import (
    FIELD_WIDTH_YD,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    extract_white_paint_ridge,
    calibrate_frame,
    plausible_homography,
    players_clustered,
    implied_player_height_ft,
    CalibrationTracker,
)
from experimental.cpu_blob_detector import detect_and_project_players
from benchmarks.evaluate_hough_week1 import HELD_OUT_LANDMARKS, evaluate_held_out


class TestCalibrationBaseline(unittest.TestCase):
    def test_nfl_rulebook_hash_centroid_spacing(self):
        """NFL inner edges are 18.5 ft apart, so 2-ft hash centroids are 20.5 ft (6.8333 yd) apart."""
        self.assertAlmostEqual(Y_NEAR_HASH_YD * 3.0, 69.75, places=4)
        self.assertAlmostEqual(Y_FAR_HASH_YD * 3.0, 90.25, places=4)
        self.assertAlmostEqual((Y_FAR_HASH_YD - Y_NEAR_HASH_YD) * 3.0, 20.50, places=4)
        self.assertAlmostEqual(FIELD_WIDTH_YD * 3.0, 160.0, places=4)

    def test_synthetic_virtual_lines_rejected_by_white_ridge_filter(self):
        """Yellow 1st-down line and blue LOS must be rejected while white yard line survives."""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = (55, 115, 55)
        img[40:250, 99:103] = (235, 235, 235)
        img[40:250, 198:203] = (0, 230, 255)
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

        H_mirror = H_valid.copy()
        H_mirror[0, 0] = -0.03
        self.assertFalse(plausible_homography(H_mirror, (900, 506)))

        clustered_pts = np.array([[25.0, 25.0], [25.5, 25.2], [24.8, 25.1], [25.1, 24.9]])
        spread_pts = np.array([[20.0, 15.0], [25.0, 25.0], [30.0, 35.0], [28.0, 18.0]])
        self.assertTrue(players_clustered(clustered_pts))
        self.assertFalse(players_clustered(spread_pts))

        boxes = [(400, 200, 20, 60), (450, 220, 20, 58), (500, 210, 20, 62)]
        h_ft = implied_player_height_ft(boxes, H_valid)
        self.assertAlmostEqual(h_ft, 5.4, places=1)

    def test_failure_reported_explicitly_on_uncalibratable_frame(self):
        """Blank turf frame with no yard lines must return success=False, H=None, and failure_reason."""
        blank_turf = np.full((506, 900, 3), (55, 115, 55), dtype=np.uint8)
        cal = calibrate_frame(blank_turf)
        self.assertFalse(cal.success)
        self.assertIsNone(cal.H)
        self.assertIsNone(cal.H_inv)
        self.assertEqual(cal.confidence, 0.0)
        self.assertEqual(cal.x_coord_mode, "uncalibrated")
        self.assertIsNotNone(cal.failure_reason)

    def test_calibration_tracker_smoothing_jump_reset_and_expiry(self):
        H1 = np.array([
            [0.03, 0.0, 10.0],
            [0.0, -0.08, 45.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        tracker = CalibrationTracker(smoothing=0.5, max_age=2, max_jump_yd=4.0)
        out1 = tracker.update(H1, (900, 506))
        np.testing.assert_allclose(out1, H1)

        # Small shift (1.0 yd < 4.0 yd): blended 50/50
        H2 = H1.copy()
        H2[0, 2] += 1.0
        out2 = tracker.update(H2, (900, 506))
        np.testing.assert_allclose(out2, 0.5 * H1 + 0.5 * H2)

        # Large shift (10.0 yd > 4.0 yd): resets to H3 without blending
        H3 = H1.copy()
        H3[0, 2] += 10.0
        out3 = tracker.update(H3, (900, 506))
        np.testing.assert_allclose(out3, H3)

        # Missing frames: kept up to max_age=2, then expires to None
        self.assertIsNotNone(tracker.update(None, (900, 506)))
        self.assertIsNotNone(tracker.update(None, (900, 506)))
        self.assertIsNone(tracker.update(None, (900, 506)))

    def test_real_nfl_frames_exact_week1_regression_numbers(self):
        """Verify exact Week-1 held-out median errors (0.275 yd, 0.350 yd, 0.057 yd) and parity."""
        cases = [
            (
                "Frame 1 (SEA vs SF — FOX Broadcast)",
                "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-5.jpg",
                15.0,
                (10.0, 40.0),
                5,
                1,
                0.275,  # Hash+VP only median error (yd)
                0.302,  # With far sideline median error (yd)
                23,
                3.63,
            ),
            (
                "Frame 2 (NYJ vs JAX — Midfield Logo, No Full Sideline)",
                "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-4.jpg",
                50.0,
                (45.0, 80.0),
                6,
                0,
                0.350,  # Hash+VP only median error (yd)
                0.301,  # With partial far sideline median error (yd)
                24,
                3.64,
            ),
            (
                "Frame 3 (NO vs CAR — All-22 Oblique + Telestrator)",
                "/home/user/image-search/nfl-all-22-film-pre-snap-formation-offen-5.png",
                80.0,
                (75.0, 105.0),
                5,
                0,
                0.057,  # Hash+VP only median error (yd)
                0.057,  # No sideline visible -> identical
                12,
                3.59,
            ),
        ]
        for (
            title, path, x0, x_span, exp_lines, exp_parity,
            exp_med_hash_only, exp_med_with_side, exp_players, exp_h_ft
        ) in cases:
            img = cv2.imread(path)
            self.assertIsNotNone(img, f"Missing test image {path}")

            cal_hash_only = calibrate_frame(img, x_start_yd=x0, use_centroid_hash_coords=True, use_sideline_if_visible=False)
            cal_with_side = calibrate_frame(img, x_start_yd=x0, use_centroid_hash_coords=True, use_sideline_if_visible=True)

            self.assertTrue(cal_hash_only.success)
            self.assertTrue(cal_with_side.success)
            self.assertEqual(len(cal_with_side.yard_lines), exp_lines)
            self.assertEqual(cal_with_side.detected_ten_yard_parity, exp_parity)
            self.assertLess(cal_with_side.ridge_orth_median_px, 1.5)
            self.assertLess(cal_with_side.hash_row_rmse_px, 1.5)

            ho_hash_only = evaluate_held_out(cal_hash_only, HELD_OUT_LANDMARKS[title])
            ho_with_side = evaluate_held_out(cal_with_side, HELD_OUT_LANDMARKS[title])
            self.assertAlmostEqual(ho_hash_only["median_err_yd"], exp_med_hash_only, places=3)
            self.assertAlmostEqual(ho_with_side["median_err_yd"], exp_med_with_side, places=3)

            players, checks = detect_and_project_players(img, cal_with_side.H, x_span)
            self.assertEqual(len(players), exp_players)
            self.assertAlmostEqual(checks["implied_median_height_ft"], exp_h_ft, places=2)
            self.assertFalse(checks["clustered_flag"])
            self.assertTrue(checks["height_plausible"])


if __name__ == "__main__":
    unittest.main()
