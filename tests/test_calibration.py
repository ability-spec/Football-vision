"""Unit and integration tests for Football-Vision Phase 1 Field Calibration Engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_vision.data_paths import (  # noqa: E402
    real_frames_skip_reason,
    resolve_nfl_frame,
)
from football_vision import (
    FIELD_WIDTH_YD,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    CalibrationResult,
    extract_white_paint_ridge,
    calibrate_frame,
    resolve_x_coord_mode_and_parity,
    image_to_field,
    field_to_image,
    plausible_homography,
    players_clustered,
    implied_player_height_ft,
    CalibrationTracker,
)
from experimental.cpu_blob_detector import detect_and_project_players
from benchmarks.evaluate_hough_week1 import HELD_OUT_LANDMARKS, evaluate_held_out


def _make_synthetic_field_without_numbers(width: int = 900, height: int = 506) -> np.ndarray:
    """Create a synthetic broadcast turf frame with 4 converging 5-yd lines and 1-yd hash ticks,
    but NO painted yard numbers, to test `relative_5yd` coordinate mode.
    """
    img = np.full((height, width, 3), (50, 115, 50), dtype=np.uint8)
    x_vp, y_vp = 450.0, -700.0
    y_mid = height / 2.0
    x_mids = [180.0, 340.0, 500.0, 660.0]

    # Draw 4 white 5-yard lines converging toward (x_vp, y_vp)
    for xm in x_mids:
        m = (xm - x_vp) / (y_mid - y_vp)
        b = xm - m * y_mid
        p1 = (int(round(m * 50 + b)), 50)
        p2 = (int(round(m * 420 + b)), 420)
        cv2.line(img, p1, p2, (235, 235, 235), 3)

    # Draw 1-yard hash ticks along two horizontal-ish rows (far row y=195, near row y=275)
    for k in range(len(x_mids) - 1):
        xm0, xm1 = x_mids[k], x_mids[k + 1]
        m0 = (xm0 - x_vp) / (y_mid - y_vp)
        b0 = xm0 - m0 * y_mid
        m1 = (xm1 - x_vp) / (y_mid - y_vp)
        b1 = xm1 - m1 * y_mid
        for t in (0.2, 0.4, 0.6, 0.8):
            mt = (1.0 - t) * m0 + t * m1
            bt = (1.0 - t) * b0 + t * b1
            for y_row in (195, 275):
                x_tick = int(round(mt * y_row + bt))
                cv2.line(img, (x_tick, y_row - 4), (x_tick, y_row + 4), (235, 235, 235), 3)
    return img


# Real NFL broadcast stills are third-party assets and are NOT vendored. They are resolved
# through ``football_vision.data_paths`` (honours FOOTBALL_VISION_NFL_FRAMES); tests that need
# them skip with an explicit reason instead of failing on a machine without the assets.
def _real_frame_path(name: str) -> str:
    path = resolve_nfl_frame(name)
    if path is None:
        raise unittest.SkipTest(real_frames_skip_reason())
    return str(path)


def _real_frame(name: str) -> np.ndarray:
    path = _real_frame_path(name)
    img = cv2.imread(path)
    assert img is not None, path
    return img


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

    def test_explicit_failure_modes_and_projection_refusal(self):
        """Every failure mode must report a specific failure_reason and refuse coordinate projection."""
        # 1. Invalid image
        cal_inv = calibrate_frame(None)
        self.assertFalse(cal_inv.success)
        self.assertEqual(cal_inv.failure_reason, "invalid_image")
        self.assertEqual(cal_inv.x_coord_mode, "uncalibrated")
        self.assertIsNone(image_to_field([[450.0, 250.0]], cal_inv))
        self.assertIsNone(field_to_image([[25.0, 25.0]], cal_inv))

        # 2. Blank turf -> no_hough_lines
        blank_turf = np.full((506, 900, 3), (55, 115, 55), dtype=np.uint8)
        cal_blank = calibrate_frame(blank_turf)
        self.assertFalse(cal_blank.success)
        self.assertEqual(cal_blank.failure_reason, "no_hough_lines")
        self.assertIsNone(cal_blank.H)
        self.assertEqual(cal_blank.confidence, 0.0)

        # 3. Only 2 yard lines -> insufficient_yard_lines
        two_lines = blank_turf.copy()
        cv2.line(two_lines, (250, 50), (200, 420), (235, 235, 235), 3)
        cv2.line(two_lines, (450, 50), (430, 420), (235, 235, 235), 3)
        cal_two = calibrate_frame(two_lines)
        self.assertFalse(cal_two.success)
        self.assertEqual(cal_two.failure_reason, "insufficient_yard_lines")

        # 4. 4 yard lines in a pencil, but zero hash marks -> insufficient_hash_ticks
        four_lines_no_hashes = blank_turf.copy()
        for xm in (180.0, 340.0, 500.0, 660.0):
            m = (xm - 450.0) / (253.0 - (-700.0))
            b = xm - m * 253.0
            cv2.line(four_lines_no_hashes, (int(m * 50 + b), 50), (int(m * 420 + b), 420), (235, 235, 235), 3)
        cal_no_hash = calibrate_frame(four_lines_no_hashes)
        self.assertFalse(cal_no_hash.success)
        self.assertEqual(cal_no_hash.failure_reason, "insufficient_hash_ticks")
        self.assertEqual(len(cal_no_hash.yard_lines), 4)

        # 5. Insufficient confidence threshold -> insufficient_confidence
        img1 = _real_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
        cal_strict = calibrate_frame(img1, x_start_yd=15.0, min_confidence=0.99)
        self.assertFalse(cal_strict.success)
        self.assertEqual(cal_strict.failure_reason, "insufficient_confidence")
        self.assertIsNone(cal_strict.H)
        self.assertEqual(cal_strict.x_coord_mode, "uncalibrated")
        self.assertIsNone(cal_strict.image_to_field([[450.0, 250.0]]))

    def test_x_coord_mode_resolution_never_guesses_absolute(self):
        """Verify relative_5yd, relative_10yd, and absolute modes under resolve_x_coord_mode_and_parity
        and on a synthetic field without painted numbers.
        """
        # Clear 10-yd parity, unverified x_start -> relative_10yd
        p, mode = resolve_x_coord_mode_and_parity(0.22, 0.01, x_start_verified=False)
        self.assertEqual((p, mode), (0, "relative_10yd"))

        # Clear 10-yd parity, verified x_start -> absolute
        p, mode = resolve_x_coord_mode_and_parity(0.01, 0.19, x_start_verified=True)
        self.assertEqual((p, mode), (1, "absolute"))

        # Weak / missing painted numbers -> parity=None, mode=relative_5yd (even if x_start_verified=True!)
        p, mode = resolve_x_coord_mode_and_parity(0.01, 0.00, x_start_verified=True)
        self.assertEqual((p, mode), (None, "relative_5yd"))

        # Ambiguous ratio (< 2.0x) -> parity=None, mode=relative_5yd
        p, mode = resolve_x_coord_mode_and_parity(0.12, 0.08, x_start_verified=False)
        self.assertEqual((p, mode), (None, "relative_5yd"))

        # Synthetic field with 4 yard lines + 24 hash ticks + NO numbers -> relative_5yd
        syn_img = _make_synthetic_field_without_numbers()
        cal_syn = calibrate_frame(syn_img, x_start_yd=20.0, use_sideline_if_visible=False)
        self.assertTrue(cal_syn.success)
        self.assertIsNone(cal_syn.detected_ten_yard_parity)
        self.assertEqual(cal_syn.x_coord_mode, "relative_5yd")
        self.assertIn("yards_gained_dx_yd", cal_syn.valid_metrics)
        self.assertNotIn("ten_yard_grid_offset_yd", cal_syn.valid_metrics)
        self.assertNotIn("absolute_yard_line_x", cal_syn.valid_metrics)

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

    def test_calibration_tracker_optical_flow_propagation_confidence_decay_and_camera_cut(self):
        """Test CalibrationTracker.update_from_result across direct observation, optical-flow bridged
        pan, confidence decay expiration, and camera-cut reset.
        """
        img1 = _real_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
        cal1 = calibrate_frame(img1, x_start_yd=15.0)
        self.assertTrue(cal1.success)

        tracker = CalibrationTracker(
            smoothing=0.5, max_age=3, max_jump_yd=4.0, decay_factor=0.80, min_confidence=0.60
        )

        # Step 1: Direct calibration on Frame 1
        res1 = tracker.update_from_result(cal1, frame=img1)
        self.assertTrue(res1.success)
        self.assertFalse(res1.is_temporally_propagated)
        self.assertAlmostEqual(res1.confidence, cal1.confidence, places=4)
        self.assertTrue(tracker.can_project())

        # Step 2: Simulate a 6-px horizontal camera pan where direct single-frame calibration dropped out
        M_pan = np.float32([[1.0, 0.0, 6.0], [0.0, 1.0, 0.0]])
        img_panned = cv2.warpAffine(img1, M_pan, (img1.shape[1], img1.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        failed_obs = CalibrationResult(
            success=False, H=None, H_inv=None, image_size=(img1.shape[1], img1.shape[0]),
            failure_reason="insufficient_yard_lines",
        )
        res2 = tracker.update_from_result(failed_obs, frame=img_panned)
        self.assertTrue(res2.success)
        self.assertTrue(res2.is_temporally_propagated)
        self.assertEqual(res2.propagation_age, 1)
        self.assertAlmostEqual(res2.confidence, round(cal1.confidence * 0.80, 4), places=4)
        # Verify optical-flow propagation shifted the projected point consistently with the 6-px pan:
        # Point (450, 250) in frame 1 moved to (456, 250) in img_panned, so projecting (456, 250) under
        # res2.H should match projecting (450, 250) under res1.H within < 0.15 yd!
        pt_f1 = res1.image_to_field([[450.0, 250.0]])[0]
        pt_f2 = res2.image_to_field([[456.0, 250.0]])[0]
        np.testing.assert_allclose(pt_f2, pt_f1, atol=0.15)

        # Step 3: Second consecutive unobserved frame -> confidence decays below min_confidence=0.60
        # (~0.899 * 0.80 * 0.80 = 0.575 < 0.60) -> tracker MUST refuse projection!
        res3 = tracker.update_from_result(failed_obs, frame=img_panned)
        self.assertFalse(res3.success)
        self.assertIsNone(res3.H)
        self.assertEqual(res3.x_coord_mode, "uncalibrated")
        self.assertEqual(res3.failure_reason, "confidence_expired")
        self.assertFalse(tracker.can_project())
        self.assertIsNone(tracker.project_image_points([[456.0, 250.0]]))

        # Step 4: Re-lock on Frame 1, then simulate a camera cut to a non-field blue graphic frame
        tracker.update_from_result(cal1, frame=img1)
        self.assertTrue(tracker.can_project())
        cut_frame = np.full_like(img1, (180, 40, 20))  # high-saturation blue graphic
        res_cut = tracker.update_from_result(failed_obs, frame=cut_frame)
        self.assertFalse(res_cut.success)
        self.assertIsNone(res_cut.H)
        self.assertEqual(res_cut.failure_reason, "camera_cut_uncalibrated")
        self.assertEqual(res_cut.confidence, 0.0)
        self.assertFalse(tracker.can_project())

    def test_real_nfl_frames_exact_week1_regression_numbers(self):
        """Verify exact Week-1 held-out median errors (0.275 yd, 0.350 yd, 0.057 yd), parity, and confidence."""
        cases = [
            (
                "Frame 1 (SEA vs SF — FOX Broadcast)",
                _real_frame_path("nfl-game-broadcast-screenshot-1st-and-10-5.jpg"),
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
                _real_frame_path("nfl-game-broadcast-screenshot-1st-and-10-4.jpg"),
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
                _real_frame_path("nfl-all-22-film-pre-snap-formation-offen-5.png"),
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
            cal_verified = calibrate_frame(
                img, x_start_yd=x0, use_centroid_hash_coords=True, use_sideline_if_visible=True, x_start_verified=True
            )

            self.assertTrue(cal_hash_only.success)
            self.assertTrue(cal_with_side.success)
            self.assertEqual(cal_with_side.x_coord_mode, "relative_10yd")
            self.assertEqual(cal_verified.x_coord_mode, "absolute")
            self.assertGreater(cal_with_side.confidence, 0.80)
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
