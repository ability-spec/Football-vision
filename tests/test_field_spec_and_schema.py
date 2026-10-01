"""Unit tests for NFL field geometry specifications, core data schemas, coordinate modes, and module boundaries."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import football_vision
from football_vision.field_spec import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_FT,
    FIELD_WIDTH_YD,
    HASH_SPACING_CENTER_FT,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    Y_NEAR_NUM_OUTER_YD,
    Y_NEAR_NUM_INNER_YD,
    Y_FAR_NUM_INNER_YD,
    Y_FAR_NUM_OUTER_YD,
    VALID_METRICS_BY_X_COORD_MODE,
    yards_to_feet,
    feet_to_yards,
)
from football_vision.schema import (
    CalibrationResult,
    Detection,
    PlayerIdentity,
    TrackState,
    FieldProjection,
    TrajectoryPoint,
    PlayMetric,
    PlayRecord,
)


class TestFieldSpecAndSchema(unittest.TestCase):
    def test_nfl_rulebook_geometry_constants_and_conversions(self):
        self.assertEqual(FIELD_LENGTH_YD, 100.0)
        self.assertEqual(FIELD_WIDTH_FT, 160.0)
        self.assertAlmostEqual(FIELD_WIDTH_YD * 3.0, 160.0, places=6)
        self.assertAlmostEqual(Y_NEAR_HASH_YD * 3.0, 69.75, places=6)
        self.assertAlmostEqual(Y_FAR_HASH_YD * 3.0, 90.25, places=6)
        self.assertAlmostEqual(HASH_SPACING_CENTER_FT, 20.50, places=6)
        self.assertAlmostEqual(Y_NEAR_NUM_OUTER_YD + Y_FAR_NUM_OUTER_YD, FIELD_WIDTH_YD, places=6)
        self.assertAlmostEqual(Y_NEAR_NUM_INNER_YD + Y_FAR_NUM_INNER_YD, FIELD_WIDTH_YD, places=6)
        self.assertAlmostEqual(yards_to_feet(10.0), 30.0, places=6)
        self.assertAlmostEqual(feet_to_yards(30.0), 10.0, places=6)

    def test_valid_metrics_by_x_coord_mode_hierarchy(self):
        abs_m = set(VALID_METRICS_BY_X_COORD_MODE["absolute"])
        rel10_m = set(VALID_METRICS_BY_X_COORD_MODE["relative_10yd"])
        rel5_m = set(VALID_METRICS_BY_X_COORD_MODE["relative_5yd"])
        uncal_m = set(VALID_METRICS_BY_X_COORD_MODE["uncalibrated"])

        self.assertEqual(uncal_m, set())
        self.assertTrue(rel5_m.issubset(rel10_m))
        self.assertTrue(rel10_m.issubset(abs_m))
        self.assertIn("yards_gained_dx_yd", rel5_m)
        self.assertIn("lateral_position_y_yd", rel5_m)
        self.assertIn("ten_yard_grid_offset_yd", rel10_m)
        self.assertNotIn("ten_yard_grid_offset_yd", rel5_m)
        self.assertIn("absolute_yard_line_x", abs_m)
        self.assertNotIn("absolute_yard_line_x", rel10_m)

    def test_schema_serialization_roundtrip_and_contracts(self):
        H = np.eye(3, dtype=np.float64)
        cal = CalibrationResult(
            success=True,
            H=H,
            H_inv=H,
            yard_lines=[{"k": 0, "x_mid": 100.0, "m": 0.0, "b": 100.0}],
            vp_yard=(450.0, -900.0),
            hash_tick_inliers=np.zeros((12, 2)),
            hash_tick_candidates=np.zeros((15, 2)),
            ridge_pixel_count=1500,
            ridge_orth_median_px=0.95,
            ridge_orth_mean_px=1.05,
            hash_row_rmse_px=0.72,
            even_idx_number_score=0.21,
            odd_idx_number_score=0.02,
            detected_ten_yard_parity=0,
            plausible_orientation_scale=True,
            runtime_ms=80.0,
            image_size=(900, 506),
            confidence=0.88,
            confidence_components={"line_support": 1.0, "ridge_residual": 0.76, "hash_support": 1.0, "hash_residual": 0.82},
            x_coord_mode="relative_10yd",
        )
        self.assertEqual(cal.vanishing_point, (450.0, -900.0))
        self.assertIs(cal.field_homography, H)
        self.assertIn("far_sideline", cal.sidelines)
        self.assertIn("rmse_px", cal.hash_marks)
        self.assertAlmostEqual(cal.residuals["ridge_orth_median_px"], 0.95)
        self.assertTrue(cal.can_project())

        # Verify image_to_field and field_to_image work when can_project() is True,
        # and refuse projection (return None) when confidence is below min_confidence.
        pts = np.array([[25.0, 30.0], [50.0, 20.0]])
        np.testing.assert_allclose(cal.image_to_field(pts), pts)
        np.testing.assert_allclose(cal.field_to_image(pts), pts)
        self.assertIsNone(cal.image_to_field(pts, min_confidence=0.95))
        self.assertIsNone(cal.field_to_image(pts, min_confidence=0.95))

        cal_dict = cal.to_dict()
        serialized = json.dumps(cal_dict)
        loaded = json.loads(serialized)
        self.assertTrue(loaded["success"])
        self.assertEqual(loaded["image_size"], [900, 506])
        self.assertEqual(loaded["hash_tick_inliers_count"], 12)
        self.assertEqual(loaded["x_coord_mode"], "relative_10yd")
        self.assertIn("yards_gained_dx_yd", loaded["valid_metrics"])
        self.assertNotIn("absolute_yard_line_x", loaded["valid_metrics"])

        det = Detection(frame_index=5, box_xyxy=(100.0, 200.0, 140.0, 280.0), cls="player", confidence=0.92)
        self.assertEqual(det.bottom_center, (120.0, 280.0))
        self.assertEqual(det.to_dict()["cls"], "player")

        ident = PlayerIdentity(
            track_id=7, team="team_a_player", team_confidence=0.95,
            jersey_number="87", jersey_confidence=0.88, evidence=["torso_lab", "ocr_frame_5"],
        )
        track = TrackState(
            track_id=7, frame_index=5, box_xyxy=(100.0, 200.0, 140.0, 280.0),
            cls="team_a_player", detection_confidence=0.92, identity=ident,
        )
        self.assertEqual(track.to_dict()["identity"]["jersey_number"], "87")

        proj = FieldProjection(
            frame_index=5, timestamp_s=0.167, track_id=7, cls="team_a_player",
            x_yd=35.0, y_yd=24.5, x_coord_mode="relative_10yd",
            detection_confidence=0.92, calibration_confidence=1.0, identity_confidence=0.88,
        )
        self.assertEqual(proj.to_dict()["x_yd"], 35.0)

        tp = TrajectoryPoint(
            track_id=7, frame_index=5, timestamp_s=0.167, x_yd=35.0, y_yd=24.5,
            vx_yd_s=6.5, vy_yd_s=1.2, speed_yd_s=6.61, confidence=0.90,
        )
        self.assertAlmostEqual(tp.to_dict()["speed_yd_s"], 6.61)

        metric = PlayMetric(
            name="formation_width_yd", value=32.4, units="yards",
            data_source="pre_snap_frame_0", confidence=0.95, limitations="Excludes off-screen wideouts",
        )
        play = PlayRecord(
            game_id="SEA_SF_2025", play_id="play_001", start_frame=0, snap_frame=15, end_frame=120,
            x_coord_mode="relative_10yd", metrics={"formation_width_yd": metric},
        )
        play_json = json.loads(json.dumps(play.to_dict()))
        self.assertAlmostEqual(play_json["metrics"]["formation_width_yd"]["value"], 32.4)

    def test_production_calibration_has_no_experimental_player_detector_dependency(self):
        """Ensure detect_and_project_players is isolated from football_vision production namespace."""
        self.assertFalse(hasattr(football_vision, "detect_and_project_players"))
        self.assertFalse(hasattr(football_vision.calibration, "detect_and_project_players"))


if __name__ == "__main__":
    unittest.main()
