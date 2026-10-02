"""Phase 3 Unit and Integration Tests for Player Detection, Footpoints,
Team Assignment, Multi-Object Tracking, and Gated Field Projection (9 tests).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from football_vision.data_paths import (  # noqa: E402
    real_frames_skip_reason,
    resolve_nfl_frame,
)
from football_vision import (
    CalibrationResult,
    FieldProjector,
    FixturePlayerDetector,
    PlayerTracker,
    TorsoTeamClassifier,
    TurfContrastPlayerDetector,
    calibrate_frame,
    extract_footpoint,
    extract_footpoints_for_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def _real_frame(name: str) -> np.ndarray:
    """Load a real NFL still, or skip the test when the unvendored asset is absent."""
    path = resolve_nfl_frame(name)
    if path is None:
        pytest.skip(real_frames_skip_reason())
    img = cv2.imread(str(path))
    assert img is not None, path
    return img
PHASE3_BENCHMARK_JSON = ROOT / "outputs" / "phase3_tracking_benchmark.json"


def _make_turf_canvas(w: int = 640, h: int = 360) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (45, 125, 45)  # Green turf BGR
    return canvas


def _paint_box(canvas: np.ndarray, bbox: tuple[float, float, float, float], bgr: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    canvas[y1:y2, x1:x2] = bgr


def test_detection_schema_and_independence_from_calibration():
    # 1. Verify no Phase 3 module imports experimental cpu_standin_perception
    for sub in ("detection", "identity", "tracking", "projection"):
        for py_file in (ROOT / "football_vision" / sub).glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "experiments" not in node.module
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "experiments" not in alias.name

    # 2. Verify detector runs purely in image space and returns canonical schema
    img = _real_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
    detector = TurfContrastPlayerDetector()
    dets = detector.detect(img, frame_id=7)
    assert len(dets) >= 10
    d0 = dets[0]
    assert d0.frame_id == 7
    assert d0.detection_id.startswith("f0007_det")
    assert len(d0.bbox) == 4
    assert 0.0 <= d0.confidence <= 1.0
    assert d0.footpoint == d0.footpoint_estimate.xy
    assert d0.detector_metadata["detector_name"] == "turf_contrast_baseline_v1"
    d_dict = d0.to_dict()
    assert set(d_dict.keys()) == {
        "frame_id",
        "detection_id",
        "bbox",
        "confidence",
        "footpoint",
        "footpoint_estimate",
        "detector_metadata",
    }


def test_footpoint_extraction_and_unreliable_flags():
    w, h = 900, 500

    # Clean mid-field upright player
    fp_clean = extract_footpoint((400.0, 200.0, 424.0, 254.0), (w, h), detection_confidence=0.92)
    assert fp_clean.u_px == 412.0
    assert fp_clean.v_px == 254.0
    assert fp_clean.is_reliable is True
    assert fp_clean.unreliable_reason is None
    assert fp_clean.confidence > 0.85

    # Bottom-border truncated player (cleats clipped below bottom edge)
    fp_trunc = extract_footpoint((300.0, 445.0, 325.0, 499.0), (w, h), detection_confidence=0.95)
    assert fp_trunc.is_partially_truncated is True
    assert fp_trunc.is_reliable is False
    assert fp_trunc.unreliable_reason == "bottom_border_truncated"
    assert fp_trunc.confidence <= 0.35

    # Sideline proximity flag
    fp_side = extract_footpoint((250.0, 20.0, 270.0, 62.0), (w, h), sideline_v_bounds_px=(70.0, 470.0))
    assert fp_side.is_near_sideline is True

    # Squat upper-body-only occluded box
    fp_squat = extract_footpoint((500.0, 220.0, 530.0, 242.0), (w, h))
    assert fp_squat.is_temporarily_occluded is True
    assert fp_squat.is_reliable is False
    assert fp_squat.unreliable_reason == "lower_body_occluded_aspect_ratio"

    # Players crossing each other: front player (y2=270) occludes lower third of back player (y2=266)
    boxes = [
        (300.0, 214.0, 324.0, 266.0),  # Back player
        (302.0, 218.0, 326.0, 270.0),  # Front crossing player covering back player's lower body
    ]
    fps = extract_footpoints_for_frame(boxes, (w, h))
    assert fps[0].is_player_crossing is True
    assert fps[0].is_temporarily_occluded is True
    assert fps[0].is_reliable is False
    assert fps[0].unreliable_reason == "player_crossing_lower_body_occluded"
    assert fps[1].is_player_crossing is True
    assert fps[1].is_reliable is True


def test_team_assignment_dark_vs_light_and_deterministic_ordering():
    canvas = _make_turf_canvas(640, 360)
    bboxes = [
        (100.0, 120.0, 130.0, 185.0),  # Light jersey 1
        (200.0, 120.0, 230.0, 185.0),  # Dark jersey 1
        (300.0, 120.0, 330.0, 185.0),  # Light jersey 2
        (400.0, 120.0, 430.0, 185.0),  # Dark jersey 2
    ]
    _paint_box(canvas, bboxes[0], (240, 240, 240))
    _paint_box(canvas, bboxes[1], (55, 25, 15))
    _paint_box(canvas, bboxes[2], (235, 238, 235))
    _paint_box(canvas, bboxes[3], (50, 20, 15))

    det = FixturePlayerDetector({0: [{"bbox": b, "confidence": 0.9} for b in bboxes]})
    dets = det.detect(canvas, frame_id=0)
    clf = TorsoTeamClassifier()
    res = clf.classify_detections(canvas, dets)

    # TEAM_A is always the darker jersey cluster, TEAM_B is always the lighter jersey cluster
    assert [r.team for r in res] == ["TEAM_B", "TEAM_A", "TEAM_B", "TEAM_A"]
    assert all(r.confidence >= 0.70 for r in res)

    # Reverse input order and verify identical deterministic team labels
    clf_rev = TorsoTeamClassifier()
    res_rev = clf_rev.classify_detections(canvas, list(reversed(dets)))
    assert [r.team for r in res_rev] == ["TEAM_A", "TEAM_B", "TEAM_A", "TEAM_B"]


def test_team_assignment_unknown_cases():
    canvas = _make_turf_canvas(640, 360)
    bboxes = [
        (60.0, 120.0, 90.0, 185.0),    # Dark TEAM_A
        (140.0, 120.0, 170.0, 185.0),  # Dark TEAM_A
        (220.0, 120.0, 250.0, 185.0),  # Light TEAM_B
        (300.0, 120.0, 330.0, 185.0),  # Light TEAM_B
        (380.0, 120.0, 410.0, 185.0),  # Pure green turf (insufficient non-turf pixels)
        (460.0, 120.0, 490.0, 185.0),  # Striped referee
        (540.0, 120.0, 570.0, 185.0),  # Mid-gray equidistant between dark and light
    ]
    _paint_box(canvas, bboxes[0], (30, 30, 30))
    _paint_box(canvas, bboxes[1], (32, 32, 32))
    _paint_box(canvas, bboxes[2], (230, 230, 230))
    _paint_box(canvas, bboxes[3], (228, 228, 228))
    # bboxes[4] left as pure green turf
    # bboxes[5]: alternating black and white vertical referee stripes
    for col in range(460, 490):
        val = 245 if ((col - 460) // 2) % 2 == 0 else 15
        canvas[120:185, col] = (val, val, val)
    # bboxes[6]: exact Lab L-channel midpoint gray (118, 118, 118 -> L ~ 126)
    _paint_box(canvas, bboxes[6], (118, 118, 118))

    det = FixturePlayerDetector({0: [{"bbox": b, "confidence": 0.9} for b in bboxes]})
    dets = det.detect(canvas, frame_id=0)
    clf = TorsoTeamClassifier()
    res = clf.classify_detections(canvas, dets)

    assert res[0].team == "TEAM_A"
    assert res[2].team == "TEAM_B"
    assert res[4].team == "UNKNOWN"
    assert res[4].reason == "insufficient_non_turf_torso_pixels"
    assert res[5].team == "UNKNOWN"
    assert res[5].reason == "striped_official_pattern"
    assert res[6].team == "UNKNOWN"
    assert res[6].reason == "ambiguous_cluster_margin"


def test_track_creation_and_persistence():
    canvas = _make_turf_canvas(640, 360)
    fixtures = {}
    for t in range(4):
        fixtures[t] = [
            {"bbox": (100.0 + 5.0 * t, 150.0, 124.0 + 5.0 * t, 202.0), "confidence": 0.92},
            {"bbox": (250.0 + 4.0 * t, 160.0, 274.0 + 4.0 * t, 212.0), "confidence": 0.88},
        ]
    det = FixturePlayerDetector(fixtures)
    tracker = PlayerTracker(max_missed_frames=3)

    for t in range(4):
        dets = det.detect(canvas, frame_id=t)
        tracks = tracker.update(dets, frame_id=t)
        assert [trk.track_id for trk in tracks] == [1, 2]
        assert all(trk.age == t + 1 for trk in tracks)
        assert all(trk.missed_frames == 0 for trk in tracks)
        if t >= 1:
            assert abs(tracks[0].velocity_uv[0] - 5.0) < 0.5
            assert abs(tracks[1].velocity_uv[0] - 4.0) < 0.5


def test_short_occlusion_recovery_and_prolonged_expiry():
    img = _real_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
    cal = calibrate_frame(img, x_start_yd=15.0)

    # Track 1: observed t=0,1 -> occluded t=2,3 (2 frames <= max_missed=3) -> recovered at t=4
    # Track 2: observed t=0,1 -> occluded t=2,3,4,5 (4 frames > max_missed=3) -> expired at t=5, new ID at t=6
    fixtures = {}
    for t in range(7):
        f_list = []
        if t not in (2, 3):
            f_list.append({"bbox": (200.0 + 3.0 * t, 220.0, 224.0 + 3.0 * t, 272.0), "confidence": 0.90})
        if t not in (2, 3, 4, 5):
            f_list.append({"bbox": (400.0 + 3.0 * t, 230.0, 424.0 + 3.0 * t, 282.0), "confidence": 0.90})
        fixtures[t] = f_list

    det = FixturePlayerDetector(fixtures)
    tracker = PlayerTracker(max_missed_frames=3)

    for t in range(7):
        dets = det.detect(img, frame_id=t)
        tracks = tracker.update(dets, frame_id=t, calibration=cal)
        by_id = {trk.track_id: trk for trk in tracks}

        if t in (2, 3):
            # Track 1 is coasting: survives in image space, but field_position is refused (None)!
            assert 1 in by_id
            assert by_id[1].missed_frames == t - 1
            assert by_id[1].footpoint_estimate.is_reliable is False
            assert by_id[1].field_position is None
            assert by_id[1].projection_available is False
        elif t == 4:
            # Track 1 recovers with its original ID and resumes valid field projection!
            assert 1 in by_id
            assert by_id[1].missed_frames == 0
            assert by_id[1].short_occlusion_recovered is True
            assert by_id[1].field_position is not None
            assert by_id[1].projection_available is True
        elif t == 5:
            # Track 2 missed 4 frames (> max_missed_frames=3) and has expired
            assert 2 not in by_id
        elif t == 6:
            # Re-appearing second player receives new track_id = 3
            assert sorted(by_id.keys()) == [1, 3]


def test_calibration_and_player_projection_integration_respects_x_coord_mode():
    img = _real_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
    cal = calibrate_frame(img, x_start_yd=15.0)
    assert cal.success is True
    assert cal.x_coord_mode == "relative_10yd"

    fp = extract_footpoint((350.0, 210.0, 374.0, 265.0), (img.shape[1], img.shape[0]), detection_confidence=0.95)
    projector = FieldProjector()
    proj, status = projector.project_footpoint(fp, cal)

    assert status == "projected"
    assert proj is not None
    assert proj.x_coord_mode == "relative_10yd"
    # Never invent an absolute yardline number when x_coord_mode != "absolute"
    assert proj.is_absolute_yardline is False
    assert proj.absolute_yardline is None
    assert "absolute_yardline" not in proj.valid_metrics

    # When x_coord_mode is explicitly "absolute", absolute_yardline is populated
    cal_abs = CalibrationResult(
        success=True,
        H=cal.H,
        H_inv=cal.H_inv,
        image_size=cal.image_size,
        confidence=cal.confidence,
        x_coord_mode="absolute",
        plausible_orientation_scale=True,
    )
    proj_abs, status_abs = projector.project_footpoint(fp, cal_abs)
    assert status_abs == "projected"
    assert proj_abs is not None
    assert proj_abs.is_absolute_yardline is True
    assert proj_abs.absolute_yardline == proj_abs.x_yd


def test_projection_refusal_on_invalid_calibration_or_unreliable_footpoint():
    img = _real_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
    cal_valid = calibrate_frame(img, x_start_yd=15.0)
    projector = FieldProjector()

    # 1. Valid calibration + unreliable (bottom-truncated) footpoint -> refused
    fp_unrel = extract_footpoint((350.0, 450.0, 374.0, 505.0), (img.shape[1], img.shape[0]))
    assert fp_unrel.is_reliable is False
    proj1, status1 = projector.project_footpoint(fp_unrel, cal_valid)
    assert proj1 is None
    assert status1 == "footpoint_unreliable:bottom_border_truncated"

    # 2. Low-confidence / expired calibration + reliable footpoint -> refused
    fp_rel = extract_footpoint((350.0, 210.0, 374.0, 265.0), (img.shape[1], img.shape[0]))
    assert fp_rel.is_reliable is True
    cal_low = CalibrationResult(
        success=True,
        H=cal_valid.H,
        H_inv=cal_valid.H_inv,
        image_size=cal_valid.image_size,
        confidence=0.32,  # Below MIN_CALIBRATION_CONFIDENCE (0.45)
        x_coord_mode="relative_10yd",
        failure_reason="low_confidence",
    )
    proj2, status2 = projector.project_footpoint(fp_rel, cal_low)
    assert proj2 is None
    assert status2.startswith("calibration_refused:")


def test_phase3_benchmark_split_metrics_and_zero_fabricated_projections():
    data = json.loads(PHASE3_BENCHMARK_JSON.read_text(encoding="utf-8"))
    assert data["calibration_layer_modified"] is False
    for sp in ("train", "val", "test"):
        s = data["splits"][sp]
        assert s["fabricated_projections"] == 0
        assert s["detection_f1"] >= 0.95
        assert s["team_accuracy"] == 1.0
    assert data["splits"]["val"]["short_occlusion_recovery_rate"] == 1.0
    assert data["splits"]["test"]["short_occlusion_recovery_rate"] == 1.0
    assert data["splits"]["test"]["unknown_abstention_accuracy"] == 1.0

    for rf in data["real_nfl_frame_audit"]:
        assert rf["fabricated_projections"] == 0
        if not rf["calibration_success"]:
            assert rf["projected_tracks_count"] == 0
