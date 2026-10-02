"""Adversarial contracts from the October audit; no external footage required."""

import numpy as np
import pytest

from football_vision import CalibrationResult, FootpointEstimate, PlayerTrack, PlayerTrajectoryBuilder
from football_vision.calibration.tracker import CalibrationTracker
from football_vision.trajectory.uncertainty import propagate_to_field_covariance
from football_vision.tracking.tracker import PlayerTracker, _InternalTrackState
from football_vision.schema import PlayerDetection


def cal(H=None, mode="relative_10yd", coordinate_id=None):
    H = np.eye(3) if H is None else np.array(H, dtype=float)
    return CalibrationResult(
        True,
        H,
        np.linalg.inv(H),
        confidence=1.0,
        plausible_orientation_scale=True,
        x_coord_mode=mode,
        coordinate_frame_id=coordinate_id,
    )


def track(frame, x=10.0, reliable=True, missed=0):
    fp = FootpointEstimate(x, 20.0, is_reliable=reliable, confidence=0.9)
    return PlayerTrack(
        track_id=1,
        frame_id=frame,
        bbox=(x - 3, 5.0, x + 3, 20.0),
        footpoint=(x, 20.0),
        footpoint_estimate=fp,
        team="UNKNOWN",
        team_confidence=0.0,
        detection_confidence=0.9,
        age=frame + 1,
        missed_frames=missed,
    )


def sample(b, frame, x=10.0, c=None, reliable=True, missed=0):
    return b.update([track(frame, x, reliable, missed)], frame_id=frame, calibration=c)[0]


def test_confidence_inflates_throughout_accepted_range():
    covs = [
        propagate_to_field_covariance(np.eye(3), (10, 20), np.eye(2), calibration_confidence=c)
        for c in (1.0, 0.8, 0.46)
    ]
    assert covs[0][0, 0] < covs[1][0, 0] < covs[2][0, 0]


@pytest.mark.parametrize(
    "mode,coordinate_id", [("absolute", None), ("relative_10yd", "new"), ("relative_10yd", None)]
)
def test_origin_changes_do_not_become_motion(mode, coordinate_id):
    b = PlayerTrajectoryBuilder()
    a = sample(b, 0, c=cal())
    H = np.eye(3)
    H[0, 2] = 10
    z = sample(b, 1, c=cal(H, mode, coordinate_id))
    assert z.coordinate_segment == a.coordinate_segment + 1
    assert z.field_position == (20.0, 20.0)
    assert z.speed_yd_s == 0 and z.distance_cum_yd == 0
    if mode == "absolute":
        assert b.trajectory(1).x_coord_mode == "uncalibrated"


def test_explicit_stable_coordinate_id_allows_camera_motion():
    b = PlayerTrajectoryBuilder()
    sample(b, 0, c=cal(coordinate_id="field"))
    H = np.eye(3)
    H[0, 2] = 2
    z = sample(b, 1, x=8, c=cal(H, coordinate_id="field"))
    assert z.coordinate_segment == 0 and z.field_position == (10.0, 20.0)
    assert z.distance_cum_yd == 0


def test_gap_expires_even_when_track_omitted_from_updates():
    b = PlayerTrajectoryBuilder(max_gap_frames=3)
    sample(b, 0, c=cal())
    b.update([], frame_id=50, calibration=None)
    z = sample(b, 100, missed=100)
    assert z.predicted_position is None and z.frames_since_measurement == 100
    z = sample(b, 101, x=30, c=cal())
    assert z.field_position == (30.0, 20.0) and z.speed_yd_s == 0
    assert z.distance_cum_yd == 0 and z.reinitialized_after_gap


def test_filter_clipping_is_reported_and_not_repeated_on_prediction():
    b = PlayerTrajectoryBuilder(max_speed_yd_s=1.0, gate_warmup_updates=10)
    sample(b, 0, c=cal())
    z = sample(b, 1, x=30, c=cal())
    assert z.speed_yd_s <= 1 and z.is_speed_clipped
    z = sample(b, 2, missed=1)
    assert not z.is_speed_clipped


def test_unreliable_detection_is_still_observed():
    z = sample(PlayerTrajectoryBuilder(), 0, c=cal(), reliable=False)
    assert z.track_state == "observed" and z.field_position is None
    assert z.provenance["footpoint_reliable"] is False


@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e20])
def test_projective_horizon_refused_without_origin_or_crash(scale):
    c = cal(np.array([[1, 0, 0], [0, 1, 0], [1, 0, -10]]) * scale)
    assert c.image_to_field([[10.0, 20.0]]) is None
    z = sample(PlayerTrajectoryBuilder(), 0, c=c)
    assert z.field_position is None and z.predicted_position is None


def test_rejections_do_not_reset_age_and_configured_limit_is_used():
    b = PlayerTrajectoryBuilder(gate_warmup_updates=0, max_consecutive_rejections=100, max_gap_frames=10)
    sample(b, 0, c=cal())
    for frame in range(1, 5):
        z = sample(b, frame, x=100, c=cal())
        assert z.is_outlier_rejected and not z.is_measurement_used
        assert z.frames_since_measurement == frame
        assert not z.reinitialized_after_rejections


def test_tracker_expires_across_sparse_frame_ids():
    t = track(0)
    d = PlayerDetection(
        frame_id=0,
        detection_id=0,
        bbox=t.bbox,
        confidence=0.9,
        footpoint=t.footpoint,
        footpoint_estimate=t.footpoint_estimate,
    )
    tracker = PlayerTracker(max_missed_frames=3)
    tracker.update([d], frame_id=0)
    assert tracker.update([], frame_id=100) == []
    assert tracker.update([d], frame_id=101)[0].track_id == 2


def test_coasting_prediction_matches_sequential_elapsed_frames():
    t = track(0)
    a = _InternalTrackState(1, 0, t.bbox, t.footpoint, t.footpoint_estimate, velocity_uv=(2.0, 1.0))
    expected = a.predict_next_bbox_and_footpoint(3)
    for _ in range(3):
        a.bbox, a.footpoint = a.predict_next_bbox_and_footpoint()
        a.missed_frames += 1
    assert np.allclose(a.bbox, expected[0])


def test_calibration_mode_switch_starts_epoch_without_blending():
    tracker = CalibrationTracker()
    first = tracker.update_from_result(cal())
    H = np.eye(3)
    H[0, 2] = 1.0
    second = tracker.update_from_result(cal(H, "absolute"))
    assert first.coordinate_frame_id != second.coordinate_frame_id
    assert np.allclose(second.H, H)
    propagated = tracker.update_from_result(CalibrationResult(False, None, None))
    assert propagated.coordinate_frame_id == second.coordinate_frame_id


@pytest.mark.parametrize("bad", [0, -1])
def test_rejection_limit_validated(bad):
    with pytest.raises(ValueError):
        PlayerTrajectoryBuilder(max_consecutive_rejections=bad)


def test_frame_order_is_enforced():
    b = PlayerTrajectoryBuilder()
    sample(b, 1, c=cal())
    with pytest.raises(ValueError):
        sample(b, 1, c=cal())


def test_detector_size_options_affect_candidates():
    import cv2
    from football_vision.detection.detector import TurfContrastPlayerDetector

    image = np.full((300, 500, 3), (30, 140, 30), dtype=np.uint8)
    cv2.rectangle(image, (220, 100), (230, 125), (5, 5, 5), -1)
    assert TurfContrastPlayerDetector(min_confidence=0).detect(image)
    assert not TurfContrastPlayerDetector(min_confidence=0, min_height_frac=0.2).detect(image)
    assert not TurfContrastPlayerDetector(min_confidence=0, min_width_frac=0.06).detect(image)


def test_cut_clears_absent_track_predictions():
    b = PlayerTrajectoryBuilder()
    sample(b, 0, c=cal())
    cut = CalibrationResult(False, None, None, failure_reason="camera_cut_uncalibrated")
    b.update([], frame_id=1, calibration=cut)
    z = sample(b, 2, missed=2)
    assert z.predicted_position is None and z.coordinate_segment == 1
