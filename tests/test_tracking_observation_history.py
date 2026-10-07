"""Motion estimates must use measured observations, never coasted positions."""

import pytest

from football_vision.schema import FootpointEstimate, PlayerDetection
from football_vision.tracking.tracker import PlayerTracker


def detection(frame, x):
    fp = FootpointEstimate(u_px=x + 10, v_px=160, is_reliable=True, confidence=0.9)
    return PlayerDetection(
        frame_id=frame, detection_id=f"f{frame}", bbox=(x, 100, x + 20, 160),
        confidence=0.9, footpoint=fp.xy, footpoint_estimate=fp,
    )


@pytest.mark.parametrize("coasted_frames", [[], [2], [2, 3]])
def test_recovery_velocity_uses_last_measured_position(coasted_frames):
    tracker = PlayerTracker(max_missed_frames=5)
    tracker.update([detection(0, 100)], frame_id=0)
    tracker.update([detection(1, 110)], frame_id=1)
    for frame in coasted_frames:
        coast = tracker.update([], frame_id=frame)[0]
        assert not coast.footpoint_estimate.is_reliable
        assert coast.field_position is None
    recovered = tracker.update([detection(4, 140)], frame_id=4)[0]
    assert recovered.track_id == 1
    assert recovered.velocity_uv == pytest.approx((10, 0))
    assert recovered.short_occlusion_recovered
    assert tracker.total_occlusion_recoveries == 1


def test_velocity_recovers_without_inheriting_wrong_prediction():
    tracker = PlayerTracker(max_missed_frames=5, velocity_smoothing=1)
    tracker.update([detection(0, 100)], frame_id=0)
    tracker.update([detection(1, 110)], frame_id=1)
    tracker.update([], frame_id=2)
    tracker.update([], frame_id=3)
    recovered = tracker.update([detection(4, 110)], frame_id=4)[0]
    assert recovered.track_id == 1
    assert recovered.velocity_uv == pytest.approx((0, 0))
    next_track = tracker.update([detection(5, 110)], frame_id=5)[0]
    assert next_track.track_id == 1
    assert next_track.velocity_uv == pytest.approx((0, 0))
