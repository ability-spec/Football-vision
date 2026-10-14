"""Configuration and provenance regression tests from post-review hardening."""

import pytest

from football_vision.tracking.tracker import PlayerTracker


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_missed_frames": True},
        {"max_missed_frames": 1.5},
        {"max_missed_frames": -1},
        {"min_iou_threshold": float("nan")},
        {"min_iou_threshold": -0.01},
        {"min_iou_threshold": 1.01},
        {"max_center_distance_px": 0.0},
        {"max_center_distance_px": float("inf")},
        {"min_match_score": -0.01},
        {"min_match_score": 1.01},
        {"min_init_confidence": -0.01},
        {"min_init_confidence": 1.01},
        {"velocity_smoothing": -0.01},
        {"velocity_smoothing": 1.01},
        {"velocity_smoothing": float("nan")},
    ],
)
def test_tracker_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PlayerTracker(**kwargs)


def test_tracker_accepts_valid_boundary_configuration():
    tracker = PlayerTracker(
        max_missed_frames=0,
        min_iou_threshold=0.0,
        max_center_distance_px=1.0,
        min_match_score=0.0,
        min_init_confidence=0.0,
        velocity_smoothing=1.0,
    )
    assert tracker.max_missed_frames == 0
    assert tracker.min_iou_threshold == 0.0
    assert tracker.velocity_smoothing == 1.0
