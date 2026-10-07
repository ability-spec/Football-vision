"""Strong detections take priority; weak ones can only recover mature identities."""

from dataclasses import replace

import pytest

from test_tracking_observation_history import detection
from football_vision.tracking.tracker import PlayerTracker


def mature_tracker():
    tracker = PlayerTracker(high_confidence_threshold=0.4, recovery_min_hits=3)
    for frame in range(3):
        tracker.update([detection(frame, 100)], frame_id=frame)
    return tracker


def test_weak_detection_recovers_mature_track_without_new_birth():
    tracker = mature_tracker()
    weak = replace(detection(3, 100), confidence=0.2)
    noise = replace(detection(3, 300), confidence=0.2)
    tracks = tracker.update([weak, noise], frame_id=3)
    assert [t.track_id for t in tracks] == [1]
    assert tracks[0].missed_frames == 0
    assert tracks[0].detection_confidence == 0.2


def test_strong_detection_wins_even_if_weak_has_better_overlap():
    tracker = mature_tracker()
    weak = replace(detection(3, 100), confidence=0.2)
    strong = detection(3, 105)
    tracks = tracker.update([weak, strong], frame_id=3)
    assert len(tracks) == 1
    assert tracks[0].bbox == strong.bbox


def test_weak_detection_cannot_confirm_tentative_track():
    tracker = PlayerTracker(high_confidence_threshold=0.4)
    tracker.update([detection(0, 100)], frame_id=0)
    track = tracker.update([replace(detection(1, 100), confidence=0.2)], frame_id=1)[0]
    assert track.hits == 1
    assert track.missed_frames == 1


def test_weak_recovery_requires_overlap_not_only_nearby_footpoint():
    tracker = mature_tracker()
    track = tracker.update([replace(detection(3, 125), confidence=0.2)], frame_id=3)[0]
    assert track.missed_frames == 1


@pytest.mark.parametrize("threshold", [True, 0.1, 0, 1.1, float("nan"), float("inf")])
def test_invalid_recovery_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        PlayerTracker(high_confidence_threshold=threshold)
