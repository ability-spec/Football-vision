"""Shot changes must not connect identities or predictions across cameras."""
from dataclasses import replace

import numpy as np
import pytest
import cv2

from football_vision.calibration.tracker import CalibrationTracker
from football_vision.schema import CalibrationResult, FootpointEstimate, PlayerDetection
from football_vision.tracking.tracker import PlayerTracker
from football_vision.trajectory.builder import PlayerTrajectoryBuilder


def calibration():
    return CalibrationResult(
        True, np.eye(3), np.eye(3), confidence=1.0,
        plausible_orientation_scale=True, x_coord_mode="absolute",
        coordinate_frame_id="field", image_size=(96, 96),
    )


def detection(frame):
    foot = FootpointEstimate(10.0, 20.0, is_reliable=True, confidence=0.9)
    return PlayerDetection(
        frame_id=frame, detection_id=str(frame), bbox=(7.0, 5.0, 13.0, 20.0),
        confidence=0.9, footpoint=foot.xy, footpoint_estimate=foot,
    )


@pytest.mark.parametrize("projectable", [False, True])
def test_cut_retires_id_even_when_new_player_has_identical_box(projectable):
    tracker = PlayerTracker()
    builder = PlayerTrajectoryBuilder()
    first = tracker.update([detection(0)], frame_id=0, calibration=calibration())
    builder.update(first, frame_id=0, calibration=calibration())
    cut = (replace(calibration(), camera_cut_detected=True) if projectable else
           CalibrationResult(False, None, None, failure_reason="camera_cut_uncalibrated"))
    second = tracker.update([detection(1)], frame_id=1, calibration=cut)
    assert second[0].track_id != first[0].track_id
    assert second[0].age == 1 and second[0].hits == 1
    samples = builder.update(second, frame_id=1, calibration=cut)
    assert samples[0].speed_yd_s == 0 and samples[0].distance_cum_yd == 0
    assert len(builder.finalize()) == 2
    # The new shot has its own continuous tracking lifecycle.
    third = tracker.update([detection(2)], frame_id=2, calibration=calibration())
    assert third[0].track_id == second[0].track_id


def test_cut_without_detections_does_not_coast_old_players():
    tracker = PlayerTracker()
    old = tracker.update([detection(0)], frame_id=0)[0].track_id
    cut = CalibrationResult(False, None, None, failure_reason="camera_cut_uncalibrated")
    assert tracker.update([], frame_id=1, calibration=cut) == []
    assert tracker.update([detection(2)], frame_id=2)[0].track_id > old


def test_successful_cut_invalidates_absent_trajectory_prediction():
    builder = PlayerTrajectoryBuilder()
    tracks = PlayerTracker().update([detection(0)], frame_id=0)
    builder.update(tracks, frame_id=0, calibration=calibration())
    builder.update([], frame_id=1, calibration=replace(calibration(), camera_cut_detected=True))
    coasted = replace(tracks[0], frame_id=2, missed_frames=2)
    sample = builder.update([coasted], frame_id=2, calibration=None)[0]
    assert sample.predicted_position is None and sample.coordinate_segment == 1


@pytest.mark.parametrize("projectable", [False, True])
def test_geometry_reports_cut_independently_of_projection_success(monkeypatch, projectable):
    tracker = CalibrationTracker()
    frame = np.full((96, 96, 3), (30, 140, 30), dtype=np.uint8)
    assert not tracker.update_from_result(calibration(), frame).camera_cut_detected
    monkeypatch.setattr("football_vision.calibration.tracker.cv2.compareHist", lambda *args: 0.0)
    observation = calibration() if projectable else CalibrationResult(False, None, None)
    result = tracker.update_from_result(observation, frame)
    assert result.camera_cut_detected
    assert result.can_project() is projectable
    assert result.to_dict()["camera_cut_detected"] is True
    # A result reused on a following non-cut frame must not repeat the cut signal.
    monkeypatch.setattr("football_vision.calibration.tracker.cv2.compareHist", lambda *args: 1.0)
    assert not tracker.update_from_result(result, frame).camera_cut_detected


def test_video_runner_exports_cut_and_separates_ids(tmp_path, monkeypatch):
    from football_vision.evaluation.video import run_video

    video = tmp_path / "shots.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 96))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.full((96, 96, 3), (30, 140, 30), dtype=np.uint8))
    finally:
        writer.release()
    results = iter([calibration(), replace(calibration(), camera_cut_detected=True), calibration()])
    monkeypatch.setattr("football_vision.evaluation.video.calibrate_frame", lambda image: calibration())
    monkeypatch.setattr(CalibrationTracker, "update_from_result", lambda *args: next(results))
    monkeypatch.setattr("football_vision.evaluation.video.TurfContrastPlayerDetector.detect",
                        lambda self, image, frame_id: [detection(frame_id)])
    frames = run_video(video, source_kind="synthetic")["frames"]
    assert [f["camera_cut_detected"] for f in frames] == [False, True, False]
    ids = [f["players"][0]["track_id"] for f in frames]
    assert ids[0] != ids[1] and ids[1] == ids[2]
