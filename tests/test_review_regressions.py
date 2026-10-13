"""Boundary failures discovered during the October code review."""
from dataclasses import replace

import numpy as np
import pytest

from football_vision.calibration.tracker import CalibrationTracker
from football_vision.evaluation.video import run_video
from football_vision.projection.projector import FieldProjector
from football_vision.schema import CalibrationResult, FootpointEstimate, PlayerDetection, PlaySegment
from football_vision.tracking.tracker import PlayerTracker
from football_vision.trajectory.builder import PlayerTrajectoryBuilder


def calibration():
    h = np.array([[.1, 0, 0], [0, -.1, 30], [0, 0, 1]])
    return CalibrationResult(True, h, np.linalg.inv(h), confidence=.9,
                             plausible_orientation_scale=True, x_coord_mode="relative_10yd",
                             image_size=(320, 240), coordinate_frame_id="field")


def detection():
    foot = FootpointEstimate(100, 160, is_reliable=True, confidence=.9)
    return PlayerDetection(0, "d0", (90, 100, 110, 160), .9, foot.xy, foot)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 1.1])
def test_invalid_confidence_never_enters_projection(confidence):
    cal = replace(calibration(), confidence=confidence)
    assert not cal.can_project()
    assert cal.image_to_field([[100, 160]]) is None
    foot = replace(detection().footpoint_estimate, confidence=confidence)
    projected, status = FieldProjector().project_footpoint(foot, calibration())
    assert projected is None and "low_footpoint_confidence" in status


def test_unknown_coordinate_mode_is_not_projectable():
    assert not replace(calibration(), x_coord_mode="typo").can_project()


@pytest.mark.parametrize("bad_frame", [-1, .5, True])
def test_invalid_frame_does_not_advance_stream(bad_frame):
    tracker = PlayerTracker()
    builder = PlayerTrajectoryBuilder()
    with pytest.raises(ValueError, match="nonnegative integer"):
        tracker.update([], frame_id=bad_frame)
    with pytest.raises(ValueError, match="nonnegative integer"):
        builder.update([], frame_id=bad_frame, calibration=None)
    tracks = tracker.update([detection()], frame_id=0)
    assert tracks[0].track_id == 1
    assert len(builder.update(tracks, frame_id=0, calibration=calibration())) == 1


def test_mismatched_teams_and_duplicate_tracks_can_be_retried():
    tracker = PlayerTracker()
    with pytest.raises(ValueError, match="team_assignments"):
        tracker.update([detection()], frame_id=0, team_assignments=[])
    tracks = tracker.update([detection()], frame_id=0)
    builder = PlayerTrajectoryBuilder()
    with pytest.raises(ValueError, match="duplicate"):
        builder.update(tracks * 2, frame_id=0, calibration=calibration())
    assert len(builder.update(tracks, frame_id=0, calibration=calibration())) == 1
    assert len(builder.finalize()[0].samples) == 1


@pytest.mark.parametrize("success", [False, True])
def test_resolution_change_resets_geometry_without_running_flow(monkeypatch, success):
    tracker = CalibrationTracker()
    image = np.full((240, 320, 3), (30, 140, 30), dtype=np.uint8)
    first = tracker.update_from_result(calibration(), image)
    def unexpected_flow(*args):
        raise AssertionError("optical flow must not run between different resolutions")
    monkeypatch.setattr("football_vision.calibration.tracker._estimate_turf_interframe_homography", unexpected_flow)
    current = calibration() if success else CalibrationResult(False, None, None)
    current = replace(current, image_size=(640, 480))
    result = tracker.update_from_result(current, np.repeat(np.repeat(image, 2, axis=0), 2, axis=1))
    assert result.camera_cut_detected
    assert result.can_project() == success
    if success:
        assert result.coordinate_frame_id != first.coordinate_frame_id
    else:
        assert result.failure_reason == "camera_cut_uncalibrated"


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0, -1])
def test_segment_rejects_invalid_frame_rate(fps):
    with pytest.raises(ValueError, match="fps"):
        PlaySegment("g", "p", 0, 2, 1, "manual", "manual", fps)


@pytest.mark.parametrize("limit", [True, 1.5, 0])
def test_video_rejects_invalid_limit_before_loading_input(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        run_video(tmp_path / "missing.avi", source_kind="synthetic", max_frames=limit)


def test_video_export_with_field_positions_imports_into_database(tmp_path, monkeypatch):
    import cv2
    from football_vision.storage import import_analysis, report

    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.full((240, 320, 3), (30, 140, 30), dtype=np.uint8))
    finally:
        writer.release()
    monkeypatch.setattr("football_vision.evaluation.video.calibrate_frame", lambda image: calibration())
    monkeypatch.setattr("football_vision.detection.detector.TurfContrastPlayerDetector.detect",
                        lambda self, image, frame_id: [replace(detection(), frame_id=frame_id)])
    result = run_video(video, source_kind="synthetic", max_frames=3)
    assert all(f["players"][0]["field_position"] is not None for f in result["frames"])
    path = tmp_path / "analysis.sqlite"
    run, inserted = import_analysis(path, result)
    assert inserted
    assert report(path, run, "coverage")[0]["field_records"] == 3


def test_builder_segment_counters_are_local_but_formation_axes_are_shared():
    from football_vision.analytics.formations import analyze_formation
    from football_vision.analytics.routes import analyze_route

    tracker = PlayerTracker()
    builder = PlayerTrajectoryBuilder()
    first = tracker.update([detection()], frame_id=0)
    builder.update(first, frame_id=0, calibration=calibration())
    old = replace(first[0], frame_id=1)
    new = replace(old, track_id=2)
    changed = replace(calibration(), coordinate_frame_id="new-field")
    samples = builder.update([old, new], frame_id=1, calibration=changed)
    assert [s.coordinate_segment for s in samples] == [1, 0]
    formation = analyze_formation(samples, frame_id=1, snap_frame=2, team_by_track={1: "O", 2: "D"})
    assert formation.status == "available"
    route = analyze_route(samples, track_id=1, start_frame=1, end_frame=1,
                          snap_frame=1, defender_samples=[samples[1]])
    assert route.separation[0]["nearest_defender_track_id"] == 2
    assert route.separation[0]["distance_yd"] == 0
