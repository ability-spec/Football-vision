"""Configuration and provenance regression tests from post-review hardening."""

import cv2
import numpy as np
import pytest

from football_vision.evaluation.prepare import prepare_annotations
from football_vision.evaluation.video import run_video
from football_vision.tracking.tracker import PlayerTracker
from football_vision.trajectory.kinematics import acceleration_from_velocity, clip_speed
from football_vision.trajectory.smoothing import ConstantVelocityFieldFilter


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


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_speed_limits_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        clip_speed((1.0, 0.0), max_speed_yd_s=bad)
    with pytest.raises(ValueError):
        ConstantVelocityFieldFilter(max_speed_yd_s=bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_acceleration_limits_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        acceleration_from_velocity((0.0, 0.0), (1.0, 0.0), 1.0, max_accel_yd_s2=bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_acceleration_rejects_invalid_dt(bad):
    with pytest.raises(ValueError):
        acceleration_from_velocity((0.0, 0.0), (1.0, 0.0), bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"process_accel_std_yd_s2": 0},
        {"gate_chi2_2dof": float("nan")},
        {"position_cov_init_yd2": -1},
        {"velocity_cov_init_yd2_s2": float("inf")},
    ],
)
def test_filter_rejects_invalid_numeric_configuration(kwargs):
    with pytest.raises(ValueError):
        ConstantVelocityFieldFilter(**kwargs)


def _write_tiny_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 96))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.full((96, 96, 3), (30, 140, 30), dtype=np.uint8))
    finally:
        writer.release()


def test_video_runner_refuses_source_change_during_decode(tmp_path, monkeypatch):
    video = tmp_path / "source.avi"
    _write_tiny_video(video)
    hashes = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr("football_vision.evaluation.provenance.sha256_file", lambda path: next(hashes))
    with pytest.raises(ValueError, match="changed during decoding"):
        run_video(video, source_kind="synthetic")


def test_annotation_preparation_refuses_source_change_and_cleans_output(tmp_path, monkeypatch):
    video = tmp_path / "source.avi"
    output = tmp_path / "labels"
    _write_tiny_video(video)
    hashes = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr("football_vision.evaluation.provenance.sha256_file", lambda path: next(hashes))
    with pytest.raises(ValueError, match="changed during decoding"):
        prepare_annotations(video, output, [0], source_kind="synthetic", split="dev")
    assert not output.exists()
