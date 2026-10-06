"""Synthetic contract tests only: none of these numbers measure football accuracy."""

from copy import deepcopy
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from benchmarks.evaluate_real_video import evaluate
from football_vision.evaluation.video import run_video


def player(tid=1, x=0, xy=None, coordinate_id="field"):
    return dict(
        track_id=tid,
        bbox=[x, 0, x + 10, 20],
        field_position=xy,
        x_coord_mode="absolute",
        coordinate_frame_id=coordinate_id,
    )


def documents():
    gt = dict(
        schema_version=1,
        video_sha256="a" * 64,
        source_kind="synthetic",
        split="test",
        annotation_method="independent_manual",
        frames=[
            dict(frame_id=0, exhaustive=True, players=[player(xy=[10, 10])]),
            dict(frame_id=1, exhaustive=True, players=[player(xy=[10, 10])]),
        ],
    )
    return gt, deepcopy(gt)


def test_missing_frames_reduce_detection_and_field_coverage():
    gt, p = documents()
    p["frames"].pop()
    p["frames"][0]["players"][0]["field_position"] = [10.5, 10]
    metrics = evaluate(gt, p)
    assert metrics["detection"]["recall"] == 0.5
    assert metrics["missing_prediction_frames"] == 1
    assert metrics["field"]["coverage"] == 0.5
    assert metrics["field"]["conditional_median_error_yd"] == 0.5
    assert metrics["field"]["fraction_all_gt_within_threshold"] == 0.5


def test_no_predictions_are_all_misses_not_perfect_zero_error():
    gt, p = documents()
    p["frames"] = []
    m = evaluate(gt, p)
    assert m["detection"]["recall"] == 0
    assert m["detection"]["precision"] is None
    assert m["field"]["conditional_median_error_yd"] is None
    assert m["field"]["coverage"] == 0


def test_coasting_and_incompatible_origins_do_not_count_as_field_success():
    gt, p = documents()
    p["frames"][0]["players"][0]["observed"] = False
    p["frames"][1]["players"][0]["coordinate_frame_id"] = "different"
    m = evaluate(gt, p)
    assert m["detection"]["recall"] == 0.5
    assert m["field"]["coverage"] == 0
    assert m["field"]["incompatible_coordinate_frames"] == 1


def test_identity_switch_and_fragment_counted():
    gt, p = documents()
    gt["frames"].append(dict(frame_id=2, exhaustive=True, players=[player()]))
    p["frames"][1]["players"] = []
    p["frames"].append(dict(frame_id=2, players=[player(tid=7)]))
    m = evaluate(gt, p)
    assert m["tracking"]["identity_switches"] == 1
    assert m["tracking"]["fragments"] == 1


def test_false_positives_in_negative_frames_are_counted():
    gt, p = documents()
    gt["frames"][1]["players"] = []
    m = evaluate(gt, p)
    assert m["detection"]["false_positives"] == 1
    assert m["detection"]["precision"] == 0.5


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "partial", "nan", "kind", "origin"])
def test_invalid_or_incomparable_inputs_are_refused(mutation):
    gt, p = documents()
    if mutation == "hash":
        p["video_sha256"] = "b" * 64
    if mutation == "duplicate":
        p["frames"].append(p["frames"][0])
    if mutation == "partial":
        gt["frames"][0]["exhaustive"] = False
    if mutation == "nan":
        p["frames"][0]["players"][0]["bbox"][0] = float("nan")
    if mutation == "kind":
        p["source_kind"] = "real"
    if mutation == "origin":
        del p["frames"][0]["players"][0]["coordinate_frame_id"]
    with pytest.raises(ValueError):
        evaluate(gt, p)


def test_video_runner_decodes_without_ground_truth(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 96))
    assert writer.isOpened(), "CI requires OpenCV's MJPEG encoder for this contract test"
    for _ in range(3):
        writer.write(np.full((96, 96, 3), (30, 140, 30), dtype=np.uint8))
    writer.release()
    result = run_video(path, source_kind="synthetic", max_frames=10)
    assert len(result["frames"]) == 3 and result["fps"] == 10
    assert len(result["video_sha256"]) == 64
    assert all(not f["calibration_projectable"] for f in result["frames"])
    assert all(f["players"] == [] for f in result["frames"])
    assert result["wall_time_s"] > 0


def test_bad_video_is_refused(tmp_path):
    path = tmp_path / "broken.avi"
    path.write_bytes(b"not video")
    with pytest.raises(ValueError):
        run_video(path, source_kind="real")


def test_evaluator_cli_writes_provenance_and_refuses_overwrite(tmp_path):
    gt, p = documents()
    a = tmp_path / "gt.json"
    b = tmp_path / "pred.json"
    out = tmp_path / "metrics.json"
    a.write_text(json.dumps(gt))
    b.write_text(json.dumps(p))
    cmd = [sys.executable, "-m", "benchmarks.evaluate_real_video", str(a), str(b), "--out", str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text())
    assert len(result["annotation_sha256"]) == 64
    assert subprocess.run(cmd, capture_output=True).returncode == 2


def test_video_play_workflow_exports_boundaries_metrics_and_review(tmp_path):
    from football_vision.visualization.__main__ import render_video

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((120, 160, 3), (30, 140, 30), dtype=np.uint8))
    writer.release()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(dict(game_id="synthetic-game", plays=[
        dict(play_id="one", start_frame=0, snap_frame=1, end_frame=3)])))
    output = tmp_path / "analysis.json"
    cmd = [sys.executable, "-m", "football_vision.evaluation.video", str(video),
           "--source-kind", "synthetic", "--max-frames", "4", "--play-labels", str(labels),
           "--out", str(output)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text())
    assert len(artifact["play_labels_sha256"]) == 64
    assert [f["play_phase"] for f in artifact["frames"]] == ["pre_snap", "snap", "play", "play_end"]
    play = artifact["plays"][0]
    assert play["game_id"] == "synthetic-game"
    assert play["metrics"]["play_duration_s"]["value"] == .3
    assert play["metrics"]["position_sample_coverage"]["value"] is None
    assert play["metrics"]["observed_track_count"]["value"] == 0
    assert [event["kind"] for event in play["events"]["events"]] == ["snap", "play_end"]
    assert render_video(video, output, tmp_path / "review.avi") == 4
    labels.write_text(json.dumps(dict(plays=[dict(play_id="one", start_frame=0, end_frame=4)])))
    output.unlink()
    refused = subprocess.run(cmd, capture_output=True, text=True)
    assert refused.returncode == 2 and "within decoded frames" in refused.stderr
    assert not output.exists()


def test_video_workflow_refuses_fractional_timestamps_and_unknown_snap(tmp_path):
    from football_vision.schema import PlayTimestampLabel

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((96, 96, 3), (30, 140, 30), dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="integer"):
        run_video(video, source_kind="synthetic", play_labels=[
            PlayTimestampLabel("one", 0.5, end_frame=2)])
    result = run_video(video, source_kind="synthetic", play_labels=[
        PlayTimestampLabel("one", 0, end_frame=2)])
    segment = result["segmentation"]["segments"][0]
    assert segment["snap_frame"] is None
    assert segment["snap_source"] == "unavailable"
    assert result["plays"][0]["routes"] == []
