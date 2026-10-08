"""Independent frame extraction and safe ownership of incomplete label output."""
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from benchmarks.evaluate_real_video import evaluate
from football_vision.evaluation.prepare import prepare_annotations
from football_vision.evaluation.video import run_video


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 96))
    assert writer.isOpened()
    try:
        for index in range(4):
            writer.write(np.full((96, 96, 3), (30 + index * 30, 140, 30), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_public_command_extracts_exact_frames_and_refuses_overwrite(video, tmp_path):
    output = tmp_path / "labels"
    command = [sys.executable, "-m", "football_vision.evaluation.prepare", str(video),
               "--out", str(output), "--frames", "3", "0", "--source-kind", "synthetic", "--split", "dev"]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    document = json.loads((output / "annotations.json").read_text())
    assert document["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert document["annotation_method"] == "pending_manual"
    assert [f["frame_id"] for f in document["frames"]] == [0, 3]
    capture = cv2.VideoCapture(str(video))
    try:
        for fid in range(4):
            ok, original = capture.read()
            assert ok
            if fid in (0, 3):
                extracted = cv2.imread(str(output / f"frame_{fid:08d}.png"))
                assert np.array_equal(extracted, original)
    finally:
        capture.release()
    before = (output / "annotations.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert (output / "annotations.json").read_bytes() == before


def test_pending_labels_cannot_be_scored_as_perfect_negatives(video, tmp_path):
    document = prepare_annotations(video, tmp_path / "labels", [0, 3], source_kind="synthetic", split="test")
    prediction = run_video(video, source_kind="synthetic")
    with pytest.raises(ValueError, match="exhaustively"):
        evaluate(document, prediction)
    for frame in document["frames"]:
        frame["exhaustive"] = True
    with pytest.raises(ValueError, match="independently annotated"):
        evaluate(document, prediction)
    # These synthetic source frames were separately reviewed as empty turf.
    document["annotation_method"] = "independent_manual"
    metrics = evaluate(document, prediction)
    assert metrics["annotated_frames"] == 2
    assert metrics["detection"]["true_positives"] == 0
    assert metrics["detection"]["precision"] is None


@pytest.mark.parametrize("frames", [[], [-1], [True], [1.5], [0, 0]])
def test_invalid_frame_selection_does_not_create_output(video, tmp_path, frames):
    output = tmp_path / "labels"
    with pytest.raises(ValueError):
        prepare_annotations(video, output, frames, source_kind="synthetic", split="dev")
    assert not output.exists()


@pytest.mark.parametrize("failure", ["past_end", "image_write"])
def test_incomplete_output_is_removed_without_touching_video(video, tmp_path, monkeypatch, failure):
    if failure == "image_write":
        monkeypatch.setattr("football_vision.evaluation.prepare.cv2.imwrite", lambda *args: False)
    output = tmp_path / "labels"
    before = video.read_bytes()
    with pytest.raises(ValueError):
        prepare_annotations(video, output, [0, 9] if failure == "past_end" else [0],
                            source_kind="synthetic", split="dev")
    assert not output.exists() and video.read_bytes() == before
