import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from benchmarks import compare_nfl_detectors as benchmark


def fixture_inputs(tmp_path):
    video = tmp_path / "1_000002_Sideline.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (320, 240))
    if not writer.isOpened():
        pytest.skip("MJPEG encoder unavailable")
    for _ in range(2):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    labels = tmp_path / "helmets.csv"
    with labels.open("w", newline="") as file:
        fields = ["video", "frame", "label", "left", "top", "width", "height", "isSidelinePlayer"]
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for frame in (1, 2):
            writer.writerow(dict(video=video.name, frame=frame, label="H1", left=10, top=10,
                                 width=5, height=5, isSidelinePlayer="FALSE"))
    return video, labels


class Detector:
    def metadata(self):
        return {"test_double": True}

    def detect(self, image, frame_id):
        return [SimpleNamespace(bbox=(5, 5, 25, 50), confidence=.8)]


def test_comparison_uses_zero_based_source_frames_and_separate_diagnostics(tmp_path, monkeypatch):
    video, labels = fixture_inputs(tmp_path)
    monkeypatch.setattr(benchmark, "create_player_detector", lambda mode, weights: Detector())
    output = tmp_path / "comparison"
    result = benchmark.compare([video], labels, tmp_path / "unused.onnx", output)
    samples = result["videos"][0]["samples"]
    assert [(s["frame_id"], s["kaggle_frame"]) for s in samples] == [(0, 1), (1, 2)]
    assert samples[0]["modes"]["yolox"]["matched_helmet_centers"] == 1
    assert result["split"] == "dev"
    assert "NOT player recall" in result["diagnostic"]
    assert json.loads((output / "comparison.json").read_text())["helmet_labels_sha256"]
    assert cv2.imread(str(output / "preview.jpg")) is not None
    with pytest.raises(FileExistsError):
        benchmark.compare([video], labels, tmp_path / "unused.onnx", output)


def test_missing_sample_labels_remove_only_new_partial_output(tmp_path, monkeypatch):
    video, labels = fixture_inputs(tmp_path)
    labels.write_text(labels.read_text().splitlines()[0] + "\n" + labels.read_text().splitlines()[1] + "\n")
    monkeypatch.setattr(benchmark, "create_player_detector", lambda mode, weights: Detector())
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="Missing sampled helmet"):
        benchmark.compare([video], labels, tmp_path / "unused.onnx", output)
    assert not output.exists()
    assert video.exists() and labels.exists()
