"""Exercise the public MVP command and its exported artifacts."""
import csv
import hashlib
import json
import subprocess
import sys

import cv2
import pytest

from football_vision.pipeline import create_demo, load_labels, run_workflow


def test_public_demo_command_exports_reviewable_artifacts(tmp_path):
    output = tmp_path / "demo"
    command = [sys.executable, "-m", "football_vision", "--demo", "--out", str(output)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    artifact = json.loads((output / "analysis.json").read_text())
    assert artifact["source_kind"] == "synthetic"
    assert artifact["frames_processed"] == 20
    assert len(artifact["plays"]) == 1
    assert any(frame["players"] for frame in artifact["frames"]), "demo must exercise perception/tracking"
    assert artifact["trajectories"]
    assert artifact["segmentation"]["fabricated_snap_frames"] == 0
    assert artifact["segmentation"]["fabricated_play_ends"] == 0
    with (output / "metrics.csv").open() as file:
        rows = {row["metric"]: row for row in csv.DictReader(file)}
    assert rows["play_duration_s"]["value"] == "1.9"
    assert rows["mean_observed_defender_separation_yd"]["value"] == ""
    report = (output / "report.html").read_text()
    assert "review.avi" in report and "Unavailable" in report
    assert 'src="preview.png"' in report
    preview = cv2.imread(str(output / "preview.png"))
    assert preview is not None and preview.shape == (720, 1360, 3)
    assert (preview[:240, :320] != preview[480:, 680:1000]).any(), "sampled frames must show motion"
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest) == 5
    for filename, digest in manifest.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    before = (output / "analysis.json").read_bytes()
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode == 2
    assert (output / "analysis.json").read_bytes() == before


def test_failed_render_cleans_only_new_output(tmp_path, monkeypatch):
    video, labels = create_demo(tmp_path)
    output = tmp_path / "result"
    def fail(*args):
        raise ValueError("encoder failed")
    monkeypatch.setattr("football_vision.pipeline.render_video", fail)
    with pytest.raises(ValueError, match="encoder failed"):
        run_workflow(video, labels, output, source_kind="synthetic")
    assert not output.exists()
    assert video.exists() and labels.exists()


@pytest.mark.parametrize("entry", [
    dict(play_id="one", start_frame=True, end_frame=2),
    dict(play_id="one", start_frame=0.5, end_frame=2),
    dict(play_id="one", start_frame=0),
    dict(play_id="", start_frame=0, end_frame=2),
])
def test_invalid_labels_are_rejected_before_video_processing(tmp_path, entry):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(dict(plays=[entry])))
    with pytest.raises(ValueError):
        load_labels(labels)


def test_report_escapes_user_labels(tmp_path):
    video, labels = create_demo(tmp_path)
    document = json.loads(labels.read_text())
    document["plays"][0]["play_id"] = "<script>alert(1)</script>"
    labels.write_text(json.dumps(document))
    output = tmp_path / "result"
    run_workflow(video, labels, output, source_kind="synthetic")
    report = (output / "report.html").read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


def test_csv_preserves_unknown_confidence_and_treats_labels_as_text(tmp_path):
    video, labels = create_demo(tmp_path)
    document = json.loads(labels.read_text())
    document["plays"][0]["play_id"] = '=HYPERLINK("https://example.com")'
    labels.write_text(json.dumps(document))
    output = tmp_path / "result"
    result = run_workflow(video, labels, output, source_kind="synthetic")
    with (output / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert all(row["play_id"].startswith("'=") for row in rows)
    assert all(row["confidence"] == "" for row in rows)
    assert result["plays"][0]["play_id"] == document["plays"][0]["play_id"]
