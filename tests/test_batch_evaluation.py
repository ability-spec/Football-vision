"""Batch contracts on synthetic media; these tests do not measure NFL quality."""
import csv
import hashlib
import json
import subprocess
import sys
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from football_vision.evaluation import batch
from football_vision.schema import FootpointEstimate, PlayerDetection


def inputs(tmp_path, *, count=4, label_frames=None, name="clip.avi"):
    video = tmp_path / name
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (160, 120))
    assert writer.isOpened(), "CI requires the MJPEG encoder"
    try:
        for _ in range(count):
            writer.write(np.full((120, 160, 3), (30, 140, 30), dtype=np.uint8))
    finally:
        writer.release()
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps(dict(schema_version=1, source_kind="synthetic", split="dev",
                                        videos=[dict(path=video.name)])))
    labels = tmp_path / "helmets.csv"
    with labels.open("w", newline="") as file:
        columns = ["video", "frame", "label", "left", "top", "width", "height", "isSidelinePlayer"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for frame in (range(1, count + 1) if label_frames is None else label_frames):
            writer.writerow(dict(video=video.name, frame=frame, label="H1", left=20, top=20,
                                 width=4, height=4, isSidelinePlayer="FALSE"))
    return video, manifest, labels


class DroppingDetector:
    """Detect once, then let the actual tracker coast through a missed frame."""
    def metadata(self):
        return dict(detector="synthetic_test_double", detector_settings={})

    def detect(self, image, frame_id):
        if frame_id != 0:
            return []
        fp = FootpointEstimate(u_px=25, v_px=80, is_reliable=True, confidence=.9)
        return [PlayerDetection(frame_id=frame_id, detection_id="first", bbox=(10, 10, 40, 80),
                                confidence=.9, footpoint=fp.xy, footpoint_estimate=fp)]


def test_full_csv_bounded_inference_and_coasting_are_separate(tmp_path, monkeypatch):
    video, manifest, labels = inputs(tmp_path)
    monkeypatch.setattr("football_vision.evaluation.video.create_player_detector",
                        lambda *args, **kwargs: DroppingDetector())
    output = tmp_path / "review"
    summary = batch.run_batch(manifest, output, max_frames=2, helmet_labels=labels)
    clip = summary["videos"][0]
    assert clip["source"]["frame_count"] == 4
    assert clip["diagnostics"]["frames_processed"] == 2
    assert clip["diagnostics"]["labeled_frames"] == 2
    assert clip["diagnostics"]["matched_helmet_centers"] == 1
    assert clip["diagnostics"]["labeled_on_field_helmets"] == 2
    artifact = json.loads((output / "clip-0001/analysis.json").read_text())
    first, second = artifact["frames"]
    assert first["frame_id"] == 0 and first["helmet_diagnostic"]["matches"][0]["player_id"] == "H1"
    assert second["players"] and not second["players"][0]["observed"]
    assert second["detections"] == []
    assert second["helmet_diagnostic"]["matched_helmet_centers"] == 0
    assert video.exists() and labels.exists()
    assert "not player recall" in summary["metric_scope"]


@pytest.mark.parametrize("label_frames,labeled,matched", [([1], 1, 0), ([4], 0, None)])
def test_missing_annotations_are_unavailable_not_negative(tmp_path, label_frames, labeled, matched):
    _, manifest, labels = inputs(tmp_path, label_frames=label_frames)
    output = tmp_path / "review"
    summary = batch.run_batch(manifest, output, max_frames=2, helmet_labels=labels)
    diagnostics = summary["videos"][0]["diagnostics"]
    assert diagnostics["labeled_frames"] == labeled
    assert diagnostics["unlabeled_frames"] == 2 - labeled
    assert diagnostics["matched_helmet_centers"] == matched
    expected = 0. if labeled else None
    assert diagnostics["helmet_center_containment_fraction"] == expected
    artifact = json.loads((output / "clip-0001/analysis.json").read_text())
    assert artifact["frames"][1]["helmet_diagnostic"] is None


def test_public_one_frame_batch_bundle_survives_download(tmp_path):
    _, manifest, _ = inputs(tmp_path, count=1)
    output = tmp_path / "review"
    command = [sys.executable, "-m", "football_vision.evaluation.batch", str(manifest),
               "--max-frames", "2", "--out", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["videos"][0]["diagnostics"]["frames_processed"] == 1
    assert summary["videos"][0]["diagnostics"]["matched_helmet_centers"] is None
    hashes = json.loads((output / "manifest.json").read_text())
    assert set(hashes) == {"summary.json", "report.md", "report.html", "clip-0001/analysis.json",
                           "clip-0001/preview.png", "clip-0001/review.avi"}
    with ZipFile(output / "bundle.zip") as archive:
        assert archive.testzip() is None
        downloaded = tmp_path / "downloaded"
        archive.extractall(downloaded)
    for filename, digest in hashes.items():
        assert hashlib.sha256((downloaded / filename).read_bytes()).hexdigest() == digest
    assert cv2.imread(str(downloaded / "clip-0001/preview.png")) is not None
    capture = cv2.VideoCapture(str(downloaded / "clip-0001/review.avi"))
    try:
        assert capture.read()[0]
        assert not capture.read()[0]
    finally:
        capture.release()
    assert 'href="clip-0001/analysis.json"' in (downloaded / "report.html").read_text()
    before = (output / "bundle.zip").read_bytes()
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode == 2
    assert (output / "bundle.zip").read_bytes() == before


def test_failure_in_second_clip_removes_only_owned_batch(tmp_path, monkeypatch):
    video, manifest, labels = inputs(tmp_path, count=1)
    second = tmp_path / "second.avi"
    second.write_bytes(video.read_bytes())
    document = json.loads(manifest.read_text())
    document["videos"].append(dict(path=second.name))
    manifest.write_text(json.dumps(document))
    original = batch.render_video
    calls = []
    def fail_second(*args):
        calls.append(args[0])
        if len(calls) == 2:
            raise ValueError("simulated encoder failure")
        return original(*args)
    monkeypatch.setattr(batch, "render_video", fail_second)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="simulated encoder failure"):
        batch.run_batch(manifest, output)
    assert calls == [video, second]
    assert not output.exists()
    assert all(path.exists() for path in (video, second, manifest, labels))


def test_batch_accepts_wholly_unlabeled_clip_without_fabricating_scores(tmp_path):
    video, manifest, labels = inputs(tmp_path, count=1)
    second = tmp_path / "no-labels.avi"
    second.write_bytes(video.read_bytes())
    document = json.loads(manifest.read_text())
    document["videos"].append(dict(path=second.name))
    manifest.write_text(json.dumps(document))
    output = tmp_path / "review"
    result = batch.run_batch(manifest, output, helmet_labels=labels)
    labeled, unlabeled = [clip["diagnostics"] for clip in result["videos"]]
    assert labeled["labeled_frames"] == 1 and labeled["matched_helmet_centers"] == 0
    assert unlabeled["labeled_frames"] == 0 and unlabeled["unlabeled_frames"] == 1
    assert unlabeled["matched_helmet_centers"] is None
    assert unlabeled["labeled_on_field_helmets"] is None
    assert unlabeled["helmet_center_containment_fraction"] is None
    analysis = json.loads((output / "clip-0002/analysis.json").read_text())
    assert analysis["frames"][0]["helmet_diagnostic"] is None
    # Other native consumers keep their strict contract by default.
    with pytest.raises(ValueError, match="no helmet labels"):
        batch.read_helmet_labels(labels, {video.name: {"frame_count": 1}, second.name: {"frame_count": 1}})


@pytest.mark.parametrize("mutation", ["schema", "kind", "split", "empty", "path", "duplicate", "alias"])
def test_invalid_manifest_fails_before_inference(tmp_path, monkeypatch, mutation):
    _, manifest, _ = inputs(tmp_path, count=1)
    document = json.loads(manifest.read_text())
    if mutation == "schema":
        document["schema_version"] = True
    elif mutation == "kind":
        document["source_kind"] = "unknown"
    elif mutation == "split":
        document["split"] = "training"
    elif mutation == "empty":
        document["videos"] = []
    elif mutation == "path":
        document["videos"][0]["path"] = 1
    elif mutation == "duplicate":
        document["videos"].append(document["videos"][0])
    else:
        document["videos"].append(dict(path="./clip.avi"))
    manifest.write_text(json.dumps(document))
    def unexpected(*args, **kwargs):
        pytest.fail("invalid manifests must not run inference")
    monkeypatch.setattr(batch, "run_video", unexpected)
    output = tmp_path / "review"
    with pytest.raises(ValueError):
        batch.run_batch(manifest, output)
    assert not output.exists()


def test_labels_outside_full_source_are_rejected_before_inference(tmp_path, monkeypatch):
    _, manifest, labels = inputs(tmp_path, label_frames=[5])
    def unexpected(*args, **kwargs):
        pytest.fail("invalid labels must not run inference")
    monkeypatch.setattr(batch, "run_video", unexpected)
    with pytest.raises(ValueError, match="outside decoded video"):
        batch.run_batch(manifest, tmp_path / "review", max_frames=2, helmet_labels=labels)
    assert not (tmp_path / "review").exists()


def test_existing_output_preserved_even_when_input_is_invalid(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "mine.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        batch.run_batch(tmp_path / "missing.json", output)
    assert sentinel.read_text() == "keep"


def test_label_and_source_provenance_are_inspectable(tmp_path):
    video, manifest, labels = inputs(tmp_path, count=1, name="<unsafe>|clip.avi")
    output = tmp_path / "review"
    result = batch.run_batch(manifest, output, helmet_labels=labels)
    assert result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result["helmet_labels_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert result["videos"][0]["source"]["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    report = (output / "report.html").read_text()
    assert "<unsafe>" not in report and "&lt;unsafe&gt;" in report
    assert "\\|" in (output / "report.md").read_text()
