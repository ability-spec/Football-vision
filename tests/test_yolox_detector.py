"""Model-profile, coordinate and workflow contracts without downloading weights."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from football_vision.detection import yolox
from football_vision.pipeline import create_demo, run_workflow


class Network:
    def __init__(self):
        self.output = np.zeros((1, 3549, 85), dtype=np.float32)

    def setPreferableBackend(self, backend):
        pass

    def setPreferableTarget(self, target):
        pass

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.output


@pytest.fixture
def model(tmp_path, monkeypatch):
    weights = tmp_path / "test-model.onnx"
    weights.write_bytes(b"test network")
    monkeypatch.setattr(yolox, "YOLOX_TINY_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    network = Network()
    monkeypatch.setattr(yolox.cv2.dnn, "readNetFromONNX", lambda path: network)
    return yolox.YoloXTinyPlayerDetector(weights), network, weights


def person(network, row, *, x=100, y=100, width=40, height=80, objectness=0.8, confidence=0.5):
    # Encode a chosen box in the release model's raw stride-8 output.
    gx, gy = row % 52, row // 52
    network.output[0, row, :4] = [x / 8 - gx, y / 8 - gy, np.log(width / 8), np.log(height / 8)]
    network.output[0, row, 4:6] = [objectness, confidence]


def test_pinned_preparation_preserves_bgr_pixel_scale_and_top_left_padding(model):
    detector, network, _ = model
    image = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)
    assert detector.detect(image) == []
    assert network.blob.shape == (1, 3, 416, 416)
    np.testing.assert_array_equal(network.blob[0, :, 0, 0], [10, 20, 30])
    np.testing.assert_array_equal(network.blob[0, :, 208, 0], [114, 114, 114])


def test_person_scores_nms_and_original_non_square_coordinates(model):
    detector, network, _ = model
    person(network, 0)
    person(network, 1, x=101, confidence=0.45)  # lower-scored duplicate
    person(network, 2, x=200, objectness=0.2, confidence=1.0)  # fails product score
    network.output[0, 3, 4] = 1.0
    network.output[0, 3, 6] = 1.0  # a confident non-person class
    results = detector.detect(np.zeros((208, 416, 3), dtype=np.uint8), frame_id=7)
    assert len(results) == 1
    assert results[0].bbox == (80.0, 60.0, 120.0, 140.0)
    assert results[0].confidence == 0.4 and results[0].frame_id == 7
    assert results[0].detector_metadata["class_name"] == "person"
    assert results[0].footpoint == (100.0, 140.0)
    # Undo the same letterbox at twice the original dimensions.
    results = detector.detect(np.zeros((416, 832, 3), dtype=np.uint8))
    assert results[0].bbox == (160.0, 120.0, 240.0, 280.0)


def test_invalid_padding_boxes_are_discarded_and_image_edges_clipped(model):
    detector, network, _ = model
    person(network, 0, y=300)  # wholly within bottom padding
    person(network, 1, x=5, width=40)
    results = detector.detect(np.zeros((208, 416, 3), dtype=np.uint8))
    assert len(results) == 1 and results[0].bbox[0] == 0.0


@pytest.mark.parametrize("mutation", ["shape", "nan", "probability"])
def test_incompatible_output_is_refused(model, mutation):
    detector, network, _ = model
    if mutation == "shape":
        network.output = np.zeros((1, 100, 85), dtype=np.float32)
    elif mutation == "nan":
        network.output[0, 0, 0] = np.nan
    else:
        network.output[0, 0, 4] = 1.1
    with pytest.raises(ValueError):
        detector.detect(np.zeros((208, 416, 3), dtype=np.uint8))


def test_loaded_network_is_not_serialized_into_results(model, tmp_path):
    _, network, weights = model
    person(network, 0)
    video, labels = create_demo(tmp_path)
    result = run_workflow(video, labels, tmp_path / "report", source_kind="synthetic",
                          detector_kind="yolox", weights=weights)
    assert result["model"]["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert result["model"]["training_dataset"] == "COCO"
    assert result["detector_accuracy_status"] == "real_video_unmeasured"
    assert result["frames"][0]["players"]
    assert result["trajectories"][0]["samples"][0]["provenance"]["detector_source"] == "pretrained_coco_person"
    assert json.loads((tmp_path / "report" / "analysis.json").read_text())["model"] == result["model"]
    report = (tmp_path / "report" / "report.html").read_text()
    assert "yolox_tiny_coco_0.1.1rc0" in report and "Untrained detector baseline" not in report


def test_wrong_or_missing_weights_never_fall_back_to_turf(tmp_path):
    with pytest.raises(ValueError, match="requires a local"):
        yolox.create_player_detector("yolox")
    weights = tmp_path / "wrong.onnx"
    weights.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="do not match"):
        yolox.create_player_detector("yolox", weights)
    with pytest.raises(ValueError, match="requires --detector"):
        yolox.create_player_detector("turf", weights)
    with pytest.raises(FileNotFoundError):
        yolox.create_player_detector("yolox", tmp_path / "absent.onnx")


@pytest.mark.parametrize("threshold", [0, -1, np.nan, 1.1])
def test_invalid_threshold_refused_before_loading_model(tmp_path, threshold):
    with pytest.raises(ValueError):
        yolox.YoloXTinyPlayerDetector(tmp_path / "absent.onnx", min_confidence=threshold)


def test_real_official_weights_smoke_when_available():
    from pathlib import Path
    weights = Path(__file__).resolve().parents[1] / "models" / "yolox_tiny.onnx"
    if not weights.exists():
        pytest.skip("optional official YOLOX weights not installed; CI does not download models")
    detector = yolox.YoloXTinyPlayerDetector(weights)
    results = detector.detect(np.full((240, 320, 3), (30, 140, 30), dtype=np.uint8))
    assert all(d.detector_metadata["class_id"] == 0 for d in results)


def test_tiles_translate_boxes_and_recompute_full_image_footpoints(model, monkeypatch):
    _, _, weights = model
    detector = yolox.create_player_detector("yolox-tiled", weights)
    calls = []

    def fake_detect(self, image, frame_id=0, **kwargs):
        calls.append(image.shape)
        if len(calls) == 5:  # Bottom right crop starts at (512, 288).
            return [SimpleNamespace(bbox=(388, 212, 428, 312), confidence=0.8)]
        return []

    monkeypatch.setattr(yolox.YoloXTinyPlayerDetector, "detect", fake_detect)
    result = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), frame_id=9)
    assert len(calls) == 5
    assert len(result) == 1
    assert result[0].bbox == (900, 500, 940, 600)
    assert result[0].footpoint == (920, 600)
    assert result[0].frame_id == 9
    assert detector.metadata()["detector_settings"]["max_inference_calls"] == 5


def test_tiles_merge_duplicate_global_boxes(model, monkeypatch):
    _, _, weights = model
    detector = yolox.YoloXTiledPlayerDetector(weights)
    offsets = iter([(0, 0), (0, 0), (0, 288), (512, 0), (512, 288)])

    def fake_detect(self, image, frame_id=0, **kwargs):
        x, y = next(offsets)
        return [SimpleNamespace(bbox=(570-x, 320-y, 620-x, 390-y), confidence=0.8)]

    monkeypatch.setattr(yolox.YoloXTinyPlayerDetector, "detect", fake_detect)
    result = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(result) == 1 and result[0].bbox == (570, 320, 620, 390)


def test_internal_crop_edge_fragments_are_not_accepted(model, monkeypatch):
    _, _, weights = model
    detector = yolox.YoloXTiledPlayerDetector(weights)
    calls = []

    def fake_detect(self, image, frame_id=0, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            return [SimpleNamespace(bbox=(0, 20, 60, 100), confidence=0.9)]
        return []

    monkeypatch.setattr(yolox.YoloXTinyPlayerDetector, "detect", fake_detect)
    assert detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8)) == []


def test_small_images_use_only_full_frame(model, monkeypatch):
    _, _, weights = model
    detector = yolox.YoloXTiledPlayerDetector(weights)
    calls = []

    def fake_detect(self, image, frame_id=0, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(yolox.YoloXTinyPlayerDetector, "detect", fake_detect)
    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []
    assert len(calls) == 1
