"""Synthetic import/scoring contracts; not evidence of football accuracy."""
import json
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from football_vision.evaluation import cvat


def inputs(tmp_path, body=None):
    images = tmp_path / "images"
    images.mkdir()
    assert cv2.imwrite(str(images / "one.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    assert cv2.imwrite(str(images / "two.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    if body is None:
        body = ('<image id="0" name="one.png" width="20" height="20">'
                '<box label="player" source="manual" xtl="1" ytl="1" xbr="10" ybr="15"/>'
                '</image><image id="1" name="two.png" width="20" height="20"/>')
    archive = tmp_path / "manual.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("annotations.xml", '<annotations><version>1.1</version>' + body + '</annotations>')
    return archive, images


class Detector:
    def metadata(self):
        return {"test_double": True}

    def detect(self, image, frame_id):
        # Same detection on a positive and a confirmed negative image.
        return [SimpleNamespace(bbox=[1, 1, 10, 15], confidence=.8)]


def test_negative_images_count_false_positives_and_import_has_no_fake_identities(tmp_path, monkeypatch):
    archive, images = inputs(tmp_path)
    monkeypatch.setattr(cvat, "create_player_detector", lambda *args: Detector())
    output = tmp_path / "results"
    result = cvat.compare_cvat(archive, images, tmp_path / "unused", output,
                               confirm_exhaustive=True, source_kind="synthetic")
    for m in result["modes"].values():
        assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (1, 1, 0)
        assert m["precision"] == .5 and m["recall"] == 1
    assert result["tracking"] is None and result["field"] is None
    annotations = json.loads((output / "annotations.json").read_text())
    assert annotations["identity_status"] == "not_annotated"
    assert "track_id" not in (output / "annotations.json").read_text()
    assert len(annotations["frames"][0]["image_sha256"]) == 64
    with pytest.raises(FileExistsError):
        cvat.compare_cvat(archive, images, tmp_path / "unused", output, confirm_exhaustive=True)
    assert (output / "metrics.json").exists()


@pytest.mark.parametrize("change", ["dimensions", "bounds", "nan", "auto", "label", "path", "duplicate"])
def test_bad_annotations_refused_before_inference(tmp_path, monkeypatch, change):
    body = ('<image id="0" name="one.png" width="20" height="20">'
            '<box label="player" source="manual" xtl="1" ytl="1" xbr="10" ybr="15"/></image>')
    edits = {"dimensions": ('width="20"', 'width="19"'),
             "bounds": ('xbr="10"', 'xbr="21"'), "nan": ('xtl="1"', 'xtl="nan"'),
             "auto": ('source="manual"', 'source="auto"'), "label": ('label="player"', 'label="referee"'),
             "path": ('name="one.png"', 'name="../one.png"')}
    if change == "duplicate":
        body += body
    else:
        body = body.replace(*edits[change])
    archive, images = inputs(tmp_path, body)
    monkeypatch.setattr(cvat, "create_player_detector", lambda *args: pytest.fail("Inference ran"))
    output = tmp_path / "results"
    with pytest.raises(ValueError):
        cvat.compare_cvat(archive, images, tmp_path / "unused", output, confirm_exhaustive=True)
    assert not output.exists()


def test_confirmation_required_and_own_failed_output_removed(tmp_path, monkeypatch):
    archive, images = inputs(tmp_path)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="Confirm exhaustive"):
        cvat.compare_cvat(archive, images, tmp_path / "unused", output)
    def fail(*args):
        raise RuntimeError("Model failed")
    monkeypatch.setattr(cvat, "create_player_detector", fail)
    with pytest.raises(RuntimeError, match="Model failed"):
        cvat.compare_cvat(archive, images, tmp_path / "unused", output, confirm_exhaustive=True)
    assert not output.exists() and archive.exists() and images.exists()


def test_matching_counts_duplicates_and_empty_cases():
    box = [0, 0, 10, 10]
    result = cvat.score_boxes([box], [box, box])
    assert result["true_positives"] == 1 and result["false_positives"] == 1
    assert cvat.score_boxes([box], [])["false_negatives"] == 1
    assert cvat.score_boxes([], [box])["false_positives"] == 1
    assert cvat.score_boxes([], [])["matches"] == []


def test_cardinality_has_priority_over_total_iou(monkeypatch):
    scores = {(0, 2): .9, (0, 3): .5, (1, 2): .5, (1, 3): .49}
    monkeypatch.setattr(cvat, "_bbox_iou", lambda a, b: scores[(a[0], b[0])])
    result = cvat.score_boxes([[0, 0, 5, 5], [1, 0, 5, 5]], [[2, 0, 5, 5], [3, 0, 5, 5]])
    assert result["true_positives"] == 2


def test_entity_declarations_and_empty_exports_refused(tmp_path):
    archive, images = inputs(tmp_path, "")
    with pytest.raises(ValueError, match="no images"):
        cvat.load_cvat(archive, images)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("annotations.xml", '<!DOCTYPE annotations [<!ENTITY x "abc">]><annotations/>')
    with pytest.raises(ValueError, match="entity"):
        cvat.load_cvat(archive, images)
