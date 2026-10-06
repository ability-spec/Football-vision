import numpy as np
import pytest

from football_vision.visualization import render_frame


def player(**overrides):
    return dict(track_id=1, bbox=[10, 20, 30, 50], observed=True,
                field_position=[20, 25], x_coord_mode="absolute",
                coordinate_frame_id="epoch", **overrides)


def test_overlay_preserves_input_and_draws_measured_field_position():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    out = render_frame(image, dict(frame_id=0, players=[player()]))
    assert out.shape == (160, 560, 3)
    assert not image.any()
    assert (out[50, 20] == [80, 220, 80]).all()
    assert (out[112, 282] == [80, 220, 80]).all()


def test_predictions_and_mixed_origins_do_not_appear_as_measurements():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    a = player()
    b = dict(a, track_id=2, coordinate_frame_id="other")
    for players in ([dict(a, observed=False, predicted_position=[20, 25])], [a, b]):
        out = render_frame(image, dict(frame_id=0, players=players))
        assert not np.any(np.all(out[72:, 216:] == [30, 100, 30], axis=2))


def test_relative_position_uses_local_window_and_invalid_box_is_refused():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    a = dict(player(), field_position=[-200, 25], x_coord_mode="relative_5yd")
    out = render_frame(image, dict(frame_id=0, players=[a]))
    assert (out[112, 380] == [80, 220, 80]).all()
    with pytest.raises(ValueError, match="positive area"):
        render_frame(image, dict(frame_id=0, players=[dict(a, bbox=[0, 0, 0, 1])]))


def test_video_render_round_trip_and_source_protection(tmp_path):
    import cv2
    import hashlib
    import json

    from football_vision.visualization.__main__ import render_video

    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 160))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((160, 200, 3), dtype=np.uint8))
    writer.release()
    predictions = tmp_path / "predictions.json"
    artifact = dict(video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(), fps=10,
                    frames=[dict(frame_id=i, players=[player()]) for i in range(2)])
    predictions.write_text(json.dumps(artifact))
    output = tmp_path / "review.avi"
    assert render_video(video, predictions, output) == 2
    capture = cv2.VideoCapture(str(output))
    try:
        ok, image = capture.read()
        assert ok and image.shape == (160, 560, 3)
    finally:
        capture.release()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        render_video(video, predictions, output)
    assert output.read_bytes() == before
    artifact["video_sha256"] = "0" * 64
    predictions.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="different video"):
        render_video(video, predictions, tmp_path / "wrong.avi")
    assert not (tmp_path / "wrong.avi").exists()


def test_fixed_relative_scale_prevents_stationary_player_from_moving():
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    stationary = dict(player(), field_position=[10, 25], x_coord_mode="relative_5yd")
    moving = dict(stationary, track_id=2, field_position=[20, 5])
    first = render_frame(image, dict(frame_id=0, players=[stationary, moving]), x_bounds=(0, 40))
    moving["field_position"] = [30, 5]
    second = render_frame(image, dict(frame_id=1, players=[stationary, moving]), x_bounds=(0, 40))
    assert np.array_equal(first[105:120, 290:310], second[105:120, 290:310])
    assert (first[112, 298] == [80, 220, 80]).all()
    with pytest.raises(ValueError, match="x_bounds"):
        render_frame(image, dict(frame_id=0, players=[]), x_bounds=(10, 10))


def test_video_export_pads_odd_render_dimensions(tmp_path, monkeypatch):
    import cv2
    import hashlib
    import json
    from football_vision.visualization.__main__ import render_video

    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 160))
    assert writer.isOpened()
    writer.write(np.zeros((160, 200, 3), dtype=np.uint8))
    writer.release()
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(dict(video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                                           fps=10, frames=[dict(frame_id=0, players=[])])))
    monkeypatch.setattr("football_vision.visualization.__main__.render_frame",
                        lambda *args, **kwargs: np.full((161, 561, 3), 200, dtype=np.uint8))
    output = tmp_path / "review.avi"
    assert render_video(video, predictions, output) == 1
    capture = cv2.VideoCapture(str(output))
    try:
        ok, decoded = capture.read()
        assert ok and decoded.shape == (162, 562, 3)
    finally:
        capture.release()
