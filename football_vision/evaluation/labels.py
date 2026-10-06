"""Strict manual timestamp parsing shared by the video workflows."""
import hashlib
import json
from pathlib import Path

from football_vision.schema import PlayTimestampLabel
from football_vision.analytics.segmentation import PlaySegmenter


def load_labels(path: Path) -> tuple[str, list[PlayTimestampLabel], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("plays"), list):
        raise ValueError("labels must contain a plays list")
    game = document.get("game_id", "")
    if not isinstance(game, str) or not document["plays"]:
        raise ValueError("game_id must be a string and plays must be nonempty")
    labels = []
    for entry in document["plays"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("play_id"), str) or not entry["play_id"]:
            raise ValueError("each play needs a nonempty string play_id")
        for name in ("start_frame", "end_frame", "snap_frame"):
            value = entry.get(name)
            if name != "snap_frame" and value is None:
                raise ValueError(f"{name} is required")
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        labels.append(PlayTimestampLabel(**entry))
    # Reject cross-play conflicts before video decoding or CV inference starts.
    labels = PlaySegmenter.validate_labels(labels)
    return game, labels, hashlib.sha256(raw).hexdigest()
