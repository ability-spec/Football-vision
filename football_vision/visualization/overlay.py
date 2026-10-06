"""Render video-runner observations without inventing field coordinates."""
from __future__ import annotations

import math
from collections.abc import Mapping

import cv2
import numpy as np

from football_vision.field_spec import FIELD_WIDTH_YD


def render_frame(image: np.ndarray, record: Mapping, *, panel_width: int = 360,
                 x_bounds: tuple[float, float] | None = None) -> np.ndarray:
    """Append a field view to a BGR frame; input pixels are never modified.

    A field view uses exactly one coordinate epoch. Relative coordinates are
    shown in a local window, never against absolute yard-line labels. Video callers
    supply fixed ``x_bounds`` for each epoch to avoid apparent motion from rescaling. Predicted
    positions are deliberately excluded from this measured-position view.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 BGR frame")
    if image.shape[0] < 96 or image.shape[1] < 1 or panel_width < 240:
        raise ValueError("frame height must be >=96 and panel width >=240")
    if x_bounds is not None and (len(x_bounds) != 2 or not all(math.isfinite(v) for v in x_bounds)
                                 or x_bounds[0] >= x_bounds[1]):
        raise ValueError("x_bounds must contain increasing finite coordinates")
    height, width = image.shape[:2]
    canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    canvas[:, :width] = image
    players = record.get("players", [])
    measured = []
    epochs = set()
    for player in players:
        box = player.get("bbox")
        if box is None or len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("player bbox must contain four finite coordinates")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("player bbox must have positive area")
        observed = player.get("observed", True)
        color = (80, 220, 80) if observed else (0, 180, 255)
        a = (int(np.clip(x1, 0, width - 1)), int(np.clip(y1, 0, height - 1)))
        b = (int(np.clip(x2, 0, width - 1)), int(np.clip(y2, 0, height - 1)))
        cv2.rectangle(canvas[:, :width], a, b, color, 1)
        state = "observed" if observed else "coasted"
        cv2.putText(canvas[:, :width], f'{player["track_id"]} {state}', a,
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        xy = player.get("field_position")
        mode = player.get("x_coord_mode")
        epoch = player.get("coordinate_frame_id")
        if (observed and xy is not None and len(xy) == 2
                and all(math.isfinite(v) for v in xy) and epoch is not None
                and mode in ("absolute", "relative_10yd", "relative_5yd")):
            measured.append(player)
            epochs.add((epoch, player.get("coordinate_segment", 0), mode))
    def label(text, y):
        cv2.putText(canvas, text, (width + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (230, 230, 230), 1)
    phase = record.get("play_phase", "")
    label(f'Frame {record["frame_id"]} {phase}', 18)
    label("Green: observed / amber: coasted", 36)
    if not measured or len(epochs) != 1:
        label("Field unavailable" if not measured else "Mixed coordinate frames", 58)
        return canvas
    mode = next(iter(epochs))[2]
    xs = [p["field_position"][0] for p in measured]
    lo, hi = (0., 100.) if mode == "absolute" else (x_bounds or (min(xs) - 5., max(xs) + 5.))
    label("Absolute field (yards)" if mode == "absolute" else "Relative field (local origin)", 58)
    left, right, top, bottom = width + 16, width + panel_width - 16, 72, height - 12
    cv2.rectangle(canvas, (left, top), (right, bottom), (30, 100, 30), -1)
    for player in measured:
        x, y = player["field_position"]
        # Out-of-field observations remain outside the view, never clamped.
        if not lo <= x <= hi or not 0 <= y <= FIELD_WIDTH_YD:
            continue
        point = (round(left + (x - lo) / (hi - lo) * (right - left)),
                 round(bottom - y / FIELD_WIDTH_YD * (bottom - top)))
        cv2.circle(canvas, point, 4, (80, 220, 80), -1)
        cv2.putText(canvas, str(player["track_id"]), point,
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    return canvas
