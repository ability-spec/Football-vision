"""Read native NFL Helmet Assignment labels without inventing whole-player boxes."""
from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import numpy as np


def read_helmet_labels(path: Path, videos: dict[str, dict]) -> dict[str, dict[int, list[dict]]]:
    """Convert one-based Kaggle frames to zero-based decoded frames explicitly."""
    result: dict[str, dict[int, list[dict]]] = {name: {} for name in videos}
    seen = set()
    required = {"video", "frame", "label", "left", "top", "width", "height", "isSidelinePlayer"}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing native helmet-label columns")
        for row in reader:
            name = row["video"]
            if name not in videos:
                continue
            frame = int(row["frame"]) - 1
            video = videos[name]
            if not 0 <= frame < video["frame_count"]:
                raise ValueError(f"Helmet frame outside decoded video: {name}")
            flag = row["isSidelinePlayer"].strip().upper()
            if flag not in {"TRUE", "FALSE"}:
                raise ValueError("Invalid sideline flag")
            if not row["label"].strip():
                raise ValueError("Missing helmet player identity")
            key = (name, frame, row["label"])
            if key in seen:
                raise ValueError("Duplicate helmet identity in a frame")
            seen.add(key)
            left, top, width, height = (float(row[c]) for c in ("left", "top", "width", "height"))
            if not all(math.isfinite(v) for v in (left, top, width, height)) or width <= 0 or height <= 0:
                raise ValueError("Invalid helmet box")
            # Edge boxes can cross image boundaries. Preserve native coordinates.
            entry = dict(player_id=row["label"], helmet_bbox=[left, top, left + width, top + height],
                         is_sideline_player=flag == "TRUE", source_frame=frame + 1)
            result[name].setdefault(frame, []).append(entry)
    if any(not frames for frames in result.values()):
        raise ValueError("A selected video has no helmet labels")
    return result


def read_player_tracking(path: Path, plays: set[tuple[int, int]]) -> dict[tuple[int, int], list[dict]]:
    """Keep sensor positions in their native frame; do not align them to video."""
    result: dict[tuple[int, int], list[dict]] = {key: [] for key in plays}
    required = {"gameKey", "playID", "player", "time", "x", "y", "event"}
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing native tracking columns")
        for row in reader:
            key = (int(row["gameKey"]), int(row["playID"]))
            if key not in plays:
                continue
            timestamp = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
            if timestamp.utcoffset() is None or not row["player"].strip():
                raise ValueError("Tracking needs timezone-aware time and player identity")
            identity = (key, row["player"], timestamp)
            if identity in seen:
                raise ValueError("Duplicate sensor sample")
            seen.add(identity)
            x, y = float(row["x"]), float(row["y"])
            if not all(math.isfinite(v) for v in (x, y)):
                raise ValueError("Nonfinite sensor position")
            result[key].append(dict(player_id=row["player"], time=timestamp.isoformat(),
                                    x=x, y=y, event=row["event"]))
    if any(not rows for rows in result.values()):
        raise ValueError("A selected play has no sensor samples")
    return result


def helmet_coverage_diagnostic(helmets: list[dict], person_boxes: list[list[float]]) -> dict:
    """One-to-one helmet-center containment; neither player recall nor precision.

    A box spanning several players can cover only one helmet here. Still, wrong
    or oversized boxes may contain helmet centers, so this cannot score identity
    or whole-player localization. Native sideline-player labels are excluded.
    """
    active = [h for h in helmets if not h["is_sideline_player"]]
    boxes = np.asarray(person_boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Invalid person box")
    matches = []
    if active and len(boxes):
        centers = np.asarray([[(h["helmet_bbox"][0] + h["helmet_bbox"][2]) / 2,
                               (h["helmet_bbox"][1] + h["helmet_bbox"][3]) / 2] for h in active])
        eligible = np.all(centers[:, None] >= boxes[None, :, :2], axis=2)
        eligible &= np.all(centers[:, None] <= boxes[None, :, 2:], axis=2)
        rows, cols = linear_sum_assignment(-eligible.astype(int))
        matches = [dict(player_id=active[r]["player_id"], prediction_index=int(c))
                   for r, c in zip(rows, cols) if eligible[r, c]]
    matched = {m["player_id"] for m in matches}
    return dict(labeled_on_field_helmets=len(active), person_candidates=len(boxes),
                matched_helmet_centers=len(matches), matches=matches,
                unmatched_helmet_ids=[h["player_id"] for h in active if h["player_id"] not in matched],
                metric_scope="one-to-one helmet-center containment; not player recall, AP or precision")
