"""Scoped timeline events with explicit evidence and abstention."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Sequence

from football_vision.analytics.routes import _coordinate, _usable
from football_vision.field_spec import FIELD_WIDTH_YD
from football_vision.schema import PlaySegment, TrajectorySample


@dataclass
class PlayEvent:
    kind: str
    frame_id: int
    source: str
    confidence: float | None = None
    track_id: int | None = None
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_events(segment: PlaySegment, samples: Sequence[TrajectorySample]) -> dict:
    """Report resolved boundaries and sideline transitions, not possession events.

    A sideline transition is a footpoint-geometry candidate, not an officiated
    out-of-bounds ruling. Both consecutive measurements must have covariance and
    their two-sigma lateral intervals must be on opposite sides of the boundary.
    """
    events = []
    if segment.snap_frame is not None:
        events.append(PlayEvent("snap", segment.snap_frame, segment.snap_source,
                                segment.snap_confidence))
    if segment.end_frame is not None:
        events.append(PlayEvent("play_end", segment.end_frame, segment.end_source,
                                segment.end_confidence))
    groups: dict[int, list[TrajectorySample]] = {}
    seen = set()
    for s in samples:
        if s.frame_id < segment.start_frame or (segment.end_frame is not None and s.frame_id > segment.end_frame):
            continue
        key = s.track_id, s.frame_id
        if key in seen:
            raise ValueError("duplicate event observation")
        seen.add(key)
        groups.setdefault(s.track_id, []).append(s)
    for track_id, track_samples in sorted(groups.items()):
        ordered = sorted(track_samples, key=lambda s: s.frame_id)
        for a, b in zip(ordered, ordered[1:]):
            if (not _usable(a) or not _usable(b) or b.frame_id != a.frame_id + 1
                    or _coordinate(a) != _coordinate(b) or b.timestamp_s <= a.timestamp_s):
                continue
            if a.covariance_xy is None or b.covariance_xy is None:
                continue
            covariances = [a.covariance_xy, b.covariance_xy]
            if any(not all(math.isfinite(v) for v in cov) or cov[0] < 0 or cov[2] < 0
                   or cov[0] * cov[2] < cov[1] ** 2 for cov in covariances):
                continue
            variances = [a.covariance_xy[2], b.covariance_xy[2]]
            if any(not math.isfinite(v) or v < 0 for v in variances):
                continue
            ay, by = a.field_position[1], b.field_position[1]
            sa, sb = (2 * math.sqrt(v) for v in variances)
            inside = ay - sa > 0 and ay + sa < FIELD_WIDTH_YD
            outside = by + sb < 0 or by - sb > FIELD_WIDTH_YD
            if inside and outside:
                events.append(PlayEvent("sideline_exit_candidate", b.frame_id,
                                        "observed_footpoint_geometry", None, track_id, {
                    "previous_frame": a.frame_id, "coordinate_frame_id": b.coordinate_frame_id,
                    "coordinate_segment": b.coordinate_segment,
                    "x_coord_mode": b.x_coord_mode, "lateral_interval_sigma": 2,
                    "limitations": "uncertainty is uncalibrated; not a rules-based ruling",
                }))
    return {
        "events": [e.to_dict() for e in sorted(events, key=lambda e: (e.frame_id, e.kind, e.track_id or -1))],
        "unavailable": {kind: "requires validated ball/possession evidence"
                        for kind in ("pass", "run", "catch", "incomplete", "tackle")},
    }
