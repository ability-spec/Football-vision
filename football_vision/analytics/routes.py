"""Observed route geometry within explicit coordinate and frame boundaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Sequence

from football_vision.schema import TrajectorySample


def _usable(s: TrajectorySample) -> bool:
    return bool(s.field_position is not None and s.position_source == "measured_smoothed"
                and s.is_measurement_used and not s.is_outlier_rejected
                and s.track_state == "observed" and s.geometry_state != "unknown"
                and s.x_coord_mode != "uncalibrated" and s.coordinate_frame_id is not None
                and math.isfinite(s.timestamp_s)
                and all(math.isfinite(v) for v in s.field_position))


def _coordinate(s: TrajectorySample) -> tuple:
    return s.coordinate_frame_id, s.coordinate_segment, s.x_coord_mode


@dataclass
class RouteAnalysis:
    track_id: int
    status: str = "unavailable"
    reason: str | None = None
    segments: list[dict] = field(default_factory=list)
    separation: list[dict] = field(default_factory=list)
    excluded_frame_ids: list[int] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_route(
    samples: Sequence[TrajectorySample], *, track_id: int,
    start_frame: int, end_frame: int, snap_frame: int,
    offense_direction: int = 1,
    defender_samples: Sequence[TrajectorySample] = (),
    crossing_y: float | None = None, stem_min_displacement_yd: float = 1.0,
    stem_dominance_ratio: float = 2.0,
) -> RouteAnalysis:
    """Measure inclusive-window motion without bridging missing observations.

    Defenders are explicit caller-selected tracks, not inferred player roles.
    Depth is forward displacement from each continuous segment's first sample.
    Separation uses same-frame, same-coordinate, same-timestamp measurements.
    """
    if not 0 <= start_frame <= end_frame or snap_frame < 0:
        raise ValueError("invalid frame window or snap")
    if offense_direction not in (-1, 1):
        raise ValueError("offense_direction must be -1 or 1")
    if not math.isfinite(stem_min_displacement_yd) or stem_min_displacement_yd <= 0:
        raise ValueError("stem minimum must be finite and positive")
    if not math.isfinite(stem_dominance_ratio) or stem_dominance_ratio <= 1:
        raise ValueError("stem dominance ratio must be finite and greater than one")
    if crossing_y is not None and not math.isfinite(crossing_y):
        raise ValueError("crossing reference must be finite")
    selected = sorted((s for s in samples if s.track_id == track_id
                       and start_frame <= s.frame_id <= end_frame), key=lambda s: s.frame_id)
    if len({s.frame_id for s in selected}) != len(selected):
        raise ValueError("duplicate route observations")
    defenders: dict[int, list[TrajectorySample]] = {}
    seen = set()
    for d in defender_samples:
        if not start_frame <= d.frame_id <= end_frame:
            continue
        key = d.track_id, d.frame_id
        if key in seen or d.track_id == track_id:
            raise ValueError("duplicate defender observation or target listed as defender")
        seen.add(key)
        if _usable(d):
            defenders.setdefault(d.frame_id, []).append(d)
    result = RouteAnalysis(track_id, provenance={
        "source": "observed_measured_trajectory", "units": "yards",
        "start_frame": start_frame, "end_frame": end_frame, "snap_frame": snap_frame,
        "offense_direction": offense_direction, "crossing_y": crossing_y,
        "stem_min_displacement_yd": stem_min_displacement_yd,
        "stem_dominance_ratio": stem_dominance_ratio,
        "defender_source": "caller_selected_tracks", "route_label": None,
        "distance_definition": "sum_of_consecutive_observed_edges",
    })
    runs: list[list[TrajectorySample]] = []
    for s in selected:
        if not _usable(s):
            result.excluded_frame_ids.append(s.frame_id)
            continue
        previous = runs[-1][-1] if runs else None
        if (previous is None or s.frame_id != previous.frame_id + 1
                or _coordinate(s) != _coordinate(previous)
                or s.timestamp_s <= previous.timestamp_s):
            runs.append([])
        runs[-1].append(s)
        candidates = [d for d in defenders.get(s.frame_id, [])
                      if _coordinate(d) == _coordinate(s)
                      and abs(d.timestamp_s - s.timestamp_s) <= 1e-6]
        nearest = min(candidates, key=lambda d: (math.dist(s.field_position, d.field_position), d.track_id)) if candidates else None
        result.separation.append({
            "frame_id": s.frame_id,
            "nearest_defender_track_id": nearest.track_id if nearest else None,
            "distance_yd": math.dist(s.field_position, nearest.field_position) if nearest else None,
            "coordinate_frame_id": s.coordinate_frame_id,
            "coordinate_segment": s.coordinate_segment, "x_coord_mode": s.x_coord_mode,
        })
    for run in runs:
        first, last = run[0], run[-1]
        x0, y0 = first.field_position
        forward = [offense_direction * (s.field_position[0] - x0) for s in run]
        dx, dy = forward[-1], last.field_position[1] - y0
        stem = "unknown"
        if len(run) > 1:
            if max(abs(dx), abs(dy)) < stem_min_displacement_yd:
                stem = "stationary"
            elif abs(dx) >= stem_dominance_ratio * abs(dy):
                stem = "longitudinal"
            elif abs(dy) >= stem_dominance_ratio * abs(dx):
                stem = "lateral"
            else:
                stem = "mixed"
        edges = list(zip(run, run[1:]))
        # Touching the reference and returning to the same side is not a crossing.
        signs = [(s.frame_id, 1 if s.field_position[1] > crossing_y else -1)
                 for s in run if crossing_y is not None and s.field_position[1] != crossing_y]
        crossings = [b[0] for a, b in zip(signs, signs[1:]) if a[1] != b[1]]
        result.segments.append({
            "start_frame": first.frame_id, "end_frame": last.frame_id,
            "coordinate_frame_id": first.coordinate_frame_id,
            "coordinate_segment": first.coordinate_segment, "x_coord_mode": first.x_coord_mode,
            "n_observations": len(run), "n_edges": len(edges),
            "duration_s": last.timestamp_s - first.timestamp_s,
            "forward_displacement_yd": dx, "lateral_displacement_yd": dy,
            "max_forward_depth_yd": max(forward), "net_stem": stem,
            "distance_yd": sum(math.dist(a.field_position, b.field_position) for a, b in edges),
            "presnap_distance_yd": sum(math.dist(a.field_position, b.field_position)
                                       for a, b in edges if b.frame_id < snap_frame),
            "crossing_frame_ids": crossings if crossing_y is not None else None,
        })
    if any(s["n_edges"] for s in result.segments):
        result.status = "available"
    else:
        result.reason = "no_consecutive_measured_positions"
    return result
