"""Observed pre-snap geometry; no formation names or inferred player roles."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence

from football_vision.schema import TrajectorySample


@dataclass
class FormationSnapshot:
    frame_id: int
    status: str
    reason: str | None = None
    coordinate_frame_id: str | None = None
    coordinate_segment: int | None = None
    x_coord_mode: str | None = None
    teams: dict = field(default_factory=dict)
    excluded_track_ids: list[int] = field(default_factory=list)
    unassigned_track_ids: list[int] = field(default_factory=list)
    defensive_box_count: int | None = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_formation(
    samples: Sequence[TrajectorySample], *, frame_id: int,
    team_by_track: Mapping[int, str], snap_frame: int,
    offense_team: str | None = None, defense_team: str | None = None,
    line_of_scrimmage_x: float | None = None, offense_direction: int | None = 1,
    box_center_y: float | None = None, box_half_width_yd: float = 5.0,
    box_depth_yd: float = 5.0,
) -> FormationSnapshot:
    """Summarize one observed frame strictly before the caller-supplied snap.

    Team roles, LOS and box center are caller annotations in the same coordinate
    frame as the samples. Counts describe visible observations, not all 11 players.
    The defensive box extends from LOS toward the defense, inclusive of edges.
    """
    if frame_id < 0 or snap_frame < 0 or frame_id >= snap_frame:
        raise ValueError("formation frame must be nonnegative and strictly before snap")
    if offense_direction is not None and (type(offense_direction) is not int or offense_direction not in (-1, 1)):
        raise ValueError("offense_direction must be -1, 1 or None")
    for name, value in (("box_half_width_yd", box_half_width_yd), ("box_depth_yd", box_depth_yd)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for value in (line_of_scrimmage_x, box_center_y):
        if value is not None and not math.isfinite(value):
            raise ValueError("LOS and box center must be finite")
    if offense_team is not None and offense_team == defense_team:
        raise ValueError("offense and defense must differ")
    if any(not isinstance(v, str) or not v for v in team_by_track.values()):
        raise ValueError("team assignments must be nonempty string labels")
    current = [s for s in samples if s.frame_id == frame_id]
    ids = [s.track_id for s in current]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate track observations in frame")
    result = FormationSnapshot(frame_id, "unavailable", provenance={
        "source": "observed_measured_trajectory", "units": "yards",
        "snap_frame": snap_frame, "team_source": "caller_mapping",
        "population": "visible_players_only", "formation_label": None,
        "offense_team": offense_team, "defense_team": defense_team,
        "line_of_scrimmage_x": line_of_scrimmage_x,
        "offense_direction": offense_direction, "box_center_y": box_center_y,
        "box_half_width_yd": box_half_width_yd, "box_depth_yd": box_depth_yd,
    })
    usable = []
    for s in current:
        if (s.field_position is None or s.position_source != "measured_smoothed"
                or not s.is_measurement_used or s.is_outlier_rejected
                or s.track_state != "observed" or s.geometry_state == "unknown"
                or s.x_coord_mode == "uncalibrated"
                or not all(math.isfinite(v) for v in s.field_position)):
            result.excluded_track_ids.append(s.track_id)
        else:
            usable.append(s)
    if not usable:
        result.reason = "no_measured_positions"
        return result
    coordinates = {(s.coordinate_frame_id, s.coordinate_segment, s.x_coord_mode) for s in usable}
    if len(coordinates) != 1 or usable[0].coordinate_frame_id is None:
        result.reason = "ambiguous_coordinate_frame"
        return result
    result.coordinate_frame_id, result.coordinate_segment, result.x_coord_mode = next(iter(coordinates))
    groups: dict[str, list[TrajectorySample]] = {}
    for s in usable:
        team = team_by_track.get(s.track_id)
        if team is None or team == "UNKNOWN":
            result.unassigned_track_ids.append(s.track_id)
        else:
            groups.setdefault(team, []).append(s)
    for team, players in sorted(groups.items()):
        positions = [s.field_position for s in players]
        xs, ys = zip(*positions)
        nearest = [min(math.dist(p, q) for j, q in enumerate(positions) if i != j)
                   for i, p in enumerate(positions)] if len(players) > 1 else []
        result.teams[team] = {
            "observed_count": len(players), "track_ids": sorted(s.track_id for s in players),
            "width_yd": max(ys) - min(ys), "depth_yd": max(xs) - min(xs),
            "centroid_xy_yd": [sum(xs) / len(xs), sum(ys) / len(ys)],
            "mean_nearest_teammate_distance_yd": sum(nearest) / len(nearest) if nearest else None,
        }
    if not groups:
        result.reason = "no_team_assignments"
        return result
    result.status = "available"
    if (defense_team in groups and line_of_scrimmage_x is not None and box_center_y is not None
            and offense_direction is not None):
        result.defensive_box_count = sum(
            0 <= offense_direction * (s.field_position[0] - line_of_scrimmage_x) <= box_depth_yd
            and abs(s.field_position[1] - box_center_y) <= box_half_width_yd
            for s in groups[defense_team]
        )
    return result
