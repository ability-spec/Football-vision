"""Play-level geometry aggregation, data coverage and provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence

from football_vision.analytics.events import detect_events
from football_vision.analytics.formations import analyze_formation
from football_vision.analytics.routes import _usable, analyze_route
from football_vision.schema import FieldTrajectory, PlaySegment


@dataclass
class MetricValue:
    value: float | int | None
    units: str
    definition: str
    source: str
    confidence: float | None = None
    limitations: str = "visible accepted observations only; real-video accuracy unmeasured"


@dataclass
class PlayAnalysis:
    game_id: str
    play_id: str
    segment: dict
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    formation: dict | None = None
    routes: list[dict] = field(default_factory=list)
    events: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_play(segment: PlaySegment, trajectories: Sequence[FieldTrajectory], *,
                 team_by_track: Mapping[int, str] | None = None,
                 offense_team: str | None = None, defense_team: str | None = None,
                 offense_direction: int = 1, line_of_scrimmage_x: float | None = None,
                 box_center_y: float | None = None, crossing_y: float | None = None) -> PlayAnalysis:
    if not math.isfinite(segment.fps) or segment.fps <= 0:
        raise ValueError("play fps must be finite and positive")
    if offense_direction not in (-1, 1):
        raise ValueError("offense_direction must be -1 or 1")
    for value in (line_of_scrimmage_x, box_center_y, crossing_y):
        if value is not None and not math.isfinite(value):
            raise ValueError("geometry annotations must be finite")
    if any(not isinstance(v, str) or not v for v in (team_by_track or {}).values()):
        raise ValueError("team assignments must be nonempty string labels")
    teams = {k: v for k, v in (team_by_track or {}).items() if v != "UNKNOWN"}
    samples = [s for t in trajectories for s in t.samples
               if s.frame_id >= segment.start_frame
               and (segment.end_frame is None or s.frame_id <= segment.end_frame)]
    if len({(s.track_id, s.frame_id) for s in samples}) != len(samples):
        raise ValueError("duplicate play observations")
    usable = [s for s in samples if _usable(s)]
    result = PlayAnalysis(segment.game_id, segment.play_id, segment.to_dict(), provenance={
        "schema_version": 1, "metric_confidence": "unmeasured; null is not zero",
        "offense_team": offense_team, "defense_team": defense_team,
        "offense_direction": offense_direction, "team_source": "caller_mapping",
    })
    def metric(name, value, units, definition, source="observed_measured_trajectory"):
        result.metrics[name] = MetricValue(value, units, definition, source)
    metric("observed_track_count", len({s.track_id for s in usable}), "tracks", "Distinct tracks with accepted positions; not unique athletes")
    metric("measured_sample_count", len(usable), "samples", "Accepted observed measured positions")
    metric("position_sample_coverage", len(usable) / len(samples) if samples else None,
           "fraction", "Accepted measured positions / supplied trajectory samples; excludes unseen players")
    metric("play_duration_s", (segment.end_frame - segment.start_frame) / segment.fps if segment.end_frame is not None else None,
           "seconds", "Elapsed time from start to end timestamps", "segmentation")
    metric("post_snap_duration_s", (segment.end_frame - segment.snap_frame) / segment.fps
           if segment.end_frame is not None and segment.snap_frame is not None else None,
           "seconds", "Elapsed time from snap to end timestamps", "segmentation")
    if segment.snap_frame is not None and segment.snap_frame > segment.start_frame:
        # Exactly the last pre-snap frame: never silently substitute an earlier view.
        result.formation = analyze_formation(samples, frame_id=segment.snap_frame - 1,
            snap_frame=segment.snap_frame, team_by_track=teams, offense_team=offense_team,
            defense_team=defense_team, offense_direction=offense_direction,
            line_of_scrimmage_x=line_of_scrimmage_x, box_center_y=box_center_y).to_dict()
    if segment.snap_frame is not None and segment.end_frame is not None:
        defenders = [s for s in samples if defense_team is not None and teams.get(s.track_id) == defense_team]
        for track_id in sorted({s.track_id for s in samples}):
            result.routes.append(analyze_route(samples, track_id=track_id,
                start_frame=segment.start_frame, end_frame=segment.end_frame,
                snap_frame=segment.snap_frame, offense_direction=offense_direction,
                defender_samples=[s for s in defenders if s.track_id != track_id]
                    if offense_team is not None and teams.get(track_id) == offense_team else (),
                crossing_y=crossing_y).to_dict())
    route_segments = [part for route in result.routes for part in route["segments"]]
    measured_edges = sum(part["n_edges"] for part in route_segments)
    metric("observed_path_length_sum_yd", sum(part["distance_yd"] for part in route_segments) if measured_edges else None,
           "yards", "Sum of per-track consecutive observed edges; never bridges gaps; not unique-athlete distance")
    metric("presnap_motion_track_count", sum(any(part["presnap_distance_yd"] >= 1.0 for part in route["segments"])
           for route in result.routes) if measured_edges else None,
           "tracks", "Tracks with at least one yard of pre-snap path within a continuous segment")
    offensive_distances = [entry["distance_yd"] for route in result.routes
        if offense_team is not None and teams.get(route["track_id"]) == offense_team
        for entry in route["separation"] if entry["distance_yd"] is not None]
    metric("mean_observed_defender_separation_yd", sum(offensive_distances) / len(offensive_distances)
           if offensive_distances else None, "yards", "Sample-weighted nearest annotated defender distance for offensive tracks")
    result.events = detect_events(segment, samples)
    return result
