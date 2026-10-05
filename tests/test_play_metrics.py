import json

from football_vision.analytics.events import detect_events
from football_vision.analytics.play import analyze_play
from football_vision.schema import FieldTrajectory, PlaySegment, TrajectorySample


def sample(frame, xy, track=1, **kwargs):
    values = dict(track_id=track, frame_id=frame, timestamp_s=frame / 10,
                  geometry_state="calibrated", x_coord_mode="absolute", track_state="observed",
                  field_position=xy, position_source="measured_smoothed", is_measurement_used=True,
                  coordinate_frame_id="a", covariance_xy=(0.01, 0, 0.01))
    values.update(kwargs)
    return TrajectorySample(**values)


def segment(**kwargs):
    values = dict(game_id="g", play_id="p", start_frame=0, end_frame=3, snap_frame=1,
                  snap_source="manual", end_source="manual", fps=10,
                  snap_confidence=1, end_confidence=1)
    values.update(kwargs)
    return PlaySegment(**values)


def trajectory(samples):
    return FieldTrajectory(1, 10, "absolute", samples)


def test_play_metrics_missing_vs_zero_and_provenance():
    result = analyze_play(segment(), [trajectory([sample(0, (10, 10)), sample(1, (11, 10))])],
                          team_by_track={1: "home"})
    assert result.metrics["position_sample_coverage"].value == 1
    assert result.metrics["play_duration_s"].value == 0.3
    assert result.formation["teams"]["home"]["observed_count"] == 1
    assert result.metrics["measured_sample_count"].confidence is None
    empty = analyze_play(segment(snap_frame=None, snap_source="unavailable"), [])
    assert empty.metrics["position_sample_coverage"].value is None
    assert empty.metrics["post_snap_duration_s"].value is None
    assert empty.formation is None and empty.routes == []
    json.dumps(result.to_dict(), allow_nan=False)


def test_scoped_events_require_consecutive_uncertainty_evidence():
    result = detect_events(segment(), [sample(1, (10, 1)), sample(2, (10, -1))])
    assert [e["kind"] for e in result["events"]] == ["snap", "sideline_exit_candidate", "play_end"]
    assert result["unavailable"]["catch"]
    for override in ({"coordinate_segment": 1}, {"covariance_xy": None}, {"track_state": "coasted"}):
        events = detect_events(segment(), [sample(1, (10, 1)), sample(2, (10, -1), **override)])
        assert not any(e["kind"] == "sideline_exit_candidate" for e in events["events"])
