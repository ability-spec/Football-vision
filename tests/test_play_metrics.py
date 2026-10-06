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


def test_presnap_motion_is_unknown_without_presnap_edges(tmp_path):
    # Post-snap motion cannot establish whether this track moved before the snap.
    result = analyze_play(segment(snap_frame=2), [trajectory([
        sample(2, (10, 10)), sample(3, (12, 10))])])
    assert result.metrics["observed_path_length_sum_yd"].value == 2
    assert result.metrics["presnap_motion_track_count"].value is None
    # The distinction must survive the user-facing CSV/HTML export.
    import csv
    from football_vision.pipeline import _write_report
    _write_report(dict(plays=[result.to_dict()], segmentation=dict(refusals=[]),
                       frames_processed=4, source_kind="synthetic", video_sha256="0" * 64), tmp_path)
    with (tmp_path / "metrics.csv").open() as file:
        rows = {row["metric"]: row for row in csv.DictReader(file)}
    assert rows["presnap_motion_track_count"]["value"] == ""
    assert "Unavailable" in (tmp_path / "report.html").read_text()
    # One pre-snap position plus the snap position is still not a pre-snap edge.
    result = analyze_play(segment(snap_frame=2), [trajectory([
        sample(1, (10, 10)), sample(2, (12, 10)), sample(3, (13, 10))])])
    assert result.metrics["presnap_motion_track_count"].value is None


def test_presnap_motion_distinguishes_observed_stillness_and_movement():
    for displacement, expected in ((0, 0), (1, 1)):
        result = analyze_play(segment(snap_frame=2), [trajectory([
            sample(0, (10, 10)), sample(1, (10 + displacement, 10)), sample(2, (20, 10))])])
        assert result.metrics["presnap_motion_track_count"].value == expected
        assert result.routes[0]["segments"][0]["presnap_n_edges"] == 1


def test_presnap_motion_does_not_bridge_coordinate_change():
    result = analyze_play(segment(snap_frame=2), [trajectory([
        sample(0, (10, 10)), sample(1, (20, 10), coordinate_segment=1),
        sample(2, (21, 10), coordinate_segment=1), sample(3, (22, 10), coordinate_segment=1)])])
    assert result.metrics["presnap_motion_track_count"].value is None
