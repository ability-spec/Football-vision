import json
import pytest
from football_vision.analytics.routes import analyze_route
from football_vision.schema import TrajectorySample


def s(frame, xy, track=1, **kwargs):
    values = dict(track_id=track, frame_id=frame, timestamp_s=frame / 10,
                  geometry_state="calibrated", x_coord_mode="relative_5yd",
                  field_position=xy, position_source="measured_smoothed",
                  is_measurement_used=True, track_state="observed",
                  coordinate_frame_id="a")
    values.update(kwargs)
    return TrajectorySample(**values)


def run(samples, **kwargs):
    return analyze_route(samples, track_id=1, start_frame=0, end_frame=9, snap_frame=2, **kwargs)


def test_route_depth_distance_and_presnap():
    r = run([s(0, (10, 0)), s(1, (7, 0)), s(2, (4, 0))], offense_direction=-1)
    seg = r.segments[0]
    assert seg["max_forward_depth_yd"] == 6
    assert seg["distance_yd"] == 6
    assert seg["presnap_distance_yd"] == 3
    assert seg["net_stem"] == "longitudinal"
    json.dumps(r.to_dict(), allow_nan=False)


def test_missing_and_coordinate_changes_never_bridge():
    r = run([s(0, (0, 0)), s(2, (100, 0)), s(3, (200, 0), coordinate_segment=1)])
    assert r.status == "unavailable"
    assert all(seg["distance_yd"] == 0 for seg in r.segments)


def test_coasting_and_predictions_excluded():
    r = run([s(0, (0, 0)), s(1, (1, 0), track_state="coasted"), s(2, (2, 0))])
    assert r.excluded_frame_ids == [1]
    assert r.status == "unavailable"


def test_separation_requires_same_frame_coordinates_and_timestamp():
    r = run([s(0, (0, 0)), s(1, (1, 0))], defender_samples=[
        s(0, (3, 4), track=2), s(0, (0, 0), track=3, coordinate_segment=1),
        s(1, (1, 0), track=2, timestamp_s=5)])
    assert r.separation[0]["distance_yd"] == 5
    assert r.separation[0]["nearest_defender_track_id"] == 2
    assert r.separation[1]["distance_yd"] is None


def test_crossing_vs_touch_and_lateral_stem():
    r = run([s(0, (0, -2)), s(1, (0, 0)), s(2, (0, -2)), s(3, (0, 3))], crossing_y=0)
    assert r.segments[0]["crossing_frame_ids"] == [3]
    assert r.segments[0]["net_stem"] == "lateral"


def test_invalid_inputs():
    with pytest.raises(ValueError):
        run([s(0, (0, 0)), s(0, (1, 1))])
    with pytest.raises(ValueError):
        run([], stem_dominance_ratio=1)
    with pytest.raises(ValueError):
        run([], defender_samples=[s(0, (1, 1))])


def test_unknown_offense_direction_preserves_geometry_without_forward_claims():
    r = run([s(0, (10, 0)), s(1, (7, 4))], offense_direction=None)
    part = r.segments[0]
    assert r.status == "available"
    assert r.provenance["offense_direction"] is None
    assert part["distance_yd"] == 5
    assert part["longitudinal_displacement_yd"] == -3
    assert part["lateral_displacement_yd"] == 4
    assert part["forward_displacement_yd"] is None
    assert part["max_forward_depth_yd"] is None
    json.dumps(r.to_dict(), allow_nan=False)
