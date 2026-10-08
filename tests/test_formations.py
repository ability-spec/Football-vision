import json
import pytest
from football_vision.analytics.formations import analyze_formation
from football_vision.schema import TrajectorySample


def sample(track, xy, **kwargs):
    values = dict(track_id=track, frame_id=1, timestamp_s=0.1,
                  geometry_state="calibrated", x_coord_mode="absolute",
                  track_state="observed", field_position=xy,
                  position_source="measured_smoothed", is_measurement_used=True,
                  coordinate_frame_id="clip-a", coordinate_segment=0)
    values.update(kwargs)
    return TrajectorySample(**values)


def analyze(samples, **kwargs):
    return analyze_formation(samples, frame_id=1, snap_frame=2,
                             team_by_track={1: "O", 2: "O", 3: "D"}, **kwargs)


def test_geometry_and_annotated_box():
    r = analyze([sample(1, (40, 20)), sample(2, (43, 24)), sample(3, (45, 25))],
                offense_team="O", defense_team="D", line_of_scrimmage_x=40, box_center_y=20)
    assert r.teams["O"]["width_yd"] == 4
    assert r.teams["O"]["depth_yd"] == 3
    assert r.teams["O"]["mean_nearest_teammate_distance_yd"] == 5
    assert r.defensive_box_count == 1
    json.dumps(r.to_dict(), allow_nan=False)


def test_prediction_and_coasting_are_excluded():
    r = analyze([sample(1, (1, 2), track_state="coasted"),
                 sample(2, None, predicted_position=(3, 4), position_source="predicted_dead_reckoning")])
    assert r.status == "unavailable"
    assert r.excluded_track_ids == [1, 2]


def test_coordinate_boundaries_refuse():
    assert analyze([sample(1, (0, 0)), sample(2, (1, 1), coordinate_segment=1)]).reason == "ambiguous_coordinate_frame"
    assert analyze([sample(1, (0, 0), coordinate_frame_id=None)]).status == "unavailable"


def test_missing_team_and_box_annotations_are_unknown():
    r = analyze([sample(1, (0, 0)), sample(9, (1, 1))])
    assert r.unassigned_track_ids == [9]
    assert r.defensive_box_count is None
    assert r.teams["O"]["mean_nearest_teammate_distance_yd"] is None


def test_direction_and_zero_box():
    r = analyze([sample(3, (45, 20))], defense_team="D", line_of_scrimmage_x=40,
                box_center_y=20, offense_direction=-1)
    assert r.defensive_box_count == 0


def test_input_validation():
    with pytest.raises(ValueError):
        analyze([sample(1, (0, 0)), sample(1, (1, 1))])
    with pytest.raises(ValueError):
        analyze_formation([], frame_id=2, snap_frame=2, team_by_track={})
    with pytest.raises(ValueError):
        analyze([], box_depth_yd=float("nan"))


def test_unknown_direction_preserves_spacing_but_not_defensive_box():
    r = analyze([sample(1, (40, 20)), sample(2, (43, 24)), sample(3, (45, 25))],
                offense_team="O", defense_team="D", line_of_scrimmage_x=40,
                box_center_y=20, offense_direction=None)
    assert r.status == "available"
    assert r.teams["O"]["width_yd"] == 4
    assert r.teams["O"]["depth_yd"] == 3
    assert r.defensive_box_count is None
    assert r.provenance["offense_direction"] is None
