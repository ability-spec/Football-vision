import csv

import pytest

from football_vision.evaluation.nfl import (
    helmet_coverage_diagnostic, read_helmet_labels, read_player_tracking,
)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def helmet():
    return dict(video="1_000002_Sideline.mp4", frame="1", label="H12", left="5", top="6",
                width="4", height="4", isSidelinePlayer="FALSE")


def test_native_frames_and_sideline_flags(tmp_path):
    row = helmet()
    other = dict(row, frame="2", isSidelinePlayer="TRUE")
    path = tmp_path / "helmets.csv"
    write_csv(path, [row, other])
    labels = read_helmet_labels(path, {row["video"]: {"frame_count": 2}})[row["video"]]
    assert labels[0][0]["source_frame"] == 1
    assert labels[0][0]["helmet_bbox"] == [5, 6, 9, 10]
    assert labels[1][0]["is_sideline_player"] is True


@pytest.mark.parametrize("changes", [{"frame": "0"}, {"frame": "3"}, {"width": "0"},
                                      {"top": "nan"}, {"isSidelinePlayer": "unknown"}])
def test_invalid_native_labels_refused(tmp_path, changes):
    row = dict(helmet(), **changes)
    path = tmp_path / "helmets.csv"
    write_csv(path, [row])
    with pytest.raises(ValueError):
        read_helmet_labels(path, {row["video"]: {"frame_count": 2}})


def test_containment_cannot_credit_one_box_for_multiple_players():
    labels = [dict(player_id=p, helmet_bbox=[x, 2, x + 2, 4], is_sideline_player=False)
              for p, x in [("H1", 2), ("H2", 6)]]
    labels.append(dict(player_id="H3", helmet_bbox=[2, 2, 4, 4], is_sideline_player=True))
    result = helmet_coverage_diagnostic(labels, [[0, 0, 10, 20]])
    assert result["matched_helmet_centers"] == 1
    assert result["labeled_on_field_helmets"] == 2
    assert len(result["unmatched_helmet_ids"]) == 1
    assert "not player recall" in result["metric_scope"]
    assert helmet_coverage_diagnostic(labels, [])["matched_helmet_centers"] == 0


def test_sensor_ids_ignore_zero_padding_without_video_alignment(tmp_path):
    path = tmp_path / "tracking.csv"
    write_csv(path, [dict(gameKey="1", playID="000002", player="H1",
                         time="2019-01-01T00:00:00Z", x="20", y="30", event="ball_snap")])
    row = read_player_tracking(path, {(1, 2)})[(1, 2)][0]
    assert row["player_id"] == "H1"
    assert row["event"] == "ball_snap"
    assert row["x"] == 20
    assert "frame_id" not in row
