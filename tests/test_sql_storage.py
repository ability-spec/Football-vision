"""SQLite ingestion must preserve evidence, identity scope and atomicity."""

from copy import deepcopy
import sqlite3

import pytest

from football_vision.storage import import_analysis, open_database, report


def document():
    player = dict(track_id=1, bbox=[10, 20, 30, 80], observed=True,
                  detection_confidence=0.9, field_position=[5, 10],
                  x_coord_mode="relative", coordinate_frame_id="camera-0", coordinate_segment=0)
    coast = dict(player, observed=False, field_position=None, detection_confidence=0.4)
    return dict(schema_version=1, video_sha256="a"*64, fps=30, source_kind="synthetic",
                frames=[dict(frame_id=0, players=[player]), dict(frame_id=1, players=[coast]),
                        dict(frame_id=2, players=[])],
                plays=[dict(play_id="one", segment=dict(start_frame=0, snap_frame=1, end_frame=2))])


def test_roundtrip_and_reports(tmp_path):
    path = tmp_path / "analysis.sqlite"
    doc = document()
    run, inserted = import_analysis(path, doc)
    assert inserted
    assert import_analysis(path, deepcopy(doc)) == (run, False)
    db = open_database(path)
    try:
        rows = db.execute("SELECT * FROM observations ORDER BY frame_id").fetchall()
        assert len(rows) == 2
        assert rows[0]["field_x"] == 5
        assert rows[1]["field_x"] is None
        assert db.execute("SELECT COUNT(*) FROM plays").fetchone()[0] == 1
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        db.close()
    assert report(path, run, "tracks")[0]["observed_frames"] == 1
    assert report(path, run, "tracks")[0]["coasted_frames"] == 1
    assert report(path, run, "coverage")[0]["field_coverage_of_observed_records"] == 1
    assert report(path, run, "runs")[0]["frames_processed"] == 3


@pytest.mark.parametrize("mutation", ["duplicate_frame", "duplicate_player", "coasted_field", "bad_box", "play_boundary"])
def test_invalid_import_rolls_back_every_row(tmp_path, mutation):
    path = tmp_path / "analysis.sqlite"
    valid_run, _ = import_analysis(path, document())
    bad = document()
    bad["package_version"] = "different-run"
    if mutation == "duplicate_frame":
        bad["frames"].append(deepcopy(bad["frames"][0]))
    elif mutation == "duplicate_player":
        bad["frames"][0]["players"].append(deepcopy(bad["frames"][0]["players"][0]))
    elif mutation == "coasted_field":
        bad["frames"][1]["players"][0]["field_position"] = [0, 0]
    elif mutation == "bad_box":
        bad["frames"][1]["players"][0]["bbox"] = [30, 20, 10, 80]
    else:
        bad["plays"][0]["segment"]["end_frame"] = 100
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        import_analysis(path, bad)
    assert len(report(path, valid_run, "runs")) == 1
    assert len(report(path, valid_run, "tracks")) == 1


def test_two_runs_keep_same_track_id_separate(tmp_path):
    path = tmp_path / "analysis.sqlite"
    first, _ = import_analysis(path, document())
    second_doc = document()
    second_doc["frames"][0]["players"][0]["track_id"] = 99
    second, _ = import_analysis(path, second_doc)
    assert first != second
    assert len(report(path, first, "runs")) == 2
    assert [r["track_id"] for r in report(path, first, "tracks")] == [1]
    assert [r["track_id"] for r in report(path, second, "tracks")] == [1, 99]


def test_unrelated_database_is_refused(tmp_path):
    path = tmp_path / "other.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(ValueError, match="unrelated"):
        open_database(path)


def test_missing_observed_flag_is_not_guessed(tmp_path):
    doc = document()
    del doc["frames"][0]["players"][0]["observed"]
    with pytest.raises(ValueError, match="observed"):
        import_analysis(tmp_path / "analysis.sqlite", doc)
