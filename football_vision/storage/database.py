"""Transactional JSON ingestion. Track identities are scoped to analysis runs."""

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
 video_sha256 TEXT PRIMARY KEY CHECK(length(video_sha256)=64)
);
CREATE TABLE IF NOT EXISTS analysis_runs (
 run_id TEXT PRIMARY KEY,
 video_sha256 TEXT NOT NULL REFERENCES videos(video_sha256),
 fps REAL NOT NULL CHECK(fps>0), source_kind TEXT NOT NULL,
 pipeline_sha256 TEXT, package_version TEXT, analysis_json TEXT NOT NULL,
 imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS frames (
 run_id TEXT NOT NULL REFERENCES analysis_runs(run_id),
 frame_id INTEGER NOT NULL CHECK(frame_id>=0),
 PRIMARY KEY(run_id,frame_id)
);
CREATE TABLE IF NOT EXISTS tracks (
 run_id TEXT NOT NULL REFERENCES analysis_runs(run_id),
 track_id INTEGER NOT NULL CHECK(track_id>0), PRIMARY KEY(run_id,track_id)
);
CREATE TABLE IF NOT EXISTS observations (
 run_id TEXT NOT NULL, track_id INTEGER NOT NULL, frame_id INTEGER NOT NULL,
 observed INTEGER NOT NULL CHECK(observed IN (0,1)),
 x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
 detection_confidence REAL CHECK(detection_confidence BETWEEN 0 AND 1),
 field_x REAL, field_y REAL, x_coord_mode TEXT,
 coordinate_frame_id TEXT, coordinate_segment INTEGER,
 PRIMARY KEY(run_id,track_id,frame_id),
 FOREIGN KEY(run_id,track_id) REFERENCES tracks(run_id,track_id),
 FOREIGN KEY(run_id,frame_id) REFERENCES frames(run_id,frame_id),
 CHECK(x2>x1 AND y2>y1),
 CHECK((field_x IS NULL)=(field_y IS NULL)),
 CHECK(field_x IS NULL OR (observed=1 AND coordinate_frame_id IS NOT NULL
                         AND x_coord_mode IN ('relative','absolute')))
);
CREATE INDEX IF NOT EXISTS observations_by_frame ON observations(run_id,frame_id);
CREATE TABLE IF NOT EXISTS plays (
 run_id TEXT NOT NULL REFERENCES analysis_runs(run_id), play_id TEXT NOT NULL,
 start_frame INTEGER NOT NULL CHECK(start_frame>=0),
 snap_frame INTEGER, end_frame INTEGER,
 play_json TEXT NOT NULL, PRIMARY KEY(run_id,play_id),
 FOREIGN KEY(run_id,start_frame) REFERENCES frames(run_id,frame_id),
 FOREIGN KEY(run_id,snap_frame) REFERENCES frames(run_id,frame_id),
 FOREIGN KEY(run_id,end_frame) REFERENCES frames(run_id,frame_id),
 CHECK(end_frame IS NULL OR end_frame>=start_frame),
 CHECK(snap_frame IS NULL OR (snap_frame>=start_frame AND
       (end_frame IS NULL OR snap_frame<=end_frame)))
);
"""


def open_database(path: Path) -> sqlite3.Connection:
    """Open schema version 1; reject databases belonging to another application."""
    db = sqlite3.connect(path)
    try:
        db.execute("PRAGMA foreign_keys=ON")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        app_id = db.execute("PRAGMA application_id").fetchone()[0]
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if (version, app_id) != (1, 1179006513) and (version != 0 or app_id != 0 or tables):
            raise ValueError("unsupported or unrelated database schema")
        db.executescript("BEGIN;" + SCHEMA +
                         "PRAGMA user_version=1; PRAGMA application_id=1179006513; COMMIT;")
        db.row_factory = sqlite3.Row
        return db
    except BaseException:
        db.close()
        raise


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def import_analysis(path: Path, document: dict) -> tuple[str, bool]:
    """Return (content-addressed run ID, inserted). Any ingestion failure rolls back."""
    if not isinstance(document, dict):
        raise ValueError("analysis must be a JSON object")
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    run_id = hashlib.sha256(canonical.encode()).hexdigest()
    video = document.get("video_sha256", "")
    if not isinstance(video, str) or not re.fullmatch("[0-9a-f]{64}", video):
        raise ValueError("video_sha256 must be a lowercase SHA-256 digest")
    fps = _number(document.get("fps"), "fps")
    if fps <= 0 or document.get("source_kind") not in ("real", "synthetic"):
        raise ValueError("positive fps and real/synthetic source_kind required")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("analysis schema_version must be 1")
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("nonempty frames list required")
    with closing(open_database(path)) as db, db:
        if db.execute("SELECT 1 FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone():
            return run_id, False
        db.execute("INSERT OR IGNORE INTO videos VALUES (?)", (video,))
        db.execute("""INSERT INTO analysis_runs
                   (run_id,video_sha256,fps,source_kind,pipeline_sha256,package_version,analysis_json)
                   VALUES (?,?,?,?,?,?,?)""",
                   (run_id, video, fps, document["source_kind"], document.get("pipeline_sha256"),
                    document.get("package_version"), canonical))
        for frame in frames:
            fid = _integer(frame["frame_id"], "frame_id")
            db.execute("INSERT INTO frames VALUES (?,?)", (run_id, fid))
            for player in frame["players"]:
                tid = _integer(player["track_id"], "track_id", 1)
                observed = player.get("observed")
                if type(observed) is not bool:
                    raise ValueError("explicit boolean observed is required")
                bbox = player["bbox"]
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError("bbox requires four coordinates")
                box = [_number(v, "bbox") for v in bbox]
                confidence = player.get("detection_confidence")
                if confidence is not None:
                    confidence = _number(confidence, "detection_confidence")
                xy = player.get("field_position")
                mode = player.get("x_coord_mode")
                origin = player.get("coordinate_frame_id")
                if xy is not None:
                    if not observed or mode not in ("relative", "absolute") or not isinstance(origin, str) or not origin:
                        raise ValueError("field position requires observed, calibrated coordinates with origin")
                    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                        raise ValueError("field_position requires two coordinates")
                    xy = [_number(v, "field_position") for v in xy]
                else:
                    xy = (None, None)
                segment = player.get("coordinate_segment")
                if segment is not None:
                    segment = _integer(segment, "coordinate_segment")
                db.execute("INSERT OR IGNORE INTO tracks VALUES (?,?)", (run_id, tid))
                db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (run_id, tid, fid, int(observed), *box, confidence, *xy, mode, origin, segment))
        for play in document.get("plays", []):
            segment = play["segment"]
            play_id = play["play_id"]
            if not isinstance(play_id, str) or not play_id:
                raise ValueError("nonempty play_id required")
            bounds = [segment.get(k) for k in ("start_frame", "snap_frame", "end_frame")]
            bounds[0] = _integer(bounds[0], "start_frame")
            for i in (1, 2):
                if bounds[i] is not None:
                    bounds[i] = _integer(bounds[i], "play boundary")
            db.execute("INSERT INTO plays VALUES (?,?,?,?,?,?)",
                       (run_id, play_id, *bounds, json.dumps(play, allow_nan=False)))
    return run_id, True


REPORTS = {
    "tracks": """SELECT track_id, MIN(frame_id) AS first_frame, MAX(frame_id) AS last_frame,
        SUM(observed) AS observed_frames, COUNT(*)-SUM(observed) AS coasted_frames,
        MAX(frame_id)-MIN(frame_id)+1-SUM(observed) AS unobserved_frames_in_lifespan,
        AVG(CASE WHEN observed=1 THEN detection_confidence END) AS mean_detection_confidence
        FROM observations WHERE run_id=? GROUP BY track_id ORDER BY track_id""",
    "coverage": """SELECT COUNT(*) AS observation_records, COALESCE(SUM(observed),0) AS observed_records,
        COALESCE(SUM(CASE WHEN observed=1 AND field_x IS NOT NULL THEN 1 ELSE 0 END),0) AS field_records,
        1.0*SUM(CASE WHEN observed=1 AND field_x IS NOT NULL THEN 1 ELSE 0 END)
        /NULLIF(SUM(observed),0) AS field_coverage_of_observed_records
        FROM observations WHERE run_id=?""",
    "runs": """SELECT r.run_id, r.video_sha256, r.fps, r.source_kind, r.pipeline_sha256,
        (SELECT COUNT(*) FROM frames f WHERE f.run_id=r.run_id) AS frames_processed,
        (SELECT COUNT(*) FROM tracks t WHERE t.run_id=r.run_id) AS track_count
        FROM analysis_runs r WHERE r.video_sha256=
        (SELECT video_sha256 FROM analysis_runs WHERE run_id=?) ORDER BY r.run_id""",
}


def report(path: Path, run_id: str, name: str) -> list[dict]:
    """Return a fixed parameterized SQL report; these counts do not measure accuracy."""
    if name not in REPORTS:
        raise ValueError("unknown report")
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        if (db.execute("PRAGMA user_version").fetchone()[0],
                db.execute("PRAGMA application_id").fetchone()[0]) != (1, 1179006513):
            raise ValueError("unsupported or unrelated database schema")
        if not db.execute("SELECT 1 FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone():
            raise ValueError("unknown run_id")
        return [dict(row) for row in db.execute(REPORTS[name], (run_id,))]
