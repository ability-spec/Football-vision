# SQLite analysis storage

No database server or additional Python dependency is required. Uses stdlib
sqlite3. Storage schema v1 is application-tagged; unrelated databases and future
schema versions are refused rather than silently modified.

## Import and query

```sh
python -m football_vision.storage import outputs/run/analysis.json --db analysis.sqlite
python -m football_vision.storage report tracks --db analysis.sqlite --run-id RUN_ID
python -m football_vision.storage report coverage --db analysis.sqlite --run-id RUN_ID
python -m football_vision.storage report runs --db analysis.sqlite --run-id RUN_ID
```

The import command prints the run ID and whether new records were inserted.
Canonical JSON SHA-256 identifies a run: formatting or key-order changes do not
duplicate it. Changes to any content, including execution timing, create a new
run. This is exact-content deduplication, not semantic deduplication.

Direct video analysis can also persist its completed result:

```sh
python -m football_vision.evaluation.video clip.mp4 --source-kind real \
  --max-frames 300 --out analysis.json --db analysis.sqlite
```

JSON is written before database import. If database import fails, the JSON
remains available and can be imported again without repeating inference.

## Tables and evidence policy

- videos: source video SHA-256.
- analysis_runs: per-run fps, source kind, code/package provenance and complete
  original analysis JSON (including model settings, metrics and limitations).
- frames: all supplied decoded frames, including those with no players.
- tracks: composite identity (run_id, track_id), never a global athlete ID.
- observations: boxes, measured/coasted flag, nullable confidence and field
  coordinates, coordinate origin and segment. Predicted boxes remain present
  as unobserved records; coasted field measurements are rejected.
- plays: frame bounds and complete original play JSON, including metrics.

Missing confidence in older exports remains NULL, not zero. A field measurement
requires a real observation, relative/absolute coordinate mode and an explicit
origin. Coordinates with different origins or segments must not be combined.
No speed or distance query is offered without compatible measured geometry.

Ingestion is one transaction per run: duplicate observations, invalid bounds
or any other row failure rolls back that entire run. Existing runs are retained.
Foreign keys, unique composite keys, box checks and a frame index are enabled.

## Report meaning

tracks reports observed and coasted records, lifespan gaps (including absent
records), and mean confidence over observed records only. coverage reports
field coverage among supplied observed records, not recall among real players.
runs lists analyses of the same source video. Different processing windows and
settings are not directly comparable accuracy measurements. Track count alone
cannot establish fewer identity switches; independent annotations are required.

## pandas example

```python
import sqlite3
import pandas as pd

with sqlite3.connect("analysis.sqlite") as connection:
    observations = pd.read_sql_query(
        "SELECT frame_id, track_id, detection_confidence FROM observations "
        "WHERE run_id = ? AND observed = 1 ORDER BY frame_id, track_id",
        connection, params=(run_id,),
    )
```

PostgreSQL, migrations beyond v1, automatic role classification and an accuracy
dashboard are later work. This implementation targets the recovered GitHub
checkout; reconcile the earlier unpushed source archive separately.
