# Code review — 2026-10-13

Reviewed base `624c7d5` with a clean initial worktree.
Baseline: 297 tests passed, 21 skipped; Ruff passed. A passing suite did not cover
several producer/consumer mismatches and malformed stream inputs.

## Confirmed defects and fixes

| Severity | Defect and consequence | Resolution and verification |
| --- | --- | --- |
| High | SQLite expected `relative`, while production calibration and trajectories emit `relative_5yd` / `relative_10yd`. Importing measured field positions failed. | Accept canonical modes in Python and SQL; migrate storage v1 to v2 transactionally, preserving historical runs. Test both relative modes, absolute mode, legacy migration, foreign keys, index recreation and decoded-video export → import → coverage report. |
| High | Formation, defender separation and field rendering treated per-track segment counters as a shared coordinate identity. New tracks start at zero even when existing tracks have already changed segments, hiding valid simultaneous geometry. | Compare different tracks by shared coordinate-frame ID and mode. Keep segment boundaries for temporal motion within a track. Formation exposes per-track counters in provenance and a null aggregate counter when they differ. Regression reproduces the mismatch using the actual trajectory builder; different origins and temporal boundaries remain refused. |
| Medium | Calibration optical flow reused cached images across resolution changes. Similar color histograms could bypass cut detection, causing an OpenCV error or stale pixel geometry. | Resolution changes emit a cut and invalidate prior geometry before optical flow. Test both successful and failed new calibration; assert no flow call across dimensions and a new coordinate epoch on success. |
| Medium | NaN footpoint confidence bypassed the low-confidence comparison and could produce a measured field position. Infinite or out-of-range calibration confidence and unknown coordinate modes could also pass geometry gating. | Require finite confidence in [0,1] and a recognized calibrated mode. Test refusal of NaN, infinity, negative and above-one confidence, and unknown modes. The matrix-only temporal update also refuses invalid observation confidence. |
| Medium | A short team-assignment list crashed after advancing tracker state; retrying the same frame then failed. | Validate assignment length before any stream mutation. Regression retries successfully and checks the first track identity. |
| Medium | Duplicate track IDs in a trajectory update emitted multiple samples and updated one filter twice at the same timestamp. | Reject duplicate IDs before advancing state; regression retries and confirms exactly one retained sample. |
| Medium | Tracker and trajectory builder accepted negative, boolean or fractional frame IDs and coerced them to integers, breaking ordering and timestamps. | Reject malformed frame IDs before mutation. Test failed input followed by a valid frame-zero update in both components. |
| Medium | PlaySegment accepted NaN/infinite FPS; video-runner frame limits were not validated as integers, and invalid frame-count metadata was converted before validation. | Reject non-finite/nonpositive play FPS, non-integer frame limits and non-finite/negative advertised counts. Regressions cover play FPS and refusal before video input loading. |

## Review scope

Read the schema, calibration tracker and plausibility boundary, detection and
footpoint construction, YOLOX preprocessing/postprocessing and tiled merging,
image-space tracking, trajectory filter/gates/kinematics and lifecycle, formation,
routes and events, video and batch runners, annotation loading/preparation and
CVAT scoring, SQLite ingestion/reports, workflow export, and visualization.
Also reviewed existing remediation and architecture-risk documents so established
refusal and coordinate contracts were preserved.

Tests exercise the important repaired module boundaries, rather than relying on
isolated fabricated storage records alone. Existing camera-cut, occlusion,
coordinate-change, event, workflow and evaluation regressions remain in the full
suite. The new storage format is distinct from analysis JSON schema version 1,
which is unchanged.

## Validation limits

Final validation: **324 passed, 21 skipped**; Ruff, `pip check` and
`git diff --check` passed. The suite gained 27 passing regression cases.

The 21 existing skipped tests need four unavailable real NFL reference images;
they are not passing tests. Synthetic decoded video validates lifecycle and
serialization, not broadcast detection accuracy. Existing documented limitations
still apply: heuristic cut detection/calibration, constant-FPS timestamps,
bounded in-memory runs, unknown athlete/team/possession identity in the CPU
workflow, and unmeasured real-video accuracy. This review does not establish that
all possible defects have been found or that the prototype is production-ready.
