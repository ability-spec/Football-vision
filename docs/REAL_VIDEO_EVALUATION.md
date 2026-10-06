# Real-video evaluation protocol

This workflow is a baseline measurement tool. It does not include footage, manual labels,
trained detector weights or a claim of real-world accuracy.

## Run the CPU pipeline

From the repository root, after installing `.[dev]`:

```bash
python -m football_vision.evaluation.video /path/to/clip.mp4 --source-kind real --max-frames 300 --out predictions.json
```

PowerShell users can replace `python` with `.\.venv\Scripts\python.exe`.
The output must not already exist. The frame cap defaults to 300; it is an explicit bounded
run, not a full-game benchmark. Frame IDs are zero-based decoded frame indices. The runner
uses the video's FPS, processes every frame up to the cap, refuses invalid/empty videos and
reports premature decode failures when container frame counts expose them. Unknown container
frame counts cannot reliably distinguish clean EOF from truncated decoding.

The output includes video SHA-256, package-source SHA-256, package/OpenCV/NumPy versions,
detector settings, decoded frame count and wall time. Wall time covers decode and inference,
not hashing or JSON serialization. `processing_fps` is throughput, not first-result latency.
Each frame also reports `camera_cut_detected`. A detected cut retires active image-space
tracks without reusing IDs and clears trajectory prediction state, even if the new shot
calibrates successfully. Tracks across different shots are not unique athlete identities.
Cut detection is heuristic; see [architecture risks](ARCHITECTURE_RISKS.md) for its limits
and the constraints on longer clips, timestamps and future parallel processing.
The runner has **no annotation input**. It uses the untrained turf-contrast detector and
relative calibration; no absolute field origin is inferred from labels.

## Review the saved observations

Render the decoded prediction window with synchronized boxes and a top-down view:

```bash
python -m football_vision.visualization /path/to/clip.mp4 predictions.json --out review.avi
```

The renderer verifies the video hash, refuses to overwrite existing files, and
checks that the MJPEG output decodes to the expected number of frames. Green boxes
are observed tracks; amber boxes are coasted tracks. The field view excludes
coasted/dead-reckoning positions and refuses to combine coordinate epochs.
Relative positions use a fixed local window per coordinate epoch without absolute
yard-line labels, so changing player spread does not rescale the view frame by frame.
Odd-sized frames are padded for MJPEG rather than cropped, and decoded output dimensions
are checked. A missing
field view means coordinates are unavailable, not that the field is empty.
This visualization is a review aid and does not measure detector accuracy.

## Analyze manually bounded plays

Provide play timestamps in decoded, zero-based frame indices. This input contains
play boundaries only, not player-position ground truth or detector annotations:

```json
{
  "game_id": "game-001",
  "plays": [
    {"play_id": "play-001", "start_frame": 0, "snap_frame": 30, "end_frame": 120}
  ]
}
```

```bash
python -m football_vision.evaluation.video /path/to/clip.mp4 --source-kind real --max-frames 300 --play-labels plays.json --out analysis.json
python -m football_vision.visualization /path/to/clip.mp4 analysis.json --out review.avi
```

The analysis artifact adds full trajectories, segmentation/refusals, per-frame
play phases, formation/route geometry, scoped events, and play metrics with units,
definitions and limitations. It records the exact timestamp-input SHA-256. All
timestamps must be integers within the decoded window; manual end frames are
required. Plays must not overlap. Omitting a snap requests collective-motion
estimation, which may refuse when measured trajectory evidence is insufficient.
No snap is invented in that case. No play is inferred without a manual start.

Team assignments and offensive direction are not inferred by this workflow;
team-dependent metrics remain unavailable. Null geometry metrics mean unavailable
evidence, not perfect accuracy or zero distance. In particular, pre-snap motion
count requires at least one
consecutive pair of accepted pre-snap positions: post-snap-only tracks do not
establish a zero pre-snap count, and coordinate changes/gaps never supply an edge.
The untrained CPU detector is
still a baseline, so a successful export does not establish real-video accuracy.
The source-video hash in the output binds the results to the analyzed clip;
users must ensure the supplied timestamps belong to that clip.

## Prepare independent labels

1. Select development and held-out test clips **by game/camera source**, not adjacent frames
   of one clip. Record motion, cuts, zoom, occlusion, lighting and crowd density strata.
2. Annotate all visible players on each selected frame, including confirmed empty frames.
   Use stable manual track IDs throughout each clip. Document the visibility/box policy and
   have another annotator check a sample. Incomplete labels cannot measure precision.
3. Annotate field locations only where independent field landmarks support them. Specify
   axes, units and origin separately. Do not use the pipeline's projected player position
   as a label. Otherwise omit `field_position` or use null.
4. Freeze the labels and metric thresholds before scoring TEST. Keep a dataset inventory with
   clip hashes, source rights, split, annotator, frame selection and annotation-policy version.
   Store recordings/labels externally; do not add third-party footage to this repository.

Minimal annotation format (values below illustrate the schema, not actual observations):

```json
{
  "schema_version": 1,
  "video_sha256": "REPLACE_WITH_THE_ACTUAL_64_CHARACTER_VIDEO_SHA256",
  "source_kind": "real",
  "split": "dev",
  "annotation_method": "independent_manual",
  "frames": [
    {
      "frame_id": 0,
      "exhaustive": true,
      "players": [
        {"track_id": "manual-player-1", "bbox": [100, 80, 125, 140], "field_position": null}
      ]
    }
  ]
}
```

To score field coordinates, both annotation and prediction records must also contain an
identical, independently checked `coordinate_frame_id`, compatible `x_coord_mode`
(`absolute`, `relative_10yd` or `relative_5yd`) and `[x,y]` in yards. The runner's automatic
epoch label alone does **not** establish agreement with a manual field frame. Establish and
record an independent landmark-based alignment in a separate preparation step, with a new
aligned prediction artifact and provenance, or leave field scoring unavailable. Never simply
rename mismatched frames to force a match. No automatic GT alignment is performed here.

## Score the saved output

```bash
python -m benchmarks.evaluate_real_video annotations.json predictions.json --iou-threshold 0.5 --field-threshold-yd 1.0 --out metrics.json
```

The scorer refuses mismatched video hashes/source kinds, duplicate frames or identities,
invalid boxes/coordinates, incomplete frame labels and missing field-frame metadata. It
records input file hashes and thresholds. Source kind and annotation independence are
**declarations**, not automatically verified authenticity guarantees.

- Detection: one-to-one IoU matching maximizes valid match count, then IoU; reports TP, FP,
  FN, precision and recall. Missing prediction frames count as misses. Unannotated prediction
  frames are excluded and counted explicitly. Coasting predictions are not detections.
- Identity: switches are changes of matched prediction ID for a manual ID across annotated
  observations. Fragments count a matched identity returning after an annotated miss. Sparse
  annotation limits temporal resolution. These are not a full MOTChallenge, HOTA or IDF1 score.
- Field position: reports coverage over **all annotated field positions**, conditional median
  and p95 error for compatible matches, incompatible-frame count, and the fraction of all
  annotated positions within the chosen error threshold. Missing detections, refusals and
  coordinate mismatches cannot improve that last denominator. Predictions from dead reckoning
  are not treated as measured field positions. No scorable positions means null error, never
  “zero error.”

Report per clip and per failure stratum before aggregates. Preserve failed runs and abstentions.
Use development clips to choose thresholds, then score TEST once. A single successful clip or
green synthetic contract suite does not establish deployment readiness.
