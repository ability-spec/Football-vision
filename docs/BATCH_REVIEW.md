# Reproducible batch video review

The batch command reviews several local clips using the same detector and
tracking configuration. It reuses the video runner and synchronized renderer.
Native helmet labels are optional: they are validated separately and scored
after inference, never passed to the detector or tracker. For play analytics, provide
manual play boundaries through the [single-video workflow](../README.md).

## Manifest and command

Save `clips.json` beside the source-video directory:

```json
{
  "schema_version": 1,
  "source_kind": "real",
  "split": "dev",
  "videos": [
    {"path": "videos/57995_000109_Sideline.mp4"},
    {"path": "videos/58102_002798_Endzone.mp4"}
  ]
}
```

Paths are relative to the manifest's directory. Use unique video basenames so
native CSV labels map to one source. `source_kind` is `real` or `synthetic`, and
`split` is `dev` or `test`. A declared test split records intent; it does not
establish independent labels or make previously inspected footage a holdout.

```bash
python -m football_vision.evaluation.batch clips.json \
  --out outputs/batch \
  --max-frames 30 \
  --detector yolox-tiled \
  --weights models/yolox_tiny.onnx \
  --helmets train_labels.csv \
  --opencv-threads 2
```

Omit `--helmets` when native labels are unavailable. The learned modes require
the verified local model described in [the detector guide](LEARNED_DETECTOR.md).
`--max-frames` limits inference and review to the decoded prefix of each clip.
Source validation still decodes the complete clip to establish full-source
frame bounds. This can take time even for a short review window; no download or
parameter tuning occurs during the run.

## Outputs and interpretation

Each `clip-0001/`, `clip-0002/`, … directory contains `analysis.json`,
`review.avi` and `preview.png`. The batch root contains `summary.json`,
`report.md`, `report.html`, `manifest.json` and `bundle.zip`. Artifact paths in
the result manifest are relative to the batch root, with SHA-256 hashes for
checking downloaded files. Sources and model settings remain in provenance.

Open each PNG directly for sampled frames. To inspect motion, download the AVI
and open it in a video player. For the combined report, download the ZIP,
extract its files together, and open `report.html` in a browser; the relative
links then reach each clip's preview and review. Use a new output directory:
an existing run is preserved, and failed runs remove only their newly created
partial output.

Native CSV frame numbers are one-based; processed frame IDs are zero-based.
Full-source bounds validate labels from the complete CSV even when inference
uses only the first few frames. Labels after that prefix are not scored.
An unlabeled processed frame has a `null` diagnostic and is excluded from
labeled-frame counts. This is different from a labeled frame with no matching
person candidates, which has zero matches. Never treat an absent annotation
as a confirmed negative frame.

Helmet-center containment uses raw person detections, not coasting track boxes.
Matching is one-to-one and excludes native sideline-player labels. It does not
measure player recall, precision, AP, box localization or identity accuracy.
Track IDs are not unique-athlete counts. Automatic calibration and camera-cut
handling remain heuristic; a projectable frame does not prove accurate field
coordinates. Without manual boundaries, the batch does not claim play
segmentation, snap timing or play metrics.

## Supplied-footage status

There are twelve supplied NFL clips from seven distinct game keys and seven
plays: the first four cover three games/plays, and the additional eight cover
four other games/plays. Count games by the filename's `gameKey` prefix, not by
the number of videos; paired camera views belong to the same play and split.
The additional eight contain 2,906 sequentially decoded source frames.

The earlier ignored local review of those eight clips used a fixed detector
configuration and recorded 40,171 one-to-one helmet-center matches across
53,843 labeled on-field helmet instances. Those counts are separate evidence
from the public batch command, not recall or AP. They repeat players across
frames. The clips have already been inspected and must not be described as an
untouched holdout after results inform tuning. Final quality evaluation needs
previously uninspected games, independent whole-player labels, consecutive
identity labels and independently aligned field geometry.

## Public-command validation (2026-10-10)

The public batch command was run with the supplied eight additional clips,
local verified YOLOX-Tiny weights, `yolox-tiled`, two OpenCV threads, the native
ground-truth `train_labels.csv` and `--max-frames 30`. It fully decoded all
2,906 source frames and inferred/exported the first 30 frames of each clip:
240 processed frames, all with native helmet labels. The new output is locally
available at `outputs/public-batch-review-20261010/`.

There were 4,069 one-to-one helmet-center matches across 4,963 labeled on-field
helmet instances in that prefix. This is separate from the earlier full-clip
review. The first half-second of each clip is not representative whole-play
evaluation; these counts do not establish player precision/recall or identity
accuracy. No thresholds were tuned during this verification.

All eight review AVIs passed decoded frame-count/dimension verification. The
27 artifact hashes were checked both on disk and inside the ZIP, including
each analysis JSON and preview PNG. Every analysis recorded the same pipeline
source hash, which matched the completed local implementation. The full
software suite passed 283 tests with 21 optional-asset checks skipped; Ruff
passed. Regression cases include a full native CSV with bounded inference,
missing frames and wholly unlabeled clips, coasting without containment credit,
one-frame export, downloaded bundle integrity and cleanup after a later clip fails.
