# Tracking reference review — 2026-10-06

Base: ability-spec/Football-vision `3a81110` (main recovered from GitHub).
This checkout does not contain the earlier unpushed YOLO/camera-motion/viewer
work from `Football-Vision-reference-update.zip`. Restore and reconcile that
archive before treating this checkout as the complete product.

## References inspected

- NFL / American football: [andrewneuwirth/football-cv](https://github.com/andrewneuwirth/football-cv),
  commit `38f6f758d3e78af3b700f144214c2d4566459c5d`, `footballcv/track.py`.
  Useful: strong-first association, weak recovery restricted to confirmed tracks,
  explicit interpolated frames and IDs that are never reused.
- Generic MOT: [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack),
  commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`,
  `yolox/tracker/byte_tracker.py`.
  Useful: a second low-confidence association stage and no weak-only births.

Both are algorithm references; no source code or model weights were copied.
Soccer repositories are not evidence of NFL accuracy.

## Changes and evidence

1. Preserve the last measured footpoint and observation frame separately from
   the mutable coasted position. Estimate velocity over the complete measured
   interval. On a synthetic stopped-player recovery, the old calculation
   reported -18.5 px/frame instead of 0. Recovery with skipped update calls
   now also increments the recovery counter.
2. Optional strong-first matching (`high_confidence_threshold=0.4`). Weak
   detections in [0.1, 0.4) may recover tracks with at least three observation
   hits and IoU >= 0.3. They cannot start tracks. Strong detections always
   receive first priority. Existing single-stage behavior remains the default.
3. The video runner lowers its detector filter to 0.1 when enabling this mode,
   and records detector and tracker thresholds in result provenance. A tracker
   cannot recover weak boxes that a detector has already discarded.

The maturity condition counts total observations, not consecutive hits or a
calibrated identity probability. It does not implement ByteTrack's Kalman
filter, appearance ReID, or NFL-specific role filtering.

## Run

```sh
python -m football_vision.evaluation.video clip.mp4 --source-kind real \
  --max-frames 300 --two-stage-tracking --out analysis.json
```

This CLI currently runs the untrained TurfContrastPlayerDetector. Its confidence
scores are heuristic. Two-stage mode is experimental, not an accuracy guarantee.
Coasted positions remain unreliable and cannot acquire field projection.

## Validation and next input

Regression tests cover observation gaps, stopped-player recovery, strong-first
priority, weak recovery, weak-only false births, tentative tracks, overlap gates,
invalid thresholds and video-runner threshold provenance. The original four
observation-history cases produced three failures before the motion fix.

Real-video accuracy has not been measured for these changes. First recover the
latest source ZIP, then compare profiles on the same clip with independent
player boxes and persistent identity labels. Report false positives, recall,
identity switches and fragmentation, rather than judging by the number of IDs.
