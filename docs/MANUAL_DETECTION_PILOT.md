# Manual player-box pilot

On 2026-10-12 the owner supplied a CVAT for images 1.1 ZIP for play
`58102_002798`: Endzone and Sideline, zero-based decoded frames 10 and 183.
The four source images are 1280 x 720 and were verified pixel-for-pixel against
sequential video decoding. There are 77 independently drawn `player` boxes.
The owner confirmed the annotations after a visual review. This is a declared
manual reference, without a second-annotator quality audit.

## Measured detection sample

Both modes use the existing fixed official COCO YOLOX-Tiny weights and settings.
No training or threshold tuning was performed. Match boxes one-to-one at
IoU >= 0.5, maximizing the count of valid matches before total IoU. Duplicate
predictions and detections of officials/bystanders count as false positives
against the player-only reference. Occluded players use visible-body boxes;
COCO person boxes can have different extents, so review matching disagreements.

| Mode | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOX | 42 | 11 | 35 | 79.2% | 54.5% |
| YOLOX tiled | 52 | 21 | 25 | 71.2% | 67.5% |

Tiled inference recovers ten additional annotated instances and adds ten
unmatched predictions. That tradeoff does not establish a universally better
detector. These are pooled counts across four development images, not full-clip,
held-out or generalization measurements. They are not the native helmet-center
containment diagnostic. The sample supports prioritizing missed/merged players
in crowded formations; it does not justify a model-training acceptance claim.

## Reproduce locally

```bash
python -m football_vision.evaluation.cvat manual.zip \
  --images outputs/mvp-annotation-starter-20261010 \
  --weights models/yolox_tiny.onnx \
  --out outputs/manual-pilot-measured \
  --confirm-exhaustive --split dev --opencv-threads 2
```

Use the uploaded archive as `manual.zip` and the exact original images with
unchanged filenames. `--confirm-exhaustive` is an explicit completeness
declaration, not an automatic quality check. The command refuses to score without
it, validates every image/box before inference, never overwrites an existing
output directory, and cleans up only its own incomplete output on failure.
The importer supports manual `player` rectangles in CVAT XML; COCO and tracked
CVAT exports are not supported by this command.

Artifacts: `annotations.json`, `metrics.json`, `report.md`. ZIP and source-image
hashes, detector profile/settings, raw predictions and individual IoU matches
make the result traceable. Annotation indices and CVAT image ids are not player
identities or source-video frame numbers. Identity and field metrics remain
explicitly unavailable. Raw media, uploaded labels and local result artifacts
remain outside normal code commits.

## Next evidence

Measure identity on a continuous video interval with stable independent player
IDs. First use the existing native helmet identities as a clearly labeled
helmet-associated identity diagnostic; it cannot establish body-box tracking
accuracy by itself. Independent body tracks are needed for that claim.
Obtain manual start/snap/end times for the selected play before play analytics.
Field error still needs independently checked spatial and temporal alignment,
with coverage/refusals beside every conditional error number.

Choose acceptance targets after reviewing this baseline. More isolated frames
improve detection sampling but do not replace identity labels, field alignment,
or an untouched game for final evaluation.
