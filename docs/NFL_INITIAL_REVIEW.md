# Initial real NFL footage review

Four owner-uploaded ZIPs were extracted locally on 2026-10-08. Media, decoded
frames and model predictions remain in ignored `outputs/nfl-upload-review-20261008/`.
The source is Kaggle NFL Health & Safety Helmet Assignment; competition terms
have not been independently reviewed. No third-party footage is added to Git.

| Video | Decoded frames | Duration (seconds) |
| --- | ---: | ---: |
| `57995_000109_Sideline.mp4` | 529 | 8.825 |
| `58102_002798_Sideline.mp4` | 366 | 6.106 |
| `57906_000718_Endzone.mp4` | 434 | 7.241 |
| `58102_002798_Endzone.mp4` | 366 | 6.106 |

All videos decode sequentially through their declared frame count. All are
1280 by 720 pixels at approximately 59.94 FPS. Video SHA-256 hashes and
game/play/view identifiers are recorded in `inventory.json`. Matching frame
counts in the paired views do not independently establish visual synchronization.

Six frames per clip were sampled: 0, 10, quarter, half, three quarters and the
last frame, rounded down where needed. The supported pinned YOLOX-Tiny COCO
person model ran on CPU on all 24 frames. Predictions, confidence values,
original-pixel boxes and inference times are in `predictions.json`; preview
contact sheets are saved separately. This is a sampled inference check, not a
full workflow, calibration or tracking validation.

Visual inspection of the Sideline previews exposes missed players in tightly
packed lines, boxes spanning overlapping players, and detections of people
outside the active play. Frame 10 of `57995_000109_Sideline.mp4` produces only
two person candidates despite many visible players. These are qualitative
observations, not measured recall or precision. Counts alone cannot establish
quality or decide whether lowering the threshold would help overall.

Exact full-resolution source PNGs and pending annotation templates are in the
four `*-manual/` directories. They contain no copied detector predictions and
remain incomplete; the evaluator must refuse them until independent annotation.
All four videos are development material because their predictions were inspected.
Both views of any play must stay in the same split; prefer splitting by game
for evaluation on unseen games. Public competition test clips may duplicate
training plays and must not be assumed to be an independent test set.

Next: independently annotate player boxes, establish an occlusion-box policy,
measure detection on these development frames, and annotate consecutive windows
before scoring identity continuity. Manually confirmed play boundaries are
needed for the complete play-report workflow. Additional unseen-game clips
are needed for held-out evaluation after detector decisions are finalized.

## Native labels supplied afterward

The owner also provided `train_labels.csv` and `train_player_tracking.csv`.
The native-label reader in `football_vision/evaluation/nfl.py` keeps helmet boxes
separate from whole-player annotations, converts one-based label frames to
zero-based decoded frames, excludes sideline labels from coverage diagnostics,
and normalizes numeric game/play keys without depending on filename zero padding.
Duplicate identities, invalid boxes, missing selected data and out-of-range
frames are rejected. Sensor timestamps must be timezone-aware; positions are
not silently transformed into our inferred field coordinates.

All four videos have native labels for every decoded frame. The sensor data
contains 22 distinct player IDs for each of the three plays. Selected rows,
source hashes, events and unmatched identities are summarized in the ignored
`native-label-audit.json`; `native-label-audit.md` is the readable report.

On decoded frame 10 of `57995_000109_Sideline.mp4`, only 2 of 22 labeled
on-field helmet centers have a one-to-one containment match in the model's
person boxes. This corroborates the visually observed failure on the packed
line. The diagnostic prevents one large box from receiving credit for several
players, but does not establish correct localization or identity; it is not
whole-player recall, precision or AP. Original boxes and native helmets are
shown together in `helmet-coverage-preview.jpg`.

Native player tracking includes `ball_snap` timestamps, but video-to-sensor
alignment and camera-to-field mapping remain unvalidated. No coordinate error
or sensor-based tracking accuracy is reported. The general dataset description
gives approximate snap alignment, which must not be treated as an exact
zero-based boundary for these clips without reviewing the actual sequence.

Validation after adding the native-label reader: 236 tests passed, 21 optional
checks skipped; Ruff passed. Eight new tests cover frame indexing, sideline
flags, invalid native rows, one-to-one containment and tracking key normalization.

## Tiled detection and complete workflow runs

`--detector yolox-tiled` now runs the verified person model on the full image and
four overlapping crops, using global NMS and rejecting internal-edge fragments.
The checked-in `benchmarks/compare_nfl_detectors.py` reproduces the comparison
without changing model thresholds or taking labels into the detector.
`verified-mode-comparison/` contains the actual supported-mode outputs.

| Clip | Baseline matches / labeled helmets | Tiled matches / labeled helmets |
| --- | ---: | ---: |
| `57995_000109_Sideline.mp4` | 48 / 130 | 91 / 130 |
| `58102_002798_Sideline.mp4` | 64 / 126 | 101 / 126 |
| `57906_000718_Endzone.mp4` | 63 / 120 | 92 / 120 |
| `58102_002798_Endzone.mp4` | 72 / 92 | 80 / 92 |
| Total, 24 sampled frames | 247 / 468 | 364 / 468 |

These are helmet-center diagnostics only, with the limitations above. Local
mean inference times were approximately 0.102 seconds for baseline and 0.465
seconds for tiled mode. Timing included a concurrent workflow run; it is not an
isolated hardware benchmark. Tiling trades CPU time for better small-person
coverage. No held-out game has been evaluated and no football fine-tuning was done.

Both views of `58102_002798` completed all 366 frames through the public workflow,
including JSON, CSV, PNG, HTML, Markdown, AVI and ZIP. Exported file hashes and ZIP
integrity were verified. The review window was manually bounded to the entire
provided clip; an exact snap was deliberately omitted. This does not establish
an independently annotated semantic play boundary.

The Sideline run produced 184 distinct observed image track IDs and 33 accepted
coordinate-frame IDs despite zero detected cuts. Its snap estimator refused
with `coordinate_frame_change`. The Endzone run produced 136 image track IDs,
no accepted measured coordinate frame and a snap refusal of
`insufficient_measured_samples`. These counts are not unique-athlete counts;
they expose fragmentation and calibration limitations. Full workflow throughput
was approximately 2.49 and 2.86 processing FPS respectively, below source FPS.

The final public-command smoke run used a bounded 60-frame Sideline window and
the default two-thread CLI budget. Its reports include observation diagnostics
and boundary refusals. `final-public-workflow/` contains the final report format;
the two complete-clip runs remain available separately as inference evidence.

Remaining acceptance work is independent whole-player localization/false-positive
evaluation, stable identity evaluation on consecutive frames, reliable camera
calibration, precise play/snap labels and final held-out evaluation. The supplied
native labels do not eliminate these requirements. Software integration and
improved sampled detection must not be presented as completed real-game quality.

Final software validation: 243 tests passed, 21 optional asset checks skipped;
Ruff and `git diff --check` passed. Controlled tests cover crop translation,
full-image footpoints, global duplicate suppression, crop-edge refusals,
small-image fallback, report diagnostics, native label indexing and safe cleanup.
An independent annotation ZIP containing original source PNGs and pending JSON
templates is prepared locally; it contains no detector predictions.
