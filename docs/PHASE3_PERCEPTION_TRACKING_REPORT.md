# Phase 3 Player Detection, Team Assignment, Tracking & Gated Projection Report

## 1. Executive Summary & Architectural Boundaries

Phase 3 builds a modular perception, team assignment, multi-object tracking, and gated field-projection layer on top of the untouched **Phase 2 calibration API** (`CalibrationResult` & `CalibrationTracker`):

- **Calibration Layer Preserved (Requirement A)**:
  - Zero lines of Phase 1 / Phase 2 production calibration code (`football_vision/calibration/homography.py`, `football_vision/calibration/tracker.py`) were modified.
  - Player detection (`BasePlayerDetector`) operates strictly in image space with zero dependency on `CalibrationResult`.
  - Field projection (`FieldProjector`) treats `CalibrationResult` as an immutable geometry contract and explicitly refuses projection (`field_position = None`) whenever calibration is invalid/expired (`not cal.can_project()`), whenever a player footpoint is unreliable (`not footpoint_estimate.is_reliable`), or whenever a track is coasting through an occlusion (`missed_frames > 0`).
- **Scope Discipline (Requirement I)**:
  - Phase 3 intentionally does **not** implement jersey number OCR, football detection, play recognition, route reconstruction, or advanced event detection.

---

## 2. Phase 3 Subsystem Architecture

| Subsystem | Module Path | Key Classes / Functions | Core Contract & Guarantees |
| :--- | :--- | :--- | :--- |
| **Player Detection** | `football_vision/detection/detector.py` | `BasePlayerDetector`, `TurfContrastPlayerDetector`, `FixturePlayerDetector` | Image-space-only detection outputting `PlayerDetection(frame_id, detection_id, bbox, confidence, footpoint, footpoint_estimate, detector_metadata)`. Separated from `experiments/cpu_standin_perception.py`. |
| **Footpoint Reliability** | `football_vision/detection/footpoint.py` | `extract_footpoint`, `extract_footpoints_for_frame`, `FootpointEstimate` | Bottom-center `(u_px, v_px)` extraction with explicit flags for `is_partially_truncated`, `is_near_sideline`, `is_player_crossing`, and `is_temporarily_occluded`. Marks `is_reliable = False` with explicit `unreliable_reason` when bottom-clipped or lower-body occluded. |
| **Team Assignment** | `football_vision/identity/team_classifier.py` | `TorsoTeamClassifier`, `TeamAssignment` | Turf-masked CIE Lab + HSV torso crop (`16%..54%` height, `20%..80%` width) with deterministic 2-centroid clustering (`TEAM_A` = darker/lower-$L$ centroid, `TEAM_B` = lighter/higher-$L$ centroid) and `UNKNOWN` abstention on striped officials, low non-turf fill, or ambiguous margin. Zero hardcoded game colors. |
| **Multi-Object Tracking** | `football_vision/tracking/tracker.py` | `PlayerTracker`, `PlayerTrack` | Deterministic Hungarian assignment over `0.55 * IoU + 0.45 * velocity_predicted_distance_score`. Survives short occlusions up to `max_missed_frames = 3` in image space while setting `footpoint_estimate.is_reliable = False` (`track_occluded_unobserved`) during coasting. |
| **Field Projection** | `football_vision/projection/projector.py` | `FieldProjector`, `ProjectedFieldPosition` | Projects `(u_px, v_px)` via `CalibrationResult.image_to_field` only when `cal.can_project()`, `footpoint_estimate.is_reliable`, and `missed_frames == 0`. Preserves `x_coord_mode` (`absolute_yardline = None` whenever `x_coord_mode != "absolute"`). |

---

## 3. Frozen `TRAIN / VAL / TEST` Benchmark Results

All detector, footpoint, team-margin, and tracker association parameters were selected on `TRAIN` / `VAL` and evaluated without modification on the frozen `TEST` split (`data/benchmarks/phase3_tracking_manifest.json` $\rightarrow$ `outputs/phase3_tracking_benchmark.json`).

### 3.1 Split-Level Aggregate Metrics (`6` Sequences, `60` Frames)

| Metric | `train` (`1` seq, `10` frames) | `val` (`2` seqs, `20` frames) | Frozen `test` (`3` seqs, `30` frames) | Evidence Type |
| :--- | :---: | :---: | :---: | :---: |
| **Detection Precision (`IoU >= 0.50`)** | `1.0000` (`60 TP, 0 FP`) | `1.0000` (`116 TP, 0 FP`) | **`0.9945`** (`181 TP, 1 FP`) | Measured (Controlled Sequence Benchmark) |
| **Detection Recall (`IoU >= 0.50`)** | `1.0000` (`0 FN`) | `0.9667` (`4 FN` dropout) | **`0.9526`** (`9 FN` dropout) | Measured (Controlled Sequence Benchmark) |
| **Detection F1 (`IoU >= 0.50`)** | `1.0000` | `0.9831` | **`0.9731`** | Measured (Controlled Sequence Benchmark) |
| **Tracking ID Switches (`IDSW`)** | **`0`** | **`0`** | **`1`** *(5-frame prolonged expiry > `max_missed=3`)* | Measured |
| **Track Fragmentations (`FRAG`)** | `0` | `2` | `3` | Measured |
| **Short-Occlusion Recovery Rate (`1..3` frames)** | — (`0/0`) | **`100.0%` (`2/2`)** | **`100.0%` (`1/1`)** | Measured |
| **Team Assignment Accuracy (`TEAM_A` / `TEAM_B`)** | **`100.0%`** | **`100.0%`** | **`100.0%`** | Measured |
| **`UNKNOWN` Abstention Accuracy (Striped Official)** | — | — | **`100.0%` (`10/10`)** | Measured |
| **Median Footpoint Error — Reliable (`px`)** | `0.800 px` (`0.816` RMSE) | `0.800 px` (`0.813` RMSE) | **`0.800 px`** (`0.815` RMSE) | Measured |
| **Median Footpoint Error — Unreliable/Truncated (`px`)** | — | `13.007 px` *(100% refused)* | — | Measured |
| **Median Projected Field-Position Error (`yd`)** | `0.0817 yd` (`0.0698` RMSE) | `0.0777 yd` (`0.0668` RMSE) | **`0.0403 yd`** (`0.0598` RMSE) | Measured (Relative to Calibrated Homography) |
| **Unreliable / Coasting Footpoints Refused** | `0` | `18` | `10` | Measured |
| **Fabricated Field Projections** | **`0`** | **`0`** | **`0`** | Measured |
| **Mean Runtime per Frame (`det / team / trk / total`)** | `0.22 / 10.07 / 0.26 / 10.54 ms` | `0.18 / 1.17 / 0.24 / 1.59 ms` | **`0.20 / 1.22 / 0.25 / 1.66 ms`** | Measured (CPU) |

### 3.2 Sequence-Level Breakdown

| Sequence ID | Split | Scenario | Det Precision | Det Recall | Det F1 | `IDSW` | `FRAG` | Short-Occ Recovered | Team Acc | `UNKNOWN` Acc | Reliable FP Err (`px`) | Field Pos Err (`yd`) | Unreliable Refused | Fabricated |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `trk_seq_01_train_linear_routes` | `train` | Clean linear motion | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `0/0` | `1.0000` | — | `0.800 px` | `0.0817 yd` | `0` | **`0`** |
| `trk_seq_02_val_short_occlusion_2f` | `val` | 2-frame dropout (`t=4..5`) + recovery (`t=6`) | `1.0000` | `0.9333` | `0.9655` | `0` | `2` | **`2/2`** | `1.0000` | — | `0.800 px` | `0.0830 yd` | `4` | **`0`** |
| `trk_seq_03_val_crossing_and_truncation` | `val` | Bottom border truncation + player crossing (`t=4..5`) | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `0/0` | `1.0000` | — | `0.800 px` | `0.0769 yd` | `14` | **`0`** |
| `trk_seq_04_test_short_occlusion_3f_and_referee` | `test` | 3-frame dropout (`t=3..5`) + striped referee | `1.0000` | `0.9571` | `0.9781` | `0` | `1` | **`1/1`** | `1.0000` | **`1.0000`** | `0.800 px` | `0.0815 yd` | `3` | **`0`** |
| `trk_seq_05_test_calibration_dropout_gating` | `test` | Calibration invalid at `t=4..6` (image tracks persist) | `1.0000` | `1.0000` | `1.0000` | `0` | `0` | `0/0` | `1.0000` | — | `0.800 px` | `0.0378 yd` | `0` *(18 cal-refused)* | **`0`** |
| `trk_seq_06_test_prolonged_occlusion_and_false_alarm` | `test` | 5-frame dropout (`> max_missed=3`) + 1 false alarm | `0.9818` | `0.9000` | `0.9391` | `1` | `2` | `0/0` | `1.0000` | — | `0.800 px` | `0.0746 yd` | `7` | **`0`** |

### 3.3 Real NFL Broadcast Single-Frame Perception & Projection Audit (`TurfContrastPlayerDetector`)

| Frame ID | Split | Calibration Valid | `x_coord_mode` | Detected Boxes | Reliable Footpoints | Unreliable Footpoints Refused | Projected Tracks | Team Counts (`A / B / UNKNOWN`) | Fabricated Projections | Runtime (`det / team / trk ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_01_sea_sf_val` | `val` | `True` (`0.8985`) | `relative_10yd` | `25` | `23` | `2` | `23` | `18 / 5 / 2` | **`0`** | `16.08 / 4.47 / 0.45 ms` |
| `rf_02_nyj_jax_val` | `val` | `True` (`0.8735`) | `relative_10yd` | `17` | `16` | `1` | `16` | `3 / 10 / 4` | **`0`** | `13.81 / 3.11 / 0.38 ms` |
| `rf_03_no_car_test` | `test` | `True` (`0.8429`) | `relative_10yd` | `23` | `22` | `1` | `22` | `5 / 14 / 4` | **`0`** | `30.97 / 4.45 / 0.40 ms` |
| `rf_04_scrum_uncal_test` | `test` | `False` (`0.0000`) | `uncalibrated` | `4` | `3` | `1` | **`0`** *(cal refused)* | `0 / 0 / 4` | **`0`** | `29.61 / 0.51 / 0.12 ms` |

---

## 4. Honest Limitations & Remaining Failure Modes

1. **Dense Line-of-Scrimmage Pileup Merging in `TurfContrastPlayerDetector` (Measured)**:
   - Because `TurfContrastPlayerDetector` uses morphological dark-contrast connected components without a trained deep object detector (e.g., YOLOv8 / RT-DETR), adjacent linemen in tight interior line scrums (`rf_02_nyj_jax_val`, `rf_04_scrum_uncal_test`) can merge into a single bounding box or be suppressed when their bounding width exceeds `max_width_frac = 0.085`. Integrating a trained person/player detector via the modular `BasePlayerDetector` interface is the primary path to resolving interior-line recall.
2. **Broadcast Watermark / On-Field Graphic False Positives (`rf_02_nyj_jax_val` `#17`)**:
   - High-contrast dark-bordered broadcast watermarks inside the upper playfield (`FOX NFL` bug at top-right of `NYJ vs JAX`) can trigger a candidate detection box. In `rf_02_nyj_jax_val`, `extract_footpoint` flagged its squat aspect ratio (`is_reliable = False`, `lower_body_occluded_aspect_ratio`) and refused field projection, but a learned detector is needed to eliminate watermark false alarms at the detection stage.
3. **Constant-Velocity Linear Motion Assumption During Occlusions (`max_missed_frames = 3`)**:
   - `PlayerTracker` predicts occluded player positions linearly using `velocity_uv`. Sharp cuts during a multi-frame occlusion (`> 3` frames) or tight crossing swaps where two same-team players exchange trajectories while both are simultaneously occluded can still cause an ID switch (`IDSW = 1` on `trk_seq_06_test_prolonged_occlusion_and_false_alarm` after a 5-frame dropout).
4. **Relative vs. Absolute Yard Coordinates (`x_coord_mode != "absolute"`)**:
   - Because Phase 2/3 does not include yard-number OCR, projected player coordinates on real NFL frames are reported in `relative_10yd` or `relative_5yd` mode, with `absolute_yardline = None` strictly enforced.
