# Phase 3 Player Detection, Team Assignment, Tracking & Gated Projection Report

## 1. Executive Summary & Architectural Boundaries

Phase 3 builds a modular perception, torso appearance clustering, multi-object tracking, and gated field-projection layer on top of the untouched **Phase 2 calibration API** (`CalibrationResult` & `CalibrationTracker`):

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
| **Torso Appearance Cluster Assignment** | `football_vision/identity/team_classifier.py` | `TorsoTeamClassifier`, `TeamAssignment` | Turf-masked CIE Lab + HSV torso crop (`16%..54%` height, `20%..80%` width) with deterministic 2-medians clustering (`TEAM_A` = darker/lower-$L$ centroid, `TEAM_B` = lighter/higher-$L$ centroid) and `UNKNOWN` abstention on striped officials, low non-turf fill, or ambiguous margin. Zero hardcoded game colors. |
| **Multi-Object Tracking** | `football_vision/tracking/tracker.py` | `PlayerTracker`, `PlayerTrack` | Deterministic Hungarian assignment over `0.55 * IoU + 0.45 * velocity_predicted_distance_score`. Survives short occlusions up to `max_missed_frames = 3` in image space while setting `footpoint_estimate.is_reliable = False` (`track_occluded_unobserved`) during coasting. |
| **Field Projection** | `football_vision/projection/projector.py` | `FieldProjector`, `ProjectedFieldPosition` | Projects `(u_px, v_px)` via `CalibrationResult.image_to_field` only when `cal.can_project()`, `footpoint_estimate.is_reliable`, and `missed_frames == 0`. Preserves `x_coord_mode` (`absolute_yardline = None` whenever `x_coord_mode != "absolute"`). |

---

## 3. Benchmark Integrity, Detector Provenance & Metric Semantics Audit

### 3.1 Detector Provenance (`TRAIN / VAL / TEST`)

Every Phase 3 evaluation block is explicitly tagged with the exact detector implementation that generated its boxes:

1. **6 Controlled Multi-Frame Sequences (`trk_seq_01`..`trk_seq_06`, `TRAIN`: 10 frames, `VAL`: 20 frames, `TEST`: 30 frames)**:
   - **Detector Implementation Used**: `FixturePlayerDetector` (`detector_name = "fixture_detector_v1"`, `source_type = "benchmark_fixture"`).
   - **Provenance & Restriction**: `FixturePlayerDetector` is a deterministic bounding-box fixture harness used strictly to isolate and test downstream `extract_footpoint`, `TorsoTeamClassifier`, `PlayerTracker` association/occlusion handling, and `FieldProjector` gating under controlled 2-frame, 3-frame, and 5-frame dropouts.
   - **Important Distinction**: The `TP / FP / FN` box counts in the 6 controlled sequences (`1.0000 / 1.0000 / 1.0000` on `train`, `1.0000 / 0.9667 / 0.9831` on `val`, `0.9945 / 0.9526 / 0.9731` on `test`) represent **fixture input pass-through / scheduled dropout rates** injected into the tracking harness (`4 FN` scheduled dropouts in `val`; `9 FN` scheduled dropouts and `1 FP` scheduled clutter box in `test`). They are **NOT** image-based player detector benchmark metrics.
2. **4 Real NFL Broadcast / All-22 Single Frames (`rf_01_sea_sf_val`, `rf_02_nyj_jax_val`, `rf_03_no_car_test`, `rf_04_scrum_uncal_test`)**:
   - **Detector Implementation Used**: `TurfContrastPlayerDetector` (`detector_name = "turf_contrast_baseline_v1"`, `source_type = "image_space_baseline"`).
   - **Provenance**: Because the frozen Phase 3 manifest does not include ground-truth player bounding boxes on `rf_01`..`rf_04`, **quantitative image-space player detection precision/recall/F1 for `TurfContrastPlayerDetector` is unmeasured (`None`) in Phase 3**.

### 3.2 Permutation-Invariant Team/Color Cluster Assignment Semantics

- `TorsoTeamClassifier` produces relative unsupervised appearance cluster labels:
  - **`TEAM_A`**: Darker / lower-luminance ($L$) torso cluster centroid in CIE Lab + HSV space.
  - **`TEAM_B`**: Lighter / higher-luminance ($L$) torso cluster centroid in CIE Lab + HSV space.
  - **`UNKNOWN`**: Ambiguous cluster margin, insufficient non-turf torso pixels, or striped official pattern.
- **Metric Semantics**: The reported `100.0%` (`1.0000`) accuracy on `TRAIN / VAL / TEST` is **permutation-invariant binary torso color/luminance cluster assignment accuracy** (`permutation_invariant_cluster_accuracy`) on the painted dark-navy vs. white torso patches in the controlled sequences. **`TEAM_A` and `TEAM_B` do NOT imply home vs. away, offense vs. defense, or any specific real-world NFL franchise identity.**

### 3.3 Formal Tracking Metric Definitions (`IDSW` vs. `FRAG`) & Classification of `TEST IDSW = 1`

In `benchmarks/evaluate_phase3_tracking.py` (lines 387–412), ground-truth objects `gid` at frame `t` are matched against observed tracks (`missed_frames == 0`) requiring `IoU >= 0.50`:

- **Formal Definition of `IDSW` (`id_switches`)**:
  - Under the evaluator's **CLEAR-MOTA lifetime identity convention**, `last_assigned_trk_id[gid]` stores the `track_id` most recently matched to ground-truth object `gid` at any earlier frame `t' < t`. When `gid` is matched at frame `t` to `best_trk`, if `last_assigned_trk_id[gid] != best_trk.track_id`, `IDSW` increments by `+1`.
- **Formal Definition of `FRAG` (`track_fragmentations`)**:
  - `was_matched_prev_frame[gid]` records whether `gid` was matched to an observed track at frame `t - 1`. When `gid` transitions from unmatched (`was_matched_prev_frame[gid] == False`) back to matched at frame `t`, `FRAG` increments by `+1`.
- **Audit of the `TEST IDSW = 1` Event (`trk_seq_06_test_prolonged_occlusion_and_false_alarm`)**:
  - In `trk_seq_06`, ground-truth player `gid = 3` is matched to `track_id = 3` at `t = 0, 1`, then drops out for **5 consecutive frames (`t = 2, 3, 4, 5, 6`)**, exceeding `max_missed_frames = 3`.
  - `PlayerTracker` coasts `track_id = 3` through `t = 2, 3, 4` (`missed_frames = 1, 2, 3`) and **intentionally terminates/expires `track_id = 3` at `t = 5`** (`missed_frames = 4 > 3`).
  - When `gid = 3` re-appears at `t = 7`, `PlayerTracker` initializes a new track (`track_id = 8`).
  - Under the evaluator's formal definition, this single event at `t = 7` is counted as **both**:
    1. **Track fragmentation (`FRAG += 1`)** because `was_matched_prev_frame[3]` was `False` at `t = 6` and became `True` at `t = 7`, **and**
    2. **Lifetime ID switch (`IDSW += 1`)** because `last_assigned_trk_id[3]` (`3`) $\neq$ `best_trk.track_id` (`8`).
  - Decomposing `IDSW` into active association swaps vs. post-expiration reinitializations across all splits (`TRAIN / VAL / TEST`):
    - **Active-track association swap ID switches (`active_association_swap_idsw`)**: **`0`** across `TRAIN`, `VAL`, and `TEST`.
    - **Post-expiration termination & reinitialization ID changes (`post_expiration_reinit_idsw`)**: **`0`** on `TRAIN`, **`0`** on `VAL`, **`1`** on `TEST`.

### 3.4 Stale-Artifact Regeneration Note (Found During This Audit)

- The audit compared the committed `outputs/phase3_tracking_benchmark.json` against a fresh run and found that the **real-frame cluster counts only** (`rf_01`, `rf_02`) were **stale**: they had been written before the `TorsoTeamClassifier` K-medians refinement (`np.median` centroids + `ambiguous_cluster_margin` priority), so they under-reported `UNKNOWN` abstentions.
- **Scope of impact**: `TRAIN / VAL / TEST` split and sequence metrics were **unaffected** (those are computed from the `FixturePlayerDetector` harness and are byte-identical). Only the 4 real-frame cluster partitions changed.
- **Resolution**: `outputs/phase3_tracking_benchmark.json` was regenerated from the current code and re-verified **deterministic across repeated runs** (all leaves identical except wall-clock runtimes). Corrected real-frame clusters: `rf_01` `15 / 5 / 5`, `rf_02` `2 / 10 / 5` (previously `18 / 5 / 2` and `3 / 10 / 4`).
- All accuracy metrics in §4 below are unchanged by this regeneration; the only values that move between runs are wall-clock runtimes.

---

## 4. Frozen `TRAIN / VAL / TEST` Benchmark Results

All parameters were frozen prior to `TEST` evaluation (`0` threshold tuning on `TEST`).

### 4.1 Split-Level Controlled Sequence Metrics (`6` Sequences, `60` Frames — `FixturePlayerDetector` Harness)

| Metric | `train` (`1` seq, `10` frames) | `val` (`2` seqs, `20` frames) | Frozen `test` (`3` seqs, `30` frames) | Provenance & Metric Semantics |
| :--- | :---: | :---: | :---: | :--- |
| **Detector Implementation Used** | `FixturePlayerDetector` | `FixturePlayerDetector` | `FixturePlayerDetector` | Test/fixture box harness (`fixture_detector_v1`) |
| **Image Detector (`TurfContrastPlayerDetector`) Precision / Recall / F1** | *Unmeasured* | *Unmeasured* | *Unmeasured* | No ground-truth player boxes on real NFL frames |
| **Fixture Input Pass-Through Precision (`IoU >= 0.50`)** | `1.0000` (`60 TP, 0 FP`) | `1.0000` (`116 TP, 0 FP`) | `0.9945` (`181 TP, 1 FP`) | Reflects scheduled fixture clutter (`1 FP` in `test`), **not** image detector precision |
| **Fixture Input Pass-Through Recall (`IoU >= 0.50`)** | `1.0000` (`0 FN`) | `0.9667` (`4 FN` dropout) | `0.9526` (`9 FN` dropout) | Reflects scheduled fixture occlusion dropouts, **not** image detector recall |
| **Fixture Input Pass-Through F1 (`IoU >= 0.50`)** | `1.0000` | `0.9831` | `0.9731` | Fixture harness pass-through F1 |
| **Lifetime Tracking ID Switches (`IDSW`, CLEAR-MOTA)** | **`0`** | **`0`** | **`1`** | Counts any lifetime `track_id` change on a `gt_id` |
| **— Active Association-Swap ID Switches (`active_association_swap_idsw`)** | **`0`** | **`0`** | **`0`** | Swaps between active non-expired tracks |
| **— Post-Expiration Reinit ID Changes (`post_expiration_reinit_idsw`)** | **`0`** | **`0`** | **`1`** | Due to 5-frame dropout `> max_missed_frames=3` |
| **Track Fragmentations (`FRAG`)** | `0` | `2` | `3` | Re-associations after any `>= 1` frame detection gap |
| **Short-Occlusion Recovery Rate (`1..3` frames)** | — (`0/0`) | `100.0%` (`2/2`) | `100.0%` (`1/1`) | Small-sample engineering check (`2/2` val, `1/1` test) |
| **Permutation-Invariant Cluster Assignment Accuracy (`TEAM_A` / `TEAM_B`)** | **`100.0%`** | **`100.0%`** | **`100.0%`** | Relative dark/light torso cluster accuracy (not real team ID) |
| **`UNKNOWN` Abstention Accuracy (Striped Official)** | — | — | **`100.0%` (`10/10`)** | 1 striped official across 10 frames in `trk_seq_04` |
| **Median Footpoint Error — Reliable (`px`)** | `0.800 px` (`0.816` RMSE) | `0.800 px` (`0.813` RMSE) | **`0.800 px`** (`0.815` RMSE) | Sub-pixel box jitter on reliable fixture boxes |
| **Median Footpoint Error — Unreliable/Truncated (`px`)** | — | `13.007 px` *(100% refused)* | — | Bottom-truncated & crossing boxes (all refused) |
| **Median Projected Field-Position Error (`yd`)** | `0.0817 yd` (`0.0698` RMSE) | `0.0777 yd` (`0.0668` RMSE) | **`0.0403 yd`** (`0.0598` RMSE) | Propagation of `0.8 px` footpoint jitter through homography `H` |
| **Unreliable / Coasting Footpoints Refused** | `0` | `18` | `10` | Quantitative refusal-gate check |
| **Mean Runtime — Detection** (`ms/frame`) | `0.14–0.31 ms` | `0.15–0.27 ms` | `0.16–0.24 ms` | Fixture pass-through harness; wall-clock, varies per run |
| **Mean Runtime — Cluster Assignment** (`ms/frame`) | `8.5–14.6 ms` | `1.0–2.3 ms` | `1.0–1.7 ms` | `train` mean inflated by CPU warm-up on the first sequence |
| **Mean Runtime — Tracking + Projection** (`ms/frame`) | `0.20–0.38 ms` | `0.19–0.44 ms` | `0.20–0.31 ms` | Hungarian association + gated projection |
| **Mean Runtime — Total** (`ms/frame`) | `8.8–15.3 ms` | `1.3–3.1 ms` | `1.4–2.2 ms` | CPU-only sandbox; observed range across `5` consecutive runs |

> **Runtime reporting note**: runtimes are wall-clock measurements taken on the shared CPU sandbox and fluctuate roughly 2× between runs; the ranges above are the observed span across `5` consecutive benchmark executions. `outputs/phase3_tracking_benchmark.json` stores the values from the most recent execution. All **accuracy** metrics (precision/recall/F1, IDSW, FRAG, cluster accuracy, footpoint error, field-position error, fabricated projections) are exactly reproducible and do **not** vary between runs.
| **Fabricated Field Projections** | **`0`** | **`0`** | **`0`** | Quantitative safety-gate invariant (`0` violations) |

### 4.2 Sequence-Level Breakdown (`FixturePlayerDetector` Harness)

| Sequence ID | Split | Scenario | Fixture Pass-Through P / R / F1 | Lifetime `IDSW` (Swap / Reinit) | `FRAG` | Short-Occ Recovered | Permutation-Invariant Cluster Acc | `UNKNOWN` Acc | Reliable FP Err (`px`) | Field Pos Err (`yd`) | Unreliable Refused | Fabricated |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `trk_seq_01_train_linear_routes` | `train` | Clean linear motion | `1.000 / 1.000 / 1.000` | `0 (0 / 0)` | `0` | `0/0` | `1.0000` | — | `0.800 px` | `0.0817 yd` | `0` | **`0`** |
| `trk_seq_02_val_short_occlusion_2f` | `val` | 2-frame dropout (`t=4..5`) + recovery (`t=6`) | `1.000 / 0.933 / 0.966` | `0 (0 / 0)` | `2` | `2/2` | `1.0000` | — | `0.800 px` | `0.0830 yd` | `4` | **`0`** |
| `trk_seq_03_val_crossing_and_truncation` | `val` | Bottom border truncation + player crossing (`t=4..5`) | `1.000 / 1.000 / 1.000` | `0 (0 / 0)` | `0` | `0/0` | `1.0000` | — | `0.800 px` | `0.0769 yd` | `14` | **`0`** |
| `trk_seq_04_test_short_occlusion_3f_and_referee` | `test` | 3-frame dropout (`t=3..5`) + striped referee | `1.000 / 0.957 / 0.978` | `0 (0 / 0)` | `1` | `1/1` | `1.0000` | `1.0000` | `0.800 px` | `0.0815 yd` | `3` | **`0`** |
| `trk_seq_05_test_calibration_dropout_gating` | `test` | Calibration invalid at `t=4..6` (image tracks persist) | `1.000 / 1.000 / 1.000` | `0 (0 / 0)` | `0` | `0/0` | `1.0000` | — | `0.800 px` | `0.0378 yd` | `0` *(18 cal-refused)* | **`0`** |
| `trk_seq_06_test_prolonged_occlusion_and_false_alarm` | `test` | 5-frame dropout (`> max_missed=3`) + 1 false alarm | `0.982 / 0.900 / 0.939` | `1 (0 / 1)` | `2` | `0/0` | `1.0000` | — | `0.800 px` | `0.0746 yd` | `7` | **`0`** |

---

## 5. Real NFL Frame Evaluation Scope (`rf_01`..`rf_04` — Integration / Smoke & Safety-Gate Audit Only)

For the 4 real NFL broadcast and All-22 frames (`rf_01_sea_sf_val`, `rf_02_nyj_jax_val`, `rf_03_no_car_test`, `rf_04_scrum_uncal_test`), **`TurfContrastPlayerDetector` (`turf_contrast_baseline_v1`)** was executed in an **end-to-end integration / smoke test and safety-gate audit**. Because these 4 frames do not have ground-truth player bounding boxes, team labels, or player yard coordinates, **they do not provide quantitative detection, tracking, team-classification, or player-position accuracy metrics**:

| Stage | What Was Evaluated on `rf_01`..`rf_04` | Evaluation Type | Quantitative Ground-Truth Accuracy Measured? |
| :--- | :--- | :--- | :---: |
| **1. Player Detection (`TurfContrastPlayerDetector`)** | Candidate box count (`25, 17, 23, 4`) & CPU runtime (`13.8..31.0 ms`) | **Smoke test only** | **No** (Unmeasured — no GT player boxes) |
| **2. Footpoint Extraction (`extract_footpoints_for_frame`)** | Reliable vs. unreliable flag counts (`23/25, 16/17, 22/23, 3/4` reliable) | **Smoke test / gate audit** | **No** (Unmeasured — no GT cleat coordinates) |
| **3. Torso Appearance Clustering (`TorsoTeamClassifier`)** | Unsupervised cluster partition counts (`TEAM_A / TEAM_B / UNKNOWN`) | **Smoke test only** | **No** (Unmeasured — no GT player team labels) |
| **4. Multi-Object Tracking (`PlayerTracker`)** | Single-frame track initialization (`frame_id = 0`, `age = 1`, `hits = 1`) | **Smoke test only** | **No** (Single still frames — no temporal association) |
| **5. Field Projection (`FieldProjector`)** | Refusal gating (`0` projected when calibration fails on `rf_04`; `0` projected when footpoint is unreliable) | **Quantitative invariant check (`0` fabricated projections)** | **Refusal invariant: Yes (`0` fabricated)**; **Position error (yd): No** |

| Frame ID | Split | Detector Used | Evaluation Type | Calibration Valid | `x_coord_mode` | Detected Boxes | Reliable FPs | Unreliable FPs Refused | Projected Tracks | Cluster Counts (`A / B / UNK`) | Fabricated Projections |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_01_sea_sf_val` | `val` | `TurfContrastPlayerDetector` | Smoke / Gate Audit | `True` (`0.8985`) | `relative_10yd` | `25` | `23` | `2` | `23` | `15 / 5 / 5` | **`0`** |
| `rf_02_nyj_jax_val` | `val` | `TurfContrastPlayerDetector` | Smoke / Gate Audit | `True` (`0.8735`) | `relative_10yd` | `17` | `16` | `1` | `16` | `2 / 10 / 5` | **`0`** |
| `rf_03_no_car_test` | `test` | `TurfContrastPlayerDetector` | Smoke / Gate Audit | `True` (`0.8429`) | `relative_10yd` | `23` | `22` | `1` | `22` | `5 / 14 / 4` | **`0`** |
| `rf_04_scrum_uncal_test` | `test` | `TurfContrastPlayerDetector` | Smoke / Gate Audit | `False` (`0.0000`) | `uncalibrated` | `4` | `3` | `1` | **`0`** *(cal refused)* | `0 / 0 / 4` | **`0`** |

---

## 6. Explicit Benchmark Scope & Algorithmic Limitations

1. **Benchmark Sample Size & Scope Limitation (Engineering Benchmark Only)**:
   - The `60` controlled sequence frames (`6` sequences) + `4` real NFL single frames constitute a **Phase 3 engineering verification benchmark** designed to test module contracts, occlusion state machines, and refusal gates. They are **not** evidence of broad NFL generalization across games, broadcasts, stadiums, or weather conditions.
2. **Small-Sample Short-Occlusion Recovery (`1 / 1` Test Case)**:
   - The reported `100.0%` short-occlusion recovery rate is based on **`2 / 2` events in `VAL`** (`trk_seq_02_val_short_occlusion_2f`) and **`1 / 1` event in `TEST`** (`trk_seq_04_test_short_occlusion_3f_and_referee`). Because `n = 1` on `TEST` (`n = 3` total), this result verifies deterministic state-machine behavior on the benchmark fixture and must **not** be presented as statistically robust occlusion recovery on real broadcast video.
3. **Unmeasured Real-Frame Image Detection Accuracy (`TurfContrastPlayerDetector`)**:
   - `TurfContrastPlayerDetector` is a heuristic morphological baseline (`bg_gray - gray > 24` + vertical closing + NMS). On real NFL frames (`rf_01`..`rf_04`), adjacent players in interior line scrums merge or get suppressed by width bounds (`rf_04_scrum_uncal_test` detects only `4` isolated boxes), and high-contrast broadcast watermarks (`FOX NFL` bug in `rf_02_nyj_jax_val`) can trigger false-positive candidate boxes (which was refused at the projection stage due to squat aspect ratio). Quantitative precision/recall on real annotated NFL player boxes remains unmeasured until a labeled player-detection dataset is evaluated.
4. **Relative Cluster Labels (`TEAM_A` / `TEAM_B`) vs. Real Team Identity**:
   - `TorsoTeamClassifier` assigns `TEAM_A` (lower luminance $L$) and `TEAM_B` (higher luminance $L$) via unsupervised 2-medians clustering. It does not know which cluster is offense, defense, home, away, or a specific NFL team.
5. **Constant-Velocity Linear Motion & Relative Yard Coordinates (`x_coord_mode != "absolute"`)**:
   - `PlayerTracker` uses linear constant-velocity prediction during coasted frames (`max_missed_frames = 3`), and `FieldProjector` outputs `relative_10yd` / `relative_5yd` coordinates (`absolute_yardline = None`) unless `x_coord_mode == "absolute"`.
