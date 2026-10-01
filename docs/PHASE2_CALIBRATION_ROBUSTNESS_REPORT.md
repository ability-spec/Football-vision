# Phase 2 Calibration Robustness & Temporal Stability Report

## 1. Executive Summary & Scope

This report documents the **Phase 2 Calibration Robustness & Temporal Benchmarking** evaluation of `Football-Vision`. Phase 2 preserves the immutable Phase 0 / Phase 1 Week-1 Hough + Vanishing-Point Pencil calibration baseline (`Frame 1: 0.275 yd`, `Frame 2: 0.350 yd`, `Frame 3: 0.057 yd`) while evaluating single-frame robustness across **12 multi-game, multi-condition frames** (`train`, `val`, and frozen `test` splits) and temporal stability across **8 12-frame stress sequences (`96` frames total)**.

- **Single-Frame Multi-Condition Benchmark (`12` frames)**:
  - **Calibrated frames**: `9 / 12` (`100%` of the `9` physically calibratable frames; `0` false negatives).
  - **Explicitly refused frames**: `3 / 12` (`100%` of the `3` physically uncalibratable frames where `< 3` yard lines or `< 2` hash rows are visible; **`0` fabricated calibrations**).
  - **Frozen `TEST` split held-out landmark error**: **`0.270 yd`** median held-out error (`0.176 yd` on `sf_07_sea_sf_low_light_shadow`, `0.378 yd` on `sf_08_nyj_jax_overexposed_glare`, `0.270 yd` on `sf_10_sea_sf_sideline_cropped`).
- **Temporal Sequence Stress Benchmark (`8` sequences, `96` frames)**:
  - **Total calibrated frames**: **`62 / 96`** (`55` direct calibration locks + `7` optical-flow propagated frames across short dropouts).
  - **Total refused frames**: **`34 / 96`** (`0 + 0 + 1 + 10 + 6 + 7 + 3 + 7 = 34` refused frames across the 8 sequences).
  - **Fabricated calibrations**: **`0 / 96`** (`can_project()` and `image_to_field()` return `False` / `None` on all `34` refused frames).

---

## 2. Dataset & Sequence Manifest (`TRAIN / VAL / TEST` Splits)

All confidence and parity thresholds (`MIN_CALIBRATION_CONFIDENCE = 0.45`, `PARITY_MIN_SCORE = 0.05`, `PARITY_MIN_RATIO = 2.0`) were frozen during Phase 1 (`VAL` split). **Zero thresholds were tuned against the `TEST` split.**

### 2.1 Single-Frame Multi-Condition Dataset (`12` Frames)

| Frame ID | Split | Game / Source | Broadcaster | Camera Angle | Resolution | Transform | Conditions | Expected Calibratable |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `sf_04_broadcast_sideline_wide` | `train` | Broadcast Sample 4 | CBS Style | `broadcast_sideline` | `758x360` | `none` | `low_resolution`, `sideline_absent`, `normal_lighting` | Yes |
| `sf_05_broadcast_cropped_vertical` | `train` | Broadcast Sample 1 | NBC Style | `broadcast_cropped` | `330x410` | `none` | `partial_field_visibility`, `low_resolution`, `sideline_absent` | Yes |
| `sf_06_all22_tight_scrum_uncalibratable` | `train` | All-22 Sample 4 | NFL All-22 | `all22_tight` | `1200x668` | `none` | `heavy_player_occlusion`, `sideline_absent` | **No** |
| `sf_01_sea_sf_fox` | `val` | SEA vs SF | FOX | `broadcast_sideline` | `900x506` | `none` | `sideline_visible`, `virtual_los`, `virtual_first_down`, `normal_lighting` | Yes |
| `sf_02_nyj_jax_fox` | `val` | NYJ vs JAX | FOX | `broadcast_sideline` | `900x507` | `none` | `midfield_logo`, `partial_sideline`, `virtual_first_down`, `normal_lighting` | Yes |
| `sf_03_no_car_all22` | `val` | NO vs CAR | NFL All-22 | `all22_high_sideline` | `1246x814` | `none` | `sideline_absent`, `telestrator_overlay`, `high_angle`, `normal_lighting` | Yes |
| `sf_07_sea_sf_low_light_shadow` | `test` | SEA vs SF | FOX | `broadcast_sideline` | `900x506` | `darken_068` | `sideline_visible`, `virtual_los`, `different_lighting` | Yes |
| `sf_08_nyj_jax_overexposed_glare` | `test` | NYJ vs JAX | FOX | `broadcast_sideline` | `900x507` | `brighten_122` | `midfield_logo`, `different_lighting`, `virtual_first_down` | Yes |
| `sf_09_no_car_720p_rescaled` | `test` | NO vs CAR | NFL All-22 | `all22_high_sideline` | `1280x720` | `resize_1280x720` | `sideline_absent`, `telestrator_overlay`, `different_resolution` | Yes |
| `sf_10_sea_sf_sideline_cropped` | `test` | SEA vs SF | FOX | `broadcast_sideline` | `900x506` | `mask_far_sideline` | `sideline_absent`, `partial_field_visibility`, `virtual_los` | Yes |
| `sf_11_endzone_tight_view_uncalibratable` | `test` | Broadcast Endzone 1 | CBS Style | `endzone_tight` | `800x450` | `none` | `different_camera_angle`, `partial_field_visibility` | **No** |
| `sf_12_heavy_occlusion_both_hashes_blocked` | `test` | SEA vs SF (Occluded) | FOX | `broadcast_sideline` | `900x506` | `occlude_both_hash_rows` | `heavy_player_occlusion`, `partial_field_visibility` | **No** |

---

## 3. Single-Frame Benchmark Results

### 3.1 Split-Level Aggregate Metrics

| Split | Total Frames | Expected Calibratable | Calibrated (`success=True`) | Refused (`success=False`) | Fabricated Calibrations | Mean Confidence (Calibrated) | Median Held-Out Error (yd) | Mean Runtime (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | `3` | `2` | `2` | `1` | **`0`** | `0.6081` | — | `120.79` |
| **`val`** | `3` | `3` | `3` | `0` | **`0`** | `0.8716` | **`0.275 yd`** | `93.01` |
| **`test` (Frozen)** | `6` | `4` | `4` | `2` | **`0`** | `0.8795` | **`0.270 yd`** | `60.88` |
| **Total / All** | **`12`** | **`9`** | **`9`** | **`3`** | **`0`** | **`0.8165`** | **`0.275 yd`** | **`83.89`** |

### 3.2 Frame-by-Frame Results

| Frame ID | Split | Success | Confidence | `x_coord_mode` | Yard Lines | Hash Inliers | Ridge Residual (px) | Hash RMSE (px) | Held-Out Error (yd, Hash+VP) | Failure Reason | Fabricated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sf_01_sea_sf_fox` | `val` | `True` | `0.8985` | `relative_10yd` | `5` | `30` | `0.963` | `0.661` | **`0.275 yd`** | `None` | `False` |
| `sf_02_nyj_jax_fox` | `val` | `True` | `0.8735` | `relative_10yd` | `6` | `25` | `1.292` | `0.731` | **`0.350 yd`** | `None` | `False` |
| `sf_03_no_car_all22` | `val` | `True` | `0.8429` | `relative_10yd` | `5` | `27` | `1.225` | `1.289` | **`0.057 yd`** | `None` | `False` |
| `sf_04_broadcast_sideline_wide` | `train` | `True` | `0.7240` | `relative_10yd` | `3` | `16` | `1.045` | `1.391` | — | `None` | `False` |
| `sf_05_broadcast_cropped_vertical` | `train` | `True` | `0.4922` | `relative_5yd` | `4` | `22` | `2.388` | `2.460` | — | `None` | `False` |
| `sf_06_all22_tight_scrum_uncalibratable` | `train` | `False` | `0.0000` | `uncalibrated` | `4` | `0` | — | — | — | `insufficient_hash_ticks` | `False` |
| `sf_07_sea_sf_low_light_shadow` | `test` | `True` | `0.8968` | `relative_10yd` | `5` | `28` | `0.996` | `0.663` | **`0.176 yd`** | `None` | `False` |
| `sf_08_nyj_jax_overexposed_glare` | `test` | `True` | `0.8644` | `relative_10yd` | `6` | `26` | `1.384` | `0.785` | **`0.378 yd`** | `None` | `False` |
| `sf_09_no_car_720p_rescaled` | `test` | `True` | `0.8580` | `relative_10yd` | `5` | `27` | `1.158` | `1.183` | — | `None` | `False` |
| `sf_10_sea_sf_sideline_cropped` | `test` | `True` | `0.8986` | `relative_10yd` | `5` | `30` | `0.962` | `0.661` | **`0.270 yd`** | `None` | `False` |
| `sf_11_endzone_tight_view_uncalibratable` | `test` | `False` | `0.0000` | `uncalibrated` | `0` | `0` | — | — | — | `insufficient_yard_lines` | `False` |
| `sf_12_heavy_occlusion_both_hashes_blocked` | `test` | `False` | `0.0000` | `uncalibrated` | `5` | `0` | — | — | — | `missing_hash_rows` | `False` |

### 3.3 Per-Condition Metrics Breakdown

| Condition Tag | Frames (`n`) | Calibrated | Refused | Failure Rate | Mean Confidence (Calibrated) | Median Held-Out Error (yd) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sideline_visible` | `2` | `2` | `0` | `0.0%` | `0.8977` | `0.226 yd` |
| `partial_sideline` | `1` | `1` | `0` | `0.0%` | `0.8735` | `0.350 yd` |
| `sideline_absent` | `6` | `5` | `1` | `16.7%` | `0.7631` | `0.164 yd` |
| `midfield_logo` | `2` | `2` | `0` | `0.0%` | `0.8690` | `0.364 yd` |
| `virtual_los` | `3` | `3` | `0` | `0.0%` | `0.8980` | `0.270 yd` |
| `virtual_first_down` | `3` | `3` | `0` | `0.0%` | `0.8788` | `0.350 yd` |
| `telestrator_overlay` | `2` | `2` | `0` | `0.0%` | `0.8504` | `0.057 yd` |
| `normal_lighting` | `4` | `4` | `0` | `0.0%` | `0.8347` | `0.275 yd` |
| `different_lighting` | `2` | `2` | `0` | `0.0%` | `0.8806` | `0.277 yd` |
| `high_angle` | `1` | `1` | `0` | `0.0%` | `0.8429` | `0.057 yd` |
| `different_resolution` | `1` | `1` | `0` | `0.0%` | `0.8580` | — |
| `low_resolution` | `2` | `2` | `0` | `0.0%` | `0.6081` | — |
| `partial_field_visibility` | `4` | `2` | `2` | `50.0%` | `0.6954` | `0.270 yd` |
| `heavy_player_occlusion` | `2` | `0` | `2` | `100.0%` | `0.0000` | — |
| `different_camera_angle` (end-zone tight) | `1` | `0` | `1` | `100.0%` | `0.0000` | — |

---

## 4. Temporal Stress-Test Benchmark (`8` Sequences, `96` Frames)

### 4.1 Per-Sequence Temporal Stability & Refusal Summary

Across the 8 12-frame stress sequences (`96` frames total), `CalibrationTracker` achieved **`62` calibrated frames** (`55` direct + `7` propagated) and **`34` explicitly refused frames** (`0 + 0 + 1 + 10 + 6 + 7 + 3 + 7 = 34`), with **`0` fabricated calibration frames**:

| Sequence ID | Split | Stress Condition | Total Frames | Direct Calibrated | Propagated (LK Flow / Prior) | Total Calibrated | Refused Frames | Expected Refused | Calibration Recoveries | Fabricated Frames |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seq_01_camera_pan` | `val` | Smooth horizontal camera pan (`0..+11 px`) | `12` | `12` | `0` | `12` | **`0`** | `0` | `0` | `0` |
| `seq_02_camera_zoom` | `val` | Progressive camera zoom (`1.00x..1.055x`) | `12` | `12` | `0` | `12` | **`0`** | `0` | `0` | `0` |
| `seq_03_short_line_dropout` | `test` | 2-frame dropout (`t=4..5`, bridged) + 3-frame dropout (`t=8..10`, expires at `t=10`) + recovery (`t=11`) | `12` | `7` | `4` | `11` | **`1`** | `1` | `1` | `0` |
| `seq_04_prolonged_dropout_expiration` | `test` | Prolonged 11-frame line dropout (`t=1` bridged, `t=2..11` refused) | `12` | `1` | `1` | `2` | **`10`** | `10` | `0` | `0` |
| `seq_05_camera_cut_and_recovery` | `test` | SEA vs SF (`t=0..2`) $\rightarrow$ 6-frame cutaway (`t=3..8` refused) $\rightarrow$ NYJ vs JAX (`t=9..11` recovery) | `12` | `6` | `0` | `6` | **`6`** | `6` | `1` | `0` |
| `seq_06_optical_flow_failure` | `test` | Direct (`t=0..2`) $\rightarrow$ 1-frame bridge (`t=3`) $\rightarrow$ 7-frame textureless washout (`t=4..10` refused) $\rightarrow$ recovery (`t=11`) | `12` | `4` | `1` | `5` | **`7`** | `7` | `1` | `0` |
| `seq_07_large_global_motion` | `test` | Direct (`t=0..3`) $\rightarrow$ 3-frame whip-pan blur (`t=4..6` refused) $\rightarrow$ 15-yd jump re-lock (`t=7..11`) | `12` | `9` | `0` | `9` | **`3`** | `3` | `1` | `0` |
| `seq_08_player_heavy_occlusion` | `test` | Direct (`t=0..2`) $\rightarrow$ 1-frame bridge (`t=3`) $\rightarrow$ 7-frame hash-row occlusion (`t=4..10` refused) $\rightarrow$ recovery (`t=11`) | `12` | `4` | `1` | `5` | **`7`** | `7` | `1` | `0` |
| **Total (`8` Sequences)** | — | **All Temporal Stress Sequences** | **`96`** | **`55`** | **`7`** | **`62`** | **`34`** | **`34`** | **`5`** | **`0`** |

- **Exact Refusal Sum Check**: `0 + 0 + 1 + 10 + 6 + 7 + 3 + 7 = 34` refused frames across the 8 sequences (`34 / 96 = 35.42%` of total temporal frames).

### 4.1.1 Failure-Reason Distribution (`3` Single-Frame Refusals, `34` Temporal Sequence Refusals)

All percentages in the temporal failure-reason table are calculated directly from the **`34` temporal sequence refusals** (`0 + 0 + 1 + 10 + 6 + 7 + 3 + 7 = 34`) and **`96` total temporal frames**:

| Failure Reason | Single-Frame Refusals (`n / 3`) | Share of Single-Frame Refusals (`% of 3`) | Temporal Sequence Refusals (`n / 34`) | Share of Temporal Refusals (`% of 34`) | Share of Total Temporal Frames (`% of 96`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `no_hough_lines` | `0` | `0.00%` | **`22`** | **`64.71%`** (`22 / 34`) | **`22.92%`** (`22 / 96`) |
| `missing_hash_rows` | `1` | `33.33%` | **`6`** | **`17.65%`** (`6 / 34`) | **`6.25%`** (`6 / 96`) |
| `camera_cut_uncalibrated` | `0` | `0.00%` | **`5`** | **`14.71%`** (`5 / 34`) | **`5.21%`** (`5 / 96`) |
| `confidence_expired` | `0` | `0.00%` | **`1`** | **`2.94%`** (`1 / 34`) | **`1.04%`** (`1 / 96`) |
| `insufficient_hash_ticks` | `1` | `33.33%` | `0` | `0.00%` (`0 / 34`) | `0.00%` (`0 / 96`) |
| `insufficient_yard_lines` | `1` | `33.33%` | `0` | `0.00%` (`0 / 34`) | `0.00%` (`0 / 96`) |
| **Total Refusals** | **`3 / 12` (`25.00%`)** | **`100.00%`** | **`34 / 96`** | **`100.00%`** (`34 / 34`) | **`35.42%`** (`34 / 96`) |

### 4.2 Calibration Recovery Examples

1. **Short-Dropout Recovery (`seq_03_short_line_dropout`)**:
   - During frames `t=4..5`, direct calibration fails (`insufficient_yard_lines`), and `CalibrationTracker` propagates the homography via Lucas-Kanade optical flow (`confidence = 0.7188` at `t=4`, `0.5750` at `t=5`). At `t=6`, direct calibration locks back on (`confidence = 0.8986`, `propagation_age = 0`). During the second dropout (`t=8..10`), propagation bridges `t=8..9` (`0.7189`, `0.5751`), expires at `t=10` (`failure_reason = "confidence_expired"`, `confidence = 0.0`), and recovers cleanly at `t=11` (`confidence = 0.8942`).
2. **Camera-Cut Reset & Cross-Game Recovery (`seq_05_camera_cut_and_recovery`)**:
   - At `t=3`, the scene cuts from `SEA vs SF` to a non-field broadcast graphic (`hist_corr < 0.55`). `CalibrationTracker` immediately flushes its state (`failure_reason = "camera_cut_uncalibrated"`), refuses all 6 cutaway frames (`t=3..8`), and immediately locks onto `NYJ vs JAX` at `t=9..11` (`confidence = 0.9016`) without blending stale `SEA vs SF` geometry.
3. **Large Global Motion Re-Lock (`seq_07_large_global_motion`)**:
   - After a 3-frame whip-pan dropout (`t=4..6`) and a `15.0 yd` lateral field shift (`> max_jump_yd = 4.0 yd`), `CalibrationTracker` detects the large spatial jump at `t=7` and snaps directly to the new observation (`confidence = 0.8985`) rather than averaging incompatible homographies.

---

## 5. Comparison Against Phase 1 Baseline

| Metric / Benchmark Gate | Phase 1 Baseline (`week1_hough_benchmark.json`) | Phase 2 Benchmark (`phase2_calibration_benchmark.json`) | Status |
| :--- | :---: | :---: | :---: |
| **Frame 1 (`SEA vs SF`) Held-Out Median Error (Hash+VP)** | `0.275 yd` (`0.8985` conf) | `0.275 yd` (`0.8985` conf) | **Identical (`0.000 yd` diff)** |
| **Frame 2 (`NYJ vs JAX`) Held-Out Median Error (Hash+VP)** | `0.350 yd` (`0.8735` conf) | `0.350 yd` (`0.8735` conf) | **Identical (`0.000 yd` diff)** |
| **Frame 3 (`NO vs CAR`) Held-Out Median Error (Hash+VP)** | `0.057 yd` (`0.8429` conf) | `0.057 yd` (`0.8429` conf) | **Identical (`0.000 yd` diff)** |
| **Multi-Condition Single-Frame Coverage** | `3` frames (`1` split) | `12` frames (`train` / `val` / `test`) | **4x expansion** |
| **Temporal Stress Sequence Coverage** | Unit test only | `8` sequences (`96` frames, `34` refusals verified) | **Full temporal benchmark** |

---

## 6. Biggest Remaining Failure Modes (Measured)

1. **Both Hash-Mark Rows Occluded (`sf_06`, `sf_12`, `seq_08`)**:
   - When dense line-of-scrimmage pileups occlude one or both of the middle-of-field hash-mark rows (`Y = 23.25 yd` and `30.083 yd`), the 2-row hash RANSAC cannot constrain the lateral $Y$ projective scale and refuses calibration (`insufficient_hash_ticks` / `missing_hash_rows`).
2. **End-Zone / Tight Zoom Views with `< 3` Yard Lines (`sf_11`)**:
   - Tight red-zone, goal-line, or player-isolation camera angles with fewer than 3 visible 5-yard lines cannot constrain the longitudinal vanishing-point pencil `m = alpha * x_mid + beta` and are explicitly refused (`insufficient_yard_lines`).
3. **Severe Darkening / Sideline-Tick vs. Hash-Row Ambiguity**:
   - In sideline views where the 1-yard sideline boundary ticks (`Y = 53.33 yd`) lie inside the turf mask (`sf_01_sea_sf_fox` has `14` sideline ticks at `y ≈ 88 px` vs. `15` far-hash ticks at `y ≈ 248 px`), severe darkening (`< 0.90x` gain) can cause 2 far-hash ticks to drop below the fixed top-hat threshold (`HASH_TH_MIN = 18`), allowing the sideline tick row (`14` ticks) to out-vote the far hash row (`13` ticks). Replacing fixed top-hat thresholds with local percentile thresholding or geometric row-spacing priors is deferred to a future calibration iteration.
4. **Lack of Absolute Yard-Number OCR (`x_coord_mode != "absolute"`)**:
   - Without yard-number recognition, calibration resolves `relative_10yd` (when 10-yard number parity is unambiguous) or `relative_5yd` (when parity is ambiguous), never `absolute` (`0..120 yd`).
