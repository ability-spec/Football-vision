# Phase 4 — Player Tracking & Field-Space Trajectories: Benchmark Report

**Status:** Phase 4 implementation complete — submitted for review (NOT marked FINAL).

Phase 4 turns per-frame player observations into trustworthy multi-frame **field-space trajectories**, built on the
untouched Phase 2 calibration API and the Phase 3 detector / footpoint / tracker / projection interfaces.
**No Phase 2 or Phase 3 algorithm was modified** (only additive Phase 4 dataclasses in `schema.py` and exports in
`__init__.py` → `v0.4.0`).

**Out of scope (left to downstream modules):** ball tracking, jersey OCR, event detection, route classification,
and prediction models were not implemented in Phase 4.

---

## 1. Requirements Coverage

| Phase 4 requirement | Implementation | Evidence / contract |
| :--- | :--- | :--- |
| **1. Track state** — persistent `track_id`, observed vs coasted, `missed_frames`, detection confidence, footpoint reliability, team cluster + confidence | `trajectory/builder.py`, `schema.TrajectorySample` | every sample carries `track_state ∈ {observed, coasted}`, `missed_frames`, `provenance.team` / `team_confidence`, and the Phase 3 footpoint reliability flags |
| **2. Temporal field projection** — project only when calibration + footpoint are valid; propagate calibration state; preserve `x_coord_mode`; never invent an absolute yardline | Phase 3 `projection.FieldProjector`, reused unchanged | `projection_status` records the exact refusal reason; `absolute_yardline = None` unless `x_coord_mode == "absolute"` (`0` violations) |
| **3. Trajectory construction** — x/y over time, frame/timestamp, smoothing, jump rejection, uncertainty, observed vs inferred samples | `trajectory/smoothing.py`, `outlier.py`, `uncertainty.py` | `field_position` = accepted smoothed measurement; `predicted_position` = labelled dead reckoning; `position_source="none"` beyond `max_gap_frames` |
| **4. Kinematics** — velocity, acceleration, direction, distance travelled | `trajectory/kinematics.py` | filter-state velocity, clamped finite-difference acceleration with clip flags, `direction_rad`, cumulative `distance_cum_yd` accrued only between accepted measurements |
| **5. Camera-motion interaction** — pan, zoom, short propagation, camera cut, occlusion, dropout | manifest sequences 3, 5, 6, 8 | `traj_seq_08` implements an explicit camera cut with refusal reason `camera_cut_uncalibrated` + measured recovery latency |
| **6. Frozen temporal benchmark** | `benchmarks/evaluate_phase4_trajectories.py`, `data/benchmarks/phase4_trajectory_manifest.json` | 8 sequences / 180 frames; `TRAIN / VAL / TEST`; no constant tuned on `TEST` |
| **7. Real NFL validation** | `real_frame_smoke_test` | single-frame integration smoke test, `quantitative_ground_truth_available: false` |
| **8. Scope** | — | no ball tracking, jersey OCR, event detection, route classification, or prediction model |
| **9. Preservation** | test suite | Phase 0/1 + Phase 2 + Phase 3 tests unchanged and green |

---

## 2. Detector Provenance (carried over from the accepted Phase 3 audit)

- Controlled sequences use **`FixturePlayerDetector` (`fixture_detector_v1`)**; Phase 4 evaluates the *trajectory layer* given player positions.
- **Image-space player-detector accuracy remains UNMEASURED.** A perfect fixture pass-through is **not** evidence that image-based detection is solved.
- The real-frame check uses `TurfContrastPlayerDetector` (`turf_contrast_baseline_v1`) as a single-frame smoke test only.
- `TEAM_A / TEAM_B` remain permutation-invariant torso-appearance clusters; no Phase 4 metric depends on real-world team identity.

### 2.1 What the ground truth is

1. Player motion is **synthetic deterministic field-space ground truth** (straight routes, tanh-blended cuts, speed ramps) integrated at 240 Hz — engineering fixtures, **not** recorded NFL trajectories.
2. Footpoints come from projecting that ground truth through the **real per-frame homography** of each base broadcast / All-22 frame, plus deterministic pixel jitter (0.4–0.8 px on most sequences; deliberately 1.6 px and 3.0 px on two `TEST` sequences).
3. Per-frame geometry is a `CalibrationResult` from the real Phase 1 calibration; camera pan/zoom is composed exactly onto `H`; propagated frames carry deliberate drift with `is_temporally_propagated=True`; refused frames are explicit failure objects (`confidence_expired`, `camera_cut_uncalibrated`).
4. Detection boxes are fixture boxes around those footpoints; the Phase 3 tracker, footpoint gate and projector are consumed unmodified.

---

## 3. Frozen Parameters (fixed a priori, before `TEST`)

| Parameter | Value | Honest status |
| :--- | :---: | :--- |
| `process_accel_std_yd_s2` | `5.0` | Process-noise acceleration scale (≈0.5 g). **Assumption**, not measured NFL biomechanics |
| `gate_chi2_2dof` | `9.21` | χ²₂ 99% quantile for the field-space innovation gate |
| `max_speed_yd_s` / `max_accel_yd_s2` | `12.0` / `25.0` | Plausibility clamps (≈24.5 mph / ≈2.6 g) |
| `max_gap_frames` | `3` | Frames of dead reckoning before a sample claims no position at all |
| `max_consecutive_rejections` | `3` | Explicit filter re-initialization after sustained disagreement |
| `propagated_drift_yd_per_age` | `0.25` | Assumed drift per propagation frame used to inflate uncertainty |
| `footpoint_edge_sigma_px` (floor `0.3`) | `1.0` | Assumed per-edge box noise |
| `camera_cut_refusal_frames` | `3` | Refused frames in the camera-cut scenario |
| Gate warm-up | `2` accepted updates | Standard track initiation; the gate **threshold is unchanged** |

**No constant was tuned on `TEST`.** One design rule was chosen on `TRAIN`/`VAL` only, before any `TEST` run, with a
frozen A/B measurement stored in the JSON (`gate_warmup_ablation`):

| Split | Gate active from 2nd update | Gate active after warm-up (shipped) |
| :--- | :---: | :---: |
| `train` false rejections | `48 / 264` (`18.2%`) | `13 / 264` (`4.9%`) |
| `val` false rejections | `1 / 246` (`0.4%`) | `0 / 246` (`0.0%`) |

### 3.1 The smoothing constant is not allowed to hide real motion

`process_accel_std_yd_s2 = 5.0` is the only constant that can attenuate genuine movement (it sets the Kalman gain
against measurement noise). Phase 4 therefore measures path length against ground truth:
`motion_preservation_ratio_vs_gt = smoothed path / GT path` = `1.027` (train), `0.97` (val), `1` (test).
The raw (unsmoothed) projection path is consistently **longer** than ground truth (`0.884` / `0.78` / `0.81` of the smoothed path) because it accumulates jitter; smoothing removes that jitter without
shortening real motion.

An A/B of that constant was measured on **`TRAIN` + `VAL` only** (stored in the JSON as `process_noise_ablation`; `TEST` was
excluded so the comparison cannot be used to tune the frozen split):

| Split | Shipped `5.0 yd/s²` (median err / path vs GT / mean σ) | Stiffer `0.5 yd/s²` |
| :--- | :---: | :---: |
| `train` | `0.0294 yd` / `1.027` / `0.0481 yd` | `0.0273 yd` / `1.024` / `0.0459 yd` |
| `val` | `0.0214 yd` / `0.97` / `0.0445 yd` | `0.0221 yd` / `0.982` / `0.0414 yd` |

The setting is therefore **not sensitive** in this range: position error and preserved path length move by less than a few
percent either way, and the stiffer setting merely reports slightly *smaller* uncertainty (i.e. more confident than warranted).
The shipped value stands, and the motion-preservation metric confirms that no constant in this range is hiding real movement.

---

## 4. Frozen `TRAIN / VAL / TEST` Results

Primary accuracy metrics use the **dominant trajectory per ground-truth player**; spurious / fragment tracks are scored
separately. Identity metrics use the same CLEAR-style definitions as the Phase 3 evaluator.

| Metric | `train` (2 seq, 44 fr) | `val` (2 seq, 44 fr) | Frozen `test` (4 seq, 92 fr) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Detector implementation | `FixturePlayerDetector (fixture_detector_v1)` | same | same | Fixture harness; image detector accuracy unmeasured |
| Samples / expected | `264` / `264` | `277` / `264` | `577` / `552` | 6 players per frame |
| **Track continuity** (position or labelled dead reckoning) | **1** | **0.966** | **0.884** | `TEST` loss comes from the camera cut (§4.1, §5.4) |
| Observed-sample ratio | 1 | 0.973 | 0.984 | player actually detected with a reliable footpoint |
| Inferred-sample ratio (dead reckoning) | 0.034 | 0.035 | 0.154 | never reported as a measured position |
| Measured completeness | 0.966 | 0.932 | 0.748 | accepted projected measurement |
| **IDSW** (lifetime, observed samples) | **`0`** | **`5`** | **`10`** | observed `track_id` change for a GT player |
| — active association swap / post-expiration reinit | `0` / `0` | `5` / `0` | `10` / `0` | see §5.6 for the definition difference vs Phase 3 |
| **FRAG** (observed after ≥1-frame observation gap) | `0` | `0` | `4` | |
| Tracks created | `12` | `15` | `32` | 6 players per sequence; extras are fragment/spurious tracks |
| Raw projection position error, median | 0.0447 yd | 0.0446 yd | 0.0599 yd | baseline without smoothing |
| **Smoothed trajectory position error, median** | **0.0294 yd** | **0.0214 yd** | **0.037 yd** | primary position metric |
| Trajectory error p90 / RMSE | 0.0699 / 0.044 yd | 0.0553 / 0.0335 yd | 0.0795 / 0.0566 yd | |
| Smoothing reduction vs raw projection | **34.23%** | **52.02%** | **38.23%** | median error |
| Image-space EMA arm (smoothing *before* projection) | 0.1427 yd | 0.123 yd | 0.1509 yd | degrades with camera motion |
| Spurious-track position error, median | — | 0.0342 yd | 0.0616 yd | reported separately |
| **Velocity** error median / RMSE | 0.2071 / 1.4943 yd/s | 0.1263 / 1.0814 yd/s | 0.1817 / 1.4477 yd/s | RMSE inflated by cut transients (§5.5) |
| Direction-of-motion error, median | 4.38° | 1.66° | 2.93° | GT speed > 0.5 yd/s |
| **Acceleration** magnitude error, median | 6.9914 yd/s² | 6.1868 yd/s² | 8.3251 yd/s² | GT median accel `1.5` / `0` / `0` yd/s²; see §5.3 |
| Distance travelled (smoothed / raw / GT, yd) | 38 / 43 / 37 | 32 / 41 / 33 | 64 / 79 / 64 | smoothing removes jitter, keeps real motion |
| Outlier events (rejected / spurious / refused / absorbed) | `0` | `3 / 1 / 0 / 0` | `1 / 1 / 1 / 0` | handling rate `TEST` = 1 |
| False rejections / gated clean samples | `13` / `264` | `0` / `246` | `56` / `462` | jitter-dominated on `TEST` (§5.2) |
| **Recovery latency** after an observed gap (median / max frames) | `—` / `0` | `2` / `3` | `2` / `2` | frames with no observation before the next one |
| Projection-recovery latency after a field-position gap (max frames) | `2` | `5` | `5` | includes the camera-cut refusal |
| Geometry states (calibrated / propagated / unknown) | `264 / 0 / 0` | `277 / 0 / 0` | `507 / 18 / 52` | explicit per-frame state |
| Samples with a position while geometry is `unknown` | **`0`** | **`0`** | **`0`** | safety invariant |
| Absolute-yardline violations (`x_coord_mode != absolute`) | **`0`** | **`0`** | **`0`** | safety invariant |
| **Fabricated trajectory samples** | **`0`** | **`0`** | **`0`** | safety invariant |
| Uncertainty coverage 68% / 95% (accepted measurements) | 63.4% / 90.8% | 76.0% / 98.5% | 49.7% / 80.1% | under-covers on `TEST` (§5.2) |
| Mean σ-ellipse major / minor | 0.0481 / 0.0097 yd | 0.0445 / 0.0096 yd | 0.0496 / 0.0114 yd | ≈0.5–0.7 ft typical major axis |
| Runtime (trajectory layer, ms/frame) | `5.11` | `3.59` | `3.38` | wall-clock CPU; varies between runs |

### 4.1 Per-Sequence Detail

| Sequence | Split | Fr | Jitter px | Tracks | Continuity | Obs / Inf | IDSW | FRAG | Smoothed med (yd) | Path vs GT | Recovery lat (med/max) | Outliers (rej/spur/ref/abs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `traj_seq_01_train_linear_and_sprint` | `train` | 24 | 0.8 | 6 | 1 | 1 / 0 | 0 | 0 | 0.0229 | 1.011 | — / 0 | — |
| `traj_seq_02_train_cuts_and_deceleration` | `train` | 20 | 0.8 | 6 | 1 | 1 / 0.07 | 0 | 0 | 0.0359 | 1.03 | — / 0 | — |
| `traj_seq_03_val_camera_pan` | `val` | 24 | 0.8 | 6 | 1 | 1 / 0 | 0 | 0 | 0.0232 | 1.021 | — / 0 | — |
| `traj_seq_04_val_jump_outliers` | `val` | 20 | 0.8 | 9 | 0.93 | 0.94 / 0.08 | 5 | 0 | 0.0196 | 0.944 | 2 / 3 | 3/1/0/0 |
| `traj_seq_05_test_calibration_dropout_and_propagation` | `test` | 24 | 0.8 | 7 | 0.94 | 1 / 0.11 | 1 | 0 | 0.0273 | 1 | — / 0 | 0/0/1/0 |
| `traj_seq_06_test_camera_pan_zoom_occlusion` | `test` | 24 | 1.6 | 7 | 0.93 | 0.99 / 0.22 | 1 | 1 | 0.058 | 1.064 | 2 / 2 | 1/0/0/0 |
| `traj_seq_07_test_high_jitter_and_occlusion` | `test` | 20 | 3.0 | 7 | 1 | 0.97 / 0.19 | 2 | 2 | 0.0467 | 1.009 | 1 / 2 | 0/1/0/0 |
| `traj_seq_08_test_camera_cut_refusal_and_recovery` | `test` | 24 | 0.4 | 11 | 0.69 | 0.98 / 0.07 | 6 | 1 | 0.0125 | 0.83 | 2 / 2 | — |

**Camera-cut scenario (`traj_seq_08`).** Hard shot change at frame 5, calibration refused for frames `[5, 6, 7]` with reason `camera_cut_uncalibrated` (`33` refused samples), **zero** field positions claimed while refused (`positioned_by_geometry_state.unknown = 0`), re-acquisition afterwards with max projection-recovery latency **5 frames**, continuity 0.69, IDSW `6`, FRAG `1` (identity is lost across the cut: Phase 4 has no re-identification module).

**Determinism:** re-running `traj_seq_04_val_jump_outliers` reproduces every metric exactly
(`deterministic = True`, differing keys `[]`).

---

## 5. Honest Limitations (read before quoting any number above)

**5.1 Fixture, not NFL motion.** Ground truth is synthetic deterministic routes projected through real homographies;
speeds, cut timings and jitter levels are engineering choices. These errors measure the *trajectory layer*, not NFL tracking accuracy.

**5.2 `TEST` uncertainty under-covers and the gate over-rejects at high jitter.** The frozen footpoint noise model assumes
1.0 px per-edge noise while two `TEST` sequences inject 1.6 px and 3.0 px, so the 68%/95% bands cover 49.7% / 80.1%
of accepted measurements (`TRAIN`/`VAL` are near nominal), and 56 of 462 gated clean samples are rejected, each costing one
dead-reckoned sample. Reported as a measurement–model mismatch; correcting it means re-freezing `footpoint_edge_sigma_px` in a later tuning phase.

**5.3 Acceleration is the weakest metric** (median error 8.3 yd/s² on `TEST`, GT median 0 yd/s²). On straight routes the error is the filter's own
process-noise floor, i.e. it reflects the assumed `process_accel_std_yd_s2` rather than measured biomechanics. Not a validated capability.

**5.4 Identity is lost across a camera cut.** `traj_seq_08` shows continuity 0.69 with 6 IDSW and 1 FRAG: positions are
refused rather than fabricated during the cut and re-acquired after it, but the Phase 3 tracker has no shot-change re-identification.

**5.5 Velocity RMSE is inflated by exactly the events it should expose** — filter lag at abrupt cuts, the 1-frame dead-reckoning
horizon, and post-cut re-acquisition — rather than by steady-state noise.

**5.6 IDSW here is not the same event as the Phase 3 `TEST` IDSW.** Phase 3's single `TEST` IDSW was a post-expiration
re-initialization (0 here) with zero active-swap switches. Phase 4 associates per sample against the full
active track set, and all 10 `TEST` IDSW are **active-swap** violations (10 / 0 split). Both definitions are exposed in the JSON.

**5.7 Candidate fixes were derived, then rejected as over-fitting.** (a) widen the gate to χ²₂ ≤ 16 to suppress jitter
rejections; (b) elongate the motion prior along the camera-pan axis to remove pan-induced false rejections; (c) raise
`process_accel_std_yd_s2` to smooth identity hops. Each would improve headline `TEST` numbers while weakening the
uncertainty model or the physical prior, so none are shipped.

**5.8 No real multi-frame broadcast clip was available**, and camera motion is an exact synthetic transform (no perspective
change, rolling shutter, or motion blur). The real-frame result is a single-frame smoke test and provides no trajectory accuracy.

---

## 6. Real NFL Frame Scope (smoke test only)

| Field | Value |
| :--- | :--- |
| `frame_id` | `rf_01_sea_sf_smoke` |
| `detector_implementation` | `turf_contrast_baseline_v1` |
| `evaluation_scope` | `single_frame_integration_smoke_test_no_ground_truth_trajectory` |
| `quantitative_ground_truth_available` | `False` |
| `calibration_success` | `True` |
| `x_coord_mode` | `relative_10yd` |
| `detections` | `25` |
| `tracks` | `25` |
| `trajectory_samples` | `25` |
| `geometry_state_counts` | `{'calibrated': 25, 'propagated': 0, 'unknown': 0}` |
| `samples_with_field_position` | `23` |
| `samples_with_absolute_yardline` | `0` |
| `fabricated_trajectory_samples` | `0` |

> Single-frame smoke test only: one still frame cannot exercise multi-frame trajectory, smoothing, camera motion, or occlusion behaviour.

---

## 7. Reproduce

```bash
python3 benchmarks/evaluate_phase4_trajectories.py   # outputs/phase4_trajectory_benchmark.json + overview PNG
python3 -m pytest -v                                # full Phase 0-4 suite
```

`tests/test_phase4_trajectories.py` (22 tests) covers geometry-state resolution, footpoint covariance scaling, homography-Jacobian
vs. finite differences, covariance PSD/inflation, filter jitter reduction and velocity recovery, speed/acceleration clamps, both
jump gates, no-position-under-unknown-geometry, jump rejection with trajectory preservation, camera-pan field stability, labelled
dead reckoning with `max_gap_frames`, `x_coord_mode` absolute-yardline enforcement, propagated-geometry uncertainty inflation,
determinism, observed/coasted track state, recovery latency, motion preservation, camera-cut refusal → recovery, and benchmark invariants.
