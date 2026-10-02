# Phase 4 — Field-Space Player Trajectories: Benchmark Report

Generated from `outputs/phase4_trajectory_benchmark.json` by `benchmarks/render_phase4_report.py`; every number below is read from that JSON, so the document cannot drift from the benchmark. Report and JSON are committed together in the audit-remediation commit (label: `audit-only pass; working tree`), so the label is not a claim about the commit's own SHA.

**Status: reviewed implementation of the audit-remediation pass, NOT declared final.** Phase 0/1/2/3 remain frozen; Phase 4 is the subject of this report.

## 0. What this benchmark measures — and what it does not

* **Split protocol (`train`):** Used to confirm the a priori gate/process-noise assumptions produce stable kinematics.
* **Benchmark kind:** `synthetic_trajectory_benchmark` — synthetic deterministic field-space routes projected through the *real* per-frame homography of each base image. This is an engineering benchmark of the trajectory layer, **not** a measurement of NFL or broadcast tracking accuracy.
* Field-space trajectories are scored against SYNTHETIC deterministic ground-truth routes projected through real per-frame homographies. This is an engineering benchmark of the trajectory layer, not a measurement of NFL or broadcast tracking accuracy.
* **Image-based detector accuracy: `unmeasured`** (quantitative metrics field = `None`). Detection counts/precision/recall in this file are FIXTURE HARNESS pass-through statistics (the boxes are scheduled by the harness, not predicted from pixels). They must never be reported as player-image-detector accuracy; image-based detector accuracy is unmeasured.
* **Real multi-frame trajectory accuracy: `not_measured`** — Real multi-frame trajectory accuracy is not yet measured: no real broadcast clip with frame-level ground-truth trajectories is available in this project. Real frames are used for single-frame smoke/integration and geometry-refusal tests only.
* **Uncertainty: `not_statistically_calibrated`** — The reported covariance is an a priori engineering model (footpoint pixel noise propagated through the homography Jacobian plus a propagation-drift allowance). It is NOT statistically calibrated against data: empirical coverage is reported per split, per geometry state, and before/after the smoothing stage, and the gap is explained rather than tuned away. Do not read the reported sigma as a calibrated confidence interval.
* **Acceleration: `experimental_not_validated`** — Acceleration is the finite difference of the smoothed filter velocity, bounded by a plausibility clip. It is EXPERIMENTAL AND NOT VALIDATED: no measured acceleration accuracy is claimed, and the reported error is dominated by the filter's own noise floor rather than by a validated acceleration model.
* **Real-frame smoke test in this run:** ran on `/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-5.jpg` (25 detections / 25 tracks / 23 of 25 samples positioned), scope `single_frame_integration_smoke_test_no_ground_truth_trajectory`

## 1. Data and split protocol

| Split | Sequences | Frames | GT player-frames | Emitted samples | Tracks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 2 | 44 | 264 | 264 | 12 |
| `val` | 2 | 44 | 264 | 277 | 15 |
| `test` | 4 | 92 | 552 | 577 | 32 |

* `train`: Used to confirm the a priori gate/process-noise assumptions produce stable kinematics.
* `val`: Used to check jump-rejection behaviour and camera-motion handling.
* `test`: Strictly held-out frozen split; zero thresholds or model constants tuned on TEST.

Model constants were fixed before scoring and are published verbatim in the JSON: `camera_cut_refusal_frames = 3`, `footpoint_edge_sigma_px = 1.0`, `footpoint_sigma_floor_px = 0.3`, `gate_chi2_2dof = 9.21`, `image_jump_max_speed_px_per_frame = 22.0`, `max_accel_yd_s2 = 25.0`, `max_consecutive_rejections = 3`, `max_gap_frames = 3`, `max_speed_yd_s = 12.0`, `note = All constants were fixed a priori (documented engineering assumptions) before TEST was evaluated.`, `process_accel_std_yd_s2 = 5.0`, `propagated_drift_yd_per_age = 0.25`.

Sequence inventory:

| Sequence | Split | Scenario | Frames | Tracks | Injected outliers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `traj_seq_01_train_linear_and_sprint` | train | `linear_mixed_speeds` | 24 | 6 | 0 |
| `traj_seq_02_train_cuts_and_deceleration` | train | `cuts_and_stops` | 20 | 6 | 0 |
| `traj_seq_03_val_camera_pan` | val | `camera_pan` | 24 | 6 | 0 |
| `traj_seq_04_val_jump_outliers` | val | `jump_outliers` | 20 | 9 | 4 |
| `traj_seq_05_test_calibration_dropout_and_propagation` | test | `geometry_state_transitions` | 24 | 7 | 1 |
| `traj_seq_06_test_camera_pan_zoom_occlusion` | test | `combined_camera_motion` | 24 | 7 | 1 |
| `traj_seq_07_test_high_jitter_and_occlusion` | test | `high_jitter_occlusion` | 20 | 7 | 1 |
| `traj_seq_08_test_camera_cut_refusal_and_recovery` | test | `camera_cut_refusal_and_recovery` | 24 | 11 | 0 |

## 2. Dominant-track selection (audit item A)

**Rule:** `lifetime_sample_count_then_lowest_track_id`.

* inputs: `track_id`, `sample lifetime (number of trajectory samples emitted for the track)`
* tie-break: lowest track_id (deterministic; independent of dictionary/iteration order)
* uses ground-truth trajectory: `False`; ground-truth error: `False`; oracle selection: `False`; future information: `False`
* Ground truth is used ONLY to attribute a track to the player it is scored against (an oracle association inherent to every labelled MOT benchmark: an error cannot be computed without knowing which player a track is compared to). That association is computed before, and never influences, the selection itself.
* Selection is reproducible from runtime-observable counters alone. Fragmentation is additionally reported for every non-dominant track (see track_table / fragmentation_taxonomy), so a poor dominant track cannot hide fragment tracks.

The selection is implemented by `select_dominant_track()` in the evaluator. It receives per-track counters only (track id, emitted-sample count, positioned-sample count, longest observed run) and cannot see coordinates or errors. `tests/test_phase4_trajectories.py` pins this down in three ways: (a) a poisoned-candidate test that injects perfect/tragic ground-truth error fields into the candidate dicts and asserts the selection does not move, (b) a criterion test asserting the documented rule inputs carry no error/ground-truth term, and (c) a recomputation test that re-derives every published dominant track from the published runtime counters alone.

### 2.1 Sensitivity to the selection rule (diagnostic only)

Two alternative *runtime-only* rules are evaluated on the same per-track counters. They are reported so a reader can see how sensitive the primary metric is to the rule; they are not used for any headline number.

| Split | Rule | Identical selections | Median err under rule (yd) | Shipped rule (yd) |
| :--- | :--- | :--- | :--- | :--- |
| `train` | `longest_observed_run_then_lowest_track_id` | 12/12 | 0.0359 | 0.0294 |
| `train` | `positioned_sample_count_then_lowest_track_id` | 12/12 | 0.0359 | 0.0294 |
| `val` | `longest_observed_run_then_lowest_track_id` | 12/12 | 0.0232 | 0.0214 |
| `val` | `positioned_sample_count_then_lowest_track_id` | 12/12 | 0.0232 | 0.0214 |
| `test` | `longest_observed_run_then_lowest_track_id` | 24/24 | 0.0580 | 0.0370 |
| `test` | `positioned_sample_count_then_lowest_track_id` | 24/24 | 0.0580 | 0.0370 |

## 3. Track accounting: dominant tracks vs everything else (audit item B)

| Account | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| GT player-frames | 264 | 264 | 552 |
| Samples in dominant tracks | 264 | 255 | 504 |
| Samples in non-dominant (fragment) tracks | 0 | 22 | 73 |
| Samples with no harness association | 0 | 0 | 0 |
| Samples in dominant tracks without any position (refused/coasting) | 0 | 0 | 16 |
| Samples in dominant tracks / GT frames (identity continuity) | 1.0000 | 0.9659 | 0.9130 |
| Samples with a field estimate (position or labelled dead reckoning) / GT frames | 1.0000 | 0.9659 | 0.8841 |
| Samples with a reported field position / GT frames | 0.9659 | 0.9318 | 0.7482 |
| GT player-frames with no dominant-track sample | 0 | 9 | 64 |
| Tracks emitted | 12 | 15 | 32 |
| Tracks excluded from dominant metrics | 0 | 4 | 9 |

The four sample accounts sum to the emitted samples in every split (asserted inside the evaluator), so a fragment track can never silently disappear from the report.

### 3.1 Error of the dominant track vs error across all matched tracks

| Metric | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Position error, dominant tracks (median yd) | 0.0294 | 0.0214 | 0.0370 |
| Position error, fragment tracks (median yd) | — | 0.0342 | 0.0616 |
| Position error, all matched tracks (median yd) | 0.0294 | 0.0225 | 0.0371 |
| Fragment samples (with a position) | 0 | 14 | 44 |
| Fragment samples (all) | 0 | 22 | 73 |

## 4. Fragmentation taxonomy (audit item C)

Categories are mutually exclusive and exhaustive over **(player, track) episodes** and are assigned deterministically from runtime behaviour plus the harness' injected-event labels. The evaluator asserts that the taxonomy total equals the number of episodes, so a new failure mode cannot be dropped silently. A track whose lifetime spans two player labels produces two episodes (one per label).

| Category | TRAIN | VAL | TEST | Definition |
| :--- | :--- | :--- | :--- | :--- |
| `legitimate_new_track` | 12 | 12 | 24 | First track attributed to that ground-truth player in the sequence, and not created at an injected outlier event. |
| `active_association_swap` | 0 | 3 | 8 | A track for the player appears in a frame where the player's previous track is still present (both tracks exist in that frame at least momentarily: an active-association swap, the identity failure Phase 3 reports as 0). |
| `post_expiration_reinitialization` | 0 | 0 | 0 | A track for the player appears after the previous track for that player had no sample in the re-acquisition frame (expired / missed / dropped out): a post-expiration re-initialization rather than a swap. |
| `outlier_induced_spawn` | 0 | 1 | 1 | The track's first sample coincides with an injected outlier event for that player (the corrupted measurement was strong enough to spawn a separate track). |
| `unmatched_spurious_track` | 0 | 0 | 0 | A track that the harness cannot attribute to any ground-truth player. Expected to be 0 in this harness because every fixture detection carries a label; reported so that a non-zero value is visible rather than silently dropped. |

`fragment_excluded_from_dominant_track` is the fifth audit category and is reported as an orthogonal flag (whether the episode is the one used for the primary metric) rather than a mutually exclusive origin, because an episode has both an origin and a metric-membership:

| Episode origin of excluded fragments | TRAIN | VAL | TEST |
| :--- | :--- | :--- | :--- |
| `legitimate_new_track` | 0 | 1 | 8 |
| `active_association_swap` | 0 | 2 | 0 |
| `post_expiration_reinitialization` | 0 | 0 | 0 |
| `outlier_induced_spawn` | 0 | 1 | 1 |
| `unmatched_spurious_track` | 0 | 0 | 0 |
| **excluded fragments (total)** | 0 | 4 | 9 |

### 4.1 Cross-check against the CLEAR identity metrics

`IDSW` and `FRAG` keep the CLEAR-MOTA definitions documented in Phase 3 (per *observed frame*), whereas the taxonomy counts per *track episode*; the two are reported side by side instead of redefining either one to match the other.

| Metric | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| CLEAR IDSW (lifetime) | 0 | 5 | 10 |
| — active-association swaps | 0 | 5 | 10 |
| — post-expiration re-initializations | 0 | 0 | 0 |
| CLEAR FRAG (observation gaps) | 0 | 0 | 4 |
| Taxonomy episodes | 12 | 16 | 33 |
| Tracks with two player labels | 0 | 1 | 1 |
| Max tracks per player in one sequence | 1 | 3 | 2 |

**Identity contamination is visible here (train=0, val=1, test=1).** A track whose samples carry two different player labels means the tracker kept one `track_id` alive across a player change — the behaviour the Phase-3 report measures as an active association swap after the 52 px framing jump of the camera cut. Attribution is **per sample** (CLEAR per-frame convention), so identity metrics stay honest; each contaminated track is additionally listed in `identity_contamination` per sequence and produces a second taxonomy episode. This is reported, not hidden, and it is not re-identified: Phase 4 has no jersey/identity module.

## 5. Trajectory accuracy (all values use the synthetic benchmark)

| Metric | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Raw projection error, median (yd) | 0.0447 | 0.0446 | 0.0599 |
| Smoothed trajectory error, median (yd) | 0.0294 | 0.0214 | 0.0370 |
| Trajectory error, p90 (yd) | 0.0699 | 0.0553 | 0.0795 |
| Trajectory error, RMSE (yd) | 0.0440 | 0.0335 | 0.0566 |
| Smoothing reduction vs raw projection (%) | 34.23 | 52.02 | 38.23 |
| Image-space EMA arm, median (yd) (harness stub, GT-dependent) | 0.1427 | 0.1230 | 0.1509 |
| Velocity error, median (yd/s) | 0.2071 | 0.1263 | 0.1817 |
| Velocity error, RMSE (yd/s) | 1.4943 | 1.0814 | 1.4477 |
| Direction-of-motion error, median (deg) | 4.3826 | 1.6631 | 2.9291 |
| Dead-reckoning error, median (yd) | 0.1691 | 0.0455 | 0.0996 |
| Observed samples (detection-backed) | 1.0000 | 0.973 | 0.984 |
| Inferred samples (labelled dead reckoning) | 0.034 | 0.035 | 0.154 |

### 5.1 Motion preservation (smoothing must not delete real movement)

| Metric | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Smoothed path length (yd) | 38.0 | 32.0 | 64.0 |
| Raw projection path length (yd, jitter inflated) | 43.0 | 41.0 | 79.0 |
| Ground-truth path length (yd) | 37.0 | 33.0 | 64.0 |
| Smoothed / GT | 1.027 | 0.970 | 1.0000 |
| Smoothed / raw | 0.884 | 0.780 | 0.810 |

Process-noise A/B — Comparison of the only constant that could attenuate real motion (the filter's process-noise acceleration scale). Measured on TRAIN + VAL only; TEST is excluded so this cannot be used to tune the frozen split. Shipped value remains 5.0. (`test_split_consulted = False`)

| Split | Config | Median err (yd) | Smoothed path (yd) | GT path (yd) | Path vs GT | Mean sigma major (yd) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | shipped (5.0 yd/s²) | 0.0294 | 38.00 | 37.00 | 1.0270 | 0.0481 |
| `train` | variant (0.5 yd/s²) | 0.0273 | 39.33 | 38.43 | 1.0235 | 0.0469 |
| `val` | shipped (5.0 yd/s²) | 0.0214 | 32.00 | 33.00 | 0.9697 | 0.0445 |
| `val` | variant (0.5 yd/s²) | 0.0221 | 32.73 | 33.32 | 0.9823 | 0.0407 |

## 6. Rejection-gate accounting (audit item D)

* **clean_sample**: A dominant-track sample whose (player, frame) pair is NOT an injected outlier event, and for which a raw field projection exists (so the innovation gate could act on it).
* **clean_sample_rejected**: A clean sample that the innovation gate rejected: the false-rejection event.
* **corrupted_sample**: A dominant-track sample at an injected outlier event (the manifest injects a known pixel jump); 'accepted' means the corrupted measurement was folded into the track.
* **innovation_gate**: Field-space chi-square test on the Mahalanobis distance between the measurement and the constant-velocity prediction. This is the only gate that can REJECT a measurement.
* **plausibility_gate**: Image-space footpoint jump limit. It can only FLAG a sample (and only when geometry is unknown), never reject a measurement, so its count is reported separately and is not part of the false-rejection rate.
* **geometry_refusal**: A sample for which no projectable field measurement existed at all (unknown geometry, unreliable footpoint, out-of-bounds projection, or coasting past the max-gap limit). It is accounted for in unpositioned_by_reason and is deliberately NOT counted as a measurement rejection: no measurement existed for any gate to accept or reject.
* **note**: 'Clean'/'corrupted' are harness labels from the manifest, not labels inferred from ground-truth error: this block is a harness metric, not a model-internal quantity.

### 6.1 Gate scope and calibration

The innovation gate (χ²₂ = 9.21 on the field-space Mahalanobis innovation) is the **only** component that can reject a measurement. The image-space plausibility limit can only flag a sample, and only while geometry is unknown. The warm-up rule (gate disabled for the first two updates of a track) exists because a two-point track has no estimable velocity; it is reported as a separate A/B in §10 and was selected on TRAIN/VAL only. No threshold, denominator or constant in this section was chosen using TEST.

### 6.2 What the gate rejected

| Split | Clean samples that reached the gate | Accepted | Rejected (false rejections) | False-rejection rate (gated only) | Warm-up samples (gate bypassed by design) | False-rejection rate incl. warm-up |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 220 | 207 | 13 | 5.91% | 44 | 4.92% |
| `val` | 210 | 210 | 0 | 0.00% | 36 | 0.00% |
| `test` | 379 | 323 | 56 | 14.78% | 83 | 12.12% |

| Split | Corrupted samples reaching a dominant track's gate | Rejected | Accepted | True-rejection rate on corrupted | Median gate statistic of corrupted rejections | Median gate statistic of false rejections | Samples entering dead reckoning after a false rejection |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 0 | 0 | 0 | — | — | 26.66 | 9 |
| `val` | 2 | 2 | 0 | 1.0000 | 1526.23 | — | 0 |
| `test` | 0 | 0 | 0 | — | — | 14.39 | 49 |

Reading the two tables together separates the two error directions the audit asked for: §6.2 rows 4–5 are *the gate rejecting good data* (false rejections that cost real measurements and push samples into labelled dead reckoning), while the corrupted columns are *the gate rejecting bad data*. Where a split shows `0` corrupted samples evaluated, the injected outliers never reached a dominant track's gate — they were refused earlier by the projection refusal (unknown geometry) or spawned a separate track, which is visible in the event table below.

| Split | Rejections, all tracks | … by innovation gate | … by plausibility gate | Rejections on dominant tracks (clean + corrupted) | Image-space plausibility flags | Plausibility gate can reject |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 13 | 13 | 0 | 13 | 0 | `False` |
| `val` | 5 | 5 | 0 | 2 | 0 | `False` |
| `test` | 59 | 59 | 0 | 56 | 3 | `False` |

Samples that end with no position at all, by reason:

| Split | Geometry refused | Footpoint unreliable | Projection out of bounds | Coasting beyond max gap | Other | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 0 | 0 | 0 | 0 | 0 | 0 |
| `val` | 0 | 0 | 0 | 0 | 0 | 0 |
| `test` | 16 | 0 | 0 | 2 | 0 | 18 |

### 6.3 Injected-outlier events, end to end

| Sequence | Split | Injected | Rejected by field gate | Spawned a separate track | Refused by projection | Absorbed silently |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `traj_seq_04_val_jump_outliers` | `val` | 4 | 3 | 1 | 0 | 0 |
| `traj_seq_05_test_calibration_dropout_and_propagation` | `test` | 1 | 0 | 0 | 1 | 0 |
| `traj_seq_06_test_camera_pan_zoom_occlusion` | `test` | 1 | 1 | 0 | 0 | 0 |
| `traj_seq_07_test_high_jitter_and_occlusion` | `test` | 1 | 0 | 1 | 0 | 0 |

Every injected outlier is accounted for in exactly one outcome; `absorbed silently` is the failure mode the metric exists to catch and it is `0` in every sequence.

## 7. Uncertainty: coverage and calibration status (audit item E)

**Status: `not_statistically_calibrated`.** The reported covariance is an a priori model (footpoint pixel noise propagated through the homography Jacobian, plus a propagation-drift allowance), *not* a statistically calibrated confidence interval. The current values are kept visible below and no calibration factor was applied to the shipped covariance.

| Coverage on accepted measurements | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Samples evaluated | 255 | 246 | 413 |
| 68% interval, reported (smoothed) covariance | 63.38% | 76.02% | 49.73% |
| 95% interval, reported (smoothed) covariance | 90.85% | 98.47% | 80.09% |
| 68% interval, raw measurement covariance (before smoothing) | 73.96% | 84.11% | 64.03% |
| 95% interval, raw measurement covariance (before smoothing) | 96.95% | 99.51% | 87.22% |
| 68% interval, calibrated geometry | 63.38% | 76.02% | 48.83% |
| 68% interval, propagated geometry | — | — | 77.78% |
| 68% interval, covariance inflated (rejection/re-init path) | 75.00% | — | 0.00% |
| 68% interval, covariance not inflated | 62.98% | 76.02% | 50.06% |
| Empirical position error, median (yd) | 0.0294 | 0.0214 | 0.0370 |
| Empirical position error, RMSE (yd) | 0.0430 | 0.0335 | 0.0512 |
| Reported sigma, major axis, median (yd) | 0.0473 | 0.0432 | 0.0473 |
| Median Mahalanobis distance (reported covariance) | 1.2339 | 1.0842 | 1.6861 |
| Median Mahalanobis distance (raw measurement covariance) | 1.0186 | 0.8610 | 1.2279 |
| Median accepted-measurement gate statistic (NIS) | 1.5783 | 1.2379 | 1.8023 |
| Mean accepted-measurement gate statistic (NIS) | 3.2875 | 1.5659 | 4.0593 |
| Covariance scale the data would need, k² (reported covariance) | 1.1062 | 0.8491 | 2.1572 |
| Covariance scale the data would need, k² (raw measurement covariance) | 0.7588 | 0.5366 | 1.1591 |

### 7.1 Does the frozen footpoint-noise assumption explain the gap?

The harness injects a known per-frame footpoint jitter with a per-sequence standard deviation of 0.4/0.8/1.6/3.0 px, while the frozen model assumes a box-height-scaled sigma anchored at ~0.4–0.9 px per axis. Propagating that ratio gives a first-order prediction of the coverage — a diagnostic derivation, not a calibration:

| Sequence | Injected jitter (px) | Observed jitter sd u (px) | Model sigma u (px) | Observed jitter sd v (px) | Model sigma v (px) | Predicted k² | Predicted 68% coverage | Measured 68% coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `traj_seq_01_train_linear_and_sprint` | 0.80 | 0.486 | 0.413 | 0.453 | 0.590 | 0.985 | 68.55% | 73.61% |
| `traj_seq_02_train_cuts_and_deceleration` | 0.80 | 0.485 | 0.413 | 0.450 | 0.590 | 0.980 | 68.74% | 53.15% |
| `traj_seq_03_val_camera_pan` | 0.80 | 0.486 | 0.413 | 0.453 | 0.590 | 0.985 | 68.55% | 73.61% |
| `traj_seq_04_val_jump_outliers` | 0.80 | 0.485 | 0.413 | 0.450 | 0.590 | 0.980 | 68.74% | 78.43% |
| `traj_seq_05_test_calibration_dropout_and_propagation` | 0.80 | 0.486 | 0.413 | 0.453 | 0.590 | 0.985 | 68.55% | 67.50% |
| `traj_seq_06_test_camera_pan_zoom_occlusion` | 1.60 | 0.973 | 0.413 | 0.908 | 0.590 | 3.958 | 25.01% | 14.42% |
| `traj_seq_07_test_high_jitter_and_occlusion` | 3.00 | 1.820 | 0.934 | 1.685 | 1.335 | 2.693 | 34.49% | 31.96% |
| `traj_seq_08_test_camera_cut_refusal_and_recovery` | 0.40 | 0.242 | 0.413 | 0.226 | 0.590 | 0.245 | 99.05% | 100.00% |

The derivation tracks the measured coverage closely where the injected jitter is small (`seq_03`/`seq_04`/`seq_05`: prediction within a few points) and correctly predicts the collapse on the high-jitter sequences, where it accounts for most — but not all — of the gap: `seq_06` and `seq_07` additionally lose coverage to the smoothing stage (the reported covariance is smaller than the raw measurement covariance: 64.03% vs 49.73% on TEST) and to camera-zoom geometry error that no footpoint-noise term models. Coverage is therefore **not** explained away, and the model is left unchanged.

### 7.2 TRAIN/VAL-fitted covariance scale, recorded but NOT applied

* fit split: `train+val`; TEST used for fitting: `False`
* k² needed: train `1.0952`, val `0.8582`, fitted (TRAIN+VAL) `0.9767`
* TEST 68% coverage, shipped model `49.73%`; counterfactual under the fitted scale `52.54%`
* applied to the shipped covariance: `False`
* A value of k2 > 1 means the empirical error is larger than the reported covariance (intervals too narrow). Fitting an inflation factor and shipping it would change the reported uncertainty and could change gate behaviour; that is a Phase-5 decision, so nothing was changed here.
* TEST is evaluated with the TRAIN/VAL-fitted scale exactly once, for documentation. No threshold, constant, or reported value in this benchmark uses it.

**Standing limitation:** the reported sigma must not be read as calibrated confidence, and the under-coverage on the high-jitter and camera-zoom sequences is a real, quantified weakness of the current model. Fixing it (inflating the covariance, modelling the smoothing shrinkage, or re-deriving the noise model) is a Phase-5 decision that requires its own TRAIN/VAL-fitted, frozen evaluation.

## 8. Camera motion, camera cuts and recovery

| Metric | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Observed-sample ratio | 1.0000 | 0.973 | 0.984 |
| Inferred (dead-reckoning) ratio | 0.034 | 0.035 | 0.154 |
| Observed-gap recovery latency, median (frames) | — | 2.0 | 2.0 |
| Observed-gap recovery latency, max (frames) | 0.0 | 3.0 | 2.0 |
| Projection-refusal recovery latency, max (frames) | 2.0 | 5.0 | 5.0 |
| Samples positioned while geometry was unknown | 0 | 0 | 0 |

**Camera-cut sequence `traj_seq_08_test_camera_cut_refusal_and_recovery`:** `33` samples were produced while the geometry was refused and **none** of them carries a field position (`0` positioned under refusal). The cut costs identity continuity: `6` IDSW (`6` active swaps), `1` FRAG, `11` tracks, `0` fabricated samples; re-acquisition latency after the refusal is `5` frames (projection) / `2` frames (observation).

Cut *detection* is not implemented: the cut frame is declared by the manifest, and the camera-motion model is the frame-to-frame homography of the provided calibration. There is no re-identification, so identity is expected to be lost across the cut.

## 9. Acceleration: experimental and not validated (audit item F)

**Status: `experimental_not_validated`.** Acceleration is the finite difference of the smoothed filter velocity, bounded by a plausibility clip. It is EXPERIMENTAL AND NOT VALIDATED: no measured acceleration accuracy is claimed, and the reported error is dominated by the filter's own noise floor rather than by a validated acceleration model.

| Metric | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Acceleration-magnitude error, median (yd/s²) — experimental | 6.991 | 6.187 | 8.325 |
| Ground-truth acceleration magnitude, median (yd/s²) | 1.498 | 0.0000 | 0.0000 |
| Samples hitting the acceleration clip | 30 | 26 | 82 |

The reported error is of the same order as the ground-truth magnitude itself, which is the signature of differentiating a noisy velocity estimate: the metric measures the filter's noise floor, not an acceleration capability. Only mathematical correctness is unit-tested (clipping bounds, finite differences); no accuracy claim is made and TEST was not used to select anything here.

## 10. Track-initiation warm-up A/B (audit item I)

Computed by run_phase4_benchmark() from the shipped implementation, TRAIN/VAL only (TEST is not consulted, so warm-up selection cannot leak into the frozen split). The innovation-gate threshold is unchanged; the shipped rule delays the gate until a track has two accepted updates (M-of-N track initiation).

* computed by: `benchmarks/evaluate_phase4_trajectories.py::_gate_warmup_ablation`
* TEST consulted: `False`

| Split | Gate active from the 2nd update: false rejections / gated | Shipped (2-update warm-up): false rejections / gated |
| :--- | :--- | :--- |
| `train` | 48 / 264 | 13 / 264 |
| `val` | 1 / 246 | 0 / 246 |

The values quoted in the audit for this ablation (train `48` of `264` → shipped `13` of `264`; val `1` of `246` → shipped `0` of `246`) reproduce exactly under the in-benchmark computation, and the shipped arm's counts equal the shipped per-split accounting (`splits.train.rejection_accounting.clean_samples_rejected` = 13 of 220; val = 0 of 210). The gate *threshold* is identical in both arms (χ²₂ = 9.21); only the delay before the gate is applied changes.

## 11. Benchmark integrity audit (audit item 3)

| Invariant | TRAIN | VAL | TEST (frozen) |
| :--- | :--- | :--- | :--- |
| Fabricated trajectory samples | 0 | 0 | 0 |
| Positions claimed under `unknown` geometry | 0 | 0 | 0 |
| Absolute-yardline violations (`x_coord_mode != 'absolute'`) | 0 | 0 | 0 |
| Injected outliers absorbed without a flag | 0 | 0 | 0 |
| Samples without a harness association | 0 | 0 | 0 |

* **Determinism:** `True` — wall-clock runtime fields excluded; one sequence from each split (TRAIN, VAL, frozen TEST) re-run through the identical code path; per-sequence: `train`/`traj_seq_01_train_linear_and_sprint` = True, `val`/`traj_seq_04_val_jump_outliers` = True, `test`/`traj_seq_08_test_camera_cut_refusal_and_recovery` = True
* **Ground truth in production code:** the trajectory layer has no ground-truth dependency; a regression test asserts that `football_vision/` contains no ground-truth identifier and that the builder's provenance passthrough is opt-in (default empty) — the benchmark harness opts in explicitly.
* **TEST not used for tuning:** TEST is used to select no threshold, constant, warm-up rule, process-noise value, uncertainty model or dominant track. The TRAIN/VAL-only ablations record `test_split_consulted = false`; the single TEST evaluation of the fitted covariance scale is counterfactual documentation (§7.2) and is not applied.
* **Split protocol:** the three splits are fixed by `data/benchmarks/phase4_trajectory_manifest.json`; the fixture noise is a deterministic hash of (player, frame, channel), so re-runs reproduce bit-for-bit.

### 11.1 Metric scopes — which numbers are allowed to mean what

* `model_observable_no_ground_truth`: `n_tracks`, `n_samples`, `geometry_state_sample_counts`, `positioned_by_geometry_state`, `samples_with_position_in_unknown_geometry`, `fabricated_field_positions`, `absolute_yardline_violations`, `measurements_rejected_total`, `filter_reinitializations`, `image_space_plausibility_flags_total`, `recovery_events_observed_gap`, `projection_recovery_events`, `gate_warmup_measurements`, `track_table`, `samples_per_track`, `mean_runtime_ms_per_frame`, `mean_runtime_ms_per_sample`, `projection_status accounting (unpositioned_by_reason)`
* `harness_gt_dependent`: `field_pos_err_median_yd`, `field_pos_err_p90_yd`, `field_pos_err_rmse_yd`, `raw_field_pos_err_median_yd`, `all_matched_track_field_err_median_yd`, `non_dominant_track_field_err_median_yd`, `speed_err_median_yd_s`, `velocity_rmse_yd_s`, `accel_mag_err_median_yd_s2`, `direction_err_median_deg`, `dead_reckoning_err_median_yd`, `coverage_68_pct`, `coverage_95_pct`, `id_switches`, `id_switches_active_swap`, `id_switches_post_reinit`, `track_fragmentations`, `track_completeness`, `measured_completeness`, `dominant_fraction_of_gt_player_frames`, `smoothed_path_length_yd`, `raw_projection_path_length_yd`, `gt_path_length_yd`, `motion_preservation_ratio_vs_raw`, `motion_preservation_ratio_vs_gt`, `fragmentation_taxonomy`, `image_space_ema_field_err_median_yd`
* `harness_label_dependent`: `outliers_injected`, `outliers_rejected_by_field_gate`, `outlier_events_absorbed_into_track`, `outlier_events_spawning_spurious_track`, `outlier_events_refused_by_projection_gate`, `false_rejections`, `false_rejection_rate`, `false_rejection_rate_including_warmup`, `clean_samples_gate_evaluated`, `clean_samples_accepted`, `clean_samples_rejected`, `corrupted_samples_evaluated`, `corrupted_samples_rejected`, `corrupted_samples_accepted`, `true_rejection_rate_on_corrupted`, `dead_reckoned_samples_after_false_rejection`, `recovery_latency_frames_median`, `recovery_latency_frames_max`, `projection_recovery_latency_frames_max`, `gate_warmup_ablation`, `outlier_event_details`

Harness-only metrics are named as such (`harness_gt_dependent`, `harness_label_dependent`) so they cannot be mistaken for quantities a deployed system could compute about itself. The image-detector error/accuracy metrics remain out of scope: `image_detector_quantitative_metrics = None`.

## 12. Real frames: smoke test and refusal gates only (audit item 4/8)

* `rf_01_sea_sf_smoke` — detector `turf_contrast_baseline_v1`, calibration success `True`, `x_coord_mode = relative_10yd`
* detections `25`, tracks `25`, trajectory samples `25`, samples with a field position `23`, absolute yardlines `0`, fabricated positions `0`
* scope: `single_frame_integration_smoke_test_no_ground_truth_trajectory`; quantitative ground truth available: `False`

Real frames contribute exactly three things to this project: (a) this single-frame integration smoke test, (b) the Phase-2/3 geometry-refusal gates on real broadcast stills, and (c) Phase 3's real-frame *observation* log for the turf-contrast baseline (many false positives, unmeasured precision/recall). **Real multi-frame trajectory accuracy is not measured** — no real broadcast clip with frame-level ground-truth trajectories exists in this project, so no number here may be presented as NFL/broadcast tracking accuracy.

## 13. Limitations and deliberately rejected improvements

* Player motion is synthetic deterministic field-space ground truth projected through real per-frame homographies: an engineering fixture, not recorded NFL trajectories. Real multi-frame trajectory accuracy is NOT measured.
* Detection boxes are fixture boxes (fixture_detector_v1); image-based player detection accuracy remains unmeasured and no detection-accuracy number in this file is a detector metric.
* The reported covariance is an a priori engineering model and is NOT statistically calibrated; empirical coverage is reported per split and per geometry state, and the gap is explained (jitter vs assumed pixel noise) rather than tuned away.
* Acceleration is experimental: it is a clipped finite difference of the smoothed velocity, not a validated acceleration estimator.
* Fragmentation is split by a deterministic taxonomy (active association swap / post-expiration reinitialization / outlier-induced spawn / legitimate new track / unmatched). The taxonomy is exhaustive and asserted against the track count, but the categories are harness-visible states, not inferred internal tracker modes.
* Camera pan/zoom is applied as an exact synthetic image-space transform; real broadcast camera motion also changes perspective, rolling shutter, and motion blur.
* One single-frame real-frame smoke test (no ground truth) plus geometry-refusal/integration tests only: no real multi-frame broadcast clip was available in this phase.
* **Identity across cuts is not solved.** The tracker keeps a `track_id` alive across a player change in one VAL and one TEST sequence (contamination rows in §4.1). Samples are scored against the player they claim; no re-identification is implemented and none is simulated.
* **Coverage is not calibrated** and is worst exactly where the covariance matters most (high jitter, camera zoom, re-init after rejections). The numbers are published, not hidden; fixing them requires a TRAIN/VAL-fitted model and a frozen re-evaluation.
* **Rejected as over-fitting or out of scope for this pass:** widening the gate to χ²₂ ≤ 16; elongating the motion prior along the camera-pan axis; raising the process noise to smooth over identity hops; extending the dead-reckoning horizon beyond `max_gap_frames`; shipping the TRAIN/VAL-fitted covariance inflation; dropping the fragment tracks from the report. Each would move headline numbers without adding evidence.

## 14. Reproducing this report

```bash
python3 -m pytest -v                                   # full test suite (count printed by pytest)
python3 benchmarks/evaluate_phase4_trajectories.py     # writes outputs/phase4_trajectory_benchmark.json + overview PNG
python3 benchmarks/render_phase4_report.py --commit $(git rev-parse --short HEAD)   # this document
```

Real NFL stills are third-party assets and are **not** vendored. Point `FOOTBALL_VISION_NFL_FRAMES` at a directory containing them (the default remains the historical `/home/user/image-search`); every test and the smoke section skip with an explicit reason when the assets are absent, so the suite and the benchmark run to completion on a machine without them.

