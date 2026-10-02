# Phase 4 — Quantitative Audit (tracking, fragmentation, gate, uncertainty)

Generated from `outputs/phase4_trajectory_benchmark.json` by `benchmarks/render_phase4_audit.py`; every number below is read from that JSON, so this document cannot drift from the benchmark. Quantitative audit of commit `quantitative audit pass (parent 3a2c076)`. Phase 4 is **under audit, not final**; Phase 0/1/2/3 are frozen and Phase 5 has not started.

* Phase-4 constants are frozen a priori: `camera_cut_refusal_frames = 3`, `footpoint_edge_sigma_px = 1.0`, `footpoint_sigma_floor_px = 0.3`, `gate_chi2_2dof = 9.21`, `image_jump_max_speed_px_per_frame = 22.0`, `max_accel_yd_s2 = 25.0`, `max_consecutive_rejections = 3`, `max_gap_frames = 3`, `max_speed_yd_s = 12.0`, `process_accel_std_yd_s2 = 5.0`, `propagated_drift_yd_per_age = 0.25`.
* Labels used in this document: **measured fact** (a frozen-benchmark number, reproducible from the committed JSON), **methodological limitation** (a property of how the number was obtained), **unmeasured quantity** (no number exists).

## 1. Dominant-track selection rule (measured fact)

* **Rule:** `lifetime_sample_count_then_lowest_track_id`
* **Inputs:** `track_id`, `sample lifetime (number of trajectory samples emitted for the track)`
* **Tie-break:** lowest track_id (deterministic; independent of dictionary/iteration order)
* **Ground truth in the rule:** trajectory `False`, error `False`, oracle selection `False`, future information `False`.
* Ground truth is used ONLY to attribute a track to the player it is scored against (an oracle association inherent to every labelled MOT benchmark: an error cannot be computed without knowing which player a track is compared to). That association is computed before, and never influences, the selection itself.

**Selection happens before any error is computed.** In `benchmarks/evaluate_phase4_trajectories.py` the sequence of operations inside `_score_samples` is fixed: per-track runtime counters are accumulated (`_track_runtime_stats`), `select_dominant_track` is called for every player, and only then does the sample loop compute ground-truth error. The selection function's arguments are counter dictionaries; it has no access to coordinates, errors, or the ground-truth table.

Proven by tests (`pytest`):

* `test_dominant_track_selection_uses_runtime_counters_only` — poisons the candidate dicts with a perfect error on a losing track and a catastrophic error on the winner; the selection does not move.
* `test_dominant_track_rule_has_no_ground_truth_term` — AST scan of the rule's code (docstring excluded) for any ground-truth/error identifier.
* `test_benchmark_dominant_track_is_reproducible_from_published_counters` — every published dominant track is re-derived from the published counters alone.

Rule sensitivity (measured fact, diagnostic only — the shipped rule is not replaced by a better-looking one):

| Split | Alternative runtime-only rule | Same pick as shipped | Median err under it (yd) | Shipped median err (yd) |
| :--- | :--- | :--- | :--- | :--- |
| `train` | `longest_observed_run_then_lowest_track_id` | 12/12 | 0.0294 (median of 2 sequence medians) | 0.0294 |
| `train` | `positioned_sample_count_then_lowest_track_id` | 12/12 | 0.0294 (median of 2 sequence medians) | 0.0294 |
| `val` | `longest_observed_run_then_lowest_track_id` | 12/12 | 0.0214 (median of 2 sequence medians) | 0.0214 |
| `val` | `positioned_sample_count_then_lowest_track_id` | 12/12 | 0.0214 (median of 2 sequence medians) | 0.0214 |
| `test` | `longest_observed_run_then_lowest_track_id` | 24/24 | 0.0370 (median of 4 sequence medians) | 0.0370 |
| `test` | `positioned_sample_count_then_lowest_track_id` | 24/24 | 0.0370 (median of 4 sequence medians) | 0.0370 |

## 2. Dominant track vs all observed tracks vs secondary fragments (measured fact)

| Split | GT players | Tracker tracks | Tracks per player | Dominant-track samples | Secondary-fragment samples | Unattributed | GT player-time covered by dominant | Emitted samples covered by all matched tracks | Excluded from dominant scoring |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 12 | 12 | 1.00 avg / 1 max | 264 across 12 tracks | 0 samples / 0 tracks | 0 | 264/264 = 100.00% | 264/264 = 100.00% | 0.00% (0/264) |
| `val` | 12 | 15 | 1.25 avg / 3 max | 255 across 12 tracks | 22 samples / 4 tracks | 0 | 255/264 = 96.59% | 277/277 = 100.00% | 7.94% (22/277) |
| `test` | 24 | 32 | 1.33 avg / 2 max | 504 across 24 tracks | 73 samples / 9 tracks | 0 | 488/552 = 88.41% | 577/577 = 100.00% | 12.65% (73/577) |

**Accounting identity (asserted in the harness, measured fact):** `dominant_track_samples_total + non_dominant_track_samples_total + unattributed_samples_total == n_samples` holds per sequence and per split, so no sample can leave the report by being fragment-only.

| Split | Dominant-track error, median (yd) | Dominant p90 (yd) | All matched tracks, median (yd) | Secondary fragments, median (yd) |
| :--- | :--- | :--- | :--- | :--- |
| `train` | 0.0294 (n=255) | 0.0699 | 0.0294 (n=255) | — (n=0 positioned of 0 samples) |
| `val` | 0.0214 (n=246) | 0.0553 | 0.0225 (n=260) | 0.0342 (n=14 positioned of 22 samples) |
| `test` | 0.0370 (n=413) | 0.0795 | 0.0371 (n=457) | 0.0616 (n=44 positioned of 73 samples) |

Reading (measured fact): the secondary-fragment error is worse than the dominant track on every split where fragments carry a position, and they are reported rather than dropped. On `test` the fragments' median error is 0.0616 yd against 0.0370 yd for the dominant track, while the *pooled* all-matched median is 0.0371 yd — i.e. the pooled figure barely moves because the fragments contribute few positioned samples, which is exactly why the fragment column is reported separately instead of being left implicit.

## 3. Fragmentation accounting (measured fact)

CLEAR-MOTA lifetime semantics are preserved: `IDSW` counts an observed sample whose `track_id` differs from the last observed `track_id` for that player; `FRAG` counts a player that was not observed at the previous frame and is observed again. The categories below are an additional, orthogonal taxonomy over (player, track) episodes; they do not redefine IDSW.

| Split | `active_association_swap` | `legitimate_new_track` | `outlier_induced_spawn` | `post_expiration_reinitialization` | `unmatched_spurious_track` | Total episodes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 0 (0.00%) | 12 (100.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 12 |
| `val` | 3 (18.75%) | 12 (75.00%) | 1 (6.25%) | 0 (0.00%) | 0 (0.00%) | 16 |
| `test` | 8 (24.24%) | 24 (72.73%) | 1 (3.03%) | 0 (0.00%) | 0 (0.00%) | 33 |

| Split | IDSW (lifetime) | … active swap | … post-expiration re-init | FRAG | Secondary fragments excluded | Max tracks per player | Tracks with 2 player labels | GT trajectory on the dominant track |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 264/264 = 100.00% |
| `val` | 5 | 5 | 0 | 0 | 4 | 3 | 1 | 255/264 = 96.59% |
| `test` | 10 | 10 | 0 | 4 | 9 | 2 | 1 | 488/552 = 88.41% |

* **Average fragments per player (measured fact):** `train` 12 episodes / 12 players = 1.000, `val` 16 episodes / 12 players = 1.333, `test` 33 episodes / 24 players = 1.375 (episodes, not tracks: a track that survives a player change is counted once per player)
* **Methodological limitation:** the taxonomy categories are harness-visible states (association behaviour plus injected-event labels), not inferred internal tracker modes.

## 4. Rejection-gate accounting (measured fact; gate unchanged)

| Split | Clean gated | Clean accepted | Clean rejected (false rej.) | False-rejection rate | … incl. warm-up denominator | Corrupted evaluated | Corrupted rejected | Corrupted accepted | True-rejection rate | False-acceptance rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 220 | 207 | 13 | 5.91% | 4.92% | 0 | 0 | 0 | — | — |
| `val` | 210 | 210 | 0 | 0.00% | 0.00% | 2 | 2 | 0 | 1.0000 | 0.0000 |
| `test` | 379 | 323 | 56 | 14.78% | 12.12% | 0 | 0 | 0 | — | — |

**Explicit denominators** (from the JSON `denominators` block, so no rate is quoted bare):

* `false_acceptance_rate_on_corrupted` — corrupted_samples_evaluated (same denominator as true_rejection_rate_on_corrupted)
* `false_rejection_rate` — clean_samples_gate_evaluated (clean samples the gate actually evaluated)
* `false_rejection_rate_including_warmup` — clean_samples_gate_evaluated + clean_samples_warmup_bypassed
* `geometry_refusal` — not reported as a rate: unpositioned_by_reason counts are absolute sample counts over the split's emitted samples
* `true_rejection_rate_on_corrupted` — corrupted_samples_evaluated (dominant-track injected events with a raw projection)

| Split | Innovation-gate rejections | Plausibility-gate rejections | Geometry refusals | Footpoint refusals | Out-of-bounds refusals | Coasting past max gap | Unpositioned total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| `val` | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `test` | 59 | 0 | 16 | 0 | 0 | 2 | 18 |

* Gate attribution is reported in two explicit scopes: all tracks vs dominant tracks (`rejections_all_tracks`, `rejections_on_dominant_tracks`). The clean/corrupted table above is a dominant-track scope; a rejection on a fragment track is not a clean-sample false rejection.
* The image-space plausibility gate **cannot reject** a measurement in this implementation: it only flags, and only while geometry is unknown (`plausibility_gate_can_reject_measurements = False`). Geometry refusals are not gate rejections either: no measurement existed to accept or reject.
* **Methodological limitation:** 'clean' and 'corrupted' are harness labels derived from the manifest's injected events, not labels inferred from model behaviour.
* Gate threshold (frozen): `gate_chi2_2dof = 9.21`.

## 5. Dead-reckoning impact (measured fact)

| Split | Positioned dominant samples | Accepted-measurement err median / RMSE (yd) | Of which re-init after rejection (median yd) | Post-gap re-init samples | Dead-reckoned from rejection | … its err median / p90 (yd) | Dead-reckoned from missing measurement | … its err median / p90 (yd) | GT error of the falsely rejected measurements (yd) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 255 | 0.0294 / 0.0440 | 4 (0.0670 yd) | 0 | 9 | 0.1691 / 0.2361 | 0 | — / — | 0.0638 |
| `val` | 246 | 0.0214 / 0.0335 | 0 (— yd) | 0 | 2 | 0.0593 / 0.0810 | 7 | 0.0455 / 0.1334 | — |
| `test` | 413 | 0.0370 / 0.0566 | 7 (0.1038 yd) | 0 | 49 | 0.1037 / 0.2485 | 26 | 0.1006 / 0.2111 | 0.0856 |

**Partition identity (asserted in the harness):** `dead_reckoning_samples_from_rejection + dead_reckoning_samples_from_missing_measurement == dead_reckoning_samples`, and every positioned dominant sample is either accepted (including the re-init-after-rejection sub-arm) or produced by the post-gap re-initialization path. This is why the two dead-reckoning causes can be compared without cherry-picking: they partition the whole dead-reckoned population.

* **Measured fact — which cause dominates:** `train` 9 of 9 dead-reckoned samples (100.00%) came from a rejected measurement, `val` 2 of 9 dead-reckoned samples (22.22%) came from a rejected measurement, `test` 49 of 75 dead-reckoned samples (65.33%) came from a rejected measurement. Rejected-measurement dead reckoning therefore dominates on `train` and `test`, while on `val` the two rejected samples are corruption events rather than false rejections of clean data (that split has zero clean false rejections).
* **Methodological limitation:** the post-gap re-initialization arm is defined in the code path but contains `0` positioned samples on `test` (`0` train / `0` val): the branch is not exercised by the frozen benchmark, so no claim is made about it. Re-initializations after persistent rejections DO occur and are reported separately in the column above.
* The falsely rejected measurements are geometrically *close* to the truth (column above): the gate is rejecting good jittered measurements, which is the false-rejection failure mode, not a quality gain.

## 6. Uncertainty diagnostics (measured fact; parameters unchanged)

**The uncertainty estimate is not statistically calibrated.** The reported covariance is an a priori engineering model (footpoint pixel noise propagated through the homography Jacobian, plus a propagation-drift allowance). It is published here so its coverage can be judged; it must not be described as a calibrated confidence interval. No parameter was retuned in this pass.

| Split | Accepted samples with coverage evaluated | 68% coverage (reported cov.) | 95% coverage (reported cov.) | 68% coverage (raw measurement cov.) | Empirical position err median / RMSE (yd) | Reported sigma major, median (yd) | Median Mahalanobis distance | Mean gate statistic (accepted clean) | k² needed for nominal coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 255 | 63.38% | 90.85% | 73.96% | 0.0294 / 0.0430 | 0.0473 | 1.2339 | 3.2875 | 1.1062 |
| `val` | 246 | 76.02% | 98.47% | 84.11% | 0.0214 / 0.0335 | 0.0432 | 1.0842 | 1.5659 | 0.8491 |
| `test` | 413 | 49.73% | 80.09% | 64.03% | 0.0370 / 0.0512 | 0.0473 | 1.6861 | 4.0593 | 2.1572 |

| Split | 68% coverage by geometry state | Coverage when covariance was inflated | Coverage when it was not inflated | Before vs after the smoothing stage (68%, raw vs reported cov.) |
| :--- | :--- | :--- | :--- | :--- |
| `train` | calibrated 63.38% (n=255) / propagated —% (n=0) | inflated 75.00% (n=4) | not inflated 62.98% (n=251) | 73.96% vs 63.38% |
| `val` | calibrated 76.02% (n=246) / propagated —% (n=0) | inflated —% (n=0) | not inflated 76.02% (n=246) | 84.11% vs 76.02% |
| `test` | calibrated 48.83% (n=395) / propagated 77.78% (n=18) | inflated 0.00% (n=7) | not inflated 50.06% (n=406) | 64.03% vs 49.73% |

* **Measured fact — before/after covariance inflation.** The JSON exposes the *inflation stage* directly, so both arms are reported without reconstructing anything: the `inflated` arm is the rejection/re-init path (covariance multiplied before being reported), the `not_inflated` arm is the ordinary path. On `test` the inflated arm is small; on `train` it is where the false rejections land. No before/after number was manufactured where the code does not expose the two states.
* **Measured fact — before/after smoothing.** The raw measurement covariance (before the filter/smoothing stage shrinks it) is carried per sample as `measurement_covariance_xy`, so the last column is a genuine before/after pair, not an estimate.

* **TRAIN/VAL-fitted scale, recorded and NOT applied:** k² train `1.0952`, val `0.8582`, fit `0.9767`; `applied_to_shipped_covariance = False`; `test_split_used_for_fitting = False`.
* **Standing limitation:** A value of k2 > 1 means the empirical error is larger than the reported covariance (intervals too narrow). Fitting an inflation factor and shipping it would change the reported uncertainty and could change gate behaviour; that is a Phase-5 decision, so nothing was changed here.
* **Unmeasured quantity:** there is no dataset with independent, real, per-frame player positions in this project, so coverage on real broadcast tracking is unmeasured. Every coverage number here is measured on synthetic fixture motion projected through real per-frame homographies.

## 7. TEST-integrity verification (code inspection + machine-checked flags)

| Question | Answer | Evidence |
| :--- | :--- | :--- |
| Used to choose thresholds? | No | All constants are published a priori in `frozen_parameters` (11 entries); the manifest records the split protocol; no code path reads `test` to set a value. |
| Used to select dominant tracks? | No | `dominant_track_selection.uses_ground_truth_error = False`; selection is a `max()` over runtime counters computed before scoring. |
| Used for warm-up selection? | No | `gate_warmup_ablation.test_split_consulted = False`; results keys = ['train', 'val']. |
| Used for uncertainty calibration? | No | `uncertainty_calibration_analysis.test_split_used_for_fitting = False`, `applied_to_shipped_covariance = False`. |
| Used to choose between alternative algorithms? | No | The two alternative dominant-track rules are marked `selection_rule_is_diagnostic_only = true` and no shipped number uses them; the process-noise A/B records `test_split_consulted = False`. |

**Where TRAIN/VAL are used for any parameter selection:** nowhere in this benchmark. The constants were fixed a priori (documented engineering assumptions) and the two ablations exist to *describe* sensitivity on TRAIN+VAL; neither ships a value. There is no tuning step in the Phase 4 pipeline, so there is nothing for TEST to leak into.

* **Methodological limitation:** the frozen `test` split was extended by one sequence (`traj_seq_08`, camera-cut refusal/recovery) in commit `21ed530`. That is disclosed as a scope decision; it is not a retune, and the pre-extension figures (68% coverage 32.0%, 95% 61.9%, false rejections 53/368) are historical and are not mixed with the current table.

## 8. Reproducibility verification (measured fact)

* **In-benchmark determinism check:** `deterministic = True`, `differing_keys = []` — one sequence per split re-run through the identical code path: `train/traj_seq_01_train_linear_and_sprin` = True, `val/traj_seq_04_val_jump_outliers` = True, `test/camera_cut_refusal_and_recovery` = True.
* **Manual double-run comparison (this pass):** the benchmark was executed twice on the frozen configuration and the two JSON payloads were compared field by field, excluding wall-clock `*_runtime_ms*` fields. Result: every trajectory metric, track count, fragmentation count, rejection count and uncertainty metric was identical; only runtime fields differed. Procedure: `python3 benchmarks/evaluate_phase4_trajectories.py` twice into two files, then compare the payloads with the runtime keys stripped.
* **Methodological limitation:** timing fields (`mean_runtime_ms_per_frame`, `mean_runtime_ms_per_sample`) are wall-clock and vary between runs, so the JSON file is not byte-stable by design; every metric field is.

## 9. Regression results (measured fact)

* `python3 -m pytest -v` — 71 passed.
* `python3 benchmarks/evaluate_phase4_trajectories.py` — exit 0; JSON + overview PNG written.
* `python3 -m ruff check .` — clean (E9 + pyflakes rules, configured in `pyproject.toml`).

| Split | Fabricated field positions | Fabricated trajectory samples | Absolute-yardline violations | Positions under unknown geometry | Outliers absorbed without a flag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 0 | 0 | 0 | 0 | 0 |
| `val` | 0 | 0 | 0 | 0 | 0 |
| `test` | 0 | 0 | 0 | 0 | 0 |

* Phase 0/1 frozen held-out frame values, Phase 2's 34 temporal refusals, and Phase 3's frozen metrics are unchanged in this pass; they are asserted by `tests/test_phase2_benchmark.py`, `tests/test_phase3_perception_tracking.py` and `tests/test_calibration.py`, which all pass.
* **Reporting defect found and corrected in this pass (harness only):** split-level path lengths were integer-truncated by the count-sum loop (`smoothed_path_length_yd` 38 instead of 39.1282 on `train`, 32 instead of 32.8256 on `val`, 64 instead of 65.4905 on `test`; the raw and ground-truth splits were truncated the same way). Path lengths are float quantities, so they are now summed as floats. A field-by-field diff against the previous commit shows exactly 15 changed numeric leaves: the nine path lengths and the six derived ratios (three `vs_raw`, three `vs_gt`), the largest of which moved by 0.0156 (`val` smoothed/GT 0.9697 → 0.9853). No per-sequence value, trajectory error, coverage, rejection count, IDSW/FRAG count or fabrication counter changed.

## 10. Remaining limitations (not fixed in this pass)

* Player motion is synthetic deterministic field-space ground truth projected through real per-frame homographies: an engineering fixture, not recorded NFL trajectories. Real multi-frame trajectory accuracy is NOT measured.
* Detection boxes are fixture boxes (fixture_detector_v1); image-based player detection accuracy remains unmeasured and no detection-accuracy number in this file is a detector metric.
* The reported covariance is an a priori engineering model and is NOT statistically calibrated; empirical coverage is reported per split and per geometry state, and the gap is explained (jitter vs assumed pixel noise) rather than tuned away.
* Acceleration is experimental: it is a clipped finite difference of the smoothed velocity, not a validated acceleration estimator.
* Fragmentation is split by a deterministic taxonomy (active association swap / post-expiration reinitialization / outlier-induced spawn / legitimate new track / unmatched). The taxonomy is exhaustive and asserted against the track count, but the categories are harness-visible states, not inferred internal tracker modes.
* Camera pan/zoom is applied as an exact synthetic image-space transform; real broadcast camera motion also changes perspective, rolling shutter, and motion blur.
* One single-frame real-frame smoke test (no ground truth) plus geometry-refusal/integration tests only: no real multi-frame broadcast clip was available in this phase.

* **Methodological limitation — secondary fragments are not repaired:** the taxonomy counts fragmentation, but there is no track merging, global re-association, or post-hoc identity repair. Fragment tracks are excluded from the dominant metric by construction and reported separately; their error profile (column in §2) is worse than the dominant track's.
* **Unmeasured quantity — secondary-fragment error is reported on very few samples** (train `0`, val `14`, test `44` positioned fragment samples). It is indicative, not a stable estimate; no conclusion should be drawn from it.
* **Methodological limitation — coverage is aggregate-with-small-n in places:** propagated-geometry coverage on `test` rests on `18` samples.
* **Unmeasured quantity — image-detector accuracy and real multi-frame trajectory accuracy.** Detection boxes are fixture boxes; real frames contribute a single-frame smoke test and geometry-refusal gates only.

---

**Status: Phase 4 under audit — not final. Phase 5 not started.**

