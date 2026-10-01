# Football-Vision — Assumptions, Coordinate Modes & Limitations (Phase 1)

This document explicitly records the geometric, photometric, and temporal assumptions of the Phase 1 `Football-Vision` Field Calibration Engine, its confidence and failure-state rules, and the exact validity of downstream metrics under each longitudinal coordinate mode.

---

## 1. Field Geometry & Coordinate Conventions

- **Coordinate System:**
  - $X \in [0, 100]\text{ yd}$ runs longitudinally from goal line to goal line (with end zones at $[-10, 0]\text{ yd}$ and $[100, 110]\text{ yd}$).
  - $Y \in [0, 53.3333]\text{ yd}$ ($[0, 160.0]\text{ ft}$) runs laterally from the near sideline ($Y = 0$) to the far sideline ($Y = 53.3333\text{ yd}$).
- **NFL Rulebook Hash-Mark Centroids (Critical Scale Invariant):**
  - Per NFL Rule 1, Section 2, the inbound lines (inner edges of the hash marks) lie $70\text{ ft }9\text{ in}$ ($70.75\text{ ft}$) from each sideline ($18.50\text{ ft}$ between inner edges), and each 1-yard hash mark extends $2.0\text{ ft}$ outward toward the sideline ($[68.75, 70.75]\text{ ft}$ near; $[89.25, 91.25]\text{ ft}$ far).
  - Because blob/ridge detectors locate the **centroid** of each 2-foot hash mark rather than its inner edge, the true hash-row centers sit at:
    $$Y_{\text{near}} = \frac{69.75}{3} = 23.2500\text{ yd}, \qquad Y_{\text{far}} = \frac{90.25}{3} = 30.0833\text{ yd}$$
    with a center-to-center spacing of $20.50\text{ ft}$ ($6.8333\text{ yd}$). Using $18.50\text{ ft}$ compresses lateral scale by $10.8\%$ (`20.50 / 18.50`) and introduces $1.31\text{–}2.72\text{ yd}$ median error on held-out sidelines and numbers.
- **Planarity Assumption:**
  - The homography $H \in \mathbb{R}^{3\times 3}$ assumes a planar playing surface ($Z = 0$), neglecting mild stadium crown (~6–12 inches from center to sideline) and radial lens distortion.

---

## 2. Explicit Longitudinal Coordinate Modes (`x_coord_mode`) & Valid Metrics

Because NFL yard lines repeat every 5 yards and painted yard numbers repeat every 10 yards (`10, 20, 30, 40, 50, 40, 30, 20, 10`), the calibration engine **never guesses absolute longitudinal field position**. Every `CalibrationResult` reports one of four `x_coord_mode` values:

1. **`"absolute"`:**
   - Assigned **only** when both 10-yard number parity is resolved (`max_score >= 0.05` and `max_score / min_score >= 2.0`) **and** the reference yard line `x_start_yd` has been independently verified (`x_start_verified=True`).
2. **`"relative_10yd"`:**
   - Assigned when 5-yard lines, 1-yard hash rows, and painted 10-yard number boxes are resolved (`detected_ten_yard_parity ∈ {0, 1}`), locking longitudinal coordinates up to a multiple of 10 yards ($X \pmod{10\text{ yd}}$) and lateral coordinates $Y \in [0, 53.3333]\text{ yd}$ absolutely.
3. **`"relative_5yd"`:**
   - Assigned when 5-yard lines and 1-yard hash rows are calibrated, but painted yard numbers are occluded, out of frame, or ambiguous (`detected_ten_yard_parity = None`), locking longitudinal coordinates up to a multiple of 5 yards ($X \pmod{5\text{ yd}}$) and lateral coordinates $Y \in [0, 53.3333]\text{ yd}$ absolutely.
4. **`"uncalibrated"`:**
   - Assigned whenever calibration fails or confidence falls below `MIN_CALIBRATION_CONFIDENCE = 0.45`. Projection is refused (`image_to_field` and `field_to_image` return `None`).

### Metric Validity Matrix by `x_coord_mode` (`VALID_METRICS_BY_X_COORD_MODE`)

| Downstream Metric / Feature | `"absolute"` | `"relative_10yd"` | `"relative_5yd"` | `"uncalibrated"` |
| :--- | :---: | :---: | :---: | :---: |
| Absolute yard line ($X \in [0, 100]\text{ yd}$) | **Valid** | Invalid | Invalid | Invalid |
| Field territory (Own vs. Opponent half) | **Valid** | Invalid | Invalid | Invalid |
| Red-zone indicator ($X \ge 80\text{ yd}$ or $\le 20\text{ yd}$) | **Valid** | Invalid | Invalid | Invalid |
| Distance to goal line | **Valid** | Invalid | Invalid | Invalid |
| Offset to nearest 10-yard line ($X \pmod{10\text{ yd}}$) | **Valid** | **Valid** | Invalid | Invalid |
| Offset to nearest 5-yard line ($X \pmod{5\text{ yd}}$) | **Valid** | **Valid** | **Valid** | Invalid |
| Yards gained / longitudinal displacement ($\Delta X$) | **Valid** | **Valid** | **Valid** | Invalid |
| Absolute lateral coordinate ($Y \in [0, 53.3333]\text{ yd}$) | **Valid** | **Valid** | **Valid** | Invalid |
| Distance to near/far sidelines & hash marks | **Valid** | **Valid** | **Valid** | Invalid |
| Player velocity, speed & acceleration ($v_x, v_y, a$) | **Valid** | **Valid** | **Valid** | Invalid |
| Player distance covered & receiver-DB separation | **Valid** | **Valid** | **Valid** | Invalid |
| Pre-snap formation width, depth, spacing & box count | **Valid** | **Valid** | **Valid** | Invalid |
| Route depth, stem angle & break geometry | **Valid** | **Valid** | **Valid** | Invalid |

---

## 3. Calibration Confidence & Explicit Failure Modes

### Geometric Confidence Score (`compute_calibration_confidence`)
When a frame passes all geometric stages and `plausible_homography()`, confidence $\in [0.0, 1.0]$ is the equal-weighted mean of four in-sample geometric quality components:
- **`line_support`:** $\text{clip}((N_{\text{yard\_lines}} - 2) / 3.0, 0, 1)$
- **`ridge_residual`:** $\text{clip}(1 - \text{ridge\_orth\_median\_px} / 4.0, 0, 1)$
- **`hash_support`:** $\text{clip}(N_{\text{hash\_inliers}} / 20.0, 0, 1)$
- **`hash_residual`:** $\text{clip}(1 - \text{hash\_row\_rmse\_px} / 4.0, 0, 1)$

If `confidence < MIN_CALIBRATION_CONFIDENCE` (`0.45`), calibration is rejected (`success=False`, `H=None`, `failure_reason="insufficient_confidence"`).

### Explicit Failure Modes (`CalibrationFailureMode`)
1. `"invalid_image"` — `img is None`, non-3-channel, or smaller than $64 \times 64$.
2. `"no_hough_lines"` — White-paint ridge mask produced zero Hough lines above accumulator threshold.
3. `"insufficient_yard_lines"` — Fewer than 3 yard lines survived vanishing-point pencil RANSAC.
4. `"insufficient_hash_ticks"` — Fewer than 10 1-yard hash-mark tick candidates were detected.
5. `"missing_hash_rows"` — RANSAC could not fit both near and far longitudinal hash rows ($\ge 5$ inliers each).
6. `"homography_solve_failed"` — `cv2.findHomography` returned `None` or a singular matrix.
7. `"implausible_homography"` — Failed `plausible_homography()` orientation or scale checks.
8. `"insufficient_confidence"` — Geometric confidence fell below `min_confidence` (`0.45`).
9. `"camera_cut_uncalibrated"` — `CalibrationTracker` detected a camera cut and the post-cut frame failed calibration.
10. `"confidence_expired"` — `CalibrationTracker` decayed confidence below `min_confidence` or exceeded `max_age` during a prolonged landmark dropout.

---

## 4. Temporal `CalibrationTracker` Behavior

- **Camera-Cut Detection:** Computes 2D HSV histogram correlation over the central playing field (`cv2.HISTCMP_CORREL < 0.70`) and checks field-space displacement (`_jump_yd > max_jump_yd = 4.0 yd`). On a cut, prior state is immediately cleared.
- **Inter-Frame Background Turf Propagation:** Tracks Shi-Tomasi corners on the green turf mask via forward-backward verified Lucas-Kanade optical flow (`fb_err < 1.5 px`) and RANSAC homography $\Delta H_{t-1 \to t}$, propagating $H_{\text{prop}} = H_{t-1} \cdot \Delta H_{t-1 \to t}^{-1}$.
- **Confidence Decay & Refusal:** Each unobserved frame multiplies confidence by `decay_factor = 0.80` (`age += 1`) and marks `is_temporally_propagated = True`. Once `confidence < min_confidence` (`0.45`) or `age > max_age` (`5`), the tracker clears `H = None`, sets `failure_reason = "confidence_expired"`, and refuses projection.

---

# Phase 4 Addendum — Field-Space Trajectory Assumptions & Limitations

## A. Geometry States (`calibrated` / `propagated` / `unknown`)

- `calibrated`: the frame's `CalibrationResult` passed `can_project()` and was fitted from that frame; a projected, accepted measurement is treated as a real field observation.
- `propagated`: the frame's calibration was carried forward by the Phase 2 `CalibrationTracker` (`is_temporally_propagated=True`). Geometry is usable, but every field covariance is inflated by `0.25 yd × propagation_age`.
- `unknown`: no `H`, failed/expired/below-floor calibration. **No field position may be claimed.** A dead-reckoned estimate may still be reported, but only in the explicitly labelled `predicted_position` field with `position_source="predicted_dead_reckoning"`.

## B. Frozen Trajectory Model Constants (a priori engineering assumptions)

| Constant | Value | Status |
| :--- | :---: | :--- |
| Process-noise acceleration scale | `5.0 yd/s²` | Assumption (~0.5 g), not measured NFL biomechanics |
| Innovation gate | `χ²₂ ≤ 9.21` (99% quantile) | Statistical gate; threshold never tuned on `TEST` |
| Speed clamp / acceleration clamp | `12.0 yd/s` / `25.0 yd/s²` | Plausibility guards |
| Dead-reckoning horizon | `3` frames | Beyond this, no position estimate is reported at all |
| Consecutive-rejection reset | `3` | Explicit, auditable filter re-initialization |
| Footpoint per-edge noise | `1.0 px` (floor `0.3 px`) | Assumption; measured to be optimistic at 3 px jitter |
| Propagated-geometry drift | `0.25 yd` per propagation frame | Assumption for uncertainty inflation |
| Gate warm-up | 2 accepted updates | Standard track initiation; threshold unchanged |

## C. Uncertainty Model

Field covariance = `J · Σ_px · Jᵀ` with the analytic homography Jacobian `J`, corrupted box-height scaling, a det-confidence inflation term, and an additive drift term for propagated geometry. The reported σ-ellipse is an *assumption-based* uncertainty, not a calibrated NOR/NEES estimate.

## D. Measured Limitations (Phase 4 `TEST`)

1. Uncertainty **under-covers** at `TEST` jitter (1.6–3.0 px): 68% band covers ~32%, 95% band ~62% of accepted measurements.
2. The chi-square gate produces false rejections when jitter exceeds the frozen ~1 px noise assumption (`53 / 368` clean `TEST` samples).
3. Acceleration error is bounded by the filter's process-noise floor and should not be quoted as a validated capability.
4. Track fragmentation is unresolved: outlier-induced track spawns and out-of-sequence associations are reported but not repaired (no track merging / global re-association).
5. The Phase 4 benchmark measures the trajectory layer on **synthetic deterministic fixture motion**; it is not evidence of NFL tracking accuracy, and image-space player-detector accuracy remains unmeasured.

---

## E. Phase 4 Addendum II — Track State, Camera Cuts, and Motion Preservation

### Track state
Every `TrajectorySample` carries `track_state ∈ {"observed", "coasted"}`:
- `observed` — the Phase 3 tracker had a detection this frame **and** the footpoint was reliable.
- `coasted` — the track was coasting (occluded / missed) or the footpoint was refused.
Observed/coasted counts, `missed_frames`, detection confidence, footpoint reliability flags, and the torso-cluster
`team` / `team_confidence` travel with the sample, so a consumer can always tell a real observation from a
model-propagated one.

### Recovery latency
- `recovery_latency_frames_*` — frames between two consecutive **observed** samples of the same track.
- `projection_recovery_latency_frames_*` — frames between two consecutive samples that carried a **field position**.
Both are reported rather than hidden; a long latency is visible as a large number instead of a silent gap.

### Camera cuts
A camera cut is modelled as an abrupt synthetic framing change plus explicit calibration refusal objects
(`failure_reason = "camera_cut_uncalibrated"`). Phase 4 guarantees: no field position is claimed while the geometry is
refused, trajectories are re-acquired afterwards, and the projection-recovery latency is reported. Phase 4 does **not**
preserve player identity across a cut (no re-identification module) and does **not** implement cut *detection* — the cut
is declared by the benchmark, not inferred.

### Motion preservation (no smoothing away of real motion)
`motion_preservation_ratio_vs_gt = smoothed path length / ground-truth path length` is reported per sequence and per split
(≈1.0 across the frozen splits). The raw unsmoothed projection path is longer than ground truth because it accumulates
jitter, which is exactly what smoothing is for. An A/B on `process_accel_std_yd_s2` (5.0 vs 0.5 yd/s², `TRAIN`+`VAL` only,
stored as `process_noise_ablation` in the benchmark JSON) shows the result is insensitive in that range and that neither
setting shortens real movement by more than ~2%.

### Identity metrics
Phase 4 evaluates the same CLEAR-style identity definitions as Phase 3 (`IDSW` on observed samples, split into
active-association swaps vs post-expiration re-initializations; `FRAG` on observation gaps). Because Phase 4 associates
per sample against the full active track set, its `IDSW` events are all active-swap violations — a *different* failure
mode from the single post-expiration re-init that Phase 3 reported on `TEST`.
