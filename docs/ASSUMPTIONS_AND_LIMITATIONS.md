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
