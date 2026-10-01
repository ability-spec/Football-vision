# Football-Vision — Assumptions & Limitations (Phase 0 Baseline)

This document explicitly records the geometric, photometric, and architectural assumptions of the Phase 0 `Football-Vision` calibration baseline and its known failure modes.

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

## 2. Photometric & Camera Assumptions

- **Green Turf Prior (`H in [28, 88], S >= 25, V >= 40`):**
  - White-paint extraction requires painted lines and hash ticks to border green turf (`turf_nearby`). Snow-covered fields, muddy/brown grass patches, or non-green end zones are rejected by this gate.
- **White-Paint Saturation Gate (`S < 65, gray > 135`):**
  - Designed to pass white field markings while rejecting high-saturation broadcast overlays (yellow 1st-down line, blue line of scrimmage, yellow All-22 telestrator drawings, and colored team logos).
  - **Known Limitation:** A *white* telestrator line, a white midfield team logo stripe, or a sunlit white jersey edge aligned with the yard-line pencil can pass the saturation gate and must be rejected by the geometric RANSAC stages.
- **Broadcast Scorebug Margin Mask (`top 5%`, `bottom 15%`):**
  - Assumes standard broadcast scorebugs sit in the top 5% or bottom 15% of the frame. Full-screen replay wipes or mid-screen stat banners are not masked.

---

## 3. Geometric Observability & Known Failure Modes

1. **Minimum Landmark Visibility:**
   - Single-frame calibration requires **at least 3 visible 5-yard lines** (`len(best_inliers) >= 3`) to constrain the 1D projective cross-ratio along the vanishing-point pencil, and **at least 10 detected 1-yard hash-mark ticks** (`>= 5` inliers on both near and far hash rows) to constrain lateral scale.
   - Tightly zoomed isolation shots, goal-line pileups, or end-zone corner views where fewer than 3 yard lines or fewer than 2 hash rows are visible will return `CalibrationResult(success=False, H=None)`.
2. **Longitudinal Periodicity (`x_coord_mode = "relative_10yd"`):**
   - Yard lines repeat every 5 yards and painted numbers repeat every 10 yards (`10, 20, 30, 40, 50, 40, 30, 20, 10`).
   - Number-presence parity (`detected_ten_yard_parity`) distinguishes 10-yard lines from 5-yard lines (`6.7x–36.6x` score ratio on the Week-1 frames), locking longitudinal coordinates up to a multiple of 10 yards ($X \pmod{10\text{ yd}}$).
   - Resolving absolute $X \in [0, 100]\text{ yd}$ without a caller-provided `x_start_yd` requires reading the painted yard digits (`10`–`50`) and directional arrow or observing the 50-yard line / goal line (scheduled for Phase 1/2).
3. **Sideline Extrapolation Lever Arm ($3.9\times$):**
   - Extrapolating from the $6.83\text{-yd}$ inter-hash band ($Y \in [23.25, 30.08]\text{ yd}$) out to the sidelines ($Y = 0$ and $Y = 53.33\text{ yd}$) carries a $3.9\times$ lever arm. A $0.7\text{-px}$ RMSE on the hash rows translates to $\sim 2.5\text{ px}$ ($\sim 0.30\text{–}0.40\text{ yd}$) at the sideline when the sideline itself is occluded.
4. **Experimental Player Detector (`experimental/cpu_blob_detector.py`):**
   - The dark-region morphological blob detector in `experimental/cpu_blob_detector.py` is a CPU-only stand-in used strictly to test physical scale/spread plausibility in environments without GPU YOLO weights. It splits or merges occluded linemen in scrums and is **not** part of the production `football_vision` library.
