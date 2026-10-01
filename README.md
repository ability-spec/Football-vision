# Football-Vision

Modular computer-vision pipeline for American football broadcast and All-22 video: zero-training projective field calibration, player/ball tracking, coordinate projection, and play-level geometry analytics.

---

## Current Status (Phases 0-4 implemented; Phase 4 under review)

- **Validated Core (`football_vision/calibration/`):**
  - 1-px morphological white-paint ridge extraction + HSV saturation gating (`S < 65`) inside turf mask
  - Full-line Hough accumulator + vanishing-point pencil RANSAC + 5-parameter Huber least-squares refinement
  - Projective cross-ratio 1-yard subdivision + guided NFL hash-mark centroid row RANSAC (`Y = 23.2500 yd` and `30.0833 yd`, `20.50 ft` center-to-center)
  - Optional far-sideline boundary refinement (`Y = 53.3333 yd`)
  - Automatic 10-yard vs. 5-yard line number-box parity detection and homography plausibility checks
- **Experimental Code (`experimental/cpu_blob_detector.py`):**
  - Dark-region morphological player detector + BGR K-Means team split used for CPU-only plausibility benchmarking when YOLO weights are not loaded.
- **Reproducible Week-1 Benchmark (`benchmarks/evaluate_hough_week1.py`):**
  - Evaluated on 26 held-out ground-truth sideline and yard-number landmarks across 3 real NFL frames (`0.275 yd`, `0.350 yd`, `0.057 yd` median held-out error).
- **Phase 2 (`benchmarks/evaluate_phase2_robustness.py`):** calibration robustness + temporal stress benchmark; `34` temporal refusals across 8 sequences (`0 + 0 + 1 + 10 + 6 + 7 + 3 + 7`), `0` fabricated calibrations.
- **Phase 3 (`benchmarks/evaluate_phase3_tracking.py`):** player detection / footpoint reliability / torso-cluster team assignment / multi-object tracking / gated field projection. Box fixtures come from `FixturePlayerDetector`; image-space detector accuracy is **unmeasured**. `0` fabricated projections.
- **Phase 4 (`benchmarks/evaluate_phase4_trajectories.py`):** field-space trajectories with persistent track state, camera-motion-safe field-space smoothing, velocity/acceleration, chi-square jump rejection, uncertainty propagation, and explicit `calibrated` / `propagated` / `unknown` geometry states. Player motion is synthetic fixture ground truth; `0` fabricated field positions and `0` absolute-yardline violations.

---

## Repository Layout

```text
Football-Vision/
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── docs/
│   ├── ASSUMPTIONS_AND_LIMITATIONS.md
│   └── WEEK1_HOUGH_CALIBRATION_REPORT.md
├── football_vision/
│   ├── field_spec.py
│   ├── schema.py
│   ├── calibration/
│   │   ├── white_ridge.py
│   │   ├── yard_lines.py
│   │   ├── hash_marks.py
│   │   ├── sidelines.py
│   │   ├── homography.py
│   │   └── tracker.py
│   ├── detection/        # Phase 3 detectors + footpoint reliability
│   ├── tracking/         # Phase 3 IoU + motion multi-object tracker
│   ├── identity/         # Phase 3 torso appearance cluster assignment
│   ├── projection/       # Phase 3 gated field projection
│   ├── trajectory/       # Phase 4 field-space trajectories, kinematics, uncertainty
│   ├── analytics/
│   └── visualization/
├── experimental/
│   └── cpu_blob_detector.py
├── benchmarks/
│   ├── evaluate_hough_week1.py
│   ├── evaluate_phase2_robustness.py
│   ├── evaluate_phase3_tracking.py
│   └── evaluate_phase4_trajectories.py
├── data/
│   └── benchmarks/
│       ├── week1_held_out_landmarks.json
│       ├── phase2_calibration_manifest.json
│       ├── phase3_tracking_manifest.json
│       └── phase4_trajectory_manifest.json
├── tests/
│   ├── test_field_spec_and_schema.py
│   ├── test_calibration.py
│   ├── test_phase2_benchmark.py
│   ├── test_phase3_perception_tracking.py
│   └── test_phase4_trajectories.py
└── outputs/
```

---

## Running Tests & Reproducing the Week-1 Benchmark

```bash
# Run the full unit & integration test suite (Phases 0-4)
python3 -m pytest -v

# Reproduce benchmarks / figures
python3 benchmarks/evaluate_hough_week1.py            # Phase 0/1 calibration
python3 benchmarks/evaluate_phase2_robustness.py      # Phase 2 robustness + temporal
python3 benchmarks/evaluate_phase3_tracking.py        # Phase 3 perception/tracking
python3 benchmarks/evaluate_phase4_trajectories.py    # Phase 4 trajectories
```

Reports: `docs/WEEK1_HOUGH_CALIBRATION_REPORT.md`, `docs/PHASE2_CALIBRATION_ROBUSTNESS_REPORT.md`,
`docs/PHASE3_PERCEPTION_TRACKING_REPORT.md`, `docs/PHASE4_TRAJECTORY_REPORT.md`.
