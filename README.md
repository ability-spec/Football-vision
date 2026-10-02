# Football-Vision

Modular computer-vision pipeline for American football broadcast and All-22 video: zero-training projective field calibration, player/ball tracking, coordinate projection, and play-level geometry analytics.

---

## Current Status (Phases 0-4 implemented; Phase 4 audit remediation committed, awaiting review)

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
- **Phase 4 (`benchmarks/evaluate_phase4_trajectories.py`):** field-space trajectories with persistent track state, camera-motion-safe field-space smoothing, velocity/acceleration, chi-square jump rejection, uncertainty propagation, and explicit `calibrated` / `propagated` / `unknown` geometry states. Player motion is **synthetic fixture ground truth** (`benchmark_kind = synthetic_trajectory_benchmark`); `0` fabricated field positions and `0` absolute-yardline violations. Real multi-frame trajectory accuracy is **not measured**. Uncertainty is **not statistically calibrated** (empirical coverage is reported per split, per geometry state and before/after smoothing). Acceleration is **experimental / not validated**. Image-based detector accuracy is **unmeasured** (`image_detector_quantitative_metrics = null`) — fixture pass-through counts are never detector accuracy. `TEST` is a frozen held-out split; nothing was tuned on it. See `docs/PHASE4_TRAJECTORY_REPORT.md` (generated from the benchmark JSON) and `docs/ASSUMPTIONS_AND_LIMITATIONS.md`.

### How to read a number in this repository

| Label | Meaning |
| :--- | :--- |
| *measured* | Produced by a benchmark on a frozen split of data with ground truth (e.g. Phase 0/1 held-out landmark error, Phase 2 temporal refusals, Phase 3 tracking metrics). |
| *synthetic benchmark* | Scored against synthetic deterministic ground truth (Phase 4 trajectories use synthetic routes projected through real per-frame homographies). |
| *smoke test* | Single-frame integration check on a real frame with **no** ground truth; it can only show "it ran / it refused". |
| *unmeasured* | No number exists; the pipeline has not been evaluated on that quantity (image detector accuracy, real multi-frame trajectory accuracy). |
| *experimental* | Implemented and unit-tested for mathematical correctness only (acceleration). |
| *not statistically calibrated* | A number is reported, but it is an a priori model rather than a fitted/validated one (Phase 4 uncertainty: empirical coverage is published, the gap is explained, and no calibration factor is applied). |
| *harness-only* | Computed by the benchmark harness from scheduled inputs or known labels, not by the pipeline (e.g. `FixturePlayerDetector` pass-through counts, injected-outlier labels). These must never be presented as detector accuracy; image-detector quantitative accuracy is *unmeasured* (`image_detector_quantitative_metrics = null`). |

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

`docs/PHASE4_TRAJECTORY_REPORT.md` is generated from `outputs/phase4_trajectory_benchmark.json`:

```bash
python3 benchmarks/evaluate_phase4_trajectories.py                  # writes the JSON + overview PNG
python3 benchmarks/render_phase4_report.py                          # renders the Markdown report
```

Real NFL stills are third-party assets and are **not vendored**. Tests and the single-frame smoke section
resolve them through `football_vision/data_paths.py`; set `FOOTBALL_VISION_NFL_FRAMES` to a directory
containing them to run the asset-dependent checks, otherwise they skip with an explicit reason.

CI (`.github/workflows/ci.yml`) runs `ruff check .` (E9 + pyflakes rules) and `pytest -v` on CPU-only
runners with no third-party assets and no model downloads.
