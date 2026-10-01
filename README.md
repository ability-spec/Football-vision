# Football-Vision

Modular computer-vision pipeline for American football broadcast and All-22 video: zero-training projective field calibration, player/ball tracking, coordinate projection, and play-level geometry analytics.

---

## Current Status (Phase 0 Baseline)

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
│   ├── detection/
│   ├── tracking/
│   ├── identity/
│   ├── projection/
│   ├── analytics/
│   └── visualization/
├── experimental/
│   └── cpu_blob_detector.py
├── benchmarks/
│   └── evaluate_hough_week1.py
├── data/
│   └── benchmarks/
│       └── week1_held_out_landmarks.json
├── tests/
│   ├── test_field_spec_and_schema.py
│   └── test_calibration.py
└── outputs/
```

---

## Running Tests & Reproducing the Week-1 Benchmark

```bash
# Run full unit & integration test suite
python3 -m unittest discover -s tests -v

# Reproduce the Week-1 calibration benchmark & diagnostic figures
python3 benchmarks/evaluate_hough_week1.py
```
