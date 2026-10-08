# Football-Vision

Modular computer-vision pipeline for American football broadcast and All-22 video: zero-training projective field calibration, player/ball tracking, coordinate projection, and play-level geometry analytics.

## CPU MVP: one workflow

After installing `.[dev]` in the virtual environment, try the self-contained synthetic demo:

```bash
source .venv/bin/activate
python -m football_vision --demo --out outputs/my-demo
```

For your own video, create `plays.json` with manually declared play boundaries:

```json
{"game_id":"game-001","plays":[{"play_id":"play-001","start_frame":0,"snap_frame":30,"end_frame":120}]}
```

```bash
python -m football_vision clip.mp4 --plays plays.json --source-kind real --max-frames 300 --out outputs/my-run
```

Open `outputs/my-run/report.html` for play metrics and artifact links. Each run creates
`analysis.json` (predictions, trajectories, segmentation, analytics and provenance),
`metrics.csv`, `review.avi` (synchronized boxes, field view and play phase),
`preview.png` (sampled frames viewable without video playback), `report.html`,
`report.md` (readable text and metric tables),
and `manifest.json` (artifact hashes). `bundle.zip` packages all those artifacts
with opening instructions. If an embedded HTML viewer blocks the report, download
the ZIP, extract all files into one folder, and open `report.html` in your browser.
Open `preview.png` directly for still frames or `review.avi` in a video player for motion.
Use a new output directory
for each run; previous results are never overwritten. Failed export attempts remove
their own incomplete output. No model download, GPU or external service is required.
CSV includes metric source, confidence and limitations. Unavailable values remain blank;
labels beginning with spreadsheet formula characters receive a leading apostrophe in
CSV only. JSON retains the exact original labels.

Frame timestamps are zero-based integers and must lie within the decoded window.
Manual starts and ends are required. Omit `snap_frame` to request motion-based snap
estimation; insufficient evidence produces a documented refusal. Unknown geometry,
team roles and possession events remain unavailable. The demo tests integration,
not football accuracy; the untrained CPU detector is a research baseline. Real-video
accuracy requires independent labeled footage using [the evaluation protocol](docs/REAL_VIDEO_EVALUATION.md).

Prepare selected frames for independent manual evaluation:

```bash
python -m football_vision.evaluation.prepare clip.mp4 --frames 0 30 60 --source-kind real --split dev --out outputs/manual-labels
```

This creates original-size PNGs and a pending annotation template; follow its editing
instructions before scoring. See [MVP completion criteria](docs/MVP_COMPLETION.md) for
what is implemented, what needs real data, and the next scope decision.

For learned person detection, download the verified local weights described in
[the YOLOX guide](docs/LEARNED_DETECTOR.md), then use the same workflow:

```bash
python -m football_vision clip.mp4 --plays plays.json --source-kind real --detector yolox --weights models/yolox_tiny.onnx --out outputs/learned-run
```

YOLOX-Tiny runs on CPU through OpenCV; it is pretrained on COCO people, not
fine-tuned for football. Player/referee/spectator separation and football accuracy
remain unvalidated. Model hashes and preprocessing profiles are recorded in results.

For small players, replace `--detector yolox` with `--detector yolox-tiled`.
This adds four overlapping crops and global duplicate suppression, at up to five
model passes per frame. Both commands accept `--opencv-threads` (default: 2).
Native helmet-center coverage improved on the supplied development clips; this
does not establish player precision/recall or stable tracking identity. See
[the first real-video review](docs/NFL_INITIAL_REVIEW.md).

The Markdown and HTML reports expose observation diagnostics and boundary
refusals. Track IDs are not counts of unique athletes; projectable frames are
not proof of accurate field coordinates. The current learned workflow is offline,
not a real-time 60-FPS system.

---

## Current Status: research prototype

**0.5.0:** Phase 10 play segmentation (manual timestamps + collective-motion onset/cessation)
is implemented, benchmarked on frozen synthetic splits and documented in
[the Phase 10 report](docs/PHASE10_PLAY_SEGMENTATION_REPORT.md).

**0.4.1 remediation:** clean-install/CI fixes, coordinate-boundary and trajectory corrections,
and an offline video evaluation workflow are implemented. The CPU MVP now connects
video processing, play analytics, exports and synchronized review. See
[the finding-by-finding remediation](docs/OCTOBER_REMEDIATION.md) and
[the real-video protocol](docs/REAL_VIDEO_EVALUATION.md).

**The benchmark numbers below are frozen historical results, not measurements of 0.4.1.**
Real multi-frame detection/tracking accuracy remains unmeasured. Analytics includes play
segmentation, observed formation/route geometry, scoped events and play metrics. The CPU MVP
connects these to video processing and synchronized review. A pretrained YOLOX-Tiny
person detector is integrated as an optional local ONNX model. Real-game validation,
football fine-tuning if needed, ball/possession analysis and a browser-based synchronized
player remain future work.


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
- **Phase 4 (`benchmarks/evaluate_phase4_trajectories.py`):** field-space trajectories with persistent track state, field-space smoothing within coordinate segments, velocity/acceleration, chi-square jump rejection, uncertainty propagation, and explicit `calibrated` / `propagated` / `unknown` geometry states. Player motion is **synthetic fixture ground truth** (`benchmark_kind = synthetic_trajectory_benchmark`); `0` fabricated field positions and `0` absolute-yardline violations. Real multi-frame trajectory accuracy is **not measured**. Uncertainty is **not statistically calibrated** (empirical coverage is reported per split, per geometry state and before/after smoothing). Acceleration is **experimental / not validated**. Image-based detector accuracy is **unmeasured** (`image_detector_quantitative_metrics = null`) — fixture pass-through counts are never detector accuracy. `TEST` is a frozen held-out split; nothing was tuned on it. See `docs/PHASE4_TRAJECTORY_REPORT.md` (generated from the benchmark JSON) and `docs/ASSUMPTIONS_AND_LIMITATIONS.md`.
- **Phase 10 (`benchmarks/evaluate_phase10_segmentation.py`):** play segmentation
  (PLAY START -> PRE-SNAP -> SNAP -> PLAY -> PLAY END). Manual play timestamps are the primary
  contract; a missing snap is estimated as the first sustained collective-motion onset after a
  fully defined quiescent baseline, and a missing play end (opt-in) as the first sustained
  collective quiet run. On the frozen synthetic splits: `5` snaps resolved / `6` refused
  (`4` false refusals on declared low-visibility strata, `2` true refusals), snap error `0`
  frames on rectangular onsets and `2` frames on a staggered onset (frozen bound `4`),
  `0` fabricated boundaries and `0` expectation violations. Player motion is synthetic fixture
  ground truth (`benchmark_kind = synthetic_segmentation_benchmark`); no image, detector or
  homography is involved and end-to-end video play segmentation is **unmeasured** (a play START
  is always a manual timestamp, so the unattended video runner does not call Phase 10).
  See `docs/PHASE10_PLAY_SEGMENTATION_REPORT.md`.


### How to read a number in this repository

| Label | Meaning |
| :--- | :--- |
| *measured* | Produced by a benchmark on a frozen split of data with ground truth (e.g. Phase 0/1 held-out landmark error, Phase 2 temporal refusals, Phase 3 tracking metrics). |
| *synthetic benchmark* | Scored against synthetic deterministic ground truth (Phase 4 trajectories use synthetic routes projected through real per-frame homographies; Phase 10 segmentation uses synthetic field-space speed profiles with fixture boundaries). |
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
│   ├── schema.py            # Phase 16A contracts + Phase 3/4/10 records
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
│   ├── analytics/        # Play segmentation, observed geometry, scoped events and metrics
│   ├── evaluation/       # Manual boundaries, source-frame extraction and video predictions
│   └── visualization/
├── experimental/
│   └── cpu_blob_detector.py
├── benchmarks/
│   ├── evaluate_hough_week1.py
│   ├── evaluate_phase2_robustness.py
│   ├── evaluate_phase3_tracking.py
│   ├── evaluate_phase4_trajectories.py
│   ├── evaluate_phase10_segmentation.py
│   └── render_phase10_report.py
├── data/
│   └── benchmarks/
│       ├── week1_held_out_landmarks.json
│       ├── phase2_calibration_manifest.json
│       ├── phase3_tracking_manifest.json
│       ├── phase4_trajectory_manifest.json
│       └── phase10_segmentation_manifest.json
├── tests/
│   ├── test_field_spec_and_schema.py
│   ├── test_calibration.py
│   ├── test_phase2_benchmark.py
│   ├── test_phase3_perception_tracking.py
│   ├── test_phase4_trajectories.py
│   └── test_phase10_play_segmentation.py
└── outputs/
```

---

## Running Tests & Reproducing the Week-1 Benchmark

```bash
# Install the package and all CPU test dependencies in a clean environment.
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip check

# Run the full unit & integration test suite
python3 -m pytest -v

# Reproduce benchmarks / figures
python3 benchmarks/evaluate_hough_week1.py            # Phase 0/1 calibration
python3 benchmarks/evaluate_phase2_robustness.py      # Phase 2 robustness + temporal
python3 benchmarks/evaluate_phase3_tracking.py        # Phase 3 perception/tracking
python3 benchmarks/evaluate_phase4_trajectories.py    # Phase 4 trajectories
python3 benchmarks/evaluate_phase10_segmentation.py   # Phase 10 play segmentation
```

Reports: `docs/WEEK1_HOUGH_CALIBRATION_REPORT.md`, `docs/PHASE2_CALIBRATION_ROBUSTNESS_REPORT.md`,
`docs/PHASE3_PERCEPTION_TRACKING_REPORT.md`, `docs/PHASE4_TRAJECTORY_REPORT.md`,
`docs/phase4_quantitative_audit.md` (Phase 4 quantitative audit: all-track vs dominant-track, fragmentation,
rejection-gate, dead-reckoning, uncertainty, TEST integrity, reproducibility),
`docs/PHASE10_PLAY_SEGMENTATION_REPORT.md` (generated from the Phase 10 benchmark JSON).

`docs/PHASE4_TRAJECTORY_REPORT.md` is generated from `outputs/phase4_trajectory_benchmark.json`:

```bash
python3 benchmarks/evaluate_phase4_trajectories.py                  # writes the JSON + overview PNG
python3 benchmarks/render_phase4_report.py                          # renders the Markdown report
python3 benchmarks/render_phase4_audit.py                           # renders the quantitative-audit doc
python3 benchmarks/render_phase10_report.py                         # renders the Phase 10 report
```

Real NFL stills are third-party assets and are **not vendored**. Tests and the single-frame smoke section
resolve them through `football_vision/data_paths.py`; set `FOOTBALL_VISION_NFL_FRAMES` to a directory
containing them to run the asset-dependent checks, otherwise they skip with an explicit reason.

CI (`.github/workflows/ci.yml`) runs `ruff check .` (E9 + pyflakes rules) and `pytest -v` on CPU-only
runners with no third-party assets and no model downloads.
