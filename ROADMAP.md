# Football-Vision — Master Development Workflow & Engineering Roadmap

**Status:** Agreed Architecture & Execution Plan (Pre-Phase 0 Implementation)  
**Hardware Target:** Local development on ASUS ROG Laptop (NVIDIA RTX 4060 8GB VRAM) + CPU fallback  
**Engineering Cadence:** ~10 hours/week  

**Implementation status (updated with Phase 10):** Stages I-II are complete (Phases 0, 16A, 1,
15A, 2, 3, 4, 7, 8, 9) including the 0.4.1 remediation. Stage III has started: **Phase 10 play
segmentation is implemented** in `football_vision/analytics/segmentation.py` with a frozen
synthetic benchmark (`benchmarks/evaluate_phase10_segmentation.py`,
`docs/PHASE10_PLAY_SEGMENTATION_REPORT.md`). Observed formation/route geometry and play metrics
are implemented. The CPU MVP entry point (`python -m football_vision`) now connects local video,
manual play boundaries, JSON/CSV export, an HTML report, and synchronized AVI review.
This delivers a bounded research workflow, not completion of every original phase: validated
learned player perception, ball/possession events, Parquet export, GPU profiling, and real-game
accuracy evaluation remain open. The synthetic demo verifies integration only.
The owner has selected learned detection as part of the MVP. The local workflow
now supports official pretrained YOLOX-Tiny COCO person weights through OpenCV DNN
on CPU, with a pinned artifact/preprocessing profile. This establishes model
integration, not football-specific perception accuracy; validation and any required
football fine-tuning remain open. See `docs/LEARNED_DETECTOR.md`.

---

## 1. Overall Product Pipeline

```text
NFL VIDEO
    ↓
FIELD CALIBRATION (Single-Frame Geometry + Temporal CalibrationTracker)
    ↓
PLAYER DETECTION  (Critical Path)        ┼──► BALL DETECTION / TRACKING (Parallel Track)
    ↓                                    │
PLAYER TRACKING   (Critical Path)        ┼──► PLAYER IDENTITY / JERSEY OCR (Parallel Track)
    ↓                                    │
TEAM CLASSIFICATION                      │
    ↓                                    │
IMAGE → FIELD COORDINATES  ◄─────────────┘
    ↓
TEMPORAL TRAJECTORIES
    ↓
PLAY-LEVEL STRUCTURE (Play Segmentation)
    ↓
FOOTBALL ANALYTICS (Formations, Routes, Play Metrics, Scoped Events)
    ↓
VISUALIZATION / EXPORT (JSON / Parquet / CSV + Synchronized Viewer)
```

---

## 2. Agreed Architectural Adjustments (Adopted)

1. **Core Data Contracts Pulled into Phase 0 / Phase 1 (`Phase 16A`):**
   - Define canonical dataclasses (`CalibrationResult`, `Detection`, `TrackState`, `FieldProjection`, `Trajectory`, `PlayRecord`) in `football_vision/schema.py` during Phase 0/1 so every downstream phase consumes typed contracts with explicit confidence and provenance from day one.
2. **Benchmark Manifest & `TRAIN / VALIDATION / TEST` Split Protocol Pulled into Early Calibration (`Phase 15A`):**
   - Establish `data/benchmarks/` and `data/splits/` before Phase 2 calibration tuning so no hyperparameters or thresholds are ever tuned on `TEST` frames.
3. **Explicit `x_coord_mode` & Temporal `CalibrationTracker` in Phase 1 / Phase 2:**
   - `CalibrationResult` explicitly distinguishes `x_coord_mode ∈ {"absolute", "relative_10yd", "relative_5yd", "uncalibrated"}` so periodic yard-line geometry never fabricates an unverified absolute yard line when numbers/50-yard line are occluded.
   - Add `CalibrationTracker` (camera-cut detection, inter-frame background turf homography propagation $\Delta H_{t-1 \to t}$, temporal smoothing, and confidence decay) in Phase 1/2 to suppress global camera pan/zoom jitter before player projection.
4. **Ball Detection (`Phase 5`) and Jersey OCR (`Phase 6`) Moved to Parallel Tracks Off the Critical Path:**
   - Execute `Phase 3 -> Phase 4 -> Phase 7 -> Phase 8 -> Phase 9 -> Phase 10 -> Phase 11/12/13` on the critical path so end-to-end player trajectories, pre-snap formations, and route analytics are never blocked by small-object football detection (`~6x10 px`) or side-profile jersey number occlusion.
   - `Phase 5` (Ball) and `Phase 6` (Identity) plug cleanly into the existing `FieldProjection` and `PlayRecord` contracts once ready.
5. **Experimental CPU Blob Detector Extracted from Production Calibration (`Phase 0`):**
   - Extract `detect_and_project_players()` out of `hough_calibrate.py` into `experimental/cpu_blob_detector.py`, keeping `football_vision/calibration/` strictly independent of player detection (Engineering Rules #5 and #8).

---

## 3. Final Agreed Phase Execution Order

All 23 original phases (`Phase 0` through `Phase 22`) and their definitions of done are preserved intact, ordered into 5 execution stages:

### Stage I — Foundation, Contracts & Field Calibration (Weeks 1–2)
1. **Phase 0 — Repository / Engineering Baseline** *(includes `Phase 16A` core data contracts & extraction of `experimental/cpu_blob_detector.py`)*
2. **Phase 1 — Field Calibration Engine** *(single-frame white-paint ridge + VP pencil + 1-yd hash rows + sideline + `x_coord_mode` + `CalibrationTracker` temporal homography)*
3. **Phase 15A — Benchmark Manifest & Split Protocol** *(strict `TRAIN / VALIDATION / TEST` split definitions & landmark ground-truth schema)*
4. **Phase 2 — Calibration Robustness / Benchmarking** *(multi-game, multi-broadcaster, occlusion/logo/telestrator evaluation + failure rate & latency metrics)*

### Stage II — Player Perception, Team Split & Field-Space Trajectories (Weeks 3–5)
5. **Phase 3 — Player Detection** *(YOLOv8/v11 player & referee detection on RTX 4060)*
6. **Phase 4 — Player Tracking** *(persistent `track_id`, occlusion handling, track lifecycle, camera-cut resets)*
7. **Phase 7 — Team Classification** *(turf-masked CIE LAB 2-means with cluster-separation gating + LOS spatial context + per-track temporal voting)*
8. **Phase 8 — Field Coordinate Projection** *(bottom-center foot projection $u,v \to x,y$ with dual detection + calibration confidence)*
9. **Phase 9 — Trajectory Engine** *(missing-frame handling, jump rejection, Kalman/Savitzky-Golay smoothing, velocity, acceleration, distance, direction)*

### Stage III — Play Structure & Measurable Football Analytics (Weeks 6–8)
10. **Phase 10 — Play Segmentation** *(PLAY START $\to$ PRE-SNAP $\to$ SNAP $\to$ PLAY $\to$ PLAY END; manual timestamps first, collective motion onset second)*
11. **Phase 11 — Pre-Snap Formation Analysis** *(offensive/defensive counts, spacing, width/depth, box count, alignment geometry)*
12. **Phase 12 — Route / Movement Analysis** *(rule-based vertical/lateral stems, route depth, crossing patterns, pre-snap motion, receiver-defender separation)*
13. **Phase 16B — Full Analytics Data Model & Provenance Export** *(`Game -> Play -> Frame / Players / Ball / Events / Metrics` serialized to JSON/Parquet/CSV)*
14. **Phase 13 — Play-Level Analytics** *(structured play metrics with explicit definition, units, data source, confidence, and limitations)*

### Stage IV — Visualization, End-to-End Pipeline, Audit & v1.0 Demo (Weeks 9–10)
15. **Phase 17 — Visualization** *(synchronized broadcast overlay + 2D top-down field + trajectories + timeline, distinguishing `detected`, `tracked`, `projected`, `inferred`, `unknown`)*
16. **Phase 18 — End-to-End Pipeline** *(single-command CLI runner from raw video to exported analytics & visualization)*
17. **Phase 20 — Failure / Uncertainty Handling Audit** *(end-to-end verification that low confidence or subsystem dropouts never fabricate coordinates)*
18. **Phase 19 — Performance Benchmarking** *(CPU vs. RTX 4060 GPU latency, memory, and bottleneck profiling)*
19. **Phase 21 — Final Demo** *(1–2 minute single-play end-to-end walkthrough)*

### Stage V — Parallel High-Difficulty Perception Tracks & Generalization (Weeks 11–16)
20. **Phase 6 — Player Identity (Parallel Track)** *(probabilistic jersey/back number OCR + multi-frame track voting + evidence provenance)*
21. **Phase 5 — Ball Detection / Tracking (Parallel Track)** *(dedicated small-object detector + temporal tracker + explicit lost-ball uncertainty)*
22. **Phase 14 — Event Detection** *(scoped rule-based + trajectory/ball events: snap, pass/run, catch/incomplete, tackle, out-of-bounds)*
23. **Phase 15B — Full Multi-Play Evaluation Dataset** *(expanded ground-truth annotations for boxes, identities, ball, trajectories, and events across `TRAIN / VAL / TEST`)*
24. **Phase 22 — Future Generalization** *(post-v1.0 evaluation of reusable core vs. sport-specific modules for Rugby, Hockey, Soccer, etc.)*

---

## 4. Final Dependency Graph

```text
[Phase 0: Repo Baseline + Phase 16A Core Schema]
       │
       ├───────────────────────────────────────────┬──────────────────────────────────────────┐
       ▼                                           ▼                                          ▼
[Phase 1: Field Calibration Engine]         [Phase 3: Player Detection]                [Phase 5: Ball Detection &
 (+ x_coord_mode & CalibrationTracker)             │                                    Tracking (Parallel Track)]
       │                                           ▼                                          │
       ├──► [Phase 15A: Split Manifest]     [Phase 4: Player Tracking]                        │
       │           │                               │                                          │
       ▼           ▼                               ├──────────────────────┐                   │
[Phase 2: Calibration Benchmark]                   ▼                      ▼                   │
       │                                    [Phase 7: Team         [Phase 6: Player Identity  │
       │                                     Classification]        (Jersey OCR - Parallel)]  │
       │                                           │                      │                   │
       └───────────────────┬───────────────────────┴──────────────────────┴───────────────────┘
                           ▼
            [Phase 8: Field Coordinate Projection]
                           │
                           ▼
               [Phase 9: Trajectory Engine]
                           │
                           ▼
              [Phase 10: Play Segmentation]
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  [Phase 11:        [Phase 12:        [Phase 16B: Full Export Schema]
   Pre-Snap          Route/Movement          │
   Formations]       Analysis]               │
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
             [Phase 13: Play-Level Analytics]
                           │
                           ├──────────────────────────────┐
                           ▼                              ▼
              [Phase 17: Visualization]        [Phase 14: Event Detection]
                           │                              │
                           └──────────────┬───────────────┘
                                          ▼
                         [Phase 18: End-to-End CLI Pipeline]
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
         [Phase 20: Failure/Uncertainty Audit]   [Phase 19: Performance Profiling]
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
                               [Phase 21: Final Demo]
                                          │
                                          ▼
                          [Phase 22: Future Generalization]
```

---

## 5. Final Milestone Schedule (~10 hrs/week)

| Milestone | Target Weeks | Included Phases | Key Deliverable / Exit Gate |
| :--- | :---: | :--- | :--- |
| **M1: Validated Calibration Baseline** | Weeks 1–2 | `0, 16A, 1, 15A, 2` | Clean `football_vision` package, `CalibrationResult` + `CalibrationTracker`, experimental code separated, 15–25 frame multi-game benchmark across `VAL`/`TEST`. |
| **M2: Player Perception & Trajectories** | Weeks 3–5 | `3, 4, 7, 8, 9` | YOLO player detection on RTX 4060, persistent `track_id` across frames, CIE LAB + LOS team assignment, projected field trajectories $(x_t, y_t, v_t, a_t)$ with confidence. |
| **M3: Play Structure & Football Geometry** | Weeks 6–8 | `10, 11, 12, 16B, 13` | Play segmentation, pre-snap formation metrics (box count, width/depth), route depth/separation features, and `PlayRecord` JSON/Parquet/CSV export with provenance. |
| **M4: End-to-End Pipeline & v1.0 Demo** | Weeks 9–10 | `17, 18, 20, 19, 21` | Synchronized broadcast + 2D radar visualization, 1-command CLI runner, uncertainty/failure audit, RTX 4060 runtime profile, and 1–2 min single-play demo. |
| **M5: Identity, Ball, Events & Expansion** | Weeks 11–16 | `6, 5, 14, 15B, 22` | Probabilistic jersey OCR voting, small-object ball tracking, scoped event detection, full multi-play dataset, and cross-sport architecture evaluation. |

---

## 6. Final Repository Structure

```text
Football-Vision/
├── pyproject.toml                        # Package metadata, dependencies, pytest configuration
├── README.md                             # Quickstart, architecture overview, benchmark summary
├── ROADMAP.md                            # This formal engineering workflow & phase specification
├── .gitignore                            # Ignores outputs/, *.mp4, *.pt, *.engine, __pycache__/
├── docs/
│   ├── ASSUMPTIONS_AND_LIMITATIONS.md    # Explicit geometric/visual assumptions & failure modes
│   └── WEEK1_HOUGH_CALIBRATION_REPORT.md # Moved from root: Week-1 de-risking report
├── football_vision/                      # Production library (strictly separated from experiments/benchmarks)
│   ├── __init__.py
│   ├── field_spec.py                     # NFL Rulebook geometry constants & coordinate conversions
│   ├── schema.py                         # Phase 16A: CalibrationResult, Detection, TrackState, FieldProjection, PlayRecord
│   ├── calibration/                      # Phase 1: Field Calibration Engine
│   │   ├── __init__.py
│   │   ├── white_ridge.py                # extract_white_paint_ridge, naive_canny_hough
│   │   ├── yard_lines.py                 # detect_yard_lines_and_vp, refine_yard_line_pencil
│   │   ├── hash_marks.py                 # detect_hash_rows_guided (1-yd cross-ratio subdivision + RANSAC)
│   │   ├── sidelines.py                  # detect_far_sideline
│   │   ├── homography.py                 # calibrate_frame, plausible_homography, players_clustered, implied_player_height_ft
│   │   └── tracker.py                    # Phase 1/2: CalibrationTracker (temporal H smoothing & cut reset)
│   ├── detection/                        # Phase 3 & 5 (stubbed in Phase 0, implemented in Phase 3/5)
│   │   └── __init__.py
│   ├── tracking/                         # Phase 4 & 5
│   │   └── __init__.py
│   ├── identity/                         # Phase 6 & 7
│   │   └── __init__.py
│   ├── projection/                       # Phase 8 & 9
│   │   └── __init__.py
│   ├── analytics/                        # Phase 10–14
│   │   └── __init__.py
│   └── visualization/                    # Phase 17
│       └── __init__.py
├── experimental/                         # Rule #5: Research/prototype code isolated from production
│   ├── __init__.py
│   └── cpu_blob_detector.py              # Extracted Week-1 dark-blob + K-Means fallback player detector
├── benchmarks/                           # Rule #8: Benchmark scripts separated from core library
│   └── evaluate_hough_week1.py           # Refactored Week-1 benchmark consuming football_vision + experimental
├── data/                                 # Phase 15A: Structured benchmark manifests & split definitions
│   └── benchmarks/
│       └── week1_held_out_landmarks.json # Extracted 26 ground-truth landmarks & frame metadata
├── tests/                                # Automated unit & integration test suite
│   ├── test_field_spec_and_schema.py     # Tests NFL geometry constants & schema serialization/validation
│   └── test_calibration.py               # Refactored & expanded from test_hough_calibrate.py
└── outputs/                              # Generated artifacts (JSON results, diagnostic PNGs, videos)
```

---

## 7. Exact Scope of Phase 0 (What Will Be Implemented Next)

When you give the green light to begin **Phase 0**, we will execute the following 9 concrete tasks **without changing any validated calibration math**:

1. **Initialize Git & Packaging Baseline:**
   - Initialize git repository in `/home/user/Football-Vision` (on `main` or `feat/phase0-baseline`), add `.gitignore` (excluding `outputs/`, weights, caches), and add `pyproject.toml` + `README.md`.
2. **Create `football_vision/field_spec.py`:**
   - Centralize all NFL Rulebook constants (`FIELD_LENGTH_YD = 100.0`, `FIELD_WIDTH_YD = 160.0 / 3.0`, `Y_NEAR_HASH_YD = 23.25`, `Y_FAR_HASH_YD = 30.0833`, yard-number bounds, plausibility thresholds) and yard $\leftrightarrow$ foot helpers.
3. **Create `football_vision/schema.py` (`Phase 16A` Core Contracts):**
   - Define typed, serializable dataclasses for `CalibrationResult` (including `confidence: float`, `x_coord_mode: str`, `failure_reason: Optional[str]`, `residuals: Dict[str, float]`), `Detection`, `TrackState`, `FieldProjection`, and `TrajectoryPoint`.
4. **Modularize the Validated Calibration Code into `football_vision/calibration/`:**
   - Split the validated calibration functions from `hough_calibrate.py` into:
     - `football_vision/calibration/white_ridge.py`
     - `football_vision/calibration/yard_lines.py`
     - `football_vision/calibration/hash_marks.py`
     - `football_vision/calibration/sidelines.py`
     - `football_vision/calibration/homography.py`
5. **Isolate Experimental Code (`Rule #5` & `Rule #8`):**
   - Move `detect_and_project_players()` out of the calibration library into `experimental/cpu_blob_detector.py`.
6. **Externalize Benchmark Data & Script (`Rule #6` & `Rule #8`):**
   - Move the 26 held-out ground-truth landmark definitions into `data/benchmarks/week1_held_out_landmarks.json` and move `evaluate_hough_week1.py` into `benchmarks/evaluate_hough_week1.py` so running `python3 benchmarks/evaluate_hough_week1.py` reproduces `outputs/week1_hough_benchmark.json` bit-for-bit.
7. **Document Assumptions & Limitations (`docs/`):**
   - Move `WEEK1_HOUGH_CALIBRATION_REPORT.md` to `docs/WEEK1_HOUGH_CALIBRATION_REPORT.md` and create `docs/ASSUMPTIONS_AND_LIMITATIONS.md` explicitly recording coordinate conventions, camera/turf assumptions, and known failure modes.
8. **Expand Unit & Integration Tests (`tests/`):**
   - Create `tests/test_field_spec_and_schema.py` and `tests/test_calibration.py` verifying that the modularized package produces the exact same calibration results and held-out errors (`0.275 yd`, `0.350 yd`, `0.057 yd` median error) as the flat Week-1 script.
9. **Remove Legacy Root Scripts Only After Parity Verification:**
   - Remove the root-level `hough_calibrate.py` and `evaluate_hough_week1.py` (or keep thin deprecation shims if desired) once all tests and the benchmark script pass.
