# Football-Vision: three-part remediation

Base reviewed: `799ba63b8eb88bbc7de5802b6487641bfcaf1f53`.
This work addresses the Football-specific audit findings. It is not a claim that every
possible defect has been found, or that unit tests establish broadcast-video accuracy.

## 1. Reproducible installation and CI

- Development dependencies now include scikit-learn, which experimental test imports require.
- Hough line decoding accepts OpenCV 4 `(N, 1, 3)` and OpenCV 5 `(N, 3)` layouts.
- Historical benchmark image paths remain provenance; tests no longer require another
  machine's absolute path to exist.
- Package metadata and runtime version use one source: `football_vision/_version.py` (`0.4.1`).
- CI covers Python 3.10/3.11/3.12 with OpenCV 4 and 5, including `pip check`.
  Local validation used Python 3.12; the remaining Python versions require CI.

## 2. Correctness and configuration

| Audit finding | Change | Remaining boundary |
|---|---|---|
| F01: missing test dependency | Install sklearn in `dev` | No trained detector weights supplied |
| F02: OpenCV return shape | Normalize Hough output layout | Geometry quality is separately unmeasured on new clips |
| F03: nonportable historical path | Check recorded provenance rather than external file existence | Asset-dependent tests still skip explicitly |
| F04: confidence inflation ineffective in accepted range | Sigma multiplier `1 + 2.5 * (1 - clipped_confidence)` | Engineering assumption, not a fitted probability model |
| F05: motion across coordinate changes | Coordinate IDs, segment boundaries, reset filters and distance anchors; temporal tracker starts new epochs | Automatic camera/origin discontinuity detection remains heuristic |
| F06: gap aging by update count | Use frame ID minus last accepted measurement; expire absent tracks on return | Long gaps lose motion continuity by design |
| F07: hidden speed clipping | Filter reports clipping, including initialization | Clipping is a plausibility limit, not an accuracy guarantee |
| F08: unreliable feet mislabeled as coasting | Observation follows detection presence; reliability is separate provenance | Detection quality still requires video evaluation |
| F09: horizon fabricated as origin / Jacobian crash | Scale-invariant homogeneous denominator guard; refuse projection at infinity | Near-horizon finite geometry can still be poorly conditioned |
| F10: rejected samples reset measurement age | Only accepted measurements reset the age | Existing explicit reacquisition policy remains; flagged in samples |
| F11: ignored rejection limit | Instantiate each rejection counter with configured limit; reject nonpositive limits | Threshold selection needs development data |
| F12: ignored detector size options | Apply configured minimum height/width; preserve prior effective default thresholds | Untrained turf-contrast detector remains a baseline |
| F13: tracker skipped-frame behavior | Elapsed-frame prediction, aging, expiration before association | This is not a replacement for a stronger appearance tracker |
| F14: inconsistent version | One version source for installation/runtime | Historical reports retain their original provenance |

### Coordinate contract

`CalibrationResult.coordinate_frame_id` identifies field axes and origin, **not** a
particular matrix. A camera pan may change the matrix while preserving this ID. An
origin, orientation or shot change must change the ID. Mode changes also break continuity.

The temporal calibration tracker emits local `calibration:<epoch>` IDs and starts a new
epoch on reacquisition, mode/input-ID change or a detected large discontinuity. These IDs
are local to one tracker stream. They are not stadium-global coordinates or evidence that
an automatic origin assignment is correct. Never reuse them across independent runs as
proof of alignment.

For direct relative calibration without IDs, a changed normalized matrix conservatively
starts a new segment. For direct absolute calibration, callers are responsible for the
claimed shared field axes. Every sample exposes `coordinate_segment`; consumers must not
connect positions across segments. A trajectory containing different modes has aggregate
`x_coord_mode="uncalibrated"`; use the per-sample mode. Cumulative distance sums accepted
within-segment intervals, omitting coordinate resets and expired gaps.

After an expired gap the next valid measurement starts a new filter, marked
`reinitialized_after_gap`. This may reacquire a previously rejected location; it does not
mean the gate proved that location correct. Camera-cut resets also clear absent tracks.

## 3. Video validation workflow

See [REAL_VIDEO_EVALUATION.md](REAL_VIDEO_EVALUATION.md) for commands and label format.
The runner decodes actual video through calibration, detection, tracking and trajectories.
The scorer stays outside the production package and requires independent annotations.

Available verification uses generated solid-turf video and adversarial synthetic metric
fixtures. These validate decoding, refusal, metric denominators and CLI behavior. **No real
multi-frame accuracy has been measured in this remediation**: no independently annotated
broadcast clips were available. The video runner deliberately uses the existing untrained
CPU detector. It does not implement ball tracking, play analytics, trained detection or a
production UI, and does not score team classification, IDF1/HOTA or landmark reprojection.

## Historical numbers and release status

The checked-in Phase 2/3/4 JSON and generated reports are frozen historical artifacts from
the old pipeline. They have **not** been regenerated for 0.4.1. In particular, uncertainty,
tracking and trajectory behavior changed, so old scores must not be presented as measurements
of this revision. Keep the old results for comparison; rerun the fixed development protocol
before looking at held-out results. Do not retune thresholds on TEST.

Local checks: clean editable installation with development extras and `pip check`; full
pytest suite under OpenCV 5 and OpenCV 4; Ruff; whitespace diff check. Exact final counts and
remote CI status are recorded in the pull request. Skipped third-party-image tests are an
explicit validation gap, not passes.

The honest product assessment remains a research prototype. These fixes improve its
correctness and reproducibility; production readiness still depends on representative,
independent real-video evaluation and closing the documented gaps.
