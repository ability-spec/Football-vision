"""Phase 4 Field-Space Trajectory tests.

Covers geometry-state resolution, uncertainty propagation, smoothing,
kinematics, jump/outlier rejection, camera-motion interaction, dead-reckoning
labelling, x_coord_mode semantics, and benchmark-integrity invariants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

from football_vision.data_paths import (  # noqa: E402
    real_frames_skip_reason,
    resolve_nfl_frame,
)
from football_vision import (
    CalibrationResult,
    FootpointEstimate,
    PlayerTrack,
    PlayerTrajectoryBuilder,
    calibrate_frame,
)
from football_vision.trajectory import (
    ConstantVelocityFieldFilter,
    FieldJumpGate,
    ImageSpaceJumpGate,
    footpoint_pixel_covariance,
    geometry_state_is_projectable,
    homography_jacobian,
    inflate_covariance,
    propagate_to_field_covariance,
    resolve_geometry_state,
    sigma_ellipse,
)
from football_vision.trajectory.outlier import (
    REASON_FIELD_GATE,
    REASON_IMAGE_GATE,
    RejectionTracker,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSON = ROOT / "outputs" / "phase4_trajectory_benchmark.json"
MANIFEST_JSON = ROOT / "data" / "benchmarks" / "phase4_trajectory_manifest.json"
BASE_IMAGE_NAME = "nfl-game-broadcast-screenshot-1st-and-10-5.jpg"
BASE_IMAGE = resolve_nfl_frame(BASE_IMAGE_NAME)  # may be None: unvendored third-party asset
FPS = 30.0
DT = 1.0 / FPS


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_calibration() -> CalibrationResult:
    import cv2

    if BASE_IMAGE is None:
        pytest.skip(real_frames_skip_reason())
    img = cv2.imread(str(BASE_IMAGE))
    assert img is not None
    cal = calibrate_frame(img, x_start_yd=15.0)
    assert cal.success and cal.can_project()
    return cal


def _calibration_with_H(
    base: CalibrationResult,
    H: np.ndarray,
    *,
    propagated: bool = False,
    age: int = 0,
    confidence: Optional[float] = None,
    mode: Optional[str] = None,
) -> CalibrationResult:
    return CalibrationResult(
        success=True,
        H=H,
        H_inv=np.linalg.inv(H),
        image_size=base.image_size,
        confidence=float(base.confidence if confidence is None else confidence),
        x_coord_mode=mode or base.x_coord_mode,
        plausible_orientation_scale=True,
        detected_ten_yard_parity=base.detected_ten_yard_parity,
        is_temporally_propagated=propagated,
        propagation_age=age,
    )


def _uncalibrated(base: CalibrationResult, reason: str = "confidence_expired") -> CalibrationResult:
    return CalibrationResult(
        success=False,
        H=None,
        H_inv=None,
        image_size=base.image_size,
        confidence=0.0,
        x_coord_mode="uncalibrated",
        failure_reason=reason,
    )


def _track_from_field_point(
    track_id: int,
    cal: CalibrationResult,
    x_yd: float,
    y_yd: float,
    *,
    jitter_uv: Tuple[float, float] = (0.0, 0.0),
    confidence: float = 0.91,
    missed_frames: int = 0,
) -> PlayerTrack:
    """Build a Phase-3-style ``PlayerTrack`` whose footpoint corresponds to a field point."""
    uv = cal.field_to_image([[x_yd, y_yd]])
    assert uv is not None
    u, v = float(uv[0][0]) + jitter_uv[0], float(uv[0][1]) + jitter_uv[1]
    scale_uv = cal.field_to_image([[x_yd, y_yd + 1.0]])
    px_per_yd = float(np.hypot(scale_uv[0][0] - u, scale_uv[0][1] - v))
    box_h = float(np.clip(2.0 * px_per_yd, 26.0, 110.0))
    box_w = 0.42 * box_h
    bbox = (u - 0.5 * box_w, v - box_h, u + 0.5 * box_w, v)
    fp = FootpointEstimate(
        u_px=u,
        v_px=v,
        is_reliable=True,
        confidence=confidence,
        is_partially_truncated=False,
        is_near_sideline=False,
        is_player_crossing=False,
        is_temporarily_occluded=missed_frames > 0,
        unreliable_reason=None,
    )
    return PlayerTrack(
        track_id=track_id,
        frame_id=0,
        bbox=bbox,
        footpoint=(u, v),
        footpoint_estimate=fp,
        team="TEAM_A",
        team_confidence=0.9,
        detection_confidence=confidence,
        age=1,
        missed_frames=missed_frames,
        detector_metadata={"detector_name": "test_fixture", "gt_id": track_id},
    )


def _run_stream(
    builder: PlayerTrajectoryBuilder,
    cals: Sequence[Optional[CalibrationResult]],
    field_points: Sequence[Tuple[float, float]],
    *,
    jitter: Sequence[Tuple[float, float]] | None = None,
    observation_cals: Sequence[Optional[CalibrationResult]] | None = None,
) -> List:
    """Feed a single-track stream through the builder.

    ``observation_cals`` (when supplied) is used only to place the synthetic image
    footpoint; the builder always receives ``cals[frame]`` as the frame geometry,
    which mirrors the real situation where the detector is geometry-independent.
    """
    samples = []
    for frame, (cal, (x_yd, y_yd)) in enumerate(zip(cals, field_points)):
        if cal is None:
            continue
        obs_cal = cal if observation_cals is None else observation_cals[frame]
        assert obs_cal is not None
        j = (0.0, 0.0) if jitter is None else jitter[frame]
        trk = _track_from_field_point(1, obs_cal, x_yd, y_yd, jitter_uv=j)
        samples.extend(builder.update([trk], frame_id=frame, calibration=cal))
    return samples


# ---------------------------------------------------------------------------
# 1. Geometry states
# ---------------------------------------------------------------------------
def test_geometry_state_resolution(base_calibration: CalibrationResult) -> None:
    assert resolve_geometry_state(base_calibration) == "calibrated"
    assert resolve_geometry_state(_calibration_with_H(base_calibration, base_calibration.H, propagated=True, age=2)) == "propagated"
    assert resolve_geometry_state(_uncalibrated(base_calibration)) == "unknown"
    assert resolve_geometry_state(None) == "unknown"
    low = _calibration_with_H(base_calibration, base_calibration.H, confidence=0.20)
    assert resolve_geometry_state(low) == "unknown"
    assert geometry_state_is_projectable("calibrated") and geometry_state_is_projectable("propagated")
    assert not geometry_state_is_projectable("unknown")


# ---------------------------------------------------------------------------
# 2. Uncertainty propagation
# ---------------------------------------------------------------------------
def test_homography_jacobian_matches_finite_differences(base_calibration: CalibrationResult) -> None:
    H = np.asarray(base_calibration.H, dtype=np.float64)

    def _apply_float64(uu: float, vv: float) -> np.ndarray:
        w = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
        return np.array(
            [
                (H[0, 0] * uu + H[0, 1] * vv + H[0, 2]) / w,
                (H[1, 0] * uu + H[1, 1] * vv + H[1, 2]) / w,
            ]
        )

    u, v = 420.0, 300.0
    J = homography_jacobian(H, (u, v))
    h = 1e-4
    p0 = _apply_float64(u, v)
    pu = _apply_float64(u + h, v)
    pv = _apply_float64(u, v + h)
    J_num = np.column_stack([(pu - p0) / h, (pv - p0) / h])
    assert np.allclose(J, J_num, rtol=1e-6, atol=1e-8)


def test_footpoint_pixel_covariance_positive_and_scales_with_box_size() -> None:
    small = footpoint_pixel_covariance((100.0, 100.0, 110.0, 126.0), detection_confidence=0.91)
    large = footpoint_pixel_covariance((100.0, 100.0, 140.0, 200.0), detection_confidence=0.91)
    for cov in (small, large):
        vals = np.linalg.eigvalsh(cov)
        assert np.all(vals > 0.0)
    assert large[0, 0] > small[0, 0]
    assert large[1, 1] > small[1, 1]
    low_conf = footpoint_pixel_covariance((100.0, 100.0, 110.0, 126.0), detection_confidence=0.35)
    assert low_conf[0, 0] > small[0, 0]


def test_covariance_propagation_is_psd_and_inflates(base_calibration: CalibrationResult) -> None:
    uv = (430.0, 301.0)
    cov_px = footpoint_pixel_covariance((418.0, 262.0, 442.0, 301.0), detection_confidence=0.91)
    cov_yd = propagate_to_field_covariance(
        base_calibration.H, uv, cov_px, calibration_confidence=base_calibration.confidence
    )
    vals = np.linalg.eigvalsh(cov_yd)
    assert np.all(vals > 0.0)
    major, minor, _ = sigma_ellipse(cov_yd)
    assert major > 0.0 and minor > 0.0 and major >= minor

    drifted = propagate_to_field_covariance(
        base_calibration.H,
        uv,
        cov_px,
        calibration_confidence=base_calibration.confidence,
        extra_field_sigma_yd=0.5,
    )
    assert sigma_ellipse(drifted)[0] > major
    inflated = inflate_covariance(cov_yd, 2.0)
    assert sigma_ellipse(inflated)[0] == pytest.approx(2.0 * major, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Smoothing filter and kinematics
# ---------------------------------------------------------------------------
def test_constant_velocity_filter_reduces_jitter_and_recovers_velocity() -> None:
    filt = ConstantVelocityFieldFilter()
    sx, sy = 0.06, 0.06  # assumed measurement sigma (yd)
    cov = np.diag([sx**2, sy**2])
    rng = np.random.default_rng(7)
    v_true = (5.0, -1.5)
    raw_err: List[float] = []
    filt_err: List[float] = []
    for t in range(40):
        ts = t * DT
        true_xy = (20.0 + v_true[0] * ts, 30.0 + v_true[1] * ts)
        noisy = (true_xy[0] + rng.normal(0, sx), true_xy[1] + rng.normal(0, sy))
        if not filt.initialized:
            filt.initialize(noisy, cov, ts)
        else:
            filt.predict(ts)
            filt.update(noisy, cov)
        raw_err.append(float(np.hypot(noisy[0] - true_xy[0], noisy[1] - true_xy[1])))
        filt_err.append(float(np.hypot(filt.position[0] - true_xy[0], filt.position[1] - true_xy[1])))
    assert np.mean(filt_err[10:]) < np.mean(raw_err[10:])
    assert filt.velocity == pytest.approx(v_true, abs=0.35)


def test_kinematics_clip_and_acceleration_bounds() -> None:
    from football_vision.trajectory import acceleration_from_velocity, clip_speed, direction_rad, speed_yd_s

    v, clipped = clip_speed((15.0, 0.0), max_speed_yd_s=12.0)
    assert clipped and speed_yd_s(v) == pytest.approx(12.0)
    assert speed_yd_s(v) == pytest.approx(12.0)
    assert direction_rad((0.0, 3.0)) == pytest.approx(np.pi / 2)

    accel, mag, acc_clipped = acceleration_from_velocity(
        (0.0, 0.0), (10.0, 0.0), DT, max_accel_yd_s2=25.0
    )
    assert acc_clipped and mag == pytest.approx(25.0)
    assert abs(accel[0]) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 4. Jump / outlier rejection
# ---------------------------------------------------------------------------
def test_field_jump_gate_behaviour() -> None:
    gate = FieldJumpGate(gate_chi2_2dof=9.21)
    cov = np.diag([0.05**2, 0.05**2])
    ok = gate.evaluate((30.0, 25.0), (30.05, 25.02), cov)
    assert ok.accepted and ok.reason is None
    jump = gate.evaluate((30.0, 25.0), (32.5, 23.0), cov)
    assert jump.rejected and jump.reason == REASON_FIELD_GATE and jump.statistic > 9.21


def test_image_space_jump_gate_flags_implausible_speed() -> None:
    gate = ImageSpaceJumpGate(max_speed_px_per_frame=22.0)
    assert gate.evaluate((100.0, 200.0), (108.0, 205.0)).accepted
    flagged = gate.evaluate((100.0, 200.0), (140.0, 200.0))
    assert flagged.rejected and flagged.reason == REASON_IMAGE_GATE
    assert gate.evaluate(None, (10.0, 10.0)).accepted

    tracker = RejectionTracker(max_consecutive_rejections=3)
    assert tracker.register(True) is False
    assert tracker.register(True) is False
    assert tracker.register(True) is True
    assert tracker.register(False) is False
    assert tracker.consecutive == 0


# ---------------------------------------------------------------------------
# 5. Builder behaviour
# ---------------------------------------------------------------------------
def test_builder_never_positions_samples_with_unknown_geometry(base_calibration: CalibrationResult) -> None:
    builder = PlayerTrajectoryBuilder(fps=FPS)
    bad = _uncalibrated(base_calibration)
    samples = _run_stream(
        builder,
        [bad, bad, bad],
        [(24.0, 30.0)] * 3,
        observation_cals=[base_calibration] * 3,
    )
    assert len(samples) == 3
    for s in samples:
        assert s.geometry_state == "unknown"
        assert s.field_position is None
        assert s.position_source == "none"
        assert s.absolute_yardline is None
        assert s.x_coord_mode == "uncalibrated"
    assert builder.fabricated_field_positions == 0


def test_builder_rejects_injected_jump_and_preserves_trajectory(base_calibration: CalibrationResult) -> None:
    builder = PlayerTrajectoryBuilder(fps=FPS)
    cal = base_calibration
    n = 14
    cals = [cal] * n
    points = [(22.0 + 0.18 * t, 30.0 + 0.05 * t) for t in range(n)]
    jitter: List[Tuple[float, float]] = [(0.0, 0.0)] * n
    jump_frame = 8
    jump_px = 26.0
    jitter[jump_frame] = (jump_px, -0.5 * jump_px)
    samples = _run_stream(builder, cals, points, jitter=jitter)

    jumped = samples[jump_frame]
    assert jumped.is_outlier_rejected, "injected 26 px jump must be rejected by the field gate"
    assert jumped.rejection_reason == REASON_FIELD_GATE
    assert jumped.field_position is None
    assert jumped.predicted_position is not None
    assert jumped.position_source == "predicted_dead_reckoning"
    assert jumped.is_measurement_used is False
    assert builder.measurements_rejected >= 1

    # Trajectory must not be dragged by the outlier: the samples before and after
    # stay close to ground truth.
    errors = []
    for i, s in enumerate(samples):
        if i == jump_frame or s.field_position is None:
            continue
        errors.append(float(np.hypot(s.field_position[0] - points[i][0], s.field_position[1] - points[i][1])))
    assert max(errors) < 0.25


def test_builder_camera_pan_keeps_field_trajectory_stable(base_calibration: CalibrationResult) -> None:
    """A field-fixed player under a panning camera must stay field-fixed."""
    builder = PlayerTrajectoryBuilder(fps=FPS)
    n = 16
    x_yd, y_yd = 26.0, 30.0
    cals = []
    image_motion: List[float] = []
    prev_uv = None
    for t in range(n):
        T = np.array([[1.0, 0.0, 5.0 * t], [0.0, 1.0, 0.8 * t], [0.0, 0.0, 1.0]])
        cal_t = _calibration_with_H(base_calibration, base_calibration.H @ T)
        cals.append(cal_t)
        uv = cal_t.field_to_image([[x_yd, y_yd]])[0]
        if prev_uv is not None:
            image_motion.append(float(np.hypot(uv[0] - prev_uv[0], uv[1] - prev_uv[1])))
        prev_uv = uv

    samples = _run_stream(builder, cals, [(x_yd, y_yd)] * n)
    assert np.mean(image_motion) > 3.0, "camera pan must move the player in image space"
    positioned = [s for s in samples if s.field_position is not None]
    assert len(positioned) >= n - 2
    for s in positioned:
        err = float(np.hypot(s.field_position[0] - x_yd, s.field_position[1] - y_yd))
        assert err < 0.25
    # Field speed must stay near zero although the image footpoint moved tens of px.
    assert max(abs(s.speed_yd_s) for s in samples) < 0.6


def test_builder_dead_reckoning_is_labelled_and_bounded(base_calibration: CalibrationResult) -> None:
    builder = PlayerTrajectoryBuilder(fps=FPS, max_gap_frames=2, process_accel_std_yd_s2=5.0)
    cal = base_calibration
    bad = _uncalibrated(cal)
    vx = 5.0
    cals = [cal] * 5 + [bad] * 3 + [cal] * 3
    points = [(22.0 + vx * (t * DT), 30.0) for t in range(len(cals))]
    samples = _run_stream(builder, cals, points, observation_cals=[cal] * len(cals))

    gap = [s for s in samples if s.geometry_state == "unknown"]
    assert len(gap) == 3
    # First two gap frames may dead-reckon; beyond max_gap_frames no position at all.
    assert gap[0].position_source == "predicted_dead_reckoning"
    assert gap[0].field_position is None and gap[0].predicted_position is not None
    assert gap[-1].position_source == "none" and gap[-1].predicted_position is None
    for s in gap:
        assert s.field_position is None
    dr_errors = [
        float(np.hypot(s.predicted_position[0] - points[s.frame_id][0], s.predicted_position[1] - points[s.frame_id][1]))
        for s in gap
        if s.predicted_position is not None
    ]
    assert dr_errors and max(dr_errors) < 0.5
    # Re-acquisition after the gap re-initialises the filter and resumes measurement.
    recovered = [s for s in samples if s.geometry_state == "calibrated" and s.frame_id >= 8]
    assert recovered and recovered[-1].is_measurement_used
    assert any(s.reinitialized_after_gap for s in samples)


def test_builder_x_coord_mode_never_invents_absolute_yardline(base_calibration: CalibrationResult) -> None:
    rel_builder = PlayerTrajectoryBuilder(fps=FPS)
    rel_samples = _run_stream(rel_builder, [base_calibration] * 4, [(25.0, 30.0)] * 4)
    assert rel_samples[-1].x_coord_mode == base_calibration.x_coord_mode
    assert all(s.absolute_yardline is None for s in rel_samples)

    abs_cal = _calibration_with_H(base_calibration, base_calibration.H, mode="absolute")
    abs_builder = PlayerTrajectoryBuilder(fps=FPS)
    abs_samples = _run_stream(abs_builder, [abs_cal] * 4, [(25.0, 30.0)] * 4)
    assert abs_samples[-1].x_coord_mode == "absolute"
    assert abs_samples[-1].absolute_yardline is not None
    assert abs_samples[-1].absolute_yardline == pytest.approx(abs_samples[-1].field_position[0], abs=1e-3)


def test_builder_propagated_geometry_is_flagged_and_inflates_uncertainty(base_calibration: CalibrationResult) -> None:
    drift = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    prop = _calibration_with_H(
        base_calibration, base_calibration.H @ drift, propagated=True, age=3
    )
    builder = PlayerTrajectoryBuilder(fps=FPS)
    samples = _run_stream(builder, [prop] * 4, [(25.0, 30.0)] * 4)
    assert all(s.geometry_state == "propagated" for s in samples)
    assert samples[-1].field_position is not None
    # Propagated geometry is usable but must not be reported as exact.
    assert samples[-1].sigma_major_yd > 0.2

    cal_builder = PlayerTrajectoryBuilder(fps=FPS)
    cal_samples = _run_stream(cal_builder, [base_calibration] * 4, [(25.0, 30.0)] * 4)
    assert cal_samples[-1].sigma_major_yd < samples[-1].sigma_major_yd


def test_builder_rejects_invalid_construction_parameters() -> None:
    """Invalid construction parameters must raise instead of silently mis-scaling output."""
    from football_vision.trajectory.builder import PlayerTrajectoryBuilder

    for bad_fps in (0.0, -30.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PlayerTrajectoryBuilder(fps=bad_fps)
    with pytest.raises(ValueError):
        PlayerTrajectoryBuilder(fps=FPS, max_gap_frames=-1)
    with pytest.raises(ValueError):
        PlayerTrajectoryBuilder(fps=FPS, gate_warmup_updates=-1)


def test_builder_is_deterministic(base_calibration: CalibrationResult) -> None:
    def _run() -> List[Tuple[float, ...]]:
        builder = PlayerTrajectoryBuilder(fps=FPS)
        samples = _run_stream(builder, [base_calibration] * 8, [(23.0 + 0.2 * t, 29.0) for t in range(8)])
        return [
            (
                round(s.field_position[0], 9) if s.field_position else -1.0,
                round(s.field_position[1], 9) if s.field_position else -1.0,
                round(s.velocity_yd_s[0], 9),
                round(s.velocity_yd_s[1], 9),
                round(s.sigma_major_yd, 9),
            )
            for s in samples
        ]

    assert _run() == _run()


# ---------------------------------------------------------------------------
# 6. Benchmark integrity
# ---------------------------------------------------------------------------
def test_phase4_benchmark_metrics_and_invariants() -> None:
    assert BENCHMARK_JSON.exists(), "run benchmarks/evaluate_phase4_trajectories.py first"
    report = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    assert report["benchmark_version"] == "phase4_v1"
    assert report["calibration_layer_modified"] is False
    assert report["image_detector_accuracy_status"] == "unmeasured"
    assert report["determinism_check"]["deterministic"] is True

    total_samples = 0
    for split in ("train", "val", "test"):
        agg = report["splits"][split]
        assert agg["fabricated_field_positions"] == 0
        assert agg["fabricated_trajectory_samples"] == 0
        assert agg["absolute_yardline_violations"] == 0
        assert agg["samples_with_position_in_unknown_geometry"] == 0
        # TRAIN/VAL are uninterrupted; TEST deliberately includes a camera cut that
        # breaks track continuity, so completeness is required to be lower there.
        assert agg["track_completeness"] > 0.95 if split != "test" else agg["track_completeness"] > 0.80
        assert agg["field_pos_err_median_yd"] is not None
        assert agg["field_pos_err_median_yd"] < agg["raw_field_pos_err_median_yd"]
        assert agg["outlier_events_absorbed_into_track"] == 0
        # Observe/infer split and identity metrics must be reported for every split.
        assert agg["observed_sample_ratio"] > 0.9
        assert 0.0 <= agg["inferred_sample_ratio"] < 1.0
        assert agg["id_switches"] == agg["id_switches_active_swap"] + agg["id_switches_post_reinit"]
        # Smoothing must not shorten real motion: path length within a few % of GT.
        assert agg["motion_preservation_ratio_vs_gt"] == pytest.approx(1.0, abs=0.15)
        total_samples += agg["n_samples"]

    assert total_samples == sum(s["n_samples"] for s in report["sequences"])
    # Geometry-state accounting must cover every split combination explicitly.
    test_states = report["splits"]["test"]["geometry_state_sample_counts"]
    assert test_states["propagated"] > 0 and test_states["unknown"] > 0
    assert report["splits"]["test"]["positioned_by_geometry_state"]["unknown"] == 0

    # The real-frame smoke test never claims quantitative accuracy.
    smoke = report["real_frame_smoke_test"]
    assert smoke["quantitative_ground_truth_available"] is False
    assert smoke["fabricated_field_positions"] == 0

    # The camera-cut scenario must exercise the refusal/recovery state machine.
    cut = next(s for s in report["sequences"] if s["scenario"] == "camera_cut_refusal_and_recovery")
    assert cut["geometry_state_sample_counts"]["unknown"] > 0
    assert cut["positioned_by_geometry_state"]["unknown"] == 0
    assert cut["id_switches"] > 0, "a hard shot change must cost identity continuity"
    assert cut["projection_recovery_latency_frames_max"] > 0
    assert cut["recovery_latency_frames_max"] > 0
    assert cut["fabricated_trajectory_samples"] == 0


def test_phase4_benchmark_provenance_is_unambiguous() -> None:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "phase4_v1"
    prov = manifest["benchmark_provenance_and_semantics"]
    assert "UNMEASURED" in prov["image_detector_accuracy"]
    assert "NOT recorded NFL" in prov["player_motion"]
    assert manifest["frozen_parameters"]["gate_chi2_2dof"] == 9.21
    assert "zero thresholds or model constants tuned on TEST" in manifest["split_protocol"]["test"]

    # TEST sequences must exist and be frozen in the manifest.
    splits = {s["split"] for s in manifest["trajectory_sequences"]}
    assert splits == {"train", "val", "test"}


def test_trajectory_layer_does_not_import_calibration_internals() -> None:
    for path in (ROOT / "football_vision" / "trajectory").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "import football_vision.calibration" not in src
        assert "from football_vision.calibration" not in src


# ---------------------------------------------------------------------------
# 7. Track state, recovery latency, and motion preservation
# ---------------------------------------------------------------------------
def test_track_state_marks_observed_vs_coasted(base_calibration: CalibrationResult) -> None:
    builder = PlayerTrajectoryBuilder(fps=FPS)
    cal = base_calibration
    # Frames 3 and 4 have no usable geometry -> the track coasts in image space.
    bad = _uncalibrated(cal)
    cals = [cal, cal, cal, bad, bad, cal, cal]
    points = [(24.0 + 0.06 * t, 30.0) for t in range(len(cals))]
    samples = _run_stream(builder, cals, points, observation_cals=[cal] * len(cals))

    frame3 = next(s for s in samples if s.frame_id == 3)
    assert frame3.track_state == "observed", "footpoint is reliable: the player was observed"
    assert frame3.geometry_state == "unknown"
    assert frame3.field_position is None and frame3.predicted_position is not None
    assert frame3.position_source == "predicted_dead_reckoning"

    # A coasted (tracker-occluded) sample is explicitly marked as such.
    occluded = PlayerTrack(**{**_track_from_field_point(1, cal, 24.0, 30.0).__dict__})
    occluded.missed_frames = 2
    occluded.footpoint_estimate = FootpointEstimate(
        u_px=occluded.footpoint[0], v_px=occluded.footpoint[1], is_reliable=False,
        confidence=0.0, is_temporarily_occluded=True, unreliable_reason="track_occluded_unobserved",
    )
    b2 = PlayerTrajectoryBuilder(fps=FPS)
    s2 = b2.update([occluded], frame_id=0, calibration=cal)
    assert s2[0].track_state == "coasted"
    assert s2[0].position_source == "none"


def test_recovery_latency_is_measured_after_an_observed_gap(base_calibration: CalibrationResult) -> None:
    builder = PlayerTrajectoryBuilder(fps=FPS)
    cal = base_calibration
    n = 14
    points = [(24.0 + 0.10 * t, 30.0) for t in range(n)]
    # Frames 6,7,8 are absent from the detection stream entirely (track dropout).
    for t in range(n):
        if t in (6, 7, 8):
            continue
        trk = _track_from_field_point(1, cal, *points[t])
        builder.update([trk], frame_id=t, calibration=cal)
    assert builder.recovery_latencies_frames == [3], builder.recovery_latencies_frames
    assert builder.projection_recovery_latencies_frames == [3]
    # Sample at frame 6 can only dead-reckon, frame 9 is measured again.
    # (Verified through a second run that keeps frame indices contiguous.)
    builder2 = PlayerTrajectoryBuilder(fps=FPS)
    for t in range(n):
        missed = 3 if t in (6, 7, 8) else 0
        trk = _track_from_field_point(1, cal, *points[t], missed_frames=missed)
        samples = builder2.update([trk], frame_id=t, calibration=cal)
        assert len(samples) == 1
    assert builder2.projection_recovery_latencies_frames == [3]


def test_motion_is_not_smoothed_away(base_calibration: CalibrationResult) -> None:
    """A constant-velocity player must keep its real speed through the filter."""
    builder = PlayerTrajectoryBuilder(fps=FPS)
    cal = base_calibration
    n = 20
    vx = 6.0
    points = [(20.0 + vx * (t * DT), 30.0 + 0.5 * (t * DT)) for t in range(n)]
    samples = _run_stream(builder, [cal] * n, points, observation_cals=[cal] * n)
    positioned = [s for s in samples if s.field_position is not None]
    assert len(positioned) >= n - 3
    # Recovered speed must match the true speed (no attenuation).
    late = [s.speed_yd_s for s in positioned[6:]]
    assert float(np.median(late)) == pytest.approx(float(np.hypot(vx, 0.5)), abs=0.35)
    # Smoothed path length must match the ground-truth route length.
    route_len = float(
        np.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    )
    smoothed_len = float(positioned[-1].distance_cum_yd)
    assert smoothed_len == pytest.approx(route_len, rel=0.05)
    # Direction of motion is reported and correct.
    assert positioned[-1].direction_rad == pytest.approx(float(np.arctan2(0.5, vx)), abs=0.05)


def test_camera_cut_refusal_then_projection_recovery() -> None:
    """Panning camera + a hard shot change: refusal during the cut, recovery after.

    This mirrors the frozen ``traj_seq_08`` scenario shape: geometry is unusable
    for three frames (camera cut), then a re-established calibration is supplied.
    """
    import cv2

    if BASE_IMAGE is None:
        pytest.skip(real_frames_skip_reason())
    img = cv2.imread(str(BASE_IMAGE))
    assert img is not None
    cal = calibrate_frame(img, x_start_yd=15.0)
    n = 16
    cut_frames = (5, 6, 7)
    drift = np.array([[1.0, 0.0, 52.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]])
    builder = PlayerTrajectoryBuilder(fps=FPS)
    x_yd, y_yd = 24.0, 30.0

    cals = []
    observation = []
    for t in range(n):
        framing = cal.H @ (drift if t >= cut_frames[0] else np.eye(3))
        obs = _calibration_with_H(cal, framing)
        observation.append(obs)
        if t in cut_frames:
            cals.append(cal)
            cals[-1] = _uncalibrated(cal, reason="camera_cut_uncalibrated")
        else:
            cals.append(obs)

    samples = _run_stream(builder, cals, [(x_yd, y_yd)] * n, observation_cals=observation)
    by_frame = {s.frame_id: s for s in samples}

    for t in cut_frames:
        assert by_frame[t].geometry_state == "unknown"
        assert by_frame[t].projection_status == "calibration_refused:camera_cut_uncalibrated"
        assert by_frame[t].field_position is None
        assert by_frame[t].x_coord_mode == "uncalibrated"
        assert by_frame[t].absolute_yardline is None

    # Recovery: from frame 8 the re-established geometry yields correct projections.
    for t in (8, 9, 10):
        assert by_frame[t].field_position is not None
        err = float(np.hypot(by_frame[t].field_position[0] - x_yd, by_frame[t].field_position[1] - y_yd))
        assert err < 0.15
    # Projection-recovery latency is reported, not hidden.
    assert builder.projection_recovery_latencies_frames[0] == 3
    assert builder.fabricated_field_positions == 0


# ---------------------------------------------------------------------------
# 7. Audit-remediation regressions (integrity of the benchmark harness itself)
# ---------------------------------------------------------------------------
def _load_evaluator():
    """Import the benchmark harness as a module (no side effects at import time)."""
    import importlib.util

    path = ROOT / "benchmarks" / "evaluate_phase4_trajectories.py"
    spec = importlib.util.spec_from_file_location("phase4_evaluator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluator():
    return _load_evaluator()


@pytest.fixture(scope="module")
def benchmark() -> dict:
    assert BENCHMARK_JSON.exists(), "run benchmarks/evaluate_phase4_trajectories.py first"
    return json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))


def test_dominant_track_selection_uses_runtime_counters_only(evaluator) -> None:
    """Regression test: the selection must not react to ground-truth-derived fields."""
    candidates = [
        {"track_id": 9, "n_samples": 5, "n_positioned": 1, "n_observed": 1, "longest_observed_run": 1},
        {"track_id": 2, "n_samples": 12, "n_positioned": 12, "n_observed": 12, "longest_observed_run": 12},
        {"track_id": 4, "n_samples": 12, "n_positioned": 3, "n_observed": 3, "longest_observed_run": 2},
    ]
    assert evaluator.select_dominant_track(candidates) == 2  # lifetime tie -> lowest track_id
    assert evaluator.select_dominant_track(list(reversed(candidates))) == 2

    # Poison the candidate dicts with ground-truth-derived fields that an oracle rule would use:
    # a perfect error on a losing track and a terrible error on the winner must not move anything.
    poisoned = [dict(candidate) for candidate in candidates]
    poisoned[2].update({"field_pos_err_yd": 0.0, "field_pos_err_median_yd": 0.0, "gt_distance_yd": 0.0})
    poisoned[1].update({"field_pos_err_yd": 99.0, "field_pos_err_median_yd": 99.0, "gt_distance_yd": 99.0})
    assert evaluator.select_dominant_track(poisoned) == 2
    assert evaluator.select_dominant_track(list(reversed(poisoned))) == 2

    selection = evaluator.DOMINANT_TRACK_SELECTION
    assert selection["rule"] == evaluator.DOMINANT_TRACK_RULE
    assert selection["uses_ground_truth_trajectory"] is False
    assert selection["uses_ground_truth_error"] is False
    assert selection["uses_oracle_track_selection"] is False
    assert selection["uses_future_information"] is False
    assert not any("err" in item or "gt" in item for item in selection["inputs"])


def test_benchmark_dominant_track_is_reproducible_from_published_counters(benchmark) -> None:
    """The recorded dominant track must be re-derivable from runtime counters alone."""
    for sequence in benchmark["sequences"]:
        dominant_by_gt = {int(gid): int(tid) for gid, tid in sequence["dominant_track_id_by_gt"].items()}
        candidates_by_gt: dict = {}
        for row in sequence["track_table"]:
            candidates_by_gt.setdefault(int(row["gt_id"]), []).append(
                {"track_id": int(row["track_id"]), "n_samples": int(row["n_samples"])}
            )
        for gid, candidates in candidates_by_gt.items():
            expected = min(
                candidates,
                key=lambda c: (-c["n_samples"], c["track_id"]),
            )["track_id"]
            assert dominant_by_gt[gid] == expected, (sequence["sequence_id"], gid)


def test_benchmark_dominant_and_fragment_accounting_is_complete(benchmark) -> None:
    for sequence in benchmark["sequences"]:
        table = sequence["track_table"]
        assert sum(int(row["n_samples"]) for row in table) == sequence["n_samples"]
        # Every emitted sample is either in a dominant track, a fragment track, or unattributed.
        assert (
            sequence["dominant_track_samples_total"]
            + sequence["non_dominant_track_samples_total"]
            + sequence["unattributed_samples_total"]
            == sequence["n_samples"]
        )
        assert sequence["dominant_track_samples_without_position"] == (
            sequence["dominant_track_samples_total"] - sequence["dominant_samples"]
        )
        assert sequence["non_dominant_track_samples"] <= sequence["non_dominant_track_samples_total"]
        dominant_rows = [row for row in table if not row["excluded_from_dominant_metrics"]]
        assert len(dominant_rows) == sequence["n_players"]
        assert sequence["fragmentation_taxonomy_total"] == len(table)
        assert sum(sequence["fragmentation_samples_per_track"].values()) == sequence["n_samples"]

    for split in ("train", "val", "test"):
        agg = benchmark["splits"][split]
        assert (
            agg["dominant_track_samples_total"]
            + agg["non_dominant_track_samples_total"]
            + agg["unattributed_samples_total"]
            == agg["n_samples"]
        )
        # Fragmentation is visible in aggregate: the origin histogram closes over ALL tracks,
        # and the exclusion cross-tab closes over all non-dominant episodes.
        # The origin histogram counts (player, track) EPISODES: every track contributes one
        # episode per player label it carries, so a track that survives a player change adds
        # exactly one extra episode. Episodes therefore close over tracks + contaminated tracks.
        assert sum(agg["fragmentation_counts_by_origin"].values()) == agg["fragmentation_taxonomy_total"]
        assert agg["fragmentation_taxonomy_total"] == agg["n_tracks"] + agg[
            "identity_contamination"
        ]["tracks_with_multiple_gt_labels"]
        assert agg["fragmentation_n_matched_tracks"] == agg["n_tracks"] - agg[
            "fragmentation_counts_by_origin"
        ]["unmatched_spurious_track"]
        assert agg["fragmentation_counts_by_origin"]["unmatched_spurious_track"] == 0
        assert agg["fragmentation_excluded_from_dominant_total"] == sum(
            agg["fragmentation_excluded_by_origin"].values()
        )
        # Episode-level partition: exactly one dominant episode per player-sequence, every
        # other episode is an excluded fragment. (A track that survives a player change can be
        # dominant for one player and an excluded fragment for the other, so this partition is
        # stated over episodes, not tracks.)
        assert agg["n_players"] + agg["fragmentation_excluded_from_dominant_total"] == agg[
            "fragmentation_taxonomy_total"
        ]
        assert agg["fragmentation_dominant_tracks_total"] <= agg["n_players"]
        assert agg["n_players"] == 6 * agg["num_sequences"]
        # Every track that is not the dominant track for some player must be excluded from the
        # dominant metrics; identity-contaminated tracks count as two episodes.
        expected_excluded = sum(
            int(sequence["fragmentation_fragment_tracks_excluded_from_dominant"])
            for sequence in benchmark["sequences"]
            if sequence["split"] == split
        )
        assert agg["fragmentation_fragment_tracks_excluded_from_dominant"] == expected_excluded
        assert agg["fragmentation_fragment_tracks_excluded_from_dominant"] > 0 or split == "train"


def test_fragmentation_taxonomy_is_exhaustive_and_internally_consistent(benchmark, evaluator) -> None:
    categories = set(evaluator.FRAGMENTATION_CATEGORIES)
    assert categories == set(evaluator.FRAGMENTATION_DEFINITIONS)
    for sequence in benchmark["sequences"]:
        taxonomy = sequence["fragmentation_taxonomy"]
        assert set(taxonomy) == categories
        assert sum(taxonomy.values()) == sequence["fragmentation_taxonomy_total"] == len(sequence["track_table"])
        counted: dict = {category: 0 for category in categories}
        for row in sequence["track_table"]:
            assert row["origin_category"] in categories
            counted[row["origin_category"]] += 1
        assert counted == taxonomy, sequence["sequence_id"]
        # Identity contamination is surfaced, not hidden by track-level attribution.
        contamination = sequence["identity_contamination"]
        multi_label = {
            int(row["track_id"])
            for row in sequence["track_table"]
            if row["track_carries_multiple_labels"]
        }
        assert len(multi_label) == contamination["tracks_with_multiple_gt_labels"]
        assert contamination["tracks_without_any_label"] == 0  # every fixture detection is labelled


def test_rejection_accounting_separates_clean_from_corrupted(benchmark) -> None:
    for split in ("train", "val", "test"):
        accounting = benchmark["splits"][split]["rejection_accounting"]
        gated = accounting["clean_samples_gate_evaluated"]
        rejected = accounting["clean_samples_rejected"]
        accepted = accounting["clean_samples_accepted"]
        warmup = accounting["clean_samples_warmup_bypassed"]
        # "Accepted" counts samples the gate accepted: rejected + accepted == gated. Samples
        # accepted while the gate was bypassed by design (warm-up) are counted separately, so a
        # reader can never mistake them for gated acceptances.
        assert rejected + accepted == gated
        assert accounting["clean_samples_accepted_during_warmup"] <= warmup
        assert accounting["false_rejection_rate"] == pytest.approx(rejected / gated, abs=1e-6)
        assert accounting["false_rejection_rate_including_warmup"] == pytest.approx(
            rejected / (gated + warmup), abs=1e-6
        )
        assert (
            accounting["corrupted_samples_rejected"] + accounting["corrupted_samples_accepted"]
            == accounting["corrupted_samples_evaluated"]
        )
        if accounting["corrupted_samples_evaluated"]:
            assert accounting["true_rejection_rate_on_corrupted"] == pytest.approx(
                accounting["corrupted_samples_rejected"] / accounting["corrupted_samples_evaluated"],
                abs=1e-6,
            )
        else:
            assert accounting["true_rejection_rate_on_corrupted"] is None
        # "Gate rejected bad data" is separated from "gate rejected good data", in two
        # explicitly named scopes (all tracks vs dominant-track accounting).
        assert accounting["rejections_all_tracks"] == (
            accounting["rejections_by_innovation_gate"]
            + accounting["rejections_by_image_space_plausibility_gate"]
        )
        assert accounting["rejections_on_dominant_tracks"] == rejected + accounting[
            "corrupted_samples_rejected"
        ]
        assert accounting["rejections_on_dominant_tracks"] == (
            accounting["rejections_by_innovation_gate_on_dominant_tracks"]
            + accounting["rejections_by_plausibility_gate_on_dominant_tracks"]
        )
        assert accounting["plausibility_gate_can_reject_measurements"] is False
        assert accounting["rejections_by_image_space_plausibility_gate"] == 0
        assert sum(accounting["unpositioned_by_reason"].values()) == accounting["unpositioned_samples"]
        assert accounting["dead_reckoned_samples_after_false_rejection"] >= 0
    assert benchmark["splits"]["train"]["rejection_accounting"]["clean_samples_rejected"] > 0
    assert benchmark["splits"]["train"]["rejection_accounting"]["dead_reckoned_samples_after_false_rejection"] > 0


def test_uncertainty_diagnostics_are_serialisable_and_labelled_uncalibrated(benchmark) -> None:
    assert benchmark["uncertainty_status"] == "not_statistically_calibrated"
    assert "not" in benchmark["uncertainty_status"]
    assert "calibrated" in benchmark["uncertainty_note"]
    for split in ("train", "val", "test"):
        diagnostics = benchmark["splits"][split]["uncertainty_diagnostics"]
        assert diagnostics["status"] == "not_statistically_calibrated"
        assert diagnostics["coverage_evaluated_samples"] > 0
        for key in (
            "coverage_68_pct_smoothed_covariance",
            "coverage_95_pct_smoothed_covariance",
            "coverage_68_pct_raw_measurement_covariance",
            "median_mahalanobis_distance_smoothed",
            "covariance_scale_needed_k2_smoothed",
            "observed_jitter_std_u_px",
            "assumed_footpoint_sigma_u_px",
        ):
            assert diagnostics[key] is not None, (split, key)
        assert 0.0 <= diagnostics["coverage_68_pct_smoothed_covariance"] <= 100.0
        assert (
            diagnostics["coverage_68_pct_smoothed_covariance"]
            <= diagnostics["coverage_95_pct_smoothed_covariance"]
        )
        assert diagnostics["covariance_scale_needed_k2_smoothed"] > 0.0
        assert set(diagnostics["coverage_68_pct_by_geometry_state"]) <= {
            "calibrated",
            "propagated",
        }
        assert "unknown" not in diagnostics["coverage_68_pct_by_geometry_state"]
        assert "unknown" not in diagnostics["coverage_samples_by_geometry_state"]
        assert set(diagnostics["coverage_68_pct_by_covariance_inflation"]) == {
            "inflated",
            "not_inflated",
        }
        assert sum(diagnostics["coverage_samples_by_covariance_inflation"].values()) == (
            diagnostics["coverage_evaluated_samples"]
        )
        # Round-trips through JSON without loss or non-finite values.
        assert json.loads(json.dumps(diagnostics)) == diagnostics
    analysis = benchmark["uncertainty_calibration_analysis"]
    assert analysis["fit_split"] == "train+val"
    assert analysis["test_split_used_for_fitting"] is False
    assert analysis["applied_to_shipped_covariance"] is False
    assert analysis["shipped_covariance_unchanged"] is True


def test_warmup_ablation_is_train_val_only(benchmark) -> None:
    ablation = benchmark["gate_warmup_ablation"]
    assert ablation["test_split_consulted"] is False
    assert sorted(ablation["results"]) == ["train", "val"]
    for split in ("train", "val"):
        result = ablation["results"][split]
        assert result["shipped_warmup_false_rejections"] == (
            benchmark["splits"][split]["rejection_accounting"]["clean_samples_rejected"]
        )
        assert result["gate_from_2nd_update_evaluated"] == result["shipped_warmup_evaluated"]
    assert (
        ablation["results"]["train"]["gate_from_2nd_update_false_rejections"]
        >= ablation["results"]["train"]["shipped_warmup_false_rejections"]
    )


def test_benchmark_labels_synthetic_versus_real_and_experimental_outputs(benchmark) -> None:
    assert benchmark["benchmark_kind"] == "synthetic_trajectory_benchmark"
    assert benchmark["real_multiframe_trajectory_accuracy"] == "not_measured"
    assert "not yet measured" in benchmark["real_multiframe_trajectory_accuracy_note"]
    assert benchmark["acceleration_status"] == "experimental_not_validated"
    assert "EXPERIMENTAL" in benchmark["acceleration_note"].upper()
    assert benchmark["image_detector_accuracy_status"] == "unmeasured"
    assert benchmark["image_detector_quantitative_metrics"] is None
    separation = benchmark["detector_metric_separation_note"]
    assert "FIXTURE HARNESS" in separation
    assert "must never" in separation
    assert "unmeasured" in separation
    fine_scope = benchmark["metric_scopes"]["model_observable_no_ground_truth"]
    coarse_scope = benchmark["metric_scopes"]["harness_gt_dependent"]
    assert "n_tracks" in fine_scope and "field_pos_err_median_yd" not in fine_scope
    assert "field_pos_err_median_yd" in coarse_scope
    assert "id_switches" in coarse_scope
    label_scope = benchmark["metric_scopes"]["harness_label_dependent"]
    assert "false_rejections" in label_scope and "outliers_injected" in label_scope
    assert "image_space_ema_field_err_median_yd" in coarse_scope
    assert benchmark["real_frame_smoke_test"]["evaluation_scope"] == (
        "single_frame_integration_smoke_test_no_ground_truth_trajectory"
    )
    assert benchmark["real_frame_smoke_test"]["quantitative_ground_truth_available"] is False


def test_phase3_fixture_metrics_are_not_presented_as_detector_accuracy() -> None:
    """Phase 3 fixture numbers are harness pass-through statistics, not detector accuracy."""
    phase3_json = ROOT / "outputs" / "phase3_tracking_benchmark.json"
    assert phase3_json.exists()
    report = json.loads(phase3_json.read_text(encoding="utf-8"))
    provenance = report["benchmark_integrity_audit"]["detector_provenance"]
    assert "NOT an image detector benchmark" in provenance["controlled_sequences_train_val_test"]
    assert "unmeasured" in provenance["real_nfl_frames_val_test"]
    for sequence in report["sequences"]:
        assert sequence["detector_implementation"] == "fixture_detector_v1"


# ---------------------------------------------------------------------------
# 8. Audit-remediation regressions added by this pass (integrity of the harness)
# ---------------------------------------------------------------------------
def test_dominant_track_rule_has_no_ground_truth_term(evaluator) -> None:
    """Structural guard: the rule's *code* names no ground-truth or error quantity."""
    import ast
    import inspect
    import re

    tree = ast.parse(inspect.getsource(evaluator.select_dominant_track))
    docstring = ast.get_docstring(tree.body[0], clean=False)
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != docstring:
            identifiers.add(node.value.lower())
    for forbidden in ("gt", "gt_id", "ground_truth", "gt_table", "error", "err", "oracle"):
        pattern = rf"(?:^|_){re.escape(forbidden)}(?:_|$)"
        hits = sorted(name for name in identifiers if re.search(pattern, name))
        assert hits == [], (forbidden, hits)


def test_benchmark_determinism_is_verified_on_every_split(benchmark) -> None:
    determinism = benchmark["determinism_check"]
    assert determinism["deterministic"] is True
    assert determinism["differing_keys"] == []
    assert {entry["split"] for entry in determinism["per_sequence"]} == {"train", "val", "test"}
    assert all(entry["deterministic"] for entry in determinism["per_sequence"])


def test_production_code_carries_no_ground_truth_identifier() -> None:
    """Ground truth lives in the harness only; the package must never read a GT label."""
    offenders = []
    for path in sorted((ROOT / "football_vision").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for token in ("gt_id", "ground_truth", "gt_table"):
            if token in source:
                offenders.append((str(path.relative_to(ROOT)), token))
    assert offenders == [], offenders


def test_invalid_trajectory_states_are_rejected() -> None:
    """A refused geometry must never silently become a fabricated field position."""
    from football_vision.schema import TrajectorySample

    with pytest.raises(ValueError):
        TrajectorySample(
            track_id=1,
            frame_id=0,
            timestamp_s=0.0,
            geometry_state="unknown",
            x_coord_mode="relative_10yd",
            field_position=(10.0, 20.0),
        )
    # ... and a claimed source with no position at all is refusable too.
    with pytest.raises(ValueError):
        TrajectorySample(
            track_id=1,
            frame_id=0,
            timestamp_s=0.0,
            geometry_state="calibrated",
            x_coord_mode="relative_10yd",
            field_position=None,
            predicted_position=None,
            position_source="measured_smoothed",
        )
    with pytest.raises(ValueError):
        TrajectorySample(
            track_id=1,
            frame_id=0,
            timestamp_s=0.0,
            geometry_state="calibrated",
            x_coord_mode="relative_5yd",
            absolute_yardline=30.0,
        )


def test_real_frame_assets_resolve_or_skip_cleanly() -> None:
    """The real-frame smoke path must resolve via the data-path API or skip with a reason."""
    from football_vision.data_paths import (
        NFL_FRAMES_ENV,
        missing_nfl_frames,
        real_frames_skip_reason,
        require_nfl_frame,
        resolve_nfl_frame,
    )

    assert resolve_nfl_frame("definitely-not-a-real-frame.jpg") is None
    with pytest.raises(FileNotFoundError) as excinfo:
        require_nfl_frame("definitely-not-a-real-frame.jpg")
    assert NFL_FRAMES_ENV in str(excinfo.value)

    payload = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    smoke = payload["real_frame_smoke_test"]
    if smoke["skipped"]:
        assert smoke["skip_reason"], "a skipped smoke test must explain itself"
    else:
        # This is provenance from the historical run, not a file dependency of
        # this machine. Actual image tests resolve assets through data_paths.
        assert isinstance(smoke["frame_path"], str) and smoke["frame_path"]
        assert Path(smoke["frame_path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}
    if missing_nfl_frames():
        assert NFL_FRAMES_ENV in real_frames_skip_reason()
    else:
        assert real_frames_skip_reason().startswith("real NFL frame assets")


# ---------------------------------------------------------------------------
# 9. Label integrity: the audit vocabulary is load-bearing, so it is tested
# ---------------------------------------------------------------------------
def test_docs_and_json_carry_the_audit_label_vocabulary() -> None:
    """Guard against documentation drifting back into overstatement."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "synthetic",
        "smoke test",
        "unmeasured",
        "not measured",
        "not statistically calibrated",
        "experimental",
        "harness-only",
    ):
        assert phrase in readme, f"README lost the audit label: {phrase!r}"

    assumptions = (ROOT / "docs" / "ASSUMPTIONS_AND_LIMITATIONS.md").read_text(encoding="utf-8").lower()
    for phrase in ("not statistically calibrated", "not measured", "unmeasured"):
        assert phrase in assumptions, phrase

    report = (ROOT / "docs" / "PHASE4_TRAJECTORY_REPORT.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "synthetic_trajectory_benchmark",
        "not_statistically_calibrated",
        "experimental_not_validated",
        "not_measured",
        "single_frame_integration_smoke_test_no_ground_truth_trajectory",
        "calibrated confidence interval",  # explicitly disclaimed, never claimed
    ):
        assert phrase in report, phrase


# ---------------------------------------------------------------------------
# 10. Quantitative-audit invariants (all-track accounting, rejection, dead reckoning)
# ---------------------------------------------------------------------------
def test_split_float_sums_match_sequence_sums(benchmark) -> None:
    """Path lengths are float quantities and must not be truncated at split level."""
    for split in ("train", "val", "test"):
        seqs = [s for s in benchmark["sequences"] if s["split"] == split]
        for key in ("smoothed_path_length_yd", "raw_projection_path_length_yd", "gt_path_length_yd"):
            exact = sum(float(s[key]) for s in seqs)
            assert benchmark["splits"][split][key] == pytest.approx(exact, abs=1e-4), (split, key)


def test_dead_reckoning_causes_partition_every_sequence_and_split(benchmark) -> None:
    """Every dead-reckoned sample is caused by a rejection or by a missing measurement."""
    for seq in benchmark["sequences"]:
        assert (
            int(seq["dead_reckoning_samples_from_rejection"])
            + int(seq["dead_reckoning_samples_from_missing_measurement"])
            == int(seq["dead_reckoning_samples"])
        )
    for split in ("train", "val", "test"):
        agg = benchmark["splits"][split]
        assert (
            int(agg["dead_reckoning_samples_from_rejection"])
            + int(agg["dead_reckoning_samples_from_missing_measurement"])
            == int(agg["dead_reckoning_samples"])
        )
        # Reconciliation with the gate accounting: rejection-caused dead reckoning is exactly the
        # false rejections that fell back to dead reckoning plus the corrupted rejections.
        accounting = agg["rejection_accounting"]
        assert int(agg["dead_reckoning_samples_from_rejection"]) == int(
            accounting["dead_reckoned_samples_after_false_rejection"]
        ) + int(accounting["corrupted_samples_rejected"])


def test_estimation_path_partition_closes(benchmark) -> None:
    """Every positioned dominant sample is an accepted measurement or a post-gap re-init sample."""
    for seq in benchmark["sequences"]:
        assert (
            int(seq["accepted_measurement_samples"]) + int(seq["post_reinit_measurement_samples"])
            == int(seq["measured_dominant_samples"])
        )
        assert int(seq["reinit_after_rejection_measurement_samples"]) <= int(
            seq["accepted_measurement_samples"]
        )
    for split in ("train", "val", "test"):
        agg = benchmark["splits"][split]
        assert (
            int(agg["accepted_measurement_samples"]) + int(agg["post_reinit_measurement_samples"])
            == int(agg["measured_dominant_samples"])
        )


def test_rejection_rates_are_complementary_with_explicit_denominators(benchmark) -> None:
    for split in ("train", "val", "test"):
        accounting = benchmark["splits"][split]["rejection_accounting"]
        evaluated = int(accounting["corrupted_samples_evaluated"])
        if evaluated:
            trr = accounting["true_rejection_rate_on_corrupted"]
            far = accounting["false_acceptance_rate_on_corrupted"]
            assert trr == pytest.approx(
                accounting["corrupted_samples_rejected"] / evaluated, abs=1e-6
            )
            assert far == pytest.approx(
                accounting["corrupted_samples_accepted"] / evaluated, abs=1e-6
            )
            assert trr + far == pytest.approx(1.0, abs=1e-6)
        else:
            assert accounting["true_rejection_rate_on_corrupted"] is None
            assert accounting["false_acceptance_rate_on_corrupted"] is None
        # Every rate must ship with its denominator in the same block.
        assert set(accounting["denominators"]) == {
            "false_rejection_rate",
            "false_rejection_rate_including_warmup",
            "true_rejection_rate_on_corrupted",
            "false_acceptance_rate_on_corrupted",
            "geometry_refusal",
        }
        for key, text in accounting["denominators"].items():
            assert "clean_samples" in text or "corrupted_samples" in text or "unpositioned" in text, key


def test_quantitative_audit_report_is_consistent_with_the_benchmark(benchmark) -> None:
    """The audit document must quote the frozen JSON, not a stale copy of it."""
    doc_path = ROOT / "docs" / "phase4_quantitative_audit.md"
    assert doc_path.exists(), "run benchmarks/render_phase4_audit.py"
    doc = doc_path.read_text(encoding="utf-8")
    splits = benchmark["splits"]
    test = splits["test"]
    for needle in (
        f"{test['coverage_68_pct']:.2f}%",
        f"{test['coverage_95_pct']:.2f}%",
        f"{100 * test['false_rejection_rate']:.2f}%",
        f"{100 * test['dominant_fraction_of_gt_player_frames']:.2f}%",
        benchmark["dominant_track_selection"]["rule"],
    ):
        assert needle in doc, needle
    # The uncalibrated-uncertainty statement must be explicit and unmistakable.
    assert "not statistically calibrated" in doc.lower()
    # The historical (pre-extension) TEST figures stay labelled as historical.
    assert "historical" in doc
