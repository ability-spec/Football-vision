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

from football_vision import (
    CalibrationResult,
    FieldProjector,
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
BASE_IMAGE = "/home/user/image-search/nfl-game-broadcast-screenshot-1st-and-10-5.jpg"
FPS = 30.0
DT = 1.0 / FPS


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_calibration() -> CalibrationResult:
    import cv2

    img = cv2.imread(BASE_IMAGE)
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
        assert agg["absolute_yardline_violations"] == 0
        assert agg["samples_with_position_in_unknown_geometry"] == 0
        assert agg["track_completeness"] > 0.90
        assert agg["field_pos_err_median_yd"] is not None
        assert agg["field_pos_err_median_yd"] < agg["raw_field_pos_err_median_yd"]
        assert agg["outlier_events_absorbed_into_track"] == 0
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
