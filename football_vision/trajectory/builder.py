"""Phase 4 field-space trajectory builder.

Consumes the Phase 3 ``PlayerTrack`` stream (which itself consumes the Phase 3
detector/tracker interfaces) and produces persistent, per-track field-space
trajectories with explicit geometry states, uncertainty, jump rejection,
smoothing, and kinematics.

Design invariants
-----------------
1. Phase 2/3 calibration code is untouched; ``CalibrationResult`` remains the
   geometry contract and ``FieldProjector`` remains the projection gate.
2. ``field_position`` is populated ONLY from an accepted measurement projected
   with a projectable geometry state (``calibrated`` / ``propagated``) and a
   reliable, non-coasting footpoint. Dead-reckoned estimates live in a separate,
   explicitly labelled ``predicted_position`` field and are never a position claim.
3. Smoothing runs exclusively in field space (yards), never in image space, so
   camera pan/zoom cannot leak into the trajectory.
4. Rejected measurements are flagged, never silently deleted. Persistent
   disagreement resets the filter explicitly (auditable via flags).
5. ``absolute_yardline`` is only populated when ``x_coord_mode == "absolute"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from football_vision.projection.projector import FieldProjector
from football_vision.schema import (
    CalibrationResult,
    FieldTrajectory,
    GeometryState,
    PlayerTrack,
    TrajectorySample,
)
from football_vision.trajectory.geometry_state import (
    geometry_state_is_projectable,
    geometry_state_propagation_age,
    resolve_geometry_state,
)
from football_vision.trajectory.kinematics import (
    DEFAULT_MAX_ACCEL_YD_S2,
    DEFAULT_MAX_SPEED_YD_S,
    acceleration_from_velocity,
    clip_speed,
    direction_rad,
    speed_yd_s,
)
from football_vision.trajectory.outlier import (
    FieldJumpGate,
    GateDecision,
    ImageSpaceJumpGate,
    RejectionTracker,
)
from football_vision.trajectory.smoothing import (
    DEFAULT_GATE_CHI2_2DOF,
    DEFAULT_PROCESS_ACCEL_STD_YD_S2,
    ConstantVelocityFieldFilter,
)
from football_vision.trajectory.uncertainty import (
    FOOTPOINT_EDGE_SIGMA_PX,
    footpoint_pixel_covariance,
    inflate_covariance,
    propagate_to_field_covariance,
    sigma_ellipse,
)

# Frozen engineering assumptions (a priori; documented in the Phase 4 report).
DEFAULT_MAX_GAP_FRAMES = 3                  # frames of dead reckoning before "no position"
DEFAULT_PROPAGATED_DRIFT_YD_PER_AGE = 0.25  # assumed optical-flow propagation drift
DEFAULT_MAX_CONSECUTIVE_REJECTIONS = 3
DEFAULT_IMAGE_JUMP_MAX_SPEED_PX_PER_FRAME = 22.0
DEFAULT_GATE_WARMUP_UPDATES = 2   # standard track initiation: state is not yet estimable


@dataclass
class _TrackTrajectoryState:
    """Persistent per-track trajectory state (survives frames, keyed by track_id)."""

    track_id: int
    filter: Optional[ConstantVelocityFieldFilter] = None
    rejections: RejectionTracker = field(default_factory=RejectionTracker)
    last_measured_xy: Optional[Tuple[float, float]] = None
    last_measured_frame: Optional[int] = None
    last_measured_velocity: Tuple[float, float] = (0.0, 0.0)
    last_reported_velocity: Tuple[float, float] = (0.0, 0.0)
    last_position: Optional[Tuple[float, float]] = None
    last_position_frame: Optional[int] = None
    last_accepted_position: Optional[Tuple[float, float]] = None
    last_uv: Optional[Tuple[float, float]] = None
    last_uv_frame: Optional[int] = None
    distance_cum_yd: float = 0.0
    frames_since_measurement: int = 0
    last_observed_frame: Optional[int] = None
    last_projectable_frame: Optional[int] = None
    reinitialized_after_gap: bool = False
    reinitialized_after_rejections: bool = False


class PlayerTrajectoryBuilder:
    """Builds field-space trajectories from a Phase 3 ``PlayerTrack`` stream."""

    def __init__(
        self,
        *,
        fps: float = 30.0,
        projector: Optional[FieldProjector] = None,
        process_accel_std_yd_s2: float = DEFAULT_PROCESS_ACCEL_STD_YD_S2,
        gate_chi2_2dof: float = DEFAULT_GATE_CHI2_2DOF,
        max_speed_yd_s: float = DEFAULT_MAX_SPEED_YD_S,
        max_accel_yd_s2: float = DEFAULT_MAX_ACCEL_YD_S2,
        max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
        max_consecutive_rejections: int = DEFAULT_MAX_CONSECUTIVE_REJECTIONS,
        propagated_drift_yd_per_age: float = DEFAULT_PROPAGATED_DRIFT_YD_PER_AGE,
        image_jump_max_speed_px_per_frame: float = DEFAULT_IMAGE_JUMP_MAX_SPEED_PX_PER_FRAME,
        footpoint_edge_sigma_px: float = FOOTPOINT_EDGE_SIGMA_PX,
        extra_sigma_px: float = 0.0,
        gate_warmup_updates: int = DEFAULT_GATE_WARMUP_UPDATES,
    ) -> None:
        self.fps = float(fps)
        self.dt_s = 1.0 / max(1e-6, float(fps))
        self.projector = projector if projector is not None else FieldProjector()
        self.max_speed_yd_s = float(max_speed_yd_s)
        self.max_accel_yd_s2 = float(max_accel_yd_s2)
        self.max_gap_frames = int(max_gap_frames)
        self.propagated_drift_yd_per_age = float(propagated_drift_yd_per_age)
        self.footpoint_edge_sigma_px = float(footpoint_edge_sigma_px)
        self.extra_sigma_px = float(extra_sigma_px)
        # Track-initiation warm-up: while a track has fewer than this many accepted
        # updates its velocity is not yet estimable, so the innovation gate is not
        # statistically meaningful and is bypassed. The gate THRESHOLD is unchanged;
        # this is standard M-of-N track initiation, not a threshold relaxation.
        self.gate_warmup_updates = int(gate_warmup_updates)
        self.warmup_measurements: int = 0

        self.field_gate = FieldJumpGate(gate_chi2_2dof)
        self.image_gate = ImageSpaceJumpGate(image_jump_max_speed_px_per_frame)
        self._filter_kwargs = dict(
            process_accel_std_yd_s2=float(process_accel_std_yd_s2),
            gate_chi2_2dof=float(gate_chi2_2dof),
            max_speed_yd_s=float(max_speed_yd_s),
        )
        self._max_consecutive_rejections = int(max_consecutive_rejections)

        self._states: Dict[int, _TrackTrajectoryState] = {}
        self._trajectories: Dict[int, FieldTrajectory] = {}
        self._detector_names: Dict[int, str] = {}

        # Auditable counters
        self.fabricated_field_positions: int = 0
        # Recovery audit: frames between the last observed sample and the next one.
        self.recovery_latencies_frames: List[int] = []
        self.projection_recovery_latencies_frames: List[int] = []
        self.samples_emitted: int = 0
        self.measurements_rejected: int = 0
        self.filter_reinitializations: int = 0
        self.warmup_measurements: int = 0
        self.runtime_ms: float = 0.0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._states.clear()
        self._trajectories.clear()
        self._detector_names.clear()
        self.fabricated_field_positions = 0
        self.recovery_latencies_frames = []
        self.projection_recovery_latencies_frames = []
        self.samples_emitted = 0
        self.measurements_rejected = 0
        self.filter_reinitializations = 0
        self.warmup_measurements = 0
        self.runtime_ms = 0.0

    # -- helpers -----------------------------------------------------------
    def _new_filter(self) -> ConstantVelocityFieldFilter:
        return ConstantVelocityFieldFilter(**self._filter_kwargs)

    def _geometry_extra_sigma_yd(self, calibration: Optional[CalibrationResult]) -> float:
        """Additive field-space drift allowance for the current geometry state."""
        age = geometry_state_propagation_age(calibration)
        return self.propagated_drift_yd_per_age * age

    # -- main entry point --------------------------------------------------
    def update(
        self,
        tracks: Sequence[PlayerTrack],
        *,
        frame_id: int,
        calibration: Optional[CalibrationResult],
    ) -> List[TrajectorySample]:
        """Ingest one frame of Phase 3 tracks and emit one trajectory sample per track."""
        t0 = time.perf_counter()
        timestamp_s = float(frame_id) / self.fps
        geometry_state: GeometryState = resolve_geometry_state(calibration)
        projectable = geometry_state_is_projectable(geometry_state)
        extra_field_sigma_yd = self._geometry_extra_sigma_yd(calibration)

        samples: List[TrajectorySample] = []

        for trk in tracks:
            st = self._states.get(trk.track_id)
            if st is None:
                st = _TrackTrajectoryState(track_id=trk.track_id)
                self._states[trk.track_id] = st

            detector_name = str(trk.detector_metadata.get("detector_name", "")) or None
            if detector_name:
                self._detector_names[trk.track_id] = detector_name

            # --- Phase 3 projection gate (unmodified, reused) --------------
            proj, projection_status = self.projector.project_footpoint(
                trk.footpoint_estimate,
                calibration,
                missed_frames=trk.missed_frames,
            )
            image_footpoint = (float(trk.footpoint[0]), float(trk.footpoint[1]))

            # image-space jump plausibility (informational; the only gate that
            # can run when geometry is unknown)
            img_decision = self.image_gate.evaluate(
                st.last_uv,
                image_footpoint,
                dt_frames=max(1, int(frame_id) - int(st.last_uv_frame))
                if st.last_uv_frame is not None
                else 1,
            )
            image_jump_flagged = img_decision.rejected and geometry_state == "unknown"

            measured_xy: Optional[Tuple[float, float]] = None
            measurement_cov: Optional[np.ndarray] = None
            if proj is not None and calibration is not None and calibration.H is not None:
                measured_xy = (float(proj.x_yd), float(proj.y_yd))
                cov_px = footpoint_pixel_covariance(
                    trk.bbox,
                    trk.footpoint_estimate,
                    detection_confidence=trk.detection_confidence,
                    edge_sigma_px=self.footpoint_edge_sigma_px,
                    extra_sigma_px=self.extra_sigma_px,
                )
                measurement_cov = propagate_to_field_covariance(
                    calibration.H,
                    image_footpoint,
                    cov_px,
                    calibration_confidence=float(calibration.confidence),
                    extra_field_sigma_yd=extra_field_sigma_yd,
                )

            # --- measurement handling --------------------------------------
            position: Optional[Tuple[float, float]] = None
            predicted: Optional[Tuple[float, float]] = None
            position_cov: Optional[np.ndarray] = None
            position_source = "none"
            is_measurement_used = False
            is_outlier_rejected = False
            rejection_reason: Optional[str] = None
            uncertainty_inflated = False
            reinit_after_gap = False
            reinit_after_rejections = False

            in_warmup = False
            if measured_xy is not None and measurement_cov is not None:
                st.frames_since_measurement = 0
                if st.filter is None or not st.filter.initialized:
                    st.filter = self._new_filter()
                    v0 = (0.0, 0.0)
                    if st.last_measured_xy is not None and st.last_measured_frame is not None:
                        gap = int(frame_id) - int(st.last_measured_frame)
                        if 0 < gap <= 2 * self.max_gap_frames:
                            v0 = (
                                (measured_xy[0] - st.last_measured_xy[0]) / (gap * self.dt_s),
                                (measured_xy[1] - st.last_measured_xy[1]) / (gap * self.dt_s),
                            )
                            v0, _ = clip_speed(v0, max_speed_yd_s=self.max_speed_yd_s)
                            reinit_after_gap = gap > 1
                    st.filter.initialize(measured_xy, measurement_cov, timestamp_s, velocity_yd_s=v0)
                    position = st.filter.position
                    position_cov = st.filter.position_covariance
                    position_source = "measured_smoothed"
                    is_measurement_used = True
                else:
                    st.filter.predict(timestamp_s)
                    in_warmup = st.filter.n_updates <= self.gate_warmup_updates
                    if in_warmup:
                        # Track initiation: accept the measurement and let the state
                        # become estimable before the innovation gate is applied.
                        self.warmup_measurements += 1
                        decision = GateDecision(accepted=True, statistic=0.0, threshold=float("inf"))
                    else:
                        decision = self.field_gate.evaluate(
                            st.filter.position,
                            measured_xy,
                            st.filter.cov[np.ix_([0, 1], [0, 1])] + measurement_cov,
                        )
                    if decision.accepted:
                        _, _, _, accepted = st.filter.update(
                            measured_xy, measurement_cov, reject_if_gated=not in_warmup
                        )
                        assert accepted
                        st.rejections.register(False)
                        position = st.filter.position
                        position_cov = st.filter.position_covariance
                        position_source = "measured_smoothed"
                        is_measurement_used = True
                    else:
                        self.measurements_rejected += 1
                        is_outlier_rejected = True
                        rejection_reason = decision.reason
                        predicted = st.filter.position
                        position_cov = inflate_covariance(st.filter.position_covariance, 1.5)
                        position_source = "predicted_dead_reckoning"
                        uncertainty_inflated = True
                        if st.rejections.register(True):
                            # Persistent disagreement: reset explicitly and re-acquire
                            st.filter.initialize(measured_xy, measurement_cov, timestamp_s)
                            self.filter_reinitializations += 1
                            reinit_after_rejections = True
                            st.rejections.register(False)
                            position = st.filter.position
                            position_cov = st.filter.position_covariance
                            predicted = None
                            position_source = "measured_smoothed"
                            is_measurement_used = True

                if is_measurement_used:
                    st.last_measured_xy = measured_xy
                    st.last_measured_frame = int(frame_id)
                    st.last_measured_velocity = st.filter.velocity
            else:
                st.frames_since_measurement += 1
                if st.filter is not None and st.filter.initialized:
                    if st.frames_since_measurement <= self.max_gap_frames:
                        st.filter.predict(timestamp_s)
                        predicted = st.filter.position
                        position_cov = inflate_covariance(
                            st.filter.position_covariance,
                            1.0 + 0.40 * st.frames_since_measurement,
                        )
                        position_source = "predicted_dead_reckoning"
                        uncertainty_inflated = True
                    else:
                        # Too stale to claim even a dead-reckoning estimate.
                        st.filter = None
                        position_source = "none"

            # --- kinematics -------------------------------------------------
            reported_position = position if position is not None else predicted
            # Velocity is reported only alongside a reported position; a sample that
            # can claim neither a measured nor a dead-reckoned position reports no
            # motion rather than implying movement that was never estimated.
            if (
                reported_position is not None
                and st.filter is not None
                and st.filter.initialized
            ):
                velocity, speed_clipped = clip_speed(
                    st.filter.velocity, max_speed_yd_s=self.max_speed_yd_s
                )
            else:
                velocity, speed_clipped = (0.0, 0.0), False

            accel_mag = 0.0
            accel_clipped = False
            if st.last_position_frame is not None and int(frame_id) != int(st.last_position_frame):
                _, accel_mag, accel_clipped = acceleration_from_velocity(
                    st.last_reported_velocity,
                    velocity,
                    self.dt_s * max(1, int(frame_id) - int(st.last_position_frame)),
                    max_accel_yd_s2=self.max_accel_yd_s2,
                )

            # --- cumulative distance (accepted measurements only) -----------
            # Distance is only accrued between two accepted measurements, so
            # dead-reckoned guesses can never inflate the distance travelled.
            if position is not None and st.last_accepted_position is not None:
                st.distance_cum_yd += float(
                    np.hypot(
                        position[0] - st.last_accepted_position[0],
                        position[1] - st.last_accepted_position[1],
                    )
                )
            if position is not None:
                st.last_accepted_position = position

            # --- assemble sample -------------------------------------------
            mode = calibration.x_coord_mode if calibration is not None else "uncalibrated"
            abs_yd: Optional[float] = None
            if position is not None and mode == "absolute":
                abs_yd = round(float(position[0]), 3)

            sigma_major = sigma_minor = 0.0
            cov_tuple: Optional[Tuple[float, float, float]] = None
            if position_cov is not None and reported_position is not None:
                sigma_major, sigma_minor, _ = sigma_ellipse(position_cov)
                cov_tuple = (
                    float(position_cov[0, 0]),
                    float(position_cov[0, 1]),
                    float(position_cov[1, 1]),
                )

            provenance: Dict[str, Any] = {
                "detector_name": trk.detector_metadata.get("detector_name"),
                "detector_source": trk.detector_metadata.get("source"),
                "team": trk.team,
                "team_confidence": float(trk.team_confidence),
            }
            for passthrough in ("gt_id", "scenario", "note"):
                if passthrough in trk.detector_metadata:
                    provenance[passthrough] = trk.detector_metadata[passthrough]

            observed = int(trk.missed_frames) == 0 and bool(trk.footpoint_estimate.is_reliable)
            if observed and st.last_observed_frame is not None:
                gap = int(frame_id) - int(st.last_observed_frame) - 1
                if gap > 0:
                    self.recovery_latencies_frames.append(gap)
            if observed:
                st.last_observed_frame = int(frame_id)

            # Field-position recovery latency: frames between two positioned samples
            # for this track (i.e. how long the trajectory had no field position).
            if position is not None:
                if st.last_projectable_frame is not None:
                    proj_gap = int(frame_id) - int(st.last_projectable_frame) - 1
                    if proj_gap > 0:
                        self.projection_recovery_latencies_frames.append(proj_gap)
                st.last_projectable_frame = int(frame_id)

            sample = TrajectorySample(
                track_id=int(trk.track_id),
                frame_id=int(frame_id),
                timestamp_s=round(timestamp_s, 6),
                geometry_state=geometry_state,
                x_coord_mode=mode,
                track_state="observed" if observed else "coasted",
                image_footpoint=image_footpoint,
                field_position=position,
                raw_field_position=measured_xy,
                predicted_position=predicted,
                position_source=position_source,
                velocity_yd_s=(round(float(velocity[0]), 4), round(float(velocity[1]), 4)),
                speed_yd_s=round(speed_yd_s(velocity), 4),
                accel_yd_s2=round(float(accel_mag), 4),
                direction_rad=round(direction_rad(velocity), 4),
                distance_cum_yd=round(float(st.distance_cum_yd), 4),
                is_acceleration_clipped=bool(accel_clipped),
                is_speed_clipped=bool(speed_clipped),
                sigma_major_yd=round(float(sigma_major), 4),
                sigma_minor_yd=round(float(sigma_minor), 4),
                covariance_xy=cov_tuple,
                uncertainty_inflated=bool(uncertainty_inflated),
                is_outlier_rejected=bool(is_outlier_rejected),
                rejection_reason=rejection_reason,
                is_measurement_used=bool(is_measurement_used),
                image_space_jump_flagged=bool(image_jump_flagged),
                missed_frames=int(trk.missed_frames),
                filter_age_frames=int(st.filter.n_updates) if st.filter is not None else 0,
                frames_since_measurement=int(st.frames_since_measurement),
                reinitialized_after_gap=bool(reinit_after_gap),
                reinitialized_after_rejections=bool(reinit_after_rejections),
                projection_status=projection_status,
                absolute_yardline=abs_yd,
                provenance=provenance,
            )

            # ---- fabrication self-check (invariant, never expected to fire) --
            if sample.field_position is not None and (
                not projectable
                or not trk.footpoint_estimate.is_reliable
                or trk.missed_frames > 0
            ):
                self.fabricated_field_positions += 1

            samples.append(sample)

            # ---- persistent trajectory bookkeeping -------------------------
            traj = self._trajectories.get(trk.track_id)
            if traj is None:
                traj = FieldTrajectory(
                    track_id=int(trk.track_id),
                    fps=self.fps,
                    x_coord_mode=mode,
                    detector_name=detector_name,
                )
                self._trajectories[trk.track_id] = traj
            traj.samples.append(sample)

            st.last_uv = image_footpoint
            st.last_uv_frame = int(frame_id)
            if reported_position is not None:
                st.last_position = reported_position
                st.last_position_frame = int(frame_id)
                st.last_reported_velocity = velocity
            self.samples_emitted += 1

        self.runtime_ms += (time.perf_counter() - t0) * 1000.0
        samples.sort(key=lambda s: s.track_id)
        return samples

    # -- outputs -----------------------------------------------------------
    def finalize(self) -> List[FieldTrajectory]:
        """Return all persistent trajectories, ordered by ``track_id``."""
        for traj in self._trajectories.values():
            traj.notes = [
                f"detector={traj.detector_name or 'unknown'}",
                f"samples={traj.n_samples} measured={traj.n_measured} predicted={traj.n_predicted}",
            ]
        return [self._trajectories[k] for k in sorted(self._trajectories)]

    def trajectory(self, track_id: int) -> Optional[FieldTrajectory]:
        return self._trajectories.get(int(track_id))
