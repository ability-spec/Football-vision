"""Canonical data contracts and schemas for Football-Vision (Phase 0 / Phase 1 / Phase 16A).

Defines standardized, serializable records across calibration, detection, tracking,
identity, field projection, trajectories, and play analytics so every subsystem
preserves provenance and explicit confidence/uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from football_vision.field_spec import (
    MIN_CALIBRATION_CONFIDENCE,
    VALID_METRICS_BY_X_COORD_MODE,
)

XCoordMode = Literal["absolute", "relative_10yd", "relative_5yd", "uncalibrated"]
ObservationState = Literal["detected", "tracked", "projected", "inferred", "unknown"]
CalibrationFailureMode = Literal[
    "invalid_image",
    "no_hough_lines",
    "insufficient_yard_lines",
    "insufficient_hash_ticks",
    "missing_hash_rows",
    "homography_solve_failed",
    "implausible_homography",
    "insufficient_confidence",
    "camera_cut_uncalibrated",
    "confidence_expired",
]


@dataclass
class CalibrationResult:
    """Standardized output of the Phase 1 Field Calibration Engine."""

    success: bool
    H: Optional[np.ndarray]                       # 3x3 image (u,v) -> field (X_yd, Y_yd)
    H_inv: Optional[np.ndarray]                   # 3x3 field (X_yd, Y_yd) -> image (u,v)
    yard_lines: List[Dict[str, Any]] = field(default_factory=list)
    far_hash_row: Optional[Dict[str, Any]] = None
    near_hash_row: Optional[Dict[str, Any]] = None
    far_sideline: Optional[Dict[str, Any]] = None
    vp_yard: Optional[Tuple[float, float]] = None
    hash_tick_inliers: Optional[np.ndarray] = None
    hash_tick_candidates: Optional[np.ndarray] = None
    ridge_pixel_count: int = 0
    ridge_orth_median_px: float = np.nan
    ridge_orth_mean_px: float = np.nan
    hash_row_rmse_px: float = np.nan
    even_idx_number_score: float = np.nan
    odd_idx_number_score: float = np.nan
    detected_ten_yard_parity: Optional[int] = None  # 0 if k=0,2,4 are 10-yd multiples; 1 if k=1,3,5
    plausible_orientation_scale: bool = False
    runtime_ms: float = 0.0
    notes: List[str] = field(default_factory=list)
    image_size: Optional[Tuple[int, int]] = None    # (width_px, height_px)
    confidence: float = 0.0
    confidence_components: Dict[str, float] = field(default_factory=dict)
    x_coord_mode: XCoordMode = "uncalibrated"
    failure_reason: Optional[str] = None
    is_temporally_propagated: bool = False
    propagation_age: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def vanishing_point(self) -> Optional[Tuple[float, float]]:
        """Alias for vp_yard matching the Phase 1 CalibrationResult specification."""
        return self.vp_yard

    @property
    def hash_marks(self) -> Dict[str, Any]:
        """Structured view of detected hash rows and inlier/candidate tick arrays."""
        return {
            "far_hash_row": self.far_hash_row,
            "near_hash_row": self.near_hash_row,
            "inliers": self.hash_tick_inliers,
            "candidates": self.hash_tick_candidates,
            "rmse_px": None if np.isnan(self.hash_row_rmse_px) else float(self.hash_row_rmse_px),
        }

    @property
    def sidelines(self) -> Dict[str, Any]:
        """Structured view of detected sidelines."""
        return {"far_sideline": self.far_sideline}

    @property
    def field_homography(self) -> Optional[np.ndarray]:
        """Alias for H (image (u,v) -> field (X_yd, Y_yd))."""
        return self.H

    @property
    def residuals(self) -> Dict[str, Optional[float]]:
        """In-sample geometric fit residuals in pixels."""
        return {
            "ridge_orth_median_px": None if np.isnan(self.ridge_orth_median_px) else float(self.ridge_orth_median_px),
            "ridge_orth_mean_px": None if np.isnan(self.ridge_orth_mean_px) else float(self.ridge_orth_mean_px),
            "hash_row_rmse_px": None if np.isnan(self.hash_row_rmse_px) else float(self.hash_row_rmse_px),
        }

    @property
    def valid_metrics(self) -> Tuple[str, ...]:
        """Return tuple of downstream analytics metrics valid under this result's coordinate mode."""
        if not self.can_project():
            return ()
        return VALID_METRICS_BY_X_COORD_MODE.get(self.x_coord_mode, ())

    def can_project(self, min_confidence: float = MIN_CALIBRATION_CONFIDENCE) -> bool:
        """Return True iff calibration succeeded, H is valid, and confidence >= min_confidence."""
        return bool(
            self.success
            and self.H is not None
            and self.H_inv is not None
            and self.plausible_orientation_scale
            and self.x_coord_mode != "uncalibrated"
            and self.confidence >= min_confidence
        )

    def image_to_field(
        self,
        pts_uv: Union[Sequence[Sequence[float]], np.ndarray],
        min_confidence: float = MIN_CALIBRATION_CONFIDENCE,
    ) -> Optional[np.ndarray]:
        """Project image pixel coordinates (N, 2) -> field coordinates (N, 2) in yards.

        Refuses projection (returns None) if calibration failed or confidence < min_confidence.
        """
        if not self.can_project(min_confidence=min_confidence):
            return None
        arr = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        proj = cv2.perspectiveTransform(arr, self.H).reshape(-1, 2)
        return proj.astype(np.float64)

    def field_to_image(
        self,
        pts_xy_yd: Union[Sequence[Sequence[float]], np.ndarray],
        min_confidence: float = MIN_CALIBRATION_CONFIDENCE,
    ) -> Optional[np.ndarray]:
        """Reproject field coordinates (N, 2) in yards -> image pixel coordinates (N, 2).

        Refuses reprojection (returns None) if calibration failed or confidence < min_confidence.
        """
        if not self.can_project(min_confidence=min_confidence):
            return None
        arr = np.asarray(pts_xy_yd, dtype=np.float32).reshape(-1, 1, 2)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        proj = cv2.perspectiveTransform(arr, self.H_inv).reshape(-1, 2)
        return proj.astype(np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration metadata, confidence, and residuals to a JSON-compatible dictionary."""
        return {
            "success": self.success,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "H": self.H.tolist() if self.H is not None else None,
            "H_inv": self.H_inv.tolist() if self.H_inv is not None else None,
            "yard_lines_count": len(self.yard_lines),
            "yard_lines": self.yard_lines,
            "far_sideline_detected": self.far_sideline is not None,
            "vp_yard": list(self.vp_yard) if self.vp_yard is not None else None,
            "hash_tick_inliers_count": int(len(self.hash_tick_inliers)) if self.hash_tick_inliers is not None else 0,
            "hash_tick_candidates_count": int(len(self.hash_tick_candidates)) if self.hash_tick_candidates is not None else 0,
            "ridge_pixel_count": self.ridge_pixel_count,
            "residuals": self.residuals,
            "ridge_orth_median_px": None if np.isnan(self.ridge_orth_median_px) else float(self.ridge_orth_median_px),
            "ridge_orth_mean_px": None if np.isnan(self.ridge_orth_mean_px) else float(self.ridge_orth_mean_px),
            "hash_row_rmse_px": None if np.isnan(self.hash_row_rmse_px) else float(self.hash_row_rmse_px),
            "even_idx_number_score": None if np.isnan(self.even_idx_number_score) else float(self.even_idx_number_score),
            "odd_idx_number_score": None if np.isnan(self.odd_idx_number_score) else float(self.odd_idx_number_score),
            "detected_ten_yard_parity": self.detected_ten_yard_parity,
            "plausible_orientation_scale": self.plausible_orientation_scale,
            "confidence": float(self.confidence),
            "confidence_components": dict(self.confidence_components),
            "x_coord_mode": self.x_coord_mode,
            "valid_metrics": list(self.valid_metrics),
            "failure_reason": self.failure_reason,
            "is_temporally_propagated": self.is_temporally_propagated,
            "propagation_age": self.propagation_age,
            "diagnostics": dict(self.diagnostics),
            "runtime_ms": float(self.runtime_ms),
            "notes": list(self.notes),
        }


@dataclass
class Detection:
    """Single-frame bounding box detection for a player, referee, or football."""

    frame_index: int
    box_xyxy: Tuple[float, float, float, float]
    cls: str
    confidence: float

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, _, x2, y2 = self.box_xyxy
        return ((x1 + x2) * 0.5, float(y2))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerIdentity:
    """Probabilistic player identity and team assignment for a tracked player."""

    track_id: int
    team: Optional[str] = None
    team_confidence: float = 0.0
    jersey_number: Optional[str] = None
    jersey_confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackState:
    """Temporal state of a tracked object at a specific frame."""

    track_id: int
    frame_index: int
    box_xyxy: Tuple[float, float, float, float]
    cls: str
    detection_confidence: float
    state: ObservationState = "tracked"
    identity: Optional[PlayerIdentity] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FieldProjection:
    """Projected real-world football field coordinates (in yards) with dual confidence."""

    frame_index: int
    timestamp_s: float
    track_id: Optional[int]
    cls: str
    x_yd: float
    y_yd: float
    x_coord_mode: XCoordMode
    detection_confidence: float
    calibration_confidence: float
    identity_confidence: float = 0.0
    state: ObservationState = "projected"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryPoint:
    """Smoothed field-space kinematic state for a single track at one timestamp."""

    track_id: int
    frame_index: int
    timestamp_s: float
    x_yd: float
    y_yd: float
    vx_yd_s: float = 0.0
    vy_yd_s: float = 0.0
    speed_yd_s: float = 0.0
    accel_yd_s2: float = 0.0
    distance_cum_yd: float = 0.0
    direction_rad: float = 0.0
    confidence: float = 0.0
    state: ObservationState = "projected"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayMetric:
    """Single play-level metric preserving units, data source, confidence, and limitations."""

    name: str
    value: Any
    units: str
    data_source: str
    confidence: float
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayRecord:
    """Structured analytics record for a segmented football play."""

    game_id: str
    play_id: str
    start_frame: int
    snap_frame: Optional[int]
    end_frame: int
    x_coord_mode: XCoordMode
    metrics: Dict[str, PlayMetric] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "play_id": self.play_id,
            "start_frame": self.start_frame,
            "snap_frame": self.snap_frame,
            "end_frame": self.end_frame,
            "x_coord_mode": self.x_coord_mode,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "events": list(self.events),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Phase 3 Schemas: Player Detection, Footpoint, Team Assignment, Tracking,
# and Field Projection
# ---------------------------------------------------------------------------

TeamLabel = Literal["TEAM_A", "TEAM_B", "UNKNOWN"]


@dataclass
class FootpointEstimate:
    """Player ground-contact estimate with explicit reliability and occlusion flags.

    Never claim a reliable field position when ``is_reliable`` is False.
    """

    u_px: float
    v_px: float
    is_reliable: bool
    confidence: float
    is_partially_truncated: bool = False
    is_near_sideline: bool = False
    is_player_crossing: bool = False
    is_temporarily_occluded: bool = False
    unreliable_reason: Optional[str] = None

    @property
    def xy(self) -> Tuple[float, float]:
        return (float(self.u_px), float(self.v_px))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerDetection:
    """Phase 3 player detection schema independent of field calibration."""

    frame_id: int
    detection_id: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    footpoint: Tuple[float, float]           # (u_px, v_px) bottom-center estimate
    footpoint_estimate: FootpointEstimate
    detector_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "detection_id": self.detection_id,
            "bbox": tuple(float(v) for v in self.bbox),
            "confidence": float(self.confidence),
            "footpoint": (float(self.footpoint[0]), float(self.footpoint[1])),
            "footpoint_estimate": self.footpoint_estimate.to_dict(),
            "detector_metadata": dict(self.detector_metadata),
        }


@dataclass
class TeamAssignment:
    """Baseline torso appearance team assignment result."""

    team: TeamLabel
    confidence: float
    torso_luma: float = 0.0
    torso_chroma: float = 0.0
    non_turf_ratio: float = 0.0
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectedFieldPosition:
    """Projected player field position preserving x_coord_mode semantics.

    Never invents an absolute yardline number when ``x_coord_mode != "absolute"``.
    """

    x_yd: float
    y_yd: float
    x_coord_mode: XCoordMode
    calibration_confidence: float
    footpoint_confidence: float
    is_absolute_yardline: bool = False
    absolute_yardline: Optional[float] = None
    valid_metrics: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def xy_yd(self) -> Tuple[float, float]:
        return (float(self.x_yd), float(self.y_yd))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerTrack:
    """Phase 3 tracked player state across frames with gated field projection."""

    track_id: int
    frame_id: int
    bbox: Tuple[float, float, float, float]
    footpoint: Tuple[float, float]
    footpoint_estimate: FootpointEstimate
    team: TeamLabel
    team_confidence: float
    detection_confidence: float
    age: int
    missed_frames: int
    hits: int = 1
    field_position: Optional[Tuple[float, float]] = None
    projected_position: Optional[ProjectedFieldPosition] = None
    x_coord_mode: XCoordMode = "uncalibrated"
    projection_available: bool = False
    projection_status: str = "unprojected"
    velocity_uv: Tuple[float, float] = (0.0, 0.0)
    short_occlusion_recovered: bool = False
    detector_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "bbox": tuple(float(v) for v in self.bbox),
            "footpoint": (float(self.footpoint[0]), float(self.footpoint[1])),
            "footpoint_estimate": self.footpoint_estimate.to_dict(),
            "team": self.team,
            "team_confidence": float(self.team_confidence),
            "detection_confidence": float(self.detection_confidence),
            "age": self.age,
            "missed_frames": self.missed_frames,
            "hits": self.hits,
            "field_position": (
                (float(self.field_position[0]), float(self.field_position[1]))
                if self.field_position is not None
                else None
            ),
            "projected_position": (
                self.projected_position.to_dict()
                if self.projected_position is not None
                else None
            ),
            "x_coord_mode": self.x_coord_mode,
            "projection_available": self.projection_available,
            "projection_status": self.projection_status,
            "velocity_uv": (float(self.velocity_uv[0]), float(self.velocity_uv[1])),
            "short_occlusion_recovered": self.short_occlusion_recovered,
            "detector_metadata": dict(self.detector_metadata),
        }



# ---------------------------------------------------------------------------
# Phase 4 Schemas: Field-Space Trajectories, Kinematics, and Uncertainty
# ---------------------------------------------------------------------------

GeometryState = Literal["calibrated", "propagated", "unknown"]
PositionSource = Literal["measured_smoothed", "predicted_dead_reckoning", "none"]
TrackObservationState = Literal["observed", "coasted"]


@dataclass
class TrajectorySample:
    """Single-frame field-space trajectory sample for one track.

    ``field_position`` is populated ONLY from an accepted measurement that was
    projected with a valid geometry state (``calibrated`` / ``propagated``).
    ``predicted_position`` is an explicitly labelled dead-reckoning estimate that
    is NEVER a claim of true position (see ``position_source``).
    """

    track_id: int
    frame_id: int
    timestamp_s: float
    geometry_state: GeometryState
    x_coord_mode: XCoordMode
    track_state: TrackObservationState = "coasted"
    image_footpoint: Optional[Tuple[float, float]] = None
    field_position: Optional[Tuple[float, float]] = None
    raw_field_position: Optional[Tuple[float, float]] = None
    predicted_position: Optional[Tuple[float, float]] = None
    position_source: PositionSource = "none"
    velocity_yd_s: Tuple[float, float] = (0.0, 0.0)
    speed_yd_s: float = 0.0
    accel_yd_s2: float = 0.0
    direction_rad: float = 0.0
    distance_cum_yd: float = 0.0
    is_acceleration_clipped: bool = False
    is_speed_clipped: bool = False
    sigma_major_yd: float = 0.0
    sigma_minor_yd: float = 0.0
    covariance_xy: Optional[Tuple[float, float, float]] = None  # (sxx, sxy, syy)
    uncertainty_inflated: bool = False
    is_outlier_rejected: bool = False
    rejection_reason: Optional[str] = None
    is_measurement_used: bool = False
    image_space_jump_flagged: bool = False
    missed_frames: int = 0
    filter_age_frames: int = 0
    frames_since_measurement: int = 0
    reinitialized_after_gap: bool = False
    reinitialized_after_rejections: bool = False
    projection_status: str = "unprojected"
    absolute_yardline: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["image_footpoint"] = (
            [float(self.image_footpoint[0]), float(self.image_footpoint[1])]
            if self.image_footpoint is not None
            else None
        )
        for key in ("field_position", "raw_field_position", "predicted_position"):
            val = getattr(self, key)
            out[key] = [float(val[0]), float(val[1])] if val is not None else None
        out["velocity_yd_s"] = [float(self.velocity_yd_s[0]), float(self.velocity_yd_s[1])]
        out["covariance_xy"] = (
            [float(v) for v in self.covariance_xy] if self.covariance_xy is not None else None
        )
        return out


@dataclass
class FieldTrajectory:
    """Persistent per-track field-space trajectory assembled from trajectory samples."""

    track_id: int
    fps: float
    x_coord_mode: XCoordMode
    samples: List[TrajectorySample] = field(default_factory=list)
    detector_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_measured(self) -> int:
        return sum(1 for s in self.samples if s.position_source == "measured_smoothed")

    @property
    def n_predicted(self) -> int:
        return sum(1 for s in self.samples if s.position_source == "predicted_dead_reckoning")

    @property
    def n_observed(self) -> int:
        return sum(1 for s in self.samples if s.track_state == "observed")

    @property
    def n_coasted(self) -> int:
        return sum(1 for s in self.samples if s.track_state == "coasted")

    def geometry_state_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"calibrated": 0, "propagated": 0, "unknown": 0}
        for s in self.samples:
            counts[s.geometry_state] = counts.get(s.geometry_state, 0) + 1
        return counts

    @property
    def total_distance_yd(self) -> float:
        return float(self.samples[-1].distance_cum_yd) if self.samples else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "fps": float(self.fps),
            "x_coord_mode": self.x_coord_mode,
            "detector_name": self.detector_name,
            "n_samples": self.n_samples,
            "n_measured": self.n_measured,
            "n_predicted": self.n_predicted,
            "n_observed": self.n_observed,
            "n_coasted": self.n_coasted,
            "geometry_state_counts": self.geometry_state_counts(),
            "total_distance_yd": self.total_distance_yd,
            "notes": list(self.notes),
            "samples": [s.to_dict() for s in self.samples],
        }
