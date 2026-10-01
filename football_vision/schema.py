"""Canonical data contracts and schemas for Football-Vision (Phase 0 / Phase 16A).

Defines standardized, serializable records across calibration, detection, tracking,
identity, field projection, trajectories, and play analytics so every subsystem
preserves provenance and explicit confidence/uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

XCoordMode = Literal["absolute", "relative_10yd", "relative_5yd", "uncalibrated"]
ObservationState = Literal["detected", "tracked", "projected", "inferred", "unknown"]


@dataclass
class CalibrationResult:
    """Standardized output of the Field Calibration Engine."""

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
    x_coord_mode: XCoordMode = "uncalibrated"
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration metadata and residuals to a JSON-compatible dictionary."""
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
            "ridge_orth_median_px": None if np.isnan(self.ridge_orth_median_px) else float(self.ridge_orth_median_px),
            "ridge_orth_mean_px": None if np.isnan(self.ridge_orth_mean_px) else float(self.ridge_orth_mean_px),
            "hash_row_rmse_px": None if np.isnan(self.hash_row_rmse_px) else float(self.hash_row_rmse_px),
            "even_idx_number_score": None if np.isnan(self.even_idx_number_score) else float(self.even_idx_number_score),
            "odd_idx_number_score": None if np.isnan(self.odd_idx_number_score) else float(self.odd_idx_number_score),
            "detected_ten_yard_parity": self.detected_ten_yard_parity,
            "plausible_orientation_scale": self.plausible_orientation_scale,
            "confidence": float(self.confidence),
            "x_coord_mode": self.x_coord_mode,
            "failure_reason": self.failure_reason,
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
