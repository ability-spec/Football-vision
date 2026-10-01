"""Jump / outlier rejection for Phase 4 field-space trajectories.

Two complementary gates are provided:

``FieldJumpGate``
    Statistical gate in field space. A projected measurement is rejected when its
    Mahalanobis distance to the constant-velocity prediction exceeds a chi-square
    (2 dof) quantile. The measurement is *never silently deleted*: the sample is
    flagged (``is_outlier_rejected``, ``rejection_reason``) and reported as a
    dead-reckoning prediction instead.

``ImageSpaceJumpGate``
    Physical plausibility gate used when no usable geometry exists (geometry state
    ``unknown``) and therefore no field-space gate is possible. It flags an image
    footpoint jump that implies an implausible image-space speed between frames.
    This gate can only *flag*; it never fabricates a field position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from football_vision.trajectory.uncertainty import mahalanobis_distance_sq

REASON_FIELD_GATE = "jump_rejected_field_mahalanobis_gate"
REASON_IMAGE_GATE = "jump_flagged_image_space_speed_gate"


@dataclass
class GateDecision:
    """Result of applying a jump gate to one measurement."""

    accepted: bool
    statistic: float
    threshold: float
    reason: Optional[str] = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


class FieldJumpGate:
    """Chi-square innovation gate in field space."""

    def __init__(self, gate_chi2_2dof: float = 9.21) -> None:
        self.gate_chi2_2dof = float(gate_chi2_2dof)

    def evaluate(
        self,
        predicted_xy_yd: Sequence[float],
        measured_xy_yd: Sequence[float],
        innovation_covariance: np.ndarray,
    ) -> GateDecision:
        nu = np.array(
            [
                float(measured_xy_yd[0]) - float(predicted_xy_yd[0]),
                float(measured_xy_yd[1]) - float(predicted_xy_yd[1]),
            ]
        )
        d2 = mahalanobis_distance_sq(nu, innovation_covariance)
        accepted = d2 <= self.gate_chi2_2dof
        return GateDecision(
            accepted=accepted,
            statistic=d2,
            threshold=self.gate_chi2_2dof,
            reason=None if accepted else REASON_FIELD_GATE,
        )


class ImageSpaceJumpGate:
    """Plausibility gate on consecutive image-space footpoints (unknown geometry only)."""

    def __init__(self, max_speed_px_per_frame: float = 22.0) -> None:
        self.max_speed_px_per_frame = float(max_speed_px_per_frame)

    def evaluate(
        self,
        previous_uv: Optional[Sequence[float]],
        current_uv: Sequence[float],
        *,
        dt_frames: int = 1,
    ) -> GateDecision:
        if previous_uv is None:
            return GateDecision(accepted=True, statistic=0.0, threshold=1.0)
        dist_px = float(
            np.hypot(
                float(current_uv[0]) - float(previous_uv[0]),
                float(current_uv[1]) - float(previous_uv[1]),
            )
        )
        limit_px = self.max_speed_px_per_frame * max(1, int(dt_frames))
        ratio = dist_px / max(1e-6, limit_px)
        flagged = ratio > 1.0
        return GateDecision(
            accepted=not flagged,
            statistic=float(ratio * ratio),
            threshold=1.0,
            reason=None if not flagged else REASON_IMAGE_GATE,
        )


@dataclass
class RejectionTracker:
    """Counts consecutive rejections so a persistent disagreement can reset the track."""

    max_consecutive_rejections: int = 3
    consecutive: int = 0

    def register(self, rejected: bool) -> bool:
        """Register a decision; returns True when the reset threshold is crossed."""
        if rejected:
            self.consecutive += 1
        else:
            self.consecutive = 0
        return self.consecutive >= self.max_consecutive_rejections
