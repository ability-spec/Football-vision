"""Geometry-state resolution for Phase 4 field-space trajectories.

Phase 4 distinguishes three explicit geometry states for every trajectory sample
so downstream consumers never confuse a directly measured field position with a
position derived from propagated (optical-flow) geometry or with a position that
simply cannot be established:

``calibrated``
    The frame's ``CalibrationResult`` passed ``can_project()`` and was fitted
    directly from that frame (``is_temporally_propagated`` is False).
``propagated``
    The frame's ``CalibrationResult`` passed ``can_project()`` but was carried
    forward from an earlier frame by the Phase 2 ``CalibrationTracker``
    (``is_temporally_propagated`` is True). Geometry is usable but carries an
    explicit propagation age and drift allowance.
``unknown``
    No usable geometry: calibration missing, failed, expired, or below the
    minimum projection confidence. No field position may be claimed.
"""

from __future__ import annotations

from typing import Optional

from football_vision.field_spec import MIN_CALIBRATION_CONFIDENCE
from football_vision.schema import CalibrationResult, GeometryState

GEOMETRY_STATES = ("calibrated", "propagated", "unknown")


def resolve_geometry_state(
    calibration: Optional[CalibrationResult],
    *,
    min_confidence: float = MIN_CALIBRATION_CONFIDENCE,
) -> GeometryState:
    """Resolve the explicit Phase 4 geometry state for a frame.

    Returns ``"unknown"`` whenever the calibration cannot support projection, so
    the trajectory layer can never silently invent field coordinates.
    """
    if calibration is None:
        return "unknown"
    if not calibration.can_project(min_confidence=min_confidence):
        return "unknown"
    if bool(calibration.is_temporally_propagated):
        return "propagated"
    return "calibrated"


def geometry_state_is_projectable(state: GeometryState) -> bool:
    """Return True iff ``state`` permits projecting a measured footpoint."""
    return state in ("calibrated", "propagated")


def geometry_state_propagation_age(calibration: Optional[CalibrationResult]) -> int:
    """Return the Phase 2 propagation age (frames) for propagated geometry, else 0."""
    if calibration is None or not bool(calibration.is_temporally_propagated):
        return 0
    return max(0, int(calibration.propagation_age))
