"""Field projection gating and coordinate semantics enforcement (Phase 3F).

Treats ``CalibrationResult`` as the strict geometry contract:
  1. Refuses projection when ``calibration`` is None, invalid, or expired.
  2. Refuses projection when ``footpoint_estimate.is_reliable`` is False.
  3. Refuses projection on coasting/occluded tracks (``missed_frames > 0``).
  4. Preserves ``x_coord_mode`` semantics and NEVER invents an ``absolute_yardline``
     when ``x_coord_mode != "absolute"``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

from football_vision.field_spec import VALID_METRICS_BY_X_COORD_MODE
from football_vision.schema import (
    CalibrationResult,
    FootpointEstimate,
    PlayerDetection,
    PlayerTrack,
    ProjectedFieldPosition,
)


class FieldProjector:
    """Projects reliable player footpoints into field coordinates via CalibrationResult."""

    def __init__(
        self,
        *,
        min_footpoint_confidence: float = 0.40,
        x_bounds_yd: Tuple[float, float] = (-15.0, 135.0),
        y_bounds_yd: Tuple[float, float] = (-10.0, 63.333333333333336),
    ) -> None:
        self.min_footpoint_confidence = float(min_footpoint_confidence)
        self.x_bounds_yd = (float(x_bounds_yd[0]), float(x_bounds_yd[1]))
        self.y_bounds_yd = (float(y_bounds_yd[0]), float(y_bounds_yd[1]))

    def project_footpoint(
        self,
        footpoint_estimate: FootpointEstimate,
        calibration: Optional[CalibrationResult],
        *,
        missed_frames: int = 0,
    ) -> Tuple[Optional[ProjectedFieldPosition], str]:
        """Project a single ``FootpointEstimate`` when both geometry and footpoint are valid.

        Returns ``(projected_position_or_None, status_string)``.
        """
        if calibration is None:
            return None, "calibration_refused:no_calibration"

        if not calibration.can_project():
            reason = calibration.failure_reason or "calibration_invalid_or_expired"
            return None, f"calibration_refused:{reason}"

        if int(missed_frames) > 0:
            return None, "footpoint_unreliable:track_occluded_unobserved"

        if not footpoint_estimate.is_reliable:
            reason = footpoint_estimate.unreliable_reason or "unreliable_footpoint"
            return None, f"footpoint_unreliable:{reason}"

        if footpoint_estimate.confidence < self.min_footpoint_confidence:
            return None, "footpoint_unreliable:low_footpoint_confidence"

        pts_yd = calibration.image_to_field([[footpoint_estimate.u_px, footpoint_estimate.v_px]])
        if pts_yd is None or len(pts_yd) == 0 or not np.all(np.isfinite(pts_yd)):
            return None, "calibration_refused:projection_failed"

        x_yd = float(pts_yd[0, 0])
        y_yd = float(pts_yd[0, 1])

        if not (self.x_bounds_yd[0] <= x_yd <= self.x_bounds_yd[1]) or not (
            self.y_bounds_yd[0] <= y_yd <= self.y_bounds_yd[1]
        ):
            return None, "projection_refused:out_of_stadium_bounds"

        mode = calibration.x_coord_mode
        is_abs = mode == "absolute"
        abs_yd: Optional[float] = round(x_yd, 3) if is_abs else None
        valid_metrics = VALID_METRICS_BY_X_COORD_MODE.get(mode, ())

        proj = ProjectedFieldPosition(
            x_yd=round(x_yd, 3),
            y_yd=round(y_yd, 3),
            x_coord_mode=mode,
            calibration_confidence=round(float(calibration.confidence), 4),
            footpoint_confidence=round(float(footpoint_estimate.confidence), 4),
            is_absolute_yardline=is_abs,
            absolute_yardline=abs_yd,
            valid_metrics=tuple(valid_metrics),
        )
        return proj, "projected"

    def project_detections(
        self,
        detections: Sequence[PlayerDetection],
        calibration: Optional[CalibrationResult],
    ) -> List[Tuple[PlayerDetection, Optional[ProjectedFieldPosition], str]]:
        """Project a list of ``PlayerDetection`` objects without mutating them."""
        results: List[Tuple[PlayerDetection, Optional[ProjectedFieldPosition], str]] = []
        for det in detections:
            proj, status = self.project_footpoint(det.footpoint_estimate, calibration, missed_frames=0)
            results.append((det, proj, status))
        return results

    def update_tracks_with_projection(
        self,
        tracks: Sequence[PlayerTrack],
        calibration: Optional[CalibrationResult],
    ) -> List[PlayerTrack]:
        """Populate ``field_position`` and ``projected_position`` on active ``PlayerTrack`` objects."""
        for trk in tracks:
            proj, status = self.project_footpoint(
                trk.footpoint_estimate,
                calibration,
                missed_frames=trk.missed_frames,
            )
            if proj is not None and calibration is not None:
                trk.field_position = proj.xy_yd
                trk.projected_position = proj
                trk.x_coord_mode = calibration.x_coord_mode
                trk.projection_available = True
                trk.projection_status = status
            else:
                trk.field_position = None
                trk.projected_position = None
                trk.x_coord_mode = (
                    calibration.x_coord_mode
                    if (calibration is not None and calibration.can_project())
                    else "uncalibrated"
                )
                trk.projection_available = False
                trk.projection_status = status
        return list(tracks)
