"""Deterministic multi-object player tracker with short-occlusion survival (Phase 3E).

Uses Hungarian bipartite matching over a combined IoU + constant-velocity
footpoint distance score. Tracks survive short occlusions up to
``max_missed_frames`` in image space while explicitly refusing fabricated
field coordinates during coasted/unobserved frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from football_vision.projection.projector import FieldProjector
from football_vision.schema import (
    CalibrationResult,
    FootpointEstimate,
    PlayerDetection,
    PlayerTrack,
    TeamAssignment,
    TeamLabel,
)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


@dataclass
class _InternalTrackState:
    track_id: int
    frame_id: int
    bbox: Tuple[float, float, float, float]
    footpoint: Tuple[float, float]
    footpoint_estimate: FootpointEstimate
    velocity_uv: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    missed_frames: int = 0
    hits: int = 1
    detection_confidence: float = 1.0
    team_votes: Dict[str, float] = field(default_factory=lambda: {"TEAM_A": 0.0, "TEAM_B": 0.0})
    last_team: TeamLabel = "UNKNOWN"
    last_team_confidence: float = 0.0
    short_occlusion_recovered: bool = False
    detector_metadata: Dict[str, Any] = field(default_factory=dict)
    last_observed_frame_id: Optional[int] = None
    last_observed_footpoint: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.last_observed_frame_id is None:
            self.last_observed_frame_id = self.frame_id
        if self.last_observed_footpoint is None:
            self.last_observed_footpoint = self.footpoint

    def predict_next_bbox_and_footpoint(self, dt_frames: int = 1) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        damp = float(0.85 ** self.missed_frames) * (1.0 - 0.85 ** dt_frames) / (1.0 - 0.85)
        vx = self.velocity_uv[0] * damp
        vy = self.velocity_uv[1] * damp
        x1, y1, x2, y2 = self.bbox
        pred_bbox = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
        pred_fp = (self.footpoint[0] + vx, self.footpoint[1] + vy)
        return pred_bbox, pred_fp

    def update_team(self, assignment: Optional[TeamAssignment]) -> None:
        if assignment is not None and assignment.team in ("TEAM_A", "TEAM_B"):
            self.team_votes[assignment.team] = (
                0.85 * self.team_votes.get(assignment.team, 0.0) + float(assignment.confidence)
            )
            other = "TEAM_B" if assignment.team == "TEAM_A" else "TEAM_A"
            self.team_votes[other] = 0.85 * self.team_votes.get(other, 0.0)

        va = self.team_votes.get("TEAM_A", 0.0)
        vb = self.team_votes.get("TEAM_B", 0.0)
        v_tot = va + vb
        if v_tot < 0.40:
            self.last_team = "UNKNOWN"
            self.last_team_confidence = round(
                float(assignment.confidence) if assignment is not None else 0.0, 4
            )
        elif va >= vb:
            self.last_team = "TEAM_A"
            dom = va / max(1e-6, v_tot)
            inst = float(assignment.confidence) if (assignment and assignment.team == "TEAM_A") else 0.70
            self.last_team_confidence = round(float(np.clip(0.55 * dom + 0.45 * inst, 0.45, 0.99)), 4)
        else:
            self.last_team = "TEAM_B"
            dom = vb / max(1e-6, v_tot)
            inst = float(assignment.confidence) if (assignment and assignment.team == "TEAM_B") else 0.70
            self.last_team_confidence = round(float(np.clip(0.55 * dom + 0.45 * inst, 0.45, 0.99)), 4)


class PlayerTracker:
    """Modular, deterministic IoU + constant-velocity multi-object tracker."""

    def __init__(
        self,
        *,
        max_missed_frames: int = 3,
        min_iou_threshold: float = 0.15,
        max_center_distance_px: float = 65.0,
        min_match_score: float = 0.22,
        min_init_confidence: float = 0.35,
        velocity_smoothing: float = 0.60,
        projector: Optional[FieldProjector] = None,
    ) -> None:
        self.max_missed_frames = int(max_missed_frames)
        self.min_iou_threshold = float(min_iou_threshold)
        self.max_center_distance_px = float(max_center_distance_px)
        self.min_match_score = float(min_match_score)
        self.min_init_confidence = float(min_init_confidence)
        self.velocity_smoothing = float(velocity_smoothing)
        self.projector = projector if projector is not None else FieldProjector()

        self._last_frame_id: Optional[int] = None
        self._next_track_id: int = 1
        self._tracks: List[_InternalTrackState] = []
        self.total_occlusion_recoveries: int = 0

    def reset(self) -> None:
        """Reset all active tracks and counters."""
        self._last_frame_id = None
        self._next_track_id = 1
        self._tracks.clear()
        self.total_occlusion_recoveries = 0

    def update(
        self,
        detections: Sequence[PlayerDetection],
        *,
        frame_id: int,
        team_assignments: Optional[Sequence[TeamAssignment]] = None,
        calibration: Optional[CalibrationResult] = None,
        include_coasting_tracks: bool = True,
    ) -> List[PlayerTrack]:
        """Associate ``detections`` with active tracks and project reliable footpoints."""
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase strictly")
        self._last_frame_id = int(frame_id)
        if calibration is not None and (
            calibration.camera_cut_detected
            or calibration.failure_reason == "camera_cut_uncalibrated"
        ):
            # Different cameras cannot establish image-space identity continuity.
            # Keep the ID allocator: resetting it would merge new players into old
            # trajectories stored by downstream consumers.
            self._tracks.clear()
        # Expire identities before association when the unobserved interval is too long.
        self._tracks = [t for t in self._tracks
                        if t.missed_frames + frame_id - t.frame_id - 1 <= self.max_missed_frames]
        n_trks = len(self._tracks)
        n_dets = len(detections)
        teams_list: List[Optional[TeamAssignment]] = (
            list(team_assignments) if team_assignments is not None else [None] * n_dets
        )

        # 1. Predict all active tracks forward to current frame
        predicted = [t.predict_next_bbox_and_footpoint(int(frame_id) - t.frame_id) for t in self._tracks]

        matched_trk_indices: Dict[int, int] = {}
        matched_det_indices: set[int] = set()

        if n_trks > 0 and n_dets > 0:
            score_mat = np.zeros((n_trks, n_dets), dtype=np.float64)
            valid_mat = np.zeros((n_trks, n_dets), dtype=bool)

            for i, trk in enumerate(self._tracks):
                pred_bbox, pred_fp = predicted[i]
                # Allow slightly larger search radius if track missed 1..3 frames
                allow_dist = self.max_center_distance_px * (1.0 + 0.35 * trk.missed_frames)
                for j, det in enumerate(detections):
                    iou = _bbox_iou(pred_bbox, det.bbox)
                    dist_px = float(np.hypot(pred_fp[0] - det.footpoint[0], pred_fp[1] - det.footpoint[1]))
                    dist_score = max(0.0, 1.0 - dist_px / allow_dist)
                    score = 0.55 * iou + 0.45 * dist_score

                    is_valid = (
                        score >= self.min_match_score
                        and ((iou >= self.min_iou_threshold) or (dist_px <= allow_dist))
                    )
                    if is_valid:
                        valid_mat[i, j] = True
                        # Tiny deterministic tie-breaker by track_id and detection index
                        score_mat[i, j] = score - 1e-7 * trk.track_id - 1e-9 * j
                    else:
                        score_mat[i, j] = -1e6

            row_ind, col_ind = linear_sum_assignment(-score_mat)
            for r, c in zip(row_ind, col_ind):
                if valid_mat[r, c]:
                    matched_trk_indices[int(r)] = int(c)
                    matched_det_indices.add(int(c))

        updated_internal: List[_InternalTrackState] = []

        # 2. Update matched and unmatched existing tracks
        for i, trk in enumerate(self._tracks):
            if i in matched_trk_indices:
                det_idx = matched_trk_indices[i]
                det = detections[det_idx]
                t_assign = teams_list[det_idx]

                dt = max(1, int(frame_id) - int(trk.frame_id))
                # Coasting mutates the working position. Keep a separate measured
                # anchor so recovery estimates motion over the full observation gap.
                observation_dt = int(frame_id) - trk.last_observed_frame_id
                obs_vx = (det.footpoint[0] - trk.last_observed_footpoint[0]) / observation_dt
                obs_vy = (det.footpoint[1] - trk.last_observed_footpoint[1]) / observation_dt
                if trk.hits == 1 and trk.missed_frames == 0:
                    new_vel = (obs_vx, obs_vy)
                else:
                    a = self.velocity_smoothing
                    new_vel = (
                        a * obs_vx + (1.0 - a) * trk.velocity_uv[0],
                        a * obs_vy + (1.0 - a) * trk.velocity_uv[1],
                    )

                recovered = observation_dt > 1
                if recovered:
                    self.total_occlusion_recoveries += 1

                trk.frame_id = int(frame_id)
                trk.bbox = det.bbox
                trk.footpoint = det.footpoint
                trk.footpoint_estimate = det.footpoint_estimate
                trk.last_observed_frame_id = int(frame_id)
                trk.last_observed_footpoint = det.footpoint
                trk.velocity_uv = (round(float(new_vel[0]), 4), round(float(new_vel[1]), 4))
                trk.age += dt
                trk.missed_frames = 0
                trk.hits += 1
                trk.detection_confidence = float(det.confidence)
                trk.short_occlusion_recovered = recovered
                trk.detector_metadata = dict(det.detector_metadata)
                trk.update_team(t_assign)
                updated_internal.append(trk)
            else:
                # Unmatched track: coast if within max_missed_frames
                dt = int(frame_id) - trk.frame_id
                trk.missed_frames += dt
                trk.age += dt
                trk.frame_id = int(frame_id)
                trk.short_occlusion_recovered = False
                if trk.missed_frames <= self.max_missed_frames:
                    pred_bbox, pred_fp = predicted[i]
                    trk.bbox = tuple(round(float(v), 3) for v in pred_bbox)
                    trk.footpoint = (round(float(pred_fp[0]), 3), round(float(pred_fp[1]), 3))
                    trk.detection_confidence = round(float(trk.detection_confidence * 0.75), 4)
                    trk.footpoint_estimate = FootpointEstimate(
                        u_px=trk.footpoint[0],
                        v_px=trk.footpoint[1],
                        is_reliable=False,
                        confidence=0.0,
                        is_partially_truncated=trk.footpoint_estimate.is_partially_truncated,
                        is_near_sideline=trk.footpoint_estimate.is_near_sideline,
                        is_player_crossing=trk.footpoint_estimate.is_player_crossing,
                        is_temporarily_occluded=True,
                        unreliable_reason="track_occluded_unobserved",
                    )
                    updated_internal.append(trk)

        # 3. Create new tracks for unmatched detections sorted left-to-right
        unmatched_dets = [
            j
            for j in range(n_dets)
            if j not in matched_det_indices and detections[j].confidence >= self.min_init_confidence
        ]
        unmatched_dets.sort(key=lambda j: (detections[j].bbox[0], detections[j].bbox[1]))

        for j in unmatched_dets:
            det = detections[j]
            t_assign = teams_list[j]
            new_trk = _InternalTrackState(
                track_id=self._next_track_id,
                frame_id=int(frame_id),
                bbox=det.bbox,
                footpoint=det.footpoint,
                footpoint_estimate=det.footpoint_estimate,
                last_observed_frame_id=int(frame_id),
                last_observed_footpoint=det.footpoint,
                velocity_uv=(0.0, 0.0),
                age=1,
                missed_frames=0,
                hits=1,
                detection_confidence=float(det.confidence),
                short_occlusion_recovered=False,
                detector_metadata=dict(det.detector_metadata),
            )
            new_trk.update_team(t_assign)
            self._next_track_id += 1
            updated_internal.append(new_trk)

        # Sort active tracks by track_id for deterministic output order
        updated_internal.sort(key=lambda t: t.track_id)
        self._tracks = updated_internal

        # 4. Build external PlayerTrack objects and run gated FieldProjector
        output_tracks: List[PlayerTrack] = []
        for t in self._tracks:
            if not include_coasting_tracks and t.missed_frames > 0:
                continue
            output_tracks.append(
                PlayerTrack(
                    track_id=t.track_id,
                    frame_id=t.frame_id,
                    bbox=t.bbox,
                    footpoint=t.footpoint,
                    footpoint_estimate=t.footpoint_estimate,
                    team=t.last_team,
                    team_confidence=t.last_team_confidence,
                    detection_confidence=t.detection_confidence,
                    age=t.age,
                    missed_frames=t.missed_frames,
                    hits=t.hits,
                    velocity_uv=t.velocity_uv,
                    short_occlusion_recovered=t.short_occlusion_recovered,
                    detector_metadata=dict(t.detector_metadata),
                )
            )

        self.projector.update_tracks_with_projection(output_tracks, calibration)
        return output_tracks

