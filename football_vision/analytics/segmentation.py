"""Phase 10 play segmentation: PLAY START -> PRE-SNAP -> SNAP -> PLAY -> PLAY END.

Scope (ROADMAP Phase 10): *"manual timestamps first, collective motion onset
second"*. This module therefore has exactly two sources of truth and never mixes
them silently:

1. **Manual timestamps** (:class:`~football_vision.schema.PlayTimestampLabel`) are
   the primary contract. A play ``start_frame`` is always manual: the segmenter
   never invents the existence of a play. ``snap_frame`` / ``end_frame`` may be
   manual too, in which case they are copied through with ``source == "manual"``.
2. **Collective motion evidence** derived from Phase 4
   :class:`~football_vision.schema.FieldTrajectory` samples fills in a missing
   snap (collective motion onset) or a missing play end (collective motion
   cessation). Every automatic estimate carries a source, a confidence and the
   evidence it was computed from.

Design invariants (same refusal-first posture as Phases 1-4)
-----------------------------------------------------------
1. **No fabricated boundaries.** If the motion evidence cannot support a snap,
   the result is ``snap_frame = None``, ``snap_source = "unavailable"`` and an
   explicit refusal reason. Frames of such a play are labelled ``snap_unknown``
   rather than being split at a guessed frame.
2. **Dead reckoning is not motion.** Only samples with
   ``position_source == "measured_smoothed"`` (an accepted measurement of an
   observed track, projected through projectable geometry) count as evidence.
   ``predicted_dead_reckoning`` samples, outlier-rejected samples, coasted
   samples and unprojected samples are counted and reported as exclusions, never
   as observations.
3. **Coordinate continuity is required.** Onset/cessation compare speeds across
   frames; that is only meaningful inside one coordinate frame. A window spanning
   more than one ``coordinate_frame_id`` (camera cut, origin/mode change), or
   mixing two coordinate frames inside one frame, is refused with
   ``coordinate_frame_change`` instead of being segmented anyway.
4. **Confidence is an a priori engineering score**, not a calibrated probability:
   observed threshold contrast scaled by usable-evidence coverage, capped below
   1.0. It is reported next to the evidence that produced it.
5. **No ground truth.** Nothing here reads or forwards a label of the true snap;
   benchmark harnesses compare against their own annotations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from football_vision.schema import (
    FieldTrajectory,
    PlayEndEstimate,
    PlaySegment,
    PlaySegmentationResult,
    PlayTimestampLabel,
    SegmentationRefusal,
    SnapEstimate,
    TrajectorySample,
    XCoordMode,
)

# ---------------------------------------------------------------------------
# Frozen engineering assumptions (a priori; documented in the Phase 10 report).
# These are plausibility thresholds for broadcast football, NOT fitted values:
# no annotated multi-play dataset was used to select them.
# ---------------------------------------------------------------------------
DEFAULT_MOTION_SPEED_THRESHOLD_YD_S: float = 1.0    # ~2 mph; set players jitter below this
DEFAULT_ONSET_MIN_FRACTION: float = 0.45            # fraction of the visible group that must move
DEFAULT_BASELINE_MAX_FRACTION: float = 0.15         # tolerated pre-snap motion (shifts/motion men)
DEFAULT_BASELINE_MIN_FRAMES: int = 10               # ~0.33 s at 30 fps of quiescent evidence
DEFAULT_MIN_SUSTAINED_FRAMES: int = 3               # onset must persist, not be a one-frame artifact
DEFAULT_MIN_TRACKS: int = 6                         # fewer visible tracks => fraction is meaningless
DEFAULT_MIN_QUIET_FRAMES: int = 6                   # sustained quiet period that ends a play
DEFAULT_MIN_PLAY_FRAMES: int = 6                    # shorter than this is not a football play
DEFAULT_CONFIDENCE_REFERENCE_CONTRAST: float = 0.5  # contrast that maps to confidence 1.0 (pre-cap)
MAX_CONFIDENCE: float = 0.99                        # never claim certainty

# Refusal reasons (stable strings; asserted by tests and reported by benchmarks).
REASON_NO_TRAJECTORIES = "no_trajectories"
REASON_NO_TRAJECTORY_FRAMES = "no_trajectory_frames"
REASON_WINDOW_TOO_SHORT = "window_too_short"
REASON_INSUFFICIENT_TRACKS = "insufficient_tracks"
REASON_INSUFFICIENT_MEASURED_SAMPLES = "insufficient_measured_samples"
REASON_BASELINE_EVIDENCE_GAPS = "baseline_evidence_gaps"
REASON_NO_QUIESCENT_BASELINE = "no_quiescent_baseline"
REASON_NO_SUSTAINED_ONSET = "no_sustained_onset"
REASON_NO_PRIOR_MOTION = "no_prior_sustained_motion"
REASON_NO_MOTION_CESSATION = "no_motion_cessation"
REASON_COORDINATE_CHANGE = "coordinate_frame_change"
REASON_AUTOMATIC_DISABLED = "automatic_estimation_disabled"
REASON_NO_SNAP_FOR_END_SEARCH = "snap_unavailable"

REFUSAL_REASONS: Tuple[str, ...] = (
    REASON_NO_TRAJECTORIES,
    REASON_NO_TRAJECTORY_FRAMES,
    REASON_WINDOW_TOO_SHORT,
    REASON_INSUFFICIENT_TRACKS,
    REASON_INSUFFICIENT_MEASURED_SAMPLES,
    REASON_BASELINE_EVIDENCE_GAPS,
    REASON_NO_QUIESCENT_BASELINE,
    REASON_NO_SUSTAINED_ONSET,
    REASON_NO_PRIOR_MOTION,
    REASON_NO_MOTION_CESSATION,
    REASON_COORDINATE_CHANGE,
    REASON_AUTOMATIC_DISABLED,
    REASON_NO_SNAP_FOR_END_SEARCH,
)


def _sample_is_motion_evidence(sample: TrajectorySample) -> bool:
    """True iff a sample is an accepted measurement of a real, observed player.

    Excludes dead reckoning (model output, would invent motion), outlier-rejected
    samples, unprojected samples and coasted tracks.
    """
    return bool(
        sample.position_source == "measured_smoothed"
        and sample.is_measurement_used
        and not sample.is_outlier_rejected
        and sample.field_position is not None
        and sample.track_state == "observed"
    )


# ---------------------------------------------------------------------------
# Motion evidence
# ---------------------------------------------------------------------------
@dataclass
class MotionEvidence:
    """Per-frame collective-motion evidence over one contiguous frame window.

    ``moving_fraction[f]`` is ``None`` (undefined) whenever fewer than
    ``min_tracks`` usable samples exist for that frame. Undefined differs from
    zero: a frame with no evidence can never support a quiescence claim.
    """

    fps: float
    speed_threshold_yd_s: float
    min_tracks: int
    start_frame: int
    end_frame: int
    frame_ids: Tuple[int, ...]
    n_usable: Tuple[int, ...]
    n_moving: Tuple[int, ...]
    moving_fraction: Tuple[Optional[float], ...]
    coordinate_frame_ids: Tuple[Optional[str], ...]
    coordinate_segments: Tuple[Optional[int], ...]
    n_tracks_total: int = 0
    n_samples_in_window: int = 0
    n_excluded_dead_reckoning: int = 0
    n_excluded_rejected: int = 0
    n_excluded_coasted: int = 0
    n_excluded_unprojected: int = 0

    # -- accessors ---------------------------------------------------------
    def index_of(self, frame_id: int) -> Optional[int]:
        f = int(frame_id)
        if f < self.start_frame or f > self.end_frame:
            return None
        return f - self.start_frame

    def is_defined(self, frame_id: int) -> bool:
        idx = self.index_of(frame_id)
        return idx is not None and self.moving_fraction[idx] is not None

    def fraction(self, frame_id: int) -> Optional[float]:
        idx = self.index_of(frame_id)
        return None if idx is None else self.moving_fraction[idx]

    def usable_count(self, frame_id: int) -> int:
        idx = self.index_of(frame_id)
        return 0 if idx is None else int(self.n_usable[idx])

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def n_defined(self) -> int:
        return sum(1 for frac in self.moving_fraction if frac is not None)

    @property
    def n_undefined(self) -> int:
        return self.n_frames - self.n_defined

    @property
    def n_usable_samples(self) -> int:
        return int(sum(self.n_usable))

    def defined_frame_ids(self) -> List[int]:
        return [
            int(self.frame_ids[i])
            for i, frac in enumerate(self.moving_fraction)
            if frac is not None
        ]

    def distinct_coordinate_frame_ids(self) -> Tuple[Optional[str], ...]:
        """Coordinate frames observed on *defined* frames, in order of appearance."""
        seen: List[Optional[str]] = []
        for i, frac in enumerate(self.moving_fraction):
            if frac is None:
                continue
            cid = self.coordinate_frame_ids[i]
            if cid not in seen:
                seen.append(cid)
        return tuple(seen)

    def distinct_coordinate_segments(self) -> Tuple[int, ...]:
        seen: List[int] = []
        for i, frac in enumerate(self.moving_fraction):
            if frac is None:
                continue
            seg = self.coordinate_segments[i]
            if seg is not None and seg not in seen:
                seen.append(int(seg))
        return tuple(sorted(seen))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": float(self.fps),
            "speed_threshold_yd_s": float(self.speed_threshold_yd_s),
            "min_tracks": int(self.min_tracks),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "n_frames": self.n_frames,
            "n_defined": self.n_defined,
            "n_undefined": self.n_undefined,
            "n_tracks_total": int(self.n_tracks_total),
            "n_samples_in_window": int(self.n_samples_in_window),
            "n_usable_samples": self.n_usable_samples,
            "n_excluded_dead_reckoning": int(self.n_excluded_dead_reckoning),
            "n_excluded_rejected": int(self.n_excluded_rejected),
            "n_excluded_coasted": int(self.n_excluded_coasted),
            "n_excluded_unprojected": int(self.n_excluded_unprojected),
            "coordinate_frame_ids": list(self.distinct_coordinate_frame_ids()),
            "coordinate_segments": list(self.distinct_coordinate_segments()),
            "frames": [
                {
                    "frame_id": int(fid),
                    "n_usable": int(nu),
                    "n_moving": int(nm),
                    "moving_fraction": None if frac is None else round(float(frac), 6),
                    "coordinate_frame_id": cid,
                    "coordinate_segment": None if seg is None else int(seg),
                }
                for fid, nu, nm, frac, cid, seg in zip(
                    self.frame_ids,
                    self.n_usable,
                    self.n_moving,
                    self.moving_fraction,
                    self.coordinate_frame_ids,
                    self.coordinate_segments,
                )
            ],
        }


def build_motion_evidence(
    trajectories: Sequence[FieldTrajectory],
    *,
    start_frame: int,
    end_frame: int,
    speed_threshold_yd_s: float = DEFAULT_MOTION_SPEED_THRESHOLD_YD_S,
    min_tracks: int = DEFAULT_MIN_TRACKS,
    fps: Optional[float] = None,
) -> MotionEvidence:
    """Aggregate Phase 4 trajectory samples into per-frame collective motion.

    Every sample inside the window is accounted for exactly once: either it is
    usable evidence, or it lands in one of the ``n_excluded_*`` buckets. That
    bookkeeping is what makes "we saw nothing" distinguishable from "nobody moved".
    """
    if int(end_frame) < int(start_frame):
        raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")
    if not np.isfinite(float(speed_threshold_yd_s)) or float(speed_threshold_yd_s) < 0.0:
        raise ValueError("speed_threshold_yd_s must be a finite number >= 0")
    if int(min_tracks) < 1:
        raise ValueError("min_tracks must be >= 1")

    fps_values = {float(t.fps) for t in trajectories}
    if fps is None:
        if len(fps_values) != 1:
            raise ValueError(
                "cannot resolve a single fps from the supplied trajectories "
                f"(found {sorted(fps_values)}); pass fps explicitly"
            )
        resolved_fps = fps_values.pop()
    else:
        resolved_fps = float(fps)
        if fps_values and fps_values != {resolved_fps}:
            raise ValueError(
                f"trajectory fps {sorted(fps_values)} disagrees with fps={resolved_fps}"
            )
    if not np.isfinite(resolved_fps) or resolved_fps <= 0.0:
        raise ValueError(f"fps must be a positive finite number, got {resolved_fps!r}")

    start, end = int(start_frame), int(end_frame)
    n_frames = end - start + 1
    per_frame_samples: List[List[TrajectorySample]] = [[] for _ in range(n_frames)]
    n_samples_in_window = 0
    for traj in trajectories:
        for sample in traj.samples:
            if start <= int(sample.frame_id) <= end:
                per_frame_samples[int(sample.frame_id) - start].append(sample)
                n_samples_in_window += 1

    n_usable: List[int] = []
    n_moving: List[int] = []
    fractions: List[Optional[float]] = []
    coord_ids: List[Optional[str]] = []
    coord_segs: List[Optional[int]] = []
    excl_dead = excl_rejected = excl_coasted = excl_unprojected = 0

    for offset in range(n_frames):
        usable = moving = 0
        frame_coord_ids: List[str] = []
        frame_coord_segs: List[int] = []
        for sample in per_frame_samples[offset]:
            if sample.is_outlier_rejected:
                excl_rejected += 1
                continue
            if sample.position_source == "predicted_dead_reckoning":
                excl_dead += 1
                continue
            if sample.position_source == "none" or sample.field_position is None:
                excl_unprojected += 1
                continue
            if sample.track_state != "observed":
                excl_coasted += 1
                continue
            if not _sample_is_motion_evidence(sample):
                # Defensive: the four checks above already cover the contract, so
                # anything landing here is an internally inconsistent sample.
                excl_unprojected += 1
                continue
            usable += 1
            if float(sample.speed_yd_s) >= float(speed_threshold_yd_s):
                moving += 1
            if sample.coordinate_frame_id is not None:
                cid = str(sample.coordinate_frame_id)
                if cid not in frame_coord_ids:
                    frame_coord_ids.append(cid)
            if sample.coordinate_segment is not None:
                seg = int(sample.coordinate_segment)
                if seg not in frame_coord_segs:
                    frame_coord_segs.append(seg)
        n_usable.append(usable)
        n_moving.append(moving)
        fractions.append(float(moving) / float(usable) if usable >= int(min_tracks) else None)
        # A frame mixing two coordinate frames cannot support a motion claim; mark
        # it inconsistent (None) so both quiescence and onset refuse on it.
        coord_ids.append(frame_coord_ids[0] if len(frame_coord_ids) == 1 else None)
        coord_segs.append(frame_coord_segs[0] if len(frame_coord_segs) == 1 else None)

    return MotionEvidence(
        fps=float(resolved_fps),
        speed_threshold_yd_s=float(speed_threshold_yd_s),
        min_tracks=int(min_tracks),
        start_frame=start,
        end_frame=end,
        frame_ids=tuple(range(start, end + 1)),
        n_usable=tuple(n_usable),
        n_moving=tuple(n_moving),
        moving_fraction=tuple(fractions),
        coordinate_frame_ids=tuple(coord_ids),
        coordinate_segments=tuple(coord_segs),
        n_tracks_total=len(trajectories),
        n_samples_in_window=n_samples_in_window,
        n_excluded_dead_reckoning=excl_dead,
        n_excluded_rejected=excl_rejected,
        n_excluded_coasted=excl_coasted,
        n_excluded_unprojected=excl_unprojected,
    )


def _last_trajectory_frame(trajectories: Sequence[FieldTrajectory]) -> Optional[int]:
    frames = [int(s.frame_id) for t in trajectories for s in t.samples]
    return max(frames) if frames else None


# ---------------------------------------------------------------------------
# Collective-motion segmenter (snap onset + play-end cessation)
# ---------------------------------------------------------------------------
@dataclass
class _WindowGuard:
    """Outcome of the pre-checks both detectors run on a frame window."""

    evidence: Optional[MotionEvidence] = None
    reason: Optional[str] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.evidence is not None and self.reason is None


class CollectiveMotionSegmenter:
    """Estimates snap onset and play-end cessation from collective field motion.

    Deliberately conservative: an onset needs a *quiescent* baseline of defined
    frames before it, and a cessation needs a *sustained* active period before it.
    Anything else is refused with an explicit reason.
    """

    def __init__(
        self,
        *,
        motion_speed_threshold_yd_s: float = DEFAULT_MOTION_SPEED_THRESHOLD_YD_S,
        onset_min_fraction: float = DEFAULT_ONSET_MIN_FRACTION,
        baseline_max_fraction: float = DEFAULT_BASELINE_MAX_FRACTION,
        baseline_min_frames: int = DEFAULT_BASELINE_MIN_FRAMES,
        min_sustained_frames: int = DEFAULT_MIN_SUSTAINED_FRAMES,
        min_tracks: int = DEFAULT_MIN_TRACKS,
        min_quiet_frames: int = DEFAULT_MIN_QUIET_FRAMES,
        min_play_frames: int = DEFAULT_MIN_PLAY_FRAMES,
        quiet_max_fraction: Optional[float] = None,
        active_min_fraction: Optional[float] = None,
        confidence_reference_contrast: float = DEFAULT_CONFIDENCE_REFERENCE_CONTRAST,
    ) -> None:
        if not 0.0 <= float(baseline_max_fraction) < float(onset_min_fraction) <= 1.0:
            raise ValueError(
                "require 0 <= baseline_max_fraction < onset_min_fraction <= 1 "
                f"(got {baseline_max_fraction!r} and {onset_min_fraction!r})"
            )
        if int(baseline_min_frames) < 1:
            raise ValueError("baseline_min_frames must be >= 1")
        if int(min_sustained_frames) < 1:
            raise ValueError("min_sustained_frames must be >= 1")
        if int(min_tracks) < 1:
            raise ValueError("min_tracks must be >= 1")
        if int(min_quiet_frames) < 1:
            raise ValueError("min_quiet_frames must be >= 1")
        if int(min_play_frames) < 1:
            raise ValueError("min_play_frames must be >= 1")
        if float(confidence_reference_contrast) <= 0.0:
            raise ValueError("confidence_reference_contrast must be > 0")
        quiet = float(baseline_max_fraction) if quiet_max_fraction is None else float(quiet_max_fraction)
        active = float(onset_min_fraction) if active_min_fraction is None else float(active_min_fraction)
        if not 0.0 <= quiet < active <= 1.0:
            raise ValueError(
                "require 0 <= quiet_max_fraction < active_min_fraction <= 1 "
                f"(got {quiet!r} and {active!r})"
            )
        self.motion_speed_threshold_yd_s = float(motion_speed_threshold_yd_s)
        self.onset_min_fraction = float(onset_min_fraction)
        self.baseline_max_fraction = float(baseline_max_fraction)
        self.baseline_min_frames = int(baseline_min_frames)
        self.min_sustained_frames = int(min_sustained_frames)
        self.min_tracks = int(min_tracks)
        self.min_quiet_frames = int(min_quiet_frames)
        self.min_play_frames = int(min_play_frames)
        self.quiet_max_fraction = quiet
        self.active_min_fraction = active
        self.confidence_reference_contrast = float(confidence_reference_contrast)

        # Audit counters (Phase 20 style): boundaries claimed without evidence.
        self.fabricated_boundaries: int = 0
        self.snap_estimates_accepted: int = 0
        self.snap_estimates_refused: int = 0
        self.play_end_estimates_accepted: int = 0
        self.play_end_estimates_refused: int = 0

    # -- evidence ----------------------------------------------------------
    def motion_evidence(
        self,
        trajectories: Sequence[FieldTrajectory],
        *,
        start_frame: int,
        end_frame: int,
        fps: Optional[float] = None,
    ) -> MotionEvidence:
        return build_motion_evidence(
            trajectories,
            start_frame=start_frame,
            end_frame=end_frame,
            speed_threshold_yd_s=self.motion_speed_threshold_yd_s,
            min_tracks=self.min_tracks,
            fps=fps,
        )

    def _guard_window(
        self,
        trajectories: Sequence[FieldTrajectory],
        *,
        start_frame: int,
        end_frame: int,
        min_frames: int,
        fps: Optional[float],
    ) -> _WindowGuard:
        """Shared refusal checks: evidence exists, window length, tracks, geometry."""
        if not trajectories:
            return _WindowGuard(reason=REASON_NO_TRAJECTORIES, detail="no Phase 4 trajectories supplied")
        if int(end_frame) < int(start_frame):
            return _WindowGuard(
                reason=REASON_WINDOW_TOO_SHORT,
                detail=f"end_frame {end_frame} precedes start_frame {start_frame}",
            )
        try:
            evidence = self.motion_evidence(
                trajectories, start_frame=start_frame, end_frame=end_frame, fps=fps
            )
        except ValueError as exc:
            return _WindowGuard(reason=REASON_NO_TRAJECTORY_FRAMES, detail=str(exc))
        if evidence.n_frames < int(min_frames):
            return _WindowGuard(
                evidence=evidence,
                reason=REASON_WINDOW_TOO_SHORT,
                detail=(
                    f"window has {evidence.n_frames} frame(s); at least {int(min_frames)} required"
                ),
            )
        if evidence.n_tracks_total < int(self.min_tracks):
            return _WindowGuard(
                evidence=evidence,
                reason=REASON_INSUFFICIENT_TRACKS,
                detail=f"{evidence.n_tracks_total} track(s) < min_tracks={self.min_tracks}",
            )
        if evidence.n_defined == 0:
            return _WindowGuard(
                evidence=evidence,
                reason=REASON_INSUFFICIENT_MEASURED_SAMPLES,
                detail=(
                    "no frame reached min_tracks usable measured samples (excluded: "
                    f"dead_reckoning={evidence.n_excluded_dead_reckoning}, "
                    f"rejected={evidence.n_excluded_rejected}, "
                    f"coasted={evidence.n_excluded_coasted}, "
                    f"unprojected={evidence.n_excluded_unprojected})"
                ),
            )
        coord_ids = evidence.distinct_coordinate_frame_ids()
        if len(coord_ids) > 1 or (len(coord_ids) == 1 and coord_ids[0] is None):
            return _WindowGuard(
                evidence=evidence,
                reason=REASON_COORDINATE_CHANGE,
                detail=(
                    "the window does not hold one consistent coordinate_frame_id "
                    f"(observed {list(coord_ids)}); motion is not comparable across it"
                ),
            )
        return _WindowGuard(evidence=evidence)

    def _confidence(self, contrast: float, coverage: float) -> float:
        score = float(contrast) / float(self.confidence_reference_contrast) * float(coverage)
        return float(round(min(MAX_CONFIDENCE, max(0.0, score)), 4))

    @staticmethod
    def _first_coord(evidence: MotionEvidence) -> Tuple[Optional[str], Optional[int]]:
        coord_ids = evidence.distinct_coordinate_frame_ids()
        segments = evidence.distinct_coordinate_segments()
        return (coord_ids[0] if coord_ids else None), (segments[0] if segments else None)

    # -- snap onset --------------------------------------------------------
    def detect_snap(
        self,
        trajectories: Sequence[FieldTrajectory],
        *,
        play_id: str,
        start_frame: int,
        end_frame: Optional[int],
        fps: Optional[float] = None,
    ) -> SnapEstimate:
        """Estimate the snap as the first sustained collective-motion onset.

        Requires ``baseline_min_frames`` consecutive *defined* quiescent frames
        after ``start_frame``, then ``min_sustained_frames`` consecutive defined
        frames at or above ``onset_min_fraction``. The returned frame is the first
        frame of the sustained-motion run.
        """
        if end_frame is None:
            last = _last_trajectory_frame(trajectories)
            if last is None:
                return self._refuse_snap(
                    play_id,
                    REASON_NO_TRAJECTORY_FRAMES,
                    "no trajectory samples available to bound the search",
                )
            search_end = max(int(start_frame), last)
        else:
            search_end = int(end_frame)

        min_frames = self.baseline_min_frames + self.min_sustained_frames
        guard = self._guard_window(
            trajectories,
            start_frame=int(start_frame),
            end_frame=search_end,
            min_frames=min_frames,
            fps=fps,
        )
        if not guard.ok:
            return self._refuse_snap(
                play_id, guard.reason or REASON_WINDOW_TOO_SHORT, guard.detail, guard.evidence
            )
        evidence = guard.evidence
        assert evidence is not None

        end_idx = evidence.n_frames - 1
        # 1) Earliest index with a fully defined, quiescent baseline behind it.
        baseline_ok_from: Optional[int] = None
        baseline_gap = False
        for i in range(self.baseline_min_frames, end_idx + 1):
            baseline = evidence.moving_fraction[:i]
            if any(frac is None for frac in baseline):
                baseline_gap = True
                continue
            if max(float(frac) for frac in baseline) <= self.baseline_max_fraction:
                baseline_ok_from = i
                break
        if baseline_ok_from is None:
            if baseline_gap:
                reason = REASON_BASELINE_EVIDENCE_GAPS
                detail = (
                    f"no candidate frame has {self.baseline_min_frames} preceding defined "
                    "frames, so pre-snap quiescence cannot be established"
                )
            else:
                reason = REASON_NO_QUIESCENT_BASELINE
                detail = (
                    f"pre-snap motion never stayed <= {self.baseline_max_fraction:.2f} of "
                    "moving tracks over a defined baseline"
                )
            return self._refuse_snap(play_id, reason, detail, evidence)

        # 2) First sustained onset at or after that index.
        onset_idx: Optional[int] = None
        for i in range(baseline_ok_from, end_idx - self.min_sustained_frames + 2):
            run = evidence.moving_fraction[i : i + self.min_sustained_frames]
            if any(frac is None for frac in run):
                continue
            if min(float(frac) for frac in run) >= self.onset_min_fraction:
                onset_idx = i
                break
        if onset_idx is None:
            return self._refuse_snap(
                play_id,
                REASON_NO_SUSTAINED_ONSET,
                (
                    f"no {self.min_sustained_frames} consecutive defined frames reached "
                    f"moving_fraction >= {self.onset_min_fraction:.2f} after a quiescent baseline"
                ),
                evidence,
            )

        baseline_fracs = [
            float(f) for f in evidence.moving_fraction[:onset_idx] if f is not None
        ]
        onset_fracs = [
            float(f)
            for f in evidence.moving_fraction[onset_idx : onset_idx + self.min_sustained_frames]
            if f is not None
        ]
        baseline_fraction = max(baseline_fracs) if baseline_fracs else 0.0
        onset_fraction = min(onset_fracs) if onset_fracs else 0.0
        confidence = self._confidence(
            onset_fraction - baseline_fraction, evidence.n_defined / float(evidence.n_frames)
        )
        frame_id = int(evidence.frame_ids[onset_idx])
        coord_id, coord_seg = self._first_coord(evidence)
        self.snap_estimates_accepted += 1
        return SnapEstimate(
            play_id=play_id,
            frame_id=frame_id,
            timestamp_s=round(frame_id / float(evidence.fps), 6),
            source="collective_motion_onset",
            confidence=confidence,
            onset_fraction=round(onset_fraction, 6),
            baseline_fraction=round(baseline_fraction, 6),
            motion_threshold_fraction=self.onset_min_fraction,
            baseline_threshold_fraction=self.baseline_max_fraction,
            motion_speed_threshold_yd_s=self.motion_speed_threshold_yd_s,
            min_sustained_frames=self.min_sustained_frames,
            n_tracks=evidence.n_tracks_total,
            n_usable_samples=evidence.n_usable_samples,
            n_undefined_frames=evidence.n_undefined,
            coordinate_frame_id=coord_id,
            coordinate_segment=coord_seg,
            notes=[
                "snap = first sustained collective-motion onset after a quiescent baseline",
                "confidence is an a priori contrast x coverage score, not a calibrated probability",
            ],
        )

    def _refuse_snap(
        self,
        play_id: str,
        reason: str,
        detail: str,
        evidence: Optional[MotionEvidence] = None,
    ) -> SnapEstimate:
        self.snap_estimates_refused += 1
        return SnapEstimate(
            play_id=play_id,
            frame_id=None,
            timestamp_s=None,
            source="unavailable",
            confidence=0.0,
            reason=reason,
            motion_threshold_fraction=self.onset_min_fraction,
            baseline_threshold_fraction=self.baseline_max_fraction,
            motion_speed_threshold_yd_s=self.motion_speed_threshold_yd_s,
            min_sustained_frames=self.min_sustained_frames,
            n_tracks=int(evidence.n_tracks_total) if evidence else 0,
            n_usable_samples=int(evidence.n_usable_samples) if evidence else 0,
            n_undefined_frames=int(evidence.n_undefined) if evidence else 0,
            notes=[f"refused: {reason}" + (f" — {detail}" if detail else "")],
        )

    # -- play end (motion cessation) ---------------------------------------
    def detect_play_end(
        self,
        trajectories: Sequence[FieldTrajectory],
        *,
        play_id: str,
        snap_frame: int,
        end_frame: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> PlayEndEstimate:
        """Estimate the dead-ball frame as the first sustained collective quiet period.

        The search starts after the snap and requires (a) a sustained active period
        first, so a play cannot be "ended" by the burst that started it, and (b) at
        least ``min_play_frames`` of play. The returned frame is the FIRST frame of
        the quiet run — not the frame at which quiescence was proven, and not a
        referee whistle timestamp (dead-ball rules are not modelled).
        """
        if end_frame is None:
            last = _last_trajectory_frame(trajectories)
            if last is None:
                return self._refuse_end(
                    play_id,
                    REASON_NO_TRAJECTORY_FRAMES,
                    "no trajectory samples available to bound the search",
                )
            search_end = last
        else:
            search_end = int(end_frame)
        search_start = int(snap_frame) + 1
        if search_end < search_start:
            return self._refuse_end(
                play_id,
                REASON_WINDOW_TOO_SHORT,
                f"search window [{search_start}, {search_end}] is empty",
            )
        guard = self._guard_window(
            trajectories,
            start_frame=search_start,
            end_frame=search_end,
            min_frames=self.min_play_frames + self.min_quiet_frames,
            fps=fps,
        )
        if not guard.ok:
            return self._refuse_end(
                play_id, guard.reason or REASON_WINDOW_TOO_SHORT, guard.detail, guard.evidence
            )
        evidence = guard.evidence
        assert evidence is not None

        # 1) The play must actually happen: a sustained active run after the snap.
        active_idx: Optional[int] = None
        for i in range(0, evidence.n_frames - self.min_sustained_frames + 1):
            run = evidence.moving_fraction[i : i + self.min_sustained_frames]
            if any(frac is None for frac in run):
                continue
            if min(float(frac) for frac in run) >= self.active_min_fraction:
                active_idx = i
                break
        if active_idx is None:
            return self._refuse_end(
                play_id,
                REASON_NO_PRIOR_MOTION,
                (
                    f"no {self.min_sustained_frames} consecutive defined frames reached "
                    f"moving_fraction >= {self.active_min_fraction:.2f} after the snap, so a "
                    "play end is not claimable"
                ),
                evidence,
            )

        # 2) First sustained quiet run that leaves a play of at least min_play_frames.
        first_quiet_idx = max(self.min_play_frames - 1, active_idx + 1)
        quiet_idx: Optional[int] = None
        for i in range(first_quiet_idx, evidence.n_frames - self.min_quiet_frames + 1):
            run = evidence.moving_fraction[i : i + self.min_quiet_frames]
            if any(frac is None for frac in run):
                continue
            if max(float(frac) for frac in run) <= self.quiet_max_fraction:
                quiet_idx = i
                break
        if quiet_idx is None:
            return self._refuse_end(
                play_id,
                REASON_NO_MOTION_CESSATION,
                (
                    f"no {self.min_quiet_frames} consecutive defined frames dropped to "
                    f"moving_fraction <= {self.quiet_max_fraction:.2f} after sustained activity"
                ),
                evidence,
            )

        active_fracs = [float(f) for f in evidence.moving_fraction[active_idx:quiet_idx] if f is not None]
        quiet_fracs = [
            float(f)
            for f in evidence.moving_fraction[quiet_idx : quiet_idx + self.min_quiet_frames]
            if f is not None
        ]
        active_fraction = max(active_fracs) if active_fracs else 0.0
        quiet_fraction = max(quiet_fracs) if quiet_fracs else 0.0
        confidence = self._confidence(
            active_fraction - quiet_fraction, evidence.n_defined / float(evidence.n_frames)
        )
        frame_id = int(evidence.frame_ids[quiet_idx])
        coord_id, coord_seg = self._first_coord(evidence)
        self.play_end_estimates_accepted += 1
        return PlayEndEstimate(
            play_id=play_id,
            frame_id=frame_id,
            timestamp_s=round(frame_id / float(evidence.fps), 6),
            source="motion_cessation",
            confidence=confidence,
            quiet_fraction=round(quiet_fraction, 6),
            active_fraction=round(active_fraction, 6),
            quiet_threshold_fraction=self.quiet_max_fraction,
            motion_speed_threshold_yd_s=self.motion_speed_threshold_yd_s,
            min_quiet_frames=self.min_quiet_frames,
            n_tracks=evidence.n_tracks_total,
            n_usable_samples=evidence.n_usable_samples,
            n_undefined_frames=evidence.n_undefined,
            coordinate_frame_id=coord_id,
            coordinate_segment=coord_seg,
            notes=[
                "play_end = first frame of a sustained collective-quiet period",
                "not a whistle timestamp; NFL dead-ball rules are not modelled",
            ],
        )

    def _refuse_end(
        self,
        play_id: str,
        reason: str,
        detail: str,
        evidence: Optional[MotionEvidence] = None,
    ) -> PlayEndEstimate:
        self.play_end_estimates_refused += 1
        return PlayEndEstimate(
            play_id=play_id,
            frame_id=None,
            timestamp_s=None,
            source="unavailable",
            confidence=0.0,
            reason=reason,
            quiet_threshold_fraction=self.quiet_max_fraction,
            motion_speed_threshold_yd_s=self.motion_speed_threshold_yd_s,
            min_quiet_frames=self.min_quiet_frames,
            n_tracks=int(evidence.n_tracks_total) if evidence else 0,
            n_usable_samples=int(evidence.n_usable_samples) if evidence else 0,
            n_undefined_frames=int(evidence.n_undefined) if evidence else 0,
            notes=[f"refused: {reason}" + (f" — {detail}" if detail else "")],
        )


# ---------------------------------------------------------------------------
# Play segmenter (the Phase 10 entry point)
# ---------------------------------------------------------------------------
class PlaySegmenter:
    """Segments plays from manual timestamps, optionally completing snap/play-end.

    ``allow_automatic_snap`` defaults True and ``allow_automatic_play_end`` False:
    the ROADMAP order is "manual timestamps first, collective motion onset second",
    and an automatic dead-ball frame is a weaker claim than an automatic snap
    (whistle and dead-ball rules are not modelled), so it stays opt-in.
    """

    def __init__(
        self,
        *,
        fps: float,
        motion_segmenter: Optional[CollectiveMotionSegmenter] = None,
        allow_automatic_snap: bool = True,
        allow_automatic_play_end: bool = False,
    ) -> None:
        if not np.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(f"fps must be a positive finite number, got {fps!r}")
        self.fps = float(fps)
        self.motion = motion_segmenter or CollectiveMotionSegmenter()
        self.allow_automatic_snap = bool(allow_automatic_snap)
        self.allow_automatic_play_end = bool(allow_automatic_play_end)

        # Audit counters (asserted by tests, reported by the benchmark).
        self.fabricated_snap_frames: int = 0
        self.fabricated_play_ends: int = 0
        self.plays_segmented: int = 0
        self.snaps_from_manual: int = 0
        self.snaps_from_motion_onset: int = 0
        self.snaps_refused: int = 0
        self.play_ends_from_manual: int = 0
        self.play_ends_from_motion_cessation: int = 0
        self.play_ends_refused: int = 0
        self.runtime_ms: float = 0.0

    def reset(self) -> None:
        """Clear audit counters (fresh stream)."""
        self.fabricated_snap_frames = 0
        self.fabricated_play_ends = 0
        self.plays_segmented = 0
        self.snaps_from_manual = 0
        self.snaps_from_motion_onset = 0
        self.snaps_refused = 0
        self.play_ends_from_manual = 0
        self.play_ends_from_motion_cessation = 0
        self.play_ends_refused = 0
        self.runtime_ms = 0.0
        self.motion.fabricated_boundaries = 0
        self.motion.snap_estimates_accepted = 0
        self.motion.snap_estimates_refused = 0
        self.motion.play_end_estimates_accepted = 0
        self.motion.play_end_estimates_refused = 0

    def counters(self) -> Dict[str, int]:
        return {
            "plays_segmented": int(self.plays_segmented),
            "snaps_from_manual": int(self.snaps_from_manual),
            "snaps_from_motion_onset": int(self.snaps_from_motion_onset),
            "snaps_refused": int(self.snaps_refused),
            "play_ends_from_manual": int(self.play_ends_from_manual),
            "play_ends_from_motion_cessation": int(self.play_ends_from_motion_cessation),
            "play_ends_refused": int(self.play_ends_refused),
            "fabricated_snap_frames": int(self.fabricated_snap_frames),
            "fabricated_play_ends": int(self.fabricated_play_ends),
            "motion_snap_accepted": int(self.motion.snap_estimates_accepted),
            "motion_snap_refused": int(self.motion.snap_estimates_refused),
            "motion_play_end_accepted": int(self.motion.play_end_estimates_accepted),
            "motion_play_end_refused": int(self.motion.play_end_estimates_refused),
            "motion_fabricated_boundaries": int(self.motion.fabricated_boundaries),
        }

    # -- label validation --------------------------------------------------
    @staticmethod
    def validate_labels(labels: Sequence[PlayTimestampLabel]) -> List[PlayTimestampLabel]:
        """Sort and validate manual labels; refuse ambiguous or overlapping plays."""
        if not labels:
            raise ValueError("at least one PlayTimestampLabel is required")
        ordered = sorted(labels, key=lambda lbl: (int(lbl.start_frame), str(lbl.play_id)))
        ids = [lbl.play_id for lbl in ordered]
        duplicates = sorted({pid for pid in ids if ids.count(pid) > 1})
        if duplicates:
            raise ValueError(f"duplicate play_id(s): {duplicates}")
        # An open-ended play (no end_frame) can only be the LAST play: its end may be
        # estimated automatically, but until then nothing can be checked against it.
        open_before_last = [lbl.play_id for lbl in ordered[:-1] if lbl.end_frame is None]
        if open_before_last:
            raise ValueError(
                f"play(s) {open_before_last} have no end_frame but are not the final play, "
                "so non-overlap cannot be verified; supply end_frame for them"
            )
        previous: Optional[PlayTimestampLabel] = None
        for lbl in ordered:
            if previous is not None:
                if previous.end_frame is None:  # unreachable: guarded above
                    raise ValueError(
                        f"play {previous.play_id!r} has no end_frame, so non-overlap with "
                        f"play {lbl.play_id!r} cannot be verified"
                    )
                if int(lbl.start_frame) <= int(previous.end_frame):
                    raise ValueError(
                        f"play {lbl.play_id!r} starts at frame {lbl.start_frame}, which "
                        f"overlaps play {previous.play_id!r} (ends at {previous.end_frame})"
                    )
            previous = lbl
        return ordered

    # -- main entry point --------------------------------------------------
    def segment(
        self,
        labels: Sequence[PlayTimestampLabel],
        trajectories: Sequence[FieldTrajectory] = (),
        *,
        game_id: str = "",
    ) -> PlaySegmentationResult:
        """Segment every labelled play, completing snap/play-end where allowed."""
        started = time.perf_counter()
        ordered = self.validate_labels(labels)
        if not self.allow_automatic_play_end:
            missing_end = [lbl.play_id for lbl in ordered if lbl.end_frame is None]
            if missing_end:
                raise ValueError(
                    f"play(s) {missing_end} have no end_frame and automatic play-end "
                    "estimation is disabled; supply end_frame or enable it"
                )
        result = PlaySegmentationResult(game_id=str(game_id), fps=self.fps)
        result.last_frame = _resolve_last_frame(ordered, trajectories)

        for lbl in ordered:
            notes: List[str] = list(lbl.notes)
            window_trajs = _trajectories_in_window(trajectories, lbl.start_frame, lbl.end_frame)

            snap_frame, snap_source, snap_confidence, snap_estimate = self._resolve_snap(
                lbl, window_trajs, result
            )
            end_frame, end_source, end_confidence, end_estimate = self._resolve_play_end(
                lbl, window_trajs, snap_frame, result
            )
            if snap_frame is None:
                notes.append(f"snap refused: {snap_estimate.reason}")
            if end_frame is None and lbl.end_frame is None:
                notes.append(f"play_end refused: {end_estimate.reason}")

            mode, coord_id, segments = _coordinate_provenance(
                window_trajs, lbl.start_frame, end_frame
            )
            segment = PlaySegment(
                game_id=str(game_id),
                play_id=lbl.play_id,
                start_frame=int(lbl.start_frame),
                end_frame=end_frame,
                snap_frame=snap_frame,
                snap_source=snap_source,
                end_source=end_source,
                fps=self.fps,
                snap_confidence=snap_confidence,
                end_confidence=end_confidence,
                x_coord_mode=mode,
                coordinate_frame_id=coord_id,
                coordinate_segments=segments,
                n_tracks=len(window_trajs),
                snap_estimate=snap_estimate,
                end_estimate=end_estimate,
                notes=notes,
            )
            # Audit: a boundary may never exist without a real source. PlaySegment
            # validation already forbids it; the counter makes the invariant
            # measurable end-to-end (Phase 20) instead of merely asserted.
            if segment.snap_frame is not None and segment.snap_source == "unavailable":
                self.fabricated_snap_frames += 1
            if segment.end_frame is not None and segment.end_source == "unavailable":
                self.fabricated_play_ends += 1

            result.segments.append(segment)
            self.plays_segmented += 1

        result.fabricated_snap_frames = int(self.fabricated_snap_frames)
        result.fabricated_play_ends = int(self.fabricated_play_ends)
        self.runtime_ms = (time.perf_counter() - started) * 1000.0
        result.runtime_ms = self.runtime_ms
        # Fails loudly if two segments ever claim the same frame.
        result.frame_phases()
        return result

    # -- per-play resolvers ------------------------------------------------
    def _resolve_snap(
        self,
        lbl: PlayTimestampLabel,
        window_trajs: Sequence[FieldTrajectory],
        result: PlaySegmentationResult,
    ) -> Tuple[Optional[int], str, float, SnapEstimate]:
        if lbl.snap_frame is not None:
            snap_frame = int(lbl.snap_frame)
            self.snaps_from_manual += 1
            return (
                snap_frame,
                "manual",
                1.0,
                SnapEstimate(
                    play_id=lbl.play_id,
                    frame_id=snap_frame,
                    timestamp_s=round(snap_frame / self.fps, 6),
                    source="manual",
                    confidence=1.0,
                    notes=["manual timestamp; not verified against motion evidence"],
                ),
            )
        if not self.allow_automatic_snap:
            self.snaps_refused += 1
            result.refusals.append(
                SegmentationRefusal(
                    scope="snap",
                    play_id=lbl.play_id,
                    reason=REASON_AUTOMATIC_DISABLED,
                    detail="allow_automatic_snap=False and no manual snap_frame",
                )
            )
            return (
                None,
                "unavailable",
                0.0,
                SnapEstimate(
                    play_id=lbl.play_id,
                    frame_id=None,
                    timestamp_s=None,
                    source="unavailable",
                    confidence=0.0,
                    reason=REASON_AUTOMATIC_DISABLED,
                    notes=["automatic snap estimation disabled by configuration"],
                ),
            )
        estimate = self.motion.detect_snap(
            window_trajs,
            play_id=lbl.play_id,
            start_frame=int(lbl.start_frame),
            end_frame=lbl.end_frame,
            fps=self.fps,
        )
        if estimate.frame_id is None:
            self.snaps_refused += 1
            result.refusals.append(
                SegmentationRefusal(
                    scope="snap",
                    play_id=lbl.play_id,
                    reason=str(estimate.reason),
                    detail="; ".join(estimate.notes),
                )
            )
        else:
            self.snaps_from_motion_onset += 1
        return estimate.frame_id, estimate.source, float(estimate.confidence), estimate

    def _resolve_play_end(
        self,
        lbl: PlayTimestampLabel,
        window_trajs: Sequence[FieldTrajectory],
        snap_frame: Optional[int],
        result: PlaySegmentationResult,
    ) -> Tuple[Optional[int], str, float, PlayEndEstimate]:
        if lbl.end_frame is not None:
            end_frame = int(lbl.end_frame)
            self.play_ends_from_manual += 1
            return (
                end_frame,
                "manual",
                1.0,
                PlayEndEstimate(
                    play_id=lbl.play_id,
                    frame_id=end_frame,
                    timestamp_s=round(end_frame / self.fps, 6),
                    source="manual",
                    confidence=1.0,
                    notes=["manual timestamp; not verified against motion evidence"],
                ),
            )
        if snap_frame is None:
            self.play_ends_refused += 1
            result.refusals.append(
                SegmentationRefusal(
                    scope="play_end",
                    play_id=lbl.play_id,
                    reason=REASON_NO_SNAP_FOR_END_SEARCH,
                    detail="no manual end_frame and no resolved snap to search from",
                )
            )
            return (
                None,
                "unavailable",
                0.0,
                PlayEndEstimate(
                    play_id=lbl.play_id,
                    frame_id=None,
                    timestamp_s=None,
                    source="unavailable",
                    confidence=0.0,
                    reason=REASON_NO_SNAP_FOR_END_SEARCH,
                    notes=["the play-end search starts after the snap, which was refused"],
                ),
            )
        estimate = self.motion.detect_play_end(
            window_trajs,
            play_id=lbl.play_id,
            snap_frame=int(snap_frame),
            end_frame=None,
            fps=self.fps,
        )
        if estimate.frame_id is None:
            self.play_ends_refused += 1
            result.refusals.append(
                SegmentationRefusal(
                    scope="play_end",
                    play_id=lbl.play_id,
                    reason=str(estimate.reason),
                    detail="; ".join(estimate.notes),
                )
            )
        else:
            self.play_ends_from_motion_cessation += 1
        return estimate.frame_id, estimate.source, float(estimate.confidence), estimate


def _resolve_last_frame(
    labels: Sequence[PlayTimestampLabel], trajectories: Sequence[FieldTrajectory]
) -> Optional[int]:
    """Highest frame index known from manual labels or trajectory evidence."""
    candidates: List[int] = [int(lbl.start_frame) for lbl in labels]
    candidates += [int(lbl.end_frame) for lbl in labels if lbl.end_frame is not None]
    last_traj = _last_trajectory_frame(trajectories)
    if last_traj is not None:
        candidates.append(int(last_traj))
    return max(candidates) if candidates else None


def _trajectories_in_window(
    trajectories: Sequence[FieldTrajectory],
    start_frame: int,
    end_frame: Optional[int],
) -> List[FieldTrajectory]:
    """Select trajectories with at least one sample inside the play window."""
    selected: List[FieldTrajectory] = []
    for traj in trajectories:
        for sample in traj.samples:
            fid = int(sample.frame_id)
            if fid < int(start_frame):
                continue
            if end_frame is not None and fid > int(end_frame):
                continue
            selected.append(traj)
            break
    return selected


def _coordinate_provenance(
    trajectories: Sequence[FieldTrajectory],
    start_frame: int,
    end_frame: Optional[int],
) -> Tuple[XCoordMode, Optional[str], Tuple[int, ...]]:
    """Aggregate coordinate provenance over a play window.

    Follows the Phase 4 rule: mixed ``x_coord_mode`` values aggregate to
    ``uncalibrated`` (consumers must use the per-sample mode), a mixed
    ``coordinate_frame_id`` aggregates to ``None``, and every coordinate segment
    touched is reported so nobody connects positions across a reset.
    """
    modes: List[str] = []
    coord_ids: List[str] = []
    segments: List[int] = []
    for traj in trajectories:
        for sample in traj.samples:
            fid = int(sample.frame_id)
            if fid < int(start_frame):
                continue
            if end_frame is not None and fid > int(end_frame):
                continue
            if sample.x_coord_mode not in modes:
                modes.append(sample.x_coord_mode)
            if sample.coordinate_frame_id is not None:
                cid = str(sample.coordinate_frame_id)
                if cid not in coord_ids:
                    coord_ids.append(cid)
            seg = int(sample.coordinate_segment)
            if seg not in segments:
                segments.append(seg)
    distinct_modes = [m for m in modes if m != "uncalibrated"]
    mode: XCoordMode = distinct_modes[0] if len(distinct_modes) == 1 else "uncalibrated"  # type: ignore[assignment]
    coord_id = coord_ids[0] if len(coord_ids) == 1 else None
    return mode, coord_id, tuple(sorted(segments))
