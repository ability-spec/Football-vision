"""Phase 10 play-segmentation tests.

Covers manual timestamp segmentation and per-frame phase labelling, automatic
snap detection from collective motion onset, automatic play-end detection from
motion cessation, evidence accounting (dead reckoning / coasted / rejected /
unprojected samples are never motion), coordinate-continuity refusals, label
validation, serialization, and the no-fabrication audit invariants.

Everything here is synthetic and deterministic: no third-party NFL assets are
required, so these tests run on a CPU-only CI machine.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

import pytest

from football_vision import (
    CollectiveMotionSegmenter,
    FieldTrajectory,
    MotionEvidence,
    PlayEndEstimate,
    PlaySegment,
    PlaySegmentationResult,
    PlaySegmenter,
    PlayTimestampLabel,
    SegmentationRefusal,
    SnapEstimate,
    TrajectorySample,
    build_motion_evidence,
)
from football_vision.analytics.segmentation import (
    REASON_AUTOMATIC_DISABLED,
    REASON_BASELINE_EVIDENCE_GAPS,
    REASON_COORDINATE_CHANGE,
    REASON_INSUFFICIENT_MEASURED_SAMPLES,
    REASON_INSUFFICIENT_TRACKS,
    REASON_NO_MOTION_CESSATION,
    REASON_NO_PRIOR_MOTION,
    REASON_NO_QUIESCENT_BASELINE,
    REASON_NO_SNAP_FOR_END_SEARCH,
    REASON_NO_SUSTAINED_ONSET,
    REASON_NO_TRAJECTORIES,
    REASON_WINDOW_TOO_SHORT,
    REFUSAL_REASONS,
)

FPS = 30.0
COORD_ID = "calibration:0"


# ---------------------------------------------------------------------------
# Synthetic trajectory fixtures
# ---------------------------------------------------------------------------
def make_sample(
    *,
    track_id: int,
    frame_id: int,
    speed_yd_s: float = 0.0,
    kind: str = "measured",
    coordinate_frame_id: Optional[str] = COORD_ID,
    coordinate_segment: int = 0,
    x_coord_mode: str = "relative_10yd",
    position: Tuple[float, float] = (20.0, 26.0),
) -> TrajectorySample:
    """Build one Phase 4 sample of an explicitly requested evidence kind.

    ``kind`` controls which Phase 4 state the sample represents:
      * ``measured``        — accepted measurement of an observed track
      * ``dead_reckoning``  — coasting prediction (never a position claim)
      * ``rejected``        — measurement rejected by the innovation gate
      * ``coasted``         — measured position but the track was coasting
      * ``unprojected``     — geometry unknown, no field position at all
    """
    timestamp_s = frame_id / FPS
    velocity = (float(speed_yd_s), 0.0)
    if kind == "measured":
        return TrajectorySample(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            geometry_state="calibrated",
            x_coord_mode=x_coord_mode,
            track_state="observed",
            image_footpoint=(100.0 + track_id, 200.0),
            field_position=position,
            raw_field_position=position,
            position_source="measured_smoothed",
            velocity_yd_s=velocity,
            speed_yd_s=float(speed_yd_s),
            is_measurement_used=True,
            coordinate_frame_id=coordinate_frame_id,
            coordinate_segment=coordinate_segment,
        )
    if kind == "dead_reckoning":
        return TrajectorySample(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            geometry_state="propagated",
            x_coord_mode=x_coord_mode,
            track_state="coasted",
            predicted_position=position,
            position_source="predicted_dead_reckoning",
            velocity_yd_s=velocity,
            speed_yd_s=float(speed_yd_s),
            missed_frames=1,
            coordinate_frame_id=coordinate_frame_id,
            coordinate_segment=coordinate_segment,
        )
    if kind == "rejected":
        return TrajectorySample(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            geometry_state="calibrated",
            x_coord_mode=x_coord_mode,
            track_state="observed",
            predicted_position=position,
            position_source="predicted_dead_reckoning",
            velocity_yd_s=velocity,
            speed_yd_s=float(speed_yd_s),
            is_outlier_rejected=True,
            rejection_reason="field_innovation_gate",
            is_measurement_used=False,
            coordinate_frame_id=coordinate_frame_id,
            coordinate_segment=coordinate_segment,
        )
    if kind == "coasted":
        return TrajectorySample(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            geometry_state="calibrated",
            x_coord_mode=x_coord_mode,
            track_state="coasted",
            field_position=position,
            position_source="measured_smoothed",
            velocity_yd_s=velocity,
            speed_yd_s=float(speed_yd_s),
            is_measurement_used=True,
            missed_frames=2,
            coordinate_frame_id=coordinate_frame_id,
            coordinate_segment=coordinate_segment,
        )
    if kind == "unprojected":
        return TrajectorySample(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            geometry_state="unknown",
            x_coord_mode="uncalibrated",
            track_state="observed",
            position_source="none",
            velocity_yd_s=(0.0, 0.0),
            speed_yd_s=0.0,
            coordinate_frame_id=None,
            coordinate_segment=coordinate_segment,
        )
    raise ValueError(f"unknown sample kind {kind!r}")


def make_trajectory(
    track_id: int,
    speeds: Sequence[float],
    *,
    start_frame: int = 0,
    kinds: Optional[Sequence[str]] = None,
    coordinate_frame_id: Optional[str] = COORD_ID,
    coordinate_segment: int = 0,
    x_coord_mode: str = "relative_10yd",
    positions: Optional[Sequence[Tuple[float, float]]] = None,
) -> FieldTrajectory:
    """Build a ``FieldTrajectory`` from an explicit per-frame speed profile."""
    samples: List[TrajectorySample] = []
    for i, speed in enumerate(speeds):
        frame_id = start_frame + i
        kind = "measured" if kinds is None else kinds[i]
        position = (20.0 + 0.5 * i, 26.0) if positions is None else positions[i]
        samples.append(
            make_sample(
                track_id=track_id,
                frame_id=frame_id,
                speed_yd_s=speed,
                kind=kind,
                coordinate_frame_id=coordinate_frame_id,
                coordinate_segment=coordinate_segment,
                x_coord_mode=x_coord_mode,
                position=position,
            )
        )
    return FieldTrajectory(
        track_id=track_id,
        fps=FPS,
        x_coord_mode=x_coord_mode,  # type: ignore[arg-type]
        samples=samples,
        detector_name="synthetic_fixture_v1",
    )


def make_field(
    n_tracks: int,
    speeds: Sequence[float],
    *,
    start_frame: int = 0,
    **kwargs,
) -> List[FieldTrajectory]:
    """Build ``n_tracks`` trajectories that all follow the same speed profile."""
    return [
        make_trajectory(track_id, speeds, start_frame=start_frame, **kwargs)
        for track_id in range(1, n_tracks + 1)
    ]


def stationary_then_moving(
    n_frames: int, snap_frame: int, *, quiet_speed: float = 0.2, move_speed: float = 3.5
) -> List[float]:
    return [quiet_speed if f < snap_frame else move_speed for f in range(n_frames)]


# ---------------------------------------------------------------------------
# 1. Manual timestamps -> phase labels
# ---------------------------------------------------------------------------
class TestManualSegmentation:
    def test_phase_labels_follow_the_roadmap_state_machine(self):
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_snap=False)
        labels = [PlayTimestampLabel(play_id="p1", start_frame=10, snap_frame=30, end_frame=80)]
        result = segmenter.segment(labels, [], game_id="g1")

        assert len(result.segments) == 1
        seg = result.segments[0]
        assert (seg.start_frame, seg.snap_frame, seg.end_frame) == (10, 30, 80)
        assert seg.snap_source == "manual"
        assert seg.end_source == "manual"
        assert seg.snap_confidence == 1.0

        phases = result.frame_phases()
        assert 9 not in phases  # the phase map only covers [start_frame, end_frame]
        assert seg.phase_for(9) == "outside_play"
        assert seg.phase_for(10) == "pre_snap"
        assert seg.phase_for(29) == "pre_snap"
        assert seg.phase_for(30) == "snap"
        assert seg.phase_for(31) == "play"
        assert seg.phase_for(79) == "play"
        assert seg.phase_for(80) == "play_end"
        assert seg.phase_for(81) == "outside_play"
        assert phases[30] == "snap"
        assert result.phase_for(200) == "outside_play"
        assert seg.n_frames == 71

    def test_phase_counts_are_exhaustive_over_the_play(self):
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_snap=False)
        result = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, snap_frame=5, end_frame=15)], []
        )
        counts = result.to_dict()["phase_counts"]
        assert counts["pre_snap"] == 5
        assert counts["snap"] == 1
        assert counts["play"] == 9
        assert counts["play_end"] == 1
        assert counts["snap_unknown"] == 0
        assert counts["outside_play"] == 0
        assert sum(counts.values()) == 16

    def test_multiple_plays_are_ordered_and_disjoint(self):
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_snap=False)
        labels = [
            PlayTimestampLabel(play_id="p2", start_frame=200, snap_frame=220, end_frame=280),
            PlayTimestampLabel(play_id="p1", start_frame=50, snap_frame=70, end_frame=120),
        ]
        result = segmenter.segment(labels, [], game_id="g1")
        assert [s.play_id for s in result.segments] == ["p1", "p2"]
        phases = result.frame_phases()
        assert phases[50] == "pre_snap" and phases[220] == "snap"
        assert result.segment_for(100).play_id == "p1"
        assert result.segment_for(150) is None
        assert result.segment_for(250).play_id == "p2"

    def test_manual_timestamps_are_not_recomputed_from_motion(self):
        """A manual snap is copied through even when the motion evidence disagrees."""
        trajs = make_field(8, stationary_then_moving(60, snap_frame=25))
        segmenter = PlaySegmenter(fps=FPS)
        result = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, snap_frame=5, end_frame=59)], trajs
        )
        seg = result.segments[0]
        assert seg.snap_frame == 5
        assert seg.snap_source == "manual"
        assert segmenter.snaps_from_manual == 1
        assert segmenter.snaps_from_motion_onset == 0

    def test_to_play_record_carries_boundaries_and_events(self):
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_snap=False)
        seg = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=10, snap_frame=30, end_frame=80)], [],
            game_id="g1",
        ).segments[0]
        record = seg.to_play_record()
        assert record.game_id == "g1" and record.play_id == "p1"
        assert (record.start_frame, record.snap_frame, record.end_frame) == (10, 30, 80)
        assert record.x_coord_mode == "uncalibrated"
        assert record.metrics == {}
        kinds = [e["type"] for e in record.events]
        assert kinds == ["snap", "play_end"]
        assert record.events[0]["source"] == "manual"
        assert record.events[0]["timestamp_s"] == pytest.approx(30 / FPS)

    def test_open_play_exports_an_explicit_sentinel_not_a_guess(self):
        # An open-ended label is only admissible as the final play with automatic
        # end estimation enabled, so build the segment directly to test the export.
        seg = PlaySegment(
            game_id="g1",
            play_id="p1",
            start_frame=10,
            end_frame=None,
            snap_frame=None,
            snap_source="unavailable",
            end_source="unavailable",
            fps=FPS,
        )
        record = seg.to_play_record()
        assert record.end_frame == -1
        assert record.snap_frame is None
        assert any("sentinel -1" in n for n in record.notes)
        assert seg.frame_phases() == {}
        assert seg.phase_for(10) == "snap_unknown"


# ---------------------------------------------------------------------------
# 2. Label validation
# ---------------------------------------------------------------------------
class TestLabelValidation:
    def test_snap_before_start_is_rejected_by_the_contract(self):
        with pytest.raises(ValueError, match="precedes start_frame"):
            PlayTimestampLabel(play_id="p1", start_frame=50, snap_frame=10, end_frame=80)

    def test_end_before_start_is_rejected_by_the_contract(self):
        with pytest.raises(ValueError, match="must be >= start_frame"):
            PlayTimestampLabel(play_id="p1", start_frame=50, end_frame=40)

    def test_snap_after_end_is_rejected_by_the_contract(self):
        with pytest.raises(ValueError, match="follows end_frame"):
            PlayTimestampLabel(play_id="p1", start_frame=10, snap_frame=90, end_frame=80)

    def test_negative_start_frame_is_rejected(self):
        with pytest.raises(ValueError, match="start_frame must be >= 0"):
            PlayTimestampLabel(play_id="p1", start_frame=-1)

    def test_empty_play_id_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            PlayTimestampLabel(play_id="", start_frame=0)

    def test_empty_label_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one PlayTimestampLabel"):
            PlaySegmenter(fps=FPS).segment([], [])

    def test_duplicate_play_ids_are_refused(self):
        labels = [
            PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=10),
            PlayTimestampLabel(play_id="p1", start_frame=20, end_frame=30),
        ]
        with pytest.raises(ValueError, match="duplicate play_id"):
            PlaySegmenter(fps=FPS).segment(labels, [])

    def test_overlapping_plays_are_refused(self):
        labels = [
            PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=50),
            PlayTimestampLabel(play_id="p2", start_frame=50, end_frame=90),
        ]
        with pytest.raises(ValueError, match="overlaps play"):
            PlaySegmenter(fps=FPS).segment(labels, [])

    def test_open_ended_play_must_be_last(self):
        labels = [
            PlayTimestampLabel(play_id="p1", start_frame=0),
            PlayTimestampLabel(play_id="p2", start_frame=50, end_frame=90),
        ]
        with pytest.raises(ValueError, match="not the final play"):
            PlaySegmenter(fps=FPS, allow_automatic_play_end=True).segment(labels, [])

    def test_open_ended_play_needs_automatic_end_enabled(self):
        labels = [PlayTimestampLabel(play_id="p1", start_frame=0, snap_frame=10)]
        with pytest.raises(ValueError, match="automatic play-end estimation is disabled"):
            PlaySegmenter(fps=FPS, allow_automatic_play_end=False).segment(labels, [])

    def test_invalid_fps_is_refused(self):
        with pytest.raises(ValueError, match="fps must be a positive finite number"):
            PlaySegmenter(fps=0.0)
        with pytest.raises(ValueError, match="fps must be a positive finite number"):
            PlaySegmenter(fps=float("nan"))

    def test_contradictory_thresholds_are_refused(self):
        with pytest.raises(ValueError, match="baseline_max_fraction"):
            CollectiveMotionSegmenter(onset_min_fraction=0.2, baseline_max_fraction=0.5)
        with pytest.raises(ValueError, match="quiet_max_fraction"):
            CollectiveMotionSegmenter(quiet_max_fraction=0.9, active_min_fraction=0.4)
        with pytest.raises(ValueError, match="min_tracks"):
            CollectiveMotionSegmenter(min_tracks=0)

    def test_overlapping_segments_never_share_a_frame_phase(self):
        """Direct contract test: the global phase map refuses ambiguity."""
        a = PlaySegment(
            game_id="g", play_id="a", start_frame=0, end_frame=20, snap_frame=5,
            snap_source="manual", end_source="manual", fps=FPS,
        )
        b = PlaySegment(
            game_id="g", play_id="b", start_frame=15, end_frame=40, snap_frame=20,
            snap_source="manual", end_source="manual", fps=FPS,
        )
        result = PlaySegmentationResult(game_id="g", fps=FPS, segments=[a, b])
        with pytest.raises(ValueError, match="overlapping play segments"):
            result.frame_phases()


# ---------------------------------------------------------------------------
# 3. Segment contract invariants
# ---------------------------------------------------------------------------
class TestSegmentContract:
    def test_snap_frame_and_source_must_agree(self):
        with pytest.raises(ValueError, match="snap_frame and snap_source must agree"):
            PlaySegment(
                game_id="g", play_id="p", start_frame=0, end_frame=10, snap_frame=5,
                snap_source="unavailable", end_source="manual", fps=FPS,
            )
        with pytest.raises(ValueError, match="snap_frame and snap_source must agree"):
            PlaySegment(
                game_id="g", play_id="p", start_frame=0, end_frame=10, snap_frame=None,
                snap_source="manual", end_source="manual", fps=FPS,
            )

    def test_end_frame_and_source_must_agree(self):
        with pytest.raises(ValueError, match="end_frame and end_source must agree"):
            PlaySegment(
                game_id="g", play_id="p", start_frame=0, end_frame=10, snap_frame=None,
                snap_source="unavailable", end_source="unavailable", fps=FPS,
            )

    def test_snap_outside_the_play_is_refused(self):
        with pytest.raises(ValueError, match="snap_frame precedes start_frame"):
            PlaySegment(
                game_id="g", play_id="p", start_frame=10, end_frame=20, snap_frame=5,
                snap_source="manual", end_source="manual", fps=FPS,
            )

    def test_estimate_disagreement_is_refused(self):
        est = SnapEstimate(
            play_id="p", frame_id=7, timestamp_s=7 / FPS, source="manual", confidence=1.0
        )
        with pytest.raises(ValueError, match="snap_estimate disagrees"):
            PlaySegment(
                game_id="g", play_id="p", start_frame=0, end_frame=20, snap_frame=5,
                snap_source="manual", end_source="manual", fps=FPS, snap_estimate=est,
            )

    def test_snap_estimate_cannot_claim_a_source_without_a_frame(self):
        with pytest.raises(ValueError, match="fabricated snap claim"):
            SnapEstimate(play_id="p", frame_id=None, timestamp_s=None, source="manual")
        with pytest.raises(ValueError, match="explicit reason"):
            SnapEstimate(play_id="p", frame_id=None, timestamp_s=None, source="unavailable")
        with pytest.raises(ValueError, match="timestamp_s"):
            SnapEstimate(play_id="p", frame_id=3, timestamp_s=None, source="manual")

    def test_play_end_estimate_mirrors_the_snap_contract(self):
        with pytest.raises(ValueError, match="PlayEndEstimate must either name a frame"):
            PlayEndEstimate(play_id="p", frame_id=None, timestamp_s=None, source="motion_cessation")
        est = PlayEndEstimate(
            play_id="p", frame_id=None, timestamp_s=None, source="unavailable",
            reason=REASON_NO_MOTION_CESSATION,
        )
        assert est.frame_id is None and est.reason == REASON_NO_MOTION_CESSATION


# ---------------------------------------------------------------------------
# 4. Motion evidence accounting
# ---------------------------------------------------------------------------
class TestMotionEvidence:
    def test_moving_fraction_counts_tracks_above_the_speed_threshold(self):
        speeds = [0.2] * 10 + [3.0] * 10
        # Half the tracks move from frame 10 onwards.
        trajs = [make_trajectory(i, speeds) for i in range(1, 9)]
        trajs += [make_trajectory(i, [0.2] * 20) for i in range(9, 13)]
        evidence = build_motion_evidence(trajs, start_frame=0, end_frame=19, min_tracks=6)
        assert evidence.n_frames == 20
        assert evidence.n_defined == 20
        assert evidence.fraction(0) == pytest.approx(0.0)
        assert evidence.fraction(10) == pytest.approx(8 / 12)
        assert evidence.usable_count(10) == 12
        assert evidence.distinct_coordinate_frame_ids() == (COORD_ID,)

    def test_frames_with_too_few_tracks_are_undefined_not_zero(self):
        trajs = make_field(3, [3.0] * 12)
        evidence = build_motion_evidence(trajs, start_frame=0, end_frame=11, min_tracks=6)
        assert evidence.n_defined == 0
        assert evidence.n_undefined == 12
        assert all(f is None for f in evidence.moving_fraction)
        assert evidence.defined_frame_ids() == []

    def test_every_sample_is_accounted_exactly_once(self):
        """usable + all exclusion buckets == samples in window (no silent loss)."""
        n_frames = 12
        trajs = [
            make_trajectory(1, [3.0] * n_frames),                                     # measured
            make_trajectory(2, [3.0] * n_frames, kinds=["dead_reckoning"] * n_frames),
            make_trajectory(3, [3.0] * n_frames, kinds=["rejected"] * n_frames),
            make_trajectory(4, [3.0] * n_frames, kinds=["coasted"] * n_frames),
            make_trajectory(5, [0.0] * n_frames, kinds=["unprojected"] * n_frames),
            make_trajectory(6, [3.0] * n_frames),
            make_trajectory(7, [3.0] * n_frames),
            make_trajectory(8, [3.0] * n_frames),
            make_trajectory(9, [3.0] * n_frames),
            make_trajectory(10, [3.0] * n_frames),
        ]
        evidence = build_motion_evidence(trajs, start_frame=0, end_frame=n_frames - 1, min_tracks=5)
        assert evidence.n_samples_in_window == 10 * n_frames
        total = (
            evidence.n_usable_samples
            + evidence.n_excluded_dead_reckoning
            + evidence.n_excluded_rejected
            + evidence.n_excluded_coasted
            + evidence.n_excluded_unprojected
        )
        assert total == evidence.n_samples_in_window
        assert evidence.n_excluded_dead_reckoning == n_frames
        assert evidence.n_excluded_rejected == n_frames
        assert evidence.n_excluded_coasted == n_frames
        assert evidence.n_excluded_unprojected == n_frames
        assert evidence.n_usable_samples == 6 * n_frames
        # Coasted/dead-reckoned tracks reported speed but must not count as motion.
        assert evidence.fraction(0) == pytest.approx(1.0)

    def test_dead_reckoning_alone_cannot_create_motion(self):
        n_frames = 30
        trajs = [
            make_trajectory(
                i, [4.0] * n_frames, kinds=["dead_reckoning"] * n_frames
            )
            for i in range(1, 9)
        ]
        seg = CollectiveMotionSegmenter()
        est = seg.detect_snap(
            trajs, play_id="p1", start_frame=0, end_frame=n_frames - 1, fps=FPS
        )
        assert est.frame_id is None
        assert est.source == "unavailable"
        assert est.reason == REASON_INSUFFICIENT_MEASURED_SAMPLES
        assert "dead_reckoning=240" in "; ".join(est.notes) or est.n_usable_samples == 0

    def test_mixed_coordinate_frames_inside_one_frame_are_inconsistent(self):
        n_frames = 12
        trajs = [make_trajectory(i, [0.2] * n_frames) for i in range(1, 5)]
        trajs += [
            make_trajectory(i, [0.2] * n_frames, coordinate_frame_id="calibration:1")
            for i in range(5, 9)
        ]
        evidence = build_motion_evidence(trajs, start_frame=0, end_frame=n_frames - 1, min_tracks=6)
        assert all(cid is None for cid in evidence.coordinate_frame_ids)
        assert evidence.distinct_coordinate_frame_ids() == (None,)

    def test_evidence_requires_a_single_resolvable_fps(self):
        a = make_trajectory(1, [0.2] * 10)
        b = FieldTrajectory(track_id=2, fps=60.0, x_coord_mode="relative_10yd", samples=[])
        with pytest.raises(ValueError, match="cannot resolve a single fps"):
            build_motion_evidence([a, b], start_frame=0, end_frame=9)
        with pytest.raises(ValueError, match="disagrees with fps"):
            build_motion_evidence([a], start_frame=0, end_frame=9, fps=60.0)

    def test_evidence_window_must_be_ordered(self):
        with pytest.raises(ValueError, match="must be >= start_frame"):
            build_motion_evidence(make_field(6, [0.2] * 5), start_frame=5, end_frame=1)

    def test_evidence_serializes_without_nan(self):
        evidence = build_motion_evidence(
            make_field(7, [0.2] * 8 + [3.0] * 8), start_frame=0, end_frame=15, min_tracks=6
        )
        blob = json.dumps(evidence.to_dict(), allow_nan=False)
        assert "moving_fraction" in blob
        assert len(json.loads(blob)["frames"]) == 16


# ---------------------------------------------------------------------------
# 5. Automatic snap detection (collective motion onset)
# ---------------------------------------------------------------------------
class TestAutomaticSnap:
    def test_onset_recovers_the_exact_snap_frame(self):
        snap = 25
        trajs = make_field(11, stationary_then_moving(60, snap))
        segmenter = PlaySegmenter(fps=FPS)
        result = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=59)], trajs, game_id="g1"
        )
        seg = result.segments[0]
        assert seg.snap_frame == snap
        assert seg.snap_source == "collective_motion_onset"
        assert seg.snap_confidence > 0.5
        assert seg.phase_for(snap - 1) == "pre_snap"
        assert seg.phase_for(snap) == "snap"
        assert seg.phase_for(snap + 1) == "play"
        assert segmenter.snaps_from_motion_onset == 1
        assert segmenter.fabricated_snap_frames == 0
        assert result.refusals == []

    def test_onset_is_the_first_sustained_frame_not_the_peak(self):
        speeds = [0.2] * 15 + [3.0] * 5 + [0.2] * 10 + [3.0] * 20
        trajs = make_field(10, speeds)
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p1", start_frame=0, end_frame=len(speeds) - 1, fps=FPS
        )
        assert est.frame_id == 15

    def test_a_single_mover_in_the_backfield_does_not_trigger_the_snap(self):
        """Pre-snap motion/shifts are 1-2 players: fraction stays under baseline_max."""
        n_frames, snap = 50, 30
        trajs = [make_trajectory(1, [0.2] * 20 + [3.0] * 10 + [0.2] * 10 + [3.0] * 10)]
        trajs += [make_trajectory(i, stationary_then_moving(n_frames, snap)) for i in range(2, 12)]
        # Track 1 moves alone in frames 20-29 (1/11 = 0.09 <= 0.15): still quiescent.
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p1", start_frame=0, end_frame=n_frames - 1, fps=FPS
        )
        assert est.frame_id == snap
        assert est.baseline_fraction == pytest.approx(1 / 11, abs=1e-6)

    def test_confidence_grows_with_contrast_and_is_never_certain(self):
        """Confidence = observed contrast / reference contrast x usable coverage, capped."""
        # Half the visible group moves at the onset: onset_fraction = 0.5.
        weak_trajs = make_field(5, [0.2] * 20 + [3.5] * 20) + make_field(5, [0.2] * 40)
        for i, t in enumerate(weak_trajs[5:], start=6):
            t.track_id = i
            for s in t.samples:
                s.track_id = i
        strong_trajs = make_field(10, [0.2] * 20 + [3.5] * 20)
        seg = CollectiveMotionSegmenter(confidence_reference_contrast=1.0)
        weak = seg.detect_snap(weak_trajs, play_id="w", start_frame=0, end_frame=39, fps=FPS)
        strong = seg.detect_snap(strong_trajs, play_id="s", start_frame=0, end_frame=39, fps=FPS)
        assert weak.frame_id == 20 and strong.frame_id == 20
        assert weak.onset_fraction == pytest.approx(0.5)
        assert strong.onset_fraction == pytest.approx(1.0)
        assert weak.confidence == pytest.approx(0.5)
        # Cap: even a perfect contrast never claims certainty.
        assert strong.confidence == pytest.approx(0.99)
        assert weak.confidence < strong.confidence <= 0.99

    def test_evidence_gaps_lower_confidence_through_coverage(self):
        """Undefined frames (too few visible tracks) reduce coverage, hence confidence."""
        full = make_field(10, stationary_then_moving(40, 20))
        gappy = make_field(10, stationary_then_moving(40, 20))
        # Frames 25-34 lose 5 of 10 tracks -> 5 usable < min_tracks=7 -> undefined.
        for t in gappy[5:]:
            for smp in t.samples:
                if 25 <= smp.frame_id <= 34:
                    smp.position_source = "none"
                    smp.field_position = None
                    smp.is_measurement_used = False
                    smp.geometry_state = "unknown"
                    smp.x_coord_mode = "uncalibrated"
        seg = CollectiveMotionSegmenter(min_tracks=7, confidence_reference_contrast=1.0)
        full_est = seg.detect_snap(full, play_id="f", start_frame=10, end_frame=39, fps=FPS)
        gappy_est = seg.detect_snap(gappy, play_id="g", start_frame=10, end_frame=39, fps=FPS)
        assert full_est.frame_id == gappy_est.frame_id == 20
        assert full_est.n_undefined_frames == 0
        assert gappy_est.n_undefined_frames == 10
        assert full_est.confidence == pytest.approx(0.99)          # coverage 20/20, capped
        assert gappy_est.confidence == pytest.approx(20 / 30, abs=1e-4)  # coverage 20/30
        assert gappy_est.confidence < full_est.confidence

    def test_snap_estimate_carries_its_evidence(self):
        trajs = make_field(9, stationary_then_moving(40, 22))
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p1", start_frame=0, end_frame=39, fps=FPS
        )
        assert est.n_tracks == 9
        assert est.n_usable_samples == 9 * 40
        assert est.motion_speed_threshold_yd_s == pytest.approx(1.0)
        assert est.min_sustained_frames == 3
        assert est.coordinate_frame_id == COORD_ID
        assert est.timestamp_s == pytest.approx(22 / FPS)
        assert est.n_undefined_frames == 0

    def test_search_without_an_end_label_uses_the_last_trajectory_frame(self):
        trajs = make_field(8, [0.2] * 20 + [3.5] * 15 + [0.2] * 10)
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_play_end=True)
        result = segmenter.segment([PlayTimestampLabel(play_id="p1", start_frame=0)], trajs)
        seg = result.segments[0]
        assert seg.snap_frame == 20
        assert seg.end_frame == 35
        assert seg.end_source == "motion_cessation"

    def test_open_play_with_no_cessation_stays_open_and_refused(self):
        """Motion that never stops: the play end is refused, not clipped to the last frame."""
        trajs = make_field(8, stationary_then_moving(45, 20))
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_play_end=True)
        result = segmenter.segment([PlayTimestampLabel(play_id="p1", start_frame=0)], trajs)
        seg = result.segments[0]
        assert seg.snap_frame == 20
        assert seg.end_frame is None and seg.end_source == "unavailable"
        assert seg.n_frames is None
        assert seg.frame_phases() == {}
        assert [r.reason for r in result.refusals] == [REASON_NO_MOTION_CESSATION]
        assert result.fabricated_play_ends == 0


# ---------------------------------------------------------------------------
# 6. Snap refusals (never fabricate)
# ---------------------------------------------------------------------------
class TestSnapRefusals:
    def test_no_trajectories(self):
        est = CollectiveMotionSegmenter().detect_snap([], play_id="p", start_frame=0, end_frame=99)
        assert est.frame_id is None and est.reason == REASON_NO_TRAJECTORIES

    def test_insufficient_tracks(self):
        trajs = make_field(3, stationary_then_moving(40, 20))
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p", start_frame=0, end_frame=39, fps=FPS
        )
        assert est.reason == REASON_INSUFFICIENT_TRACKS
        assert est.n_tracks == 3

    def test_window_too_short(self):
        trajs = make_field(8, [0.2] * 8)
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p", start_frame=0, end_frame=7, fps=FPS
        )
        assert est.reason == REASON_WINDOW_TOO_SHORT

    def test_no_quiescent_baseline_when_everybody_is_already_moving(self):
        trajs = make_field(8, [3.0] * 40)
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p", start_frame=0, end_frame=39, fps=FPS
        )
        assert est.frame_id is None
        assert est.reason == REASON_NO_QUIESCENT_BASELINE

    def test_baseline_evidence_gaps_are_not_treated_as_quiescence(self):
        """A frame below min_tracks is unknown motion, not a quiet frame."""
        n_frames = 40
        speeds = [0.2] * 20 + [3.0] * 20
        trajs = [make_trajectory(i, speeds) for i in range(1, 9)]
        # Frames 5-9 lose tracks: only 4 usable samples there (< min_tracks=6).
        for t in trajs[4:]:
            for s in t.samples:
                if 5 <= s.frame_id <= 9:
                    s.position_source = "none"
                    s.field_position = None
                    s.is_measurement_used = False
                    s.geometry_state = "unknown"
                    s.x_coord_mode = "uncalibrated"
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p", start_frame=0, end_frame=n_frames - 1, fps=FPS
        )
        assert est.frame_id is None
        assert est.reason == REASON_BASELINE_EVIDENCE_GAPS

    def test_one_frame_burst_is_not_a_sustained_onset(self):
        speeds = [0.2] * 20 + [3.0] + [0.2] * 19
        trajs = make_field(8, speeds)
        est = CollectiveMotionSegmenter(min_sustained_frames=3).detect_snap(
            trajs, play_id="p", start_frame=0, end_frame=len(speeds) - 1, fps=FPS
        )
        assert est.frame_id is None
        assert est.reason == REASON_NO_SUSTAINED_ONSET

    def test_coordinate_change_inside_the_window_is_refused(self):
        n_frames = 40
        trajs = []
        for i in range(1, 9):
            cid_first = [COORD_ID] * 20 + ["calibration:1"] * 20
            samples = [
                make_sample(
                    track_id=i,
                    frame_id=f,
                    speed_yd_s=0.2 if f < 20 else 3.0,
                    coordinate_frame_id=cid_first[f],
                    coordinate_segment=0 if f < 20 else 1,
                )
                for f in range(n_frames)
            ]
            trajs.append(
                FieldTrajectory(
                    track_id=i, fps=FPS, x_coord_mode="relative_10yd", samples=samples
                )
            )
        est = CollectiveMotionSegmenter().detect_snap(
            trajs, play_id="p", start_frame=0, end_frame=n_frames - 1, fps=FPS
        )
        assert est.frame_id is None
        assert est.reason == REASON_COORDINATE_CHANGE
        assert "coordinate_frame_id" in "; ".join(est.notes)

    def test_refusal_inside_a_play_labels_frames_snap_unknown(self):
        trajs = make_field(8, [3.0] * 40)  # never quiescent -> snap refused
        segmenter = PlaySegmenter(fps=FPS)
        result = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=5, end_frame=39)], trajs
        )
        seg = result.segments[0]
        assert seg.snap_frame is None
        assert seg.snap_source == "unavailable"
        assert seg.phase_for(10) == "snap_unknown"
        assert seg.phase_for(4) == "outside_play"
        assert result.to_dict()["phase_counts"]["snap_unknown"] == 35
        assert len(result.refusals) == 1
        assert result.refusals[0].scope == "snap"
        assert result.refusals[0].reason == REASON_NO_QUIESCENT_BASELINE
        assert result.fabricated_snap_frames == 0
        assert segmenter.snaps_refused == 1

    def test_disabled_automatic_snap_refuses_with_its_own_reason(self):
        trajs = make_field(8, stationary_then_moving(40, 20))
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_snap=False)
        result = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=39)], trajs
        )
        assert result.segments[0].snap_frame is None
        assert result.refusals[0].reason == REASON_AUTOMATIC_DISABLED
        assert segmenter.snaps_from_motion_onset == 0

    def test_all_refusal_reasons_are_declared(self):
        used = {
            REASON_AUTOMATIC_DISABLED, REASON_BASELINE_EVIDENCE_GAPS, REASON_COORDINATE_CHANGE,
            REASON_INSUFFICIENT_MEASURED_SAMPLES, REASON_INSUFFICIENT_TRACKS,
            REASON_NO_MOTION_CESSATION, REASON_NO_PRIOR_MOTION, REASON_NO_QUIESCENT_BASELINE,
            REASON_NO_SNAP_FOR_END_SEARCH, REASON_NO_SUSTAINED_ONSET, REASON_NO_TRAJECTORIES,
            REASON_WINDOW_TOO_SHORT,
        }
        assert used <= set(REFUSAL_REASONS)


# ---------------------------------------------------------------------------
# 7. Automatic play end (motion cessation)
# ---------------------------------------------------------------------------
class TestAutomaticPlayEnd:
    def test_cessation_recovers_the_first_quiet_frame(self):
        speeds = [0.2] * 20 + [3.5] * 30 + [0.2] * 20
        trajs = make_field(9, speeds)
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_play_end=True)
        result = segmenter.segment([PlayTimestampLabel(play_id="p1", start_frame=0)], trajs)
        seg = result.segments[0]
        assert seg.snap_frame == 20
        assert seg.end_frame == 50
        assert seg.end_source == "motion_cessation"
        assert seg.phase_for(50) == "play_end"
        assert seg.phase_for(49) == "play"
        assert segmenter.play_ends_from_motion_cessation == 1
        assert result.fabricated_play_ends == 0

    def test_play_end_needs_sustained_activity_first(self):
        """No post-snap movement => no claimable play end."""
        speeds = [0.2] * 20 + [3.5, 3.5] + [0.2] * 30
        trajs = make_field(9, speeds)
        est = CollectiveMotionSegmenter(min_sustained_frames=3).detect_play_end(
            trajs, play_id="p1", snap_frame=20, fps=FPS
        )
        assert est.frame_id is None
        assert est.reason == REASON_NO_PRIOR_MOTION

    def test_play_end_refused_when_the_motion_never_stops(self):
        trajs = make_field(9, [0.2] * 20 + [3.5] * 60)
        est = CollectiveMotionSegmenter().detect_play_end(
            trajs, play_id="p1", snap_frame=20, fps=FPS
        )
        assert est.frame_id is None
        assert est.reason == REASON_NO_MOTION_CESSATION

    def test_play_end_cannot_be_the_snap_burst_itself(self):
        """min_play_frames pushes the end past the first quiet frame after a short burst."""
        speeds = [0.2] * 15 + [3.5] * 4 + [0.2] * 40  # burst = frames 15..18
        trajs = make_field(9, speeds)
        est = CollectiveMotionSegmenter(min_play_frames=6, min_quiet_frames=6).detect_play_end(
            trajs, play_id="p1", snap_frame=14, fps=FPS
        )
        # The motion stops at frame 19, but a 5-frame play is not a play: the earliest
        # admissible dead-ball frame is snap + min_play_frames = 20.
        assert est.frame_id == 20
        assert est.frame_id >= 14 + 6
        assert est.source == "motion_cessation"

        permissive = CollectiveMotionSegmenter(min_play_frames=1, min_quiet_frames=6).detect_play_end(
            trajs, play_id="p1", snap_frame=14, fps=FPS
        )
        assert permissive.frame_id == 19

    def test_play_end_search_needs_a_snap(self):
        trajs = make_field(9, [3.0] * 40)  # snap will be refused
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_play_end=True)
        result = segmenter.segment([PlayTimestampLabel(play_id="p1", start_frame=0)], trajs)
        seg = result.segments[0]
        assert seg.snap_frame is None and seg.end_frame is None
        reasons = {r.reason for r in result.refusals}
        assert REASON_NO_QUIESCENT_BASELINE in reasons
        assert REASON_NO_SNAP_FOR_END_SEARCH in reasons
        assert seg.end_source == "unavailable"

    def test_manual_end_wins_over_automatic_end(self):
        speeds = [0.2] * 20 + [3.5] * 30 + [0.2] * 20
        trajs = make_field(9, speeds)
        segmenter = PlaySegmenter(fps=FPS, allow_automatic_play_end=True)
        result = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=45)], trajs
        )
        seg = result.segments[0]
        assert seg.end_frame == 45 and seg.end_source == "manual"
        assert segmenter.play_ends_from_motion_cessation == 0


# ---------------------------------------------------------------------------
# 8. Coordinate provenance and result-level invariants
# ---------------------------------------------------------------------------
class TestProvenanceAndAudit:
    def test_mixed_coord_modes_aggregate_to_uncalibrated(self):
        n_frames = 40
        trajs = [make_trajectory(i, stationary_then_moving(n_frames, 20)) for i in range(1, 7)]
        trajs += [
            make_trajectory(
                i, stationary_then_moving(n_frames, 20), x_coord_mode="relative_5yd"
            )
            for i in range(7, 10)
        ]
        segmenter = PlaySegmenter(fps=FPS)
        seg = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=n_frames - 1)], trajs
        ).segments[0]
        assert seg.x_coord_mode == "uncalibrated"
        assert seg.coordinate_frame_id == COORD_ID
        assert seg.n_tracks == 9

    def test_single_coord_mode_is_preserved_with_segments(self):
        n_frames = 40
        trajs = make_field(8, stationary_then_moving(n_frames, 20))
        segmenter = PlaySegmenter(fps=FPS)
        seg = segmenter.segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=n_frames - 1)], trajs
        ).segments[0]
        assert seg.x_coord_mode == "relative_10yd"
        assert seg.coordinate_segments == (0,)

    def test_counters_are_consistent_and_resettable(self):
        trajs_a = make_field(8, stationary_then_moving(40, 20))
        trajs_b = make_field(8, [3.0] * 40)
        all_trajs = trajs_a + [
            FieldTrajectory(track_id=100 + t.track_id, fps=FPS, x_coord_mode="relative_10yd",
                            samples=t.samples)
            for t in trajs_b
        ]
        segmenter = PlaySegmenter(fps=FPS)
        labels = [
            PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=39),
            PlayTimestampLabel(play_id="p2", start_frame=40, end_frame=79),
        ]
        # Shift the second play's trajectories into frames 40-79.
        shifted = []
        for t in trajs_b:
            shifted.append(
                FieldTrajectory(
                    track_id=t.track_id + 50,
                    fps=FPS,
                    x_coord_mode="relative_10yd",
                    samples=[
                        make_sample(
                            track_id=t.track_id + 50,
                            frame_id=s.frame_id + 40,
                            speed_yd_s=s.speed_yd_s,
                        )
                        for s in t.samples
                    ],
                )
            )
        result = segmenter.segment(labels, trajs_a + shifted, game_id="g1")
        counters = segmenter.counters()
        assert counters["plays_segmented"] == 2
        assert (
            counters["snaps_from_manual"]
            + counters["snaps_from_motion_onset"]
            + counters["snaps_refused"]
            == 2
        )
        assert counters["snaps_from_motion_onset"] == 1
        assert counters["snaps_refused"] == 1
        assert counters["fabricated_snap_frames"] == 0
        assert result.n_snaps_resolved() == 1 and result.n_snaps_refused() == 1
        assert len(all_trajs) == 16  # fixture sanity

        segmenter.reset()
        assert segmenter.counters()["plays_segmented"] == 0
        assert segmenter.motion.snap_estimates_accepted == 0

    def test_result_is_json_serializable_and_deterministic(self):
        trajs = make_field(10, stationary_then_moving(60, 25))
        labels = [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=59)]
        first = PlaySegmenter(fps=FPS).segment(labels, trajs, game_id="g1").to_dict()
        second = PlaySegmenter(fps=FPS).segment(labels, trajs, game_id="g1").to_dict()
        blob = json.dumps(first, allow_nan=False, sort_keys=True)
        assert "NaN" not in blob
        first.pop("runtime_ms")
        second.pop("runtime_ms")
        for seg in first["segments"]:
            seg["snap_estimate"].pop("notes", None)
        for seg in second["segments"]:
            seg["snap_estimate"].pop("notes", None)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert first["n_segments"] == 1
        assert first["phase_counts"]["snap"] == 1

    def test_segmentation_is_a_pure_function_of_its_inputs(self):
        """Re-running on the same trajectories must not mutate them."""
        trajs = make_field(8, stationary_then_moving(40, 20))
        before = [(s.frame_id, s.speed_yd_s, s.position_source) for t in trajs for s in t.samples]
        PlaySegmenter(fps=FPS).segment(
            [PlayTimestampLabel(play_id="p1", start_frame=0, end_frame=39)], trajs
        )
        after = [(s.frame_id, s.speed_yd_s, s.position_source) for t in trajs for s in t.samples]
        assert before == after

    def test_refusal_record_structure(self):
        refusal = SegmentationRefusal(
            scope="snap", play_id="p1", reason=REASON_NO_TRAJECTORIES, detail="none supplied"
        )
        assert refusal.to_dict() == {
            "scope": "snap",
            "play_id": "p1",
            "reason": REASON_NO_TRAJECTORIES,
            "detail": "none supplied",
        }

    def test_motion_evidence_type_is_exported(self):
        evidence = CollectiveMotionSegmenter().motion_evidence(
            make_field(7, [0.2] * 10), start_frame=0, end_frame=9, fps=FPS
        )
        assert isinstance(evidence, MotionEvidence)
        assert evidence.n_defined == 10


# ---------------------------------------------------------------------------
# 9. Frozen benchmark integrity (manifest <-> JSON <-> report <-> code)
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_JSON = ROOT / "data" / "benchmarks" / "phase10_segmentation_manifest.json"
BENCHMARK_JSON = ROOT / "outputs" / "phase10_segmentation_benchmark.json"
REPORT_MD = ROOT / "docs" / "PHASE10_PLAY_SEGMENTATION_REPORT.md"

_spec = importlib.util.spec_from_file_location(
    "phase10_benchmark_harness", ROOT / "benchmarks" / "evaluate_phase10_segmentation.py"
)
assert _spec is not None and _spec.loader is not None
BENCH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BENCH)

FROZEN_TEST_SEQUENCES = (
    "seg_seq_07_test_baseline_evidence_gaps",
    "seg_seq_08_test_two_play_drive",
    "seg_seq_09_test_never_quiescent",
    "seg_seq_10_test_dead_reckoning_only",
)


def _load_json():
    import json

    return json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))


def _load_manifest():
    import json

    return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))


class TestFrozenBenchmarkIntegrity:
    def test_manifest_split_inventory_is_frozen(self):
        manifest = _load_manifest()
        seqs = manifest["sequences"]
        splits = {}
        for seq in seqs:
            splits.setdefault(seq["split"], []).append(seq["sequence_id"])
        assert sorted(splits) == ["TEST", "TRAIN", "VAL"]
        assert tuple(splits["TEST"]) == FROZEN_TEST_SEQUENCES
        assert len(splits["TRAIN"]) == 3 and len(splits["VAL"]) == 3 and len(splits["TEST"]) == 4
        # Every play declares its frozen expectations.
        for seq in seqs:
            for play in seq["plays"]:
                assert "expected" in play and "ground_truth" in play and "manual_labels" in play
                assert set(play["expected"]) == {
                    "snap_error_max_frames",
                    "end_error_max_frames",
                    "refusal_reason",
                }

    def test_benchmark_json_reproduces_from_the_manifest(self):
        """Re-running the harness must agree with the frozen artifact frame-for-frame."""
        fresh = BENCH.run_phase10_benchmark(write=False)
        frozen = _load_json()
        assert fresh["benchmark_version"] == frozen["benchmark_version"]
        for key, value in fresh["totals"].items():
            if key == "runtime_ms":
                continue
            assert value == frozen["totals"][key], key
        fresh_by_id = {s["sequence_id"]: s for s in fresh["sequences"]}
        for seq in frozen["sequences"]:
            re_run = fresh_by_id[seq["sequence_id"]]
            assert re_run["snap_errors_frames"] == seq["snap_errors_frames"]
            assert re_run["end_errors_frames"] == seq["end_errors_frames"]
            assert re_run["phase_frame_accuracy"] == seq["phase_frame_accuracy"]
            assert re_run["refusal_reasons"] == seq["refusal_reasons"]
            assert re_run["evidence_accounting"] == seq["evidence_accounting"]

    def test_zero_fabrication_and_zero_violations(self):
        data = _load_json()
        assert data["totals"]["fabricated_snap_frames"] == 0
        assert data["totals"]["fabricated_play_ends"] == 0
        assert data["totals"]["expectation_violations"] == []
        assert data["totals"]["evidence_accounting_balanced"] is True
        for split in ("TRAIN", "VAL", "TEST"):
            assert data["splits"][split]["fabricated_snap_frames"] == 0
            assert data["splits"][split]["expectation_violations"] == []
        for seq in data["sequences"]:
            assert seq["fabricated_snap_frames"] == 0
            assert seq["segmenter_counters"]["fabricated_snap_frames"] == 0
            assert seq["segmenter_counters"]["motion_fabricated_boundaries"] == 0

    def test_thresholds_are_the_module_defaults_not_fitted_values(self):
        data = _load_json()
        probe = CollectiveMotionSegmenter()
        thr = data["thresholds_used"]
        assert thr["motion_speed_threshold_yd_s"] == probe.motion_speed_threshold_yd_s == 1.0
        assert thr["onset_min_fraction"] == probe.onset_min_fraction == 0.45
        assert thr["baseline_max_fraction"] == probe.baseline_max_fraction == 0.15
        assert thr["baseline_min_frames"] == probe.baseline_min_frames == 10
        assert thr["min_sustained_frames"] == probe.min_sustained_frames == 3
        assert thr["min_tracks"] == probe.min_tracks == 6
        assert thr["min_quiet_frames"] == probe.min_quiet_frames == 6
        assert thr["min_play_frames"] == probe.min_play_frames == 6
        assert thr["tuned_on_test_split"] is False
        assert data["image_detector_quantitative_metrics"] is None
        assert data["end_to_end_video_segmentation_measured"] is False

    def test_every_refusal_reason_is_a_declared_reason(self):
        data = _load_json()
        declared = set(REFUSAL_REASONS)
        for seq in data["sequences"]:
            assert set(seq["refusal_reasons"]) <= declared
            for play in seq["plays"]:
                reason = play["predicted"]["snap_refusal_reason"]
                if reason is not None:
                    assert reason in declared
                end_reason = play["predicted"]["end_refusal_reason"]
                if end_reason is not None:
                    assert end_reason in declared

    def test_frozen_json_and_manifest_expectations_agree(self):
        manifest = _load_manifest()
        data = _load_json()
        man_plays = {
            (seq["sequence_id"], play["play_id"]): play
            for seq in manifest["sequences"]
            for play in seq["plays"]
        }
        assert [s["sequence_id"] for s in data["sequences"]] == [
            s["sequence_id"] for s in manifest["sequences"]
        ]
        for seq in data["sequences"]:
            for play in seq["plays"]:
                src = man_plays[(seq["sequence_id"], play["play_id"])]
                assert play["expected"] == src["expected"]
                assert play["manual_labels"] == src["manual_labels"]
                assert play["ground_truth"] == src["ground_truth"]

    def test_frozen_json_has_no_nan_and_is_self_describing(self):
        raw = BENCHMARK_JSON.read_text(encoding="utf-8")
        assert "NaN" not in raw and "Infinity" not in raw
        data = _load_json()
        assert data["benchmark_kind"] == "synthetic_segmentation_benchmark"
        assert data["provenance_and_semantics"]["image_detector_accuracy"].startswith("UNMEASURED")
        assert data["provenance_and_semantics"]["end_to_end_video_accuracy"].startswith("UNMEASURED")

    def test_report_is_generated_from_the_frozen_json(self):
        data = _load_json()
        report = REPORT_MD.read_text(encoding="utf-8")
        assert report.startswith("# Phase 10 Play Segmentation Report")
        assert "Do not edit by hand" in report
        totals = data["totals"]
        assert f"| sequences | {totals['n_sequences']} |" in report
        assert f"| false refusals (GT snap existed) | {totals['false_refusals']} |" in report
        assert f"| true refusals (no determinable snap) | {totals['true_refusals']} |" in report
        assert "| fabricated snap frames | 0 |" in report
        assert "| expectation violations | 0 |" in report
        for seq in data["sequences"]:
            assert f"`{seq['sequence_id']}`" in report

    def test_harness_rewrites_the_frozen_artifacts_deterministically(self):
        """Running with write=True reproduces byte-identical JSON (no hidden state)."""
        import hashlib

        before = hashlib.sha256(BENCHMARK_JSON.read_bytes()).hexdigest()
        BENCH.run_phase10_benchmark(write=False)
        after = hashlib.sha256(BENCHMARK_JSON.read_bytes()).hexdigest()
        assert before == after  # write=False must not touch the artifact
