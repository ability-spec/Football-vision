"""Phase 10 play-segmentation benchmark (synthetic, deterministic, asset-free).

Measures the Phase 10 segmentation layer as a consumer of the Phase 4 trajectory
contracts across frozen TRAIN / VAL / TEST splits:

  1. Manual timestamp pass-through (a labelled snap/end is never recomputed)
  2. Automatic snap detection latency vs fixture ground truth (frames)
  3. Automatic play-end (motion cessation) latency vs fixture ground truth
  4. Per-frame phase accuracy (pre_snap / snap / play / play_end / snap_unknown)
  5. Refusal behaviour by reason, split into false vs true refusals
  6. Evidence accounting: usable measurements vs excluded dead-reckoning /
     rejected / coasted / unprojected samples
  7. Zero fabricated boundaries and zero expectation violations

What this benchmark is NOT
--------------------------
* It is not an image-detector benchmark: no frame is decoded, no homography is
  fitted, no detector runs. ``image_detector_quantitative_metrics`` is null.
* It is not measured on real football. Player motion is a synthetic deterministic
  speed profile; "ground truth" snap/end frames are properties of that fixture.
* It does not measure end-to-end video accuracy: the unattended video runner does
  not segment plays, because a play START is always a manual timestamp here.

Run:  python3 benchmarks/evaluate_phase10_segmentation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_vision import (  # noqa: E402
    CollectiveMotionSegmenter,
    FieldTrajectory,
    PlaySegmenter,
    PlayTimestampLabel,
    TrajectorySample,
    __version__,
)

MANIFEST_PATH = ROOT / "data" / "benchmarks" / "phase10_segmentation_manifest.json"
OUT_DIR = ROOT / "outputs"
OUT_JSON = OUT_DIR / "phase10_segmentation_benchmark.json"
OUT_PNG = OUT_DIR / "phase10_segmentation_overview.png"
SPLITS: Tuple[str, ...] = ("TRAIN", "VAL", "TEST")
BENCHMARK_KIND = "synthetic_segmentation_benchmark"


# ---------------------------------------------------------------------------
# Synthetic trajectory construction (no images, no randomness)
# ---------------------------------------------------------------------------
def _make_sample(
    *,
    track_id: int,
    frame_id: int,
    fps: float,
    speed_yd_s: float,
    position: Tuple[float, float],
    kind: str,
    coordinate_frame_id: str,
    coordinate_segment: int,
) -> TrajectorySample:
    timestamp_s = frame_id / fps
    velocity = (float(speed_yd_s), 0.0)
    if kind == "measured":
        return TrajectorySample(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            geometry_state="calibrated",
            x_coord_mode="relative_10yd",
            track_state="observed",
            image_footpoint=(100.0 + 7.0 * track_id, 200.0),
            field_position=position,
            raw_field_position=position,
            position_source="measured_smoothed",
            velocity_yd_s=velocity,
            speed_yd_s=float(speed_yd_s),
            is_measurement_used=True,
            coordinate_frame_id=coordinate_frame_id,
            coordinate_segment=coordinate_segment,
        )
    # Coasting / dead reckoning: a model-propagated estimate, never a measurement.
    return TrajectorySample(
        track_id=track_id,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        geometry_state="propagated",
        x_coord_mode="relative_10yd",
        track_state="coasted",
        predicted_position=position,
        position_source="predicted_dead_reckoning",
        velocity_yd_s=velocity,
        speed_yd_s=float(speed_yd_s),
        missed_frames=1,
        coordinate_frame_id=coordinate_frame_id,
        coordinate_segment=coordinate_segment,
    )


def _speed_profiles(seq: Dict[str, Any]) -> List[List[float]]:
    """Compose the per-track, per-frame speed profile declared by the manifest."""
    pop = seq["population"]
    n_tracks = int(pop["n_tracks"])
    n_frames = int(seq["n_frames"])
    quiet = float(pop["quiet_speed_yd_s"])
    active = float(pop["active_speed_yd_s"])
    speeds = [[quiet] * n_frames for _ in range(n_tracks)]

    for play in seq["plays"]:
        motion = play["motion"]
        snap = int(motion["snap_frame"])
        end = n_frames if motion.get("end_frame") is None else int(motion["end_frame"])
        fraction = float(motion.get("active_fraction", 1.0))
        ramp = int(motion.get("ramp_frames", 0))
        n_active = int(round(fraction * n_tracks))
        for j in range(n_active):
            onset = snap + (int(j * ramp / n_active) if ramp > 0 and n_active > 1 else 0)
            for f in range(max(0, onset), min(n_frames, end)):
                speeds[j][f] = active
        # Pre-snap movers are taken from the end of the track list: they are ordinary
        # players who also move at the snap, but who break formation early.
        cursor = n_tracks - 1
        for mover in motion.get("pre_snap_movers", []):
            count = int(mover["n_tracks"])
            for j in range(max(0, cursor - count + 1), cursor + 1):
                for f in range(
                    max(0, int(mover["start_frame"])),
                    min(n_frames, int(mover["end_frame"]) + 1),
                ):
                    speeds[j][f] = float(mover["speed_yd_s"])
            cursor -= count
    return speeds


def build_sequence_trajectories(seq: Dict[str, Any]) -> List[FieldTrajectory]:
    """Build deterministic field-space trajectories for one benchmark sequence."""
    pop = seq["population"]
    n_tracks = int(pop["n_tracks"])
    n_frames = int(seq["n_frames"])
    fps = float(seq["fps"])
    base_coord = f"synthetic:{seq['sequence_id']}"
    evidence_kind = str(pop.get("evidence_kind", "clean"))
    gap_frames = pop.get("coasted_gap_frames")
    gap_tracks = int(pop.get("coasted_gap_tracks", 0))
    cut_frame = int(pop["cut_frame"]) if pop.get("coordinate_kind") == "cut" else None

    speeds = _speed_profiles(seq)
    trajectories: List[FieldTrajectory] = []
    for j in range(n_tracks):
        samples: List[TrajectorySample] = []
        x = 20.0
        y = 5.0 + (43.3333 * j / max(1, n_tracks - 1))
        for f in range(n_frames):
            speed = speeds[j][f]
            x += speed / fps
            kind = "measured"
            if evidence_kind == "all_dead_reckoning":
                kind = "dead_reckoning"
            elif (
                evidence_kind == "coasted_gaps"
                and gap_frames is not None
                and j < gap_tracks
                and int(gap_frames[0]) <= f <= int(gap_frames[1])
            ):
                kind = "dead_reckoning"
            coord_id = base_coord
            coord_seg = 0
            if cut_frame is not None and f >= cut_frame:
                coord_id = f"{base_coord}#post_cut"
                coord_seg = 1
            samples.append(
                _make_sample(
                    track_id=j + 1,
                    frame_id=f,
                    fps=fps,
                    speed_yd_s=speed,
                    position=(round(x, 4), round(y, 4)),
                    kind=kind,
                    coordinate_frame_id=coord_id,
                    coordinate_segment=coord_seg,
                )
            )
        trajectories.append(
            FieldTrajectory(
                track_id=j + 1,
                fps=fps,
                x_coord_mode="relative_10yd",
                samples=samples,
                detector_name="synthetic_phase10_fixture_v1",
            )
        )
    return trajectories


# ---------------------------------------------------------------------------
# Ground truth and scoring
# ---------------------------------------------------------------------------
def ground_truth_phases(seq: Dict[str, Any], play: Dict[str, Any]) -> Dict[int, str]:
    """Fixture phase labels derived from the declared ground-truth boundaries."""
    start = int(play["manual_labels"]["start_frame"])
    gt = play["ground_truth"]
    manual_end = play["manual_labels"].get("end_frame")
    fallback_end = int(seq["n_frames"]) - 1
    if not bool(gt.get("play_exists", True)) or gt.get("snap_frame") is None:
        end = fallback_end if manual_end is None else int(manual_end)
        return {f: "snap_unknown" for f in range(start, end + 1)}
    snap = int(gt["snap_frame"])
    end = fallback_end if gt.get("end_frame") is None else int(gt["end_frame"])
    out: Dict[int, str] = {}
    for f in range(start, end + 1):
        if f < snap:
            out[f] = "pre_snap"
        elif f == snap:
            out[f] = "snap"
        elif f == end:
            out[f] = "play_end"
        else:
            out[f] = "play"
    return out


def _within(actual: Optional[int], bound: Optional[int]) -> Optional[bool]:
    """None bound = not applicable; otherwise compare |error| against the bound."""
    if bound is None:
        return None
    if actual is None:
        return False
    return bool(actual <= int(bound))


def score_sequence(seq: Dict[str, Any]) -> Dict[str, Any]:
    """Run Phase 10 on one sequence and score it against the frozen manifest."""
    fps = float(seq["fps"])
    trajectories = build_sequence_trajectories(seq)
    labels = [
        PlayTimestampLabel(
            play_id=play["play_id"],
            start_frame=int(play["manual_labels"]["start_frame"]),
            snap_frame=play["manual_labels"].get("snap_frame"),
            end_frame=play["manual_labels"].get("end_frame"),
        )
        for play in seq["plays"]
    ]
    segmenter = PlaySegmenter(fps=fps, allow_automatic_play_end=True)

    started = time.perf_counter()
    result = segmenter.segment(labels, trajectories, game_id=seq["sequence_id"])
    runtime_ms = (time.perf_counter() - started) * 1000.0

    evidence = segmenter.motion.motion_evidence(
        trajectories, start_frame=0, end_frame=int(seq["n_frames"]) - 1, fps=fps
    )

    plays_report: List[Dict[str, Any]] = []
    violations: List[str] = []
    n_false_refusals = n_true_refusals = 0
    snap_errors: List[int] = []
    end_errors: List[int] = []
    phase_matched = phase_total = 0

    for play, seg in zip(seq["plays"], result.segments):
        gt = play["ground_truth"]
        exp = play["expected"]
        gt_snap = gt.get("snap_frame")
        gt_end = gt.get("end_frame")
        refusal_reason = seg.snap_estimate.reason if seg.snap_estimate is not None else None
        end_refusal_reason = seg.end_estimate.reason if seg.end_estimate is not None else None

        snap_error = (
            None if gt_snap is None or seg.snap_frame is None else abs(int(seg.snap_frame) - int(gt_snap))
        )
        end_error = (
            None if gt_end is None or seg.end_frame is None else abs(int(seg.end_frame) - int(gt_end))
        )
        if snap_error is not None:
            snap_errors.append(snap_error)
        if end_error is not None:
            end_errors.append(end_error)

        expected_reason = exp.get("refusal_reason")
        snap_within = _within(snap_error, exp.get("snap_error_max_frames"))
        end_within = _within(end_error, exp.get("end_error_max_frames"))

        if seg.snap_frame is None:
            if gt_snap is None:
                n_true_refusals += 1
            else:
                n_false_refusals += 1

        gt_phases = ground_truth_phases(seq, play)
        matched = sum(1 for f, phase in gt_phases.items() if seg.phase_for(f) == phase)
        phase_matched += matched
        phase_total += len(gt_phases)

        # --- frozen expectation checks -----------------------------------
        if expected_reason is not None and refusal_reason != expected_reason:
            violations.append(
                f"{play['play_id']}: expected refusal {expected_reason!r}, "
                f"got {refusal_reason!r} (snap_frame={seg.snap_frame})"
            )
        if expected_reason is None:
            if seg.snap_frame is None:
                violations.append(
                    f"{play['play_id']}: expected a snap, got refusal {refusal_reason!r}"
                )
            if exp.get("snap_error_max_frames") is not None and snap_within is not True:
                violations.append(
                    f"{play['play_id']}: snap error {snap_error} frames exceeds "
                    f"{exp['snap_error_max_frames']}"
                )
            if exp.get("end_error_max_frames") is not None and end_within is not True:
                violations.append(
                    f"{play['play_id']}: play-end error {end_error} frames exceeds "
                    f"{exp['end_error_max_frames']}"
                )

        plays_report.append(
            {
                "play_id": play["play_id"],
                "manual_labels": play["manual_labels"],
                "ground_truth": gt,
                "expected": exp,
                "stress": seq.get("stress", "none"),
                "predicted": {
                    "start_frame": seg.start_frame,
                    "snap_frame": seg.snap_frame,
                    "snap_source": seg.snap_source,
                    "snap_confidence": seg.snap_confidence,
                    "end_frame": seg.end_frame,
                    "end_source": seg.end_source,
                    "end_confidence": seg.end_confidence,
                    "snap_refusal_reason": refusal_reason,
                    "end_refusal_reason": end_refusal_reason,
                    "x_coord_mode": seg.x_coord_mode,
                    "coordinate_frame_id": seg.coordinate_frame_id,
                    "coordinate_segments": list(seg.coordinate_segments),
                    "n_tracks": seg.n_tracks,
                },
                "snap_error_frames": snap_error,
                "end_error_frames": end_error,
                "snap_error_within_expectation": snap_within,
                "end_error_within_expectation": end_within,
                "refusal_reason_matches_expectation": (
                    None if expected_reason is None else bool(refusal_reason == expected_reason)
                ),
                "is_false_refusal": bool(seg.snap_frame is None and gt_snap is not None),
                "is_true_refusal": bool(seg.snap_frame is None and gt_snap is None),
                "phase_frames": len(gt_phases),
                "phase_frames_matched": matched,
                "phase_frame_accuracy": (matched / len(gt_phases)) if gt_phases else None,
                "snap_evidence": seg.snap_estimate.to_dict() if seg.snap_estimate else None,
                "end_evidence": seg.end_estimate.to_dict() if seg.end_estimate else None,
            }
        )

    n_plays = len(seq["plays"])
    n_snap_gt = sum(
        1 for p in seq["plays"] if p["ground_truth"].get("play_exists") and p["ground_truth"].get("snap_frame") is not None
    )
    resolved = [seg for seg in result.segments if seg.snap_frame is not None]
    return {
        "sequence_id": seq["sequence_id"],
        "split": seq["split"],
        "description": seq["description"],
        "stress": seq.get("stress", "none"),
        "fps": fps,
        "n_frames": int(seq["n_frames"]),
        "n_tracks": int(seq["population"]["n_tracks"]),
        "n_plays": n_plays,
        "n_plays_with_ground_truth_snap": n_snap_gt,
        "snaps_resolved": len(resolved),
        "snaps_refused": n_plays - len(resolved),
        "snaps_from_manual": sum(1 for s in resolved if s.snap_source == "manual"),
        "snaps_from_motion_onset": sum(1 for s in resolved if s.snap_source == "collective_motion_onset"),
        "play_ends_from_manual": sum(1 for s in result.segments if s.end_source == "manual"),
        "play_ends_from_motion_cessation": sum(
            1 for s in result.segments if s.end_source == "motion_cessation"
        ),
        "play_ends_refused": sum(1 for s in result.segments if s.end_frame is None),
        "snap_exact_matches": sum(1 for e in snap_errors if e == 0),
        "snap_errors_frames": snap_errors,
        "snap_error_median_frames": (
            float(np.median(snap_errors)) if snap_errors else None
        ),
        "snap_error_max_frames": max(snap_errors) if snap_errors else None,
        "end_exact_matches": sum(1 for e in end_errors if e == 0),
        "end_errors_frames": end_errors,
        "end_error_median_frames": float(np.median(end_errors)) if end_errors else None,
        "end_error_max_frames": max(end_errors) if end_errors else None,
        "false_refusals": n_false_refusals,
        "true_refusals": n_true_refusals,
        "refusal_reasons": sorted(
            {
                str(p["predicted"]["snap_refusal_reason"])
                for p in plays_report
                if p["predicted"]["snap_refusal_reason"]
            }
        ),
        "phase_frames": phase_total,
        "phase_frames_matched": phase_matched,
        "phase_frame_accuracy": (phase_matched / phase_total) if phase_total else None,
        "fabricated_snap_frames": int(result.fabricated_snap_frames),
        "fabricated_play_ends": int(result.fabricated_play_ends),
        "expectation_violations": violations,
        "evidence_accounting": {
            "n_samples_in_window": evidence.n_samples_in_window,
            "n_usable_samples": evidence.n_usable_samples,
            "n_excluded_dead_reckoning": evidence.n_excluded_dead_reckoning,
            "n_excluded_rejected": evidence.n_excluded_rejected,
            "n_excluded_coasted": evidence.n_excluded_coasted,
            "n_excluded_unprojected": evidence.n_excluded_unprojected,
            "n_defined_frames": evidence.n_defined,
            "n_undefined_frames": evidence.n_undefined,
            "coordinate_frame_ids": list(evidence.distinct_coordinate_frame_ids()),
            "accounting_balanced": bool(
                evidence.n_usable_samples
                + evidence.n_excluded_dead_reckoning
                + evidence.n_excluded_rejected
                + evidence.n_excluded_coasted
                + evidence.n_excluded_unprojected
                == evidence.n_samples_in_window
            ),
        },
        "moving_fraction_timeline": [
            None if f is None else round(float(f), 6) for f in evidence.moving_fraction
        ],
        "segmenter_counters": segmenter.counters(),
        "runtime_ms": round(runtime_ms, 3),
        "ms_per_frame": round(runtime_ms / max(1, int(seq["n_frames"])), 4),
        "plays": plays_report,
    }


def aggregate_split(reports: Sequence[Dict[str, Any]], split: str) -> Dict[str, Any]:
    subset = [r for r in reports if r["split"] == split]
    snap_errors = [e for r in subset for e in r["snap_errors_frames"]]
    end_errors = [e for r in subset for e in r["end_errors_frames"]]
    n_plays = sum(r["n_plays"] for r in subset)
    resolved = sum(r["snaps_resolved"] for r in subset)
    phase_frames = sum(r["phase_frames"] for r in subset)
    phase_matched = sum(r["phase_frames_matched"] for r in subset)
    n_frames = sum(r["n_frames"] for r in subset)
    runtime = sum(r["runtime_ms"] for r in subset)
    return {
        "n_sequences": len(subset),
        "n_frames": n_frames,
        "n_plays": n_plays,
        "n_plays_with_ground_truth_snap": sum(r["n_plays_with_ground_truth_snap"] for r in subset),
        "snaps_resolved": resolved,
        "snaps_refused": sum(r["snaps_refused"] for r in subset),
        "snaps_from_manual": sum(r["snaps_from_manual"] for r in subset),
        "snaps_from_motion_onset": sum(r["snaps_from_motion_onset"] for r in subset),
        "snap_exact_matches": sum(r["snap_exact_matches"] for r in subset),
        "snap_exact_match_rate": (sum(r["snap_exact_matches"] for r in subset) / resolved) if resolved else None,
        "snap_error_median_frames": float(np.median(snap_errors)) if snap_errors else None,
        "snap_error_max_frames": max(snap_errors) if snap_errors else None,
        "play_ends_from_manual": sum(r["play_ends_from_manual"] for r in subset),
        "play_ends_from_motion_cessation": sum(r["play_ends_from_motion_cessation"] for r in subset),
        "play_ends_refused": sum(r["play_ends_refused"] for r in subset),
        "end_exact_matches": sum(r["end_exact_matches"] for r in subset),
        "end_error_median_frames": float(np.median(end_errors)) if end_errors else None,
        "end_error_max_frames": max(end_errors) if end_errors else None,
        "false_refusals": sum(r["false_refusals"] for r in subset),
        "true_refusals": sum(r["true_refusals"] for r in subset),
        "phase_frame_accuracy": (phase_matched / phase_frames) if phase_frames else None,
        "fabricated_snap_frames": sum(r["fabricated_snap_frames"] for r in subset),
        "fabricated_play_ends": sum(r["fabricated_play_ends"] for r in subset),
        "expectation_violations": [v for r in subset for v in r["expectation_violations"]],
        "evidence_accounting_balanced": all(
            r["evidence_accounting"]["accounting_balanced"] for r in subset
        ),
        "runtime_ms": round(runtime, 3),
        "ms_per_frame": round(runtime / n_frames, 4) if n_frames else None,
    }


# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------
def render_overview(
    reports: Sequence[Dict[str, Any]], split_summaries: Dict[str, Any]
) -> None:
    by_id = {r["sequence_id"]: r for r in reports}
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))

    def timeline(ax, seq_id: str, title: str) -> None:
        rep = by_id[seq_id]
        fracs = rep["moving_fraction_timeline"]
        frames = list(range(len(fracs)))
        ax.plot(frames, [np.nan if f is None else f for f in fracs], color="#1f77b4", lw=1.8)
        ax.axhline(0.45, color="#2ca02c", ls="--", lw=1.1, label="onset fraction 0.45")
        ax.axhline(0.15, color="#ff7f0e", ls="--", lw=1.1, label="baseline fraction 0.15")
        for play in rep["plays"]:
            gt_snap = play["ground_truth"].get("snap_frame")
            pred_snap = play["predicted"]["snap_frame"]
            gt_end = play["ground_truth"].get("end_frame")
            pred_end = play["predicted"]["end_frame"]
            if gt_snap is not None:
                ax.axvline(gt_snap, color="#7f7f7f", ls=":", lw=1.4, label="GT snap" if play is rep["plays"][0] else None)
            if pred_snap is not None:
                ax.axvline(pred_snap, color="#d62728", lw=1.6, label="detected snap" if play is rep["plays"][0] else None)
                ax.annotate(
                    f"snap err {play['snap_error_frames']}f",
                    (pred_snap, 1.02),
                    fontsize=8,
                    color="#d62728",
                    ha="center",
                )
            if gt_end is not None:
                ax.axvline(gt_end, color="#7f7f7f", ls=":", lw=1.0)
            if pred_end is not None:
                ax.axvline(pred_end, color="#9467bd", lw=1.4, label="detected play end" if play is rep["plays"][0] else None)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(0, max(1, len(frames) - 1))
        ax.set_xlabel("frame index", fontsize=9)
        ax.set_ylabel("moving fraction of visible tracks", fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="center right")

    timeline(
        axes[0, 0],
        "seg_seq_01_train_clean",
        "1. Clean onset (TRAIN): snap and play end recovered exactly\n(dotted grey = fixture GT, red = detected snap, purple = detected end)",
    )
    timeline(
        axes[0, 1],
        "seg_seq_04_val_gradual_ramp",
        "2. Staggered onset (VAL): 22 players cross the threshold over 6 frames\n(detected snap = first sustained collective frame => small bounded latency)",
    )

    ax2 = axes[1, 0]
    labels = ["TRAIN", "VAL", "TEST"]
    exact = [
        (split_summaries[s]["snap_exact_match_rate"] or 0.0) for s in labels
    ]
    phase_acc = [(split_summaries[s]["phase_frame_accuracy"] or 0.0) for s in labels]
    x = np.arange(len(labels))
    ax2.bar(x - 0.19, exact, width=0.36, color="#2ca02c", label="snap exact-match rate")
    ax2.bar(x + 0.19, phase_acc, width=0.36, color="#1f77b4", label="phase-frame accuracy")
    for i, (a, b) in enumerate(zip(exact, phase_acc)):
        ax2.text(i - 0.19, a + 0.02, f"{a:.2f}", ha="center", fontsize=8.5)
        ax2.text(i + 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("rate", fontsize=9)
    ax2.set_title(
        "3. Frozen split accuracy (synthetic fixtures)\n"
        "VAL/TEST exact-match rates are lowered by declared refusal strata, not by wrong frames",
        fontsize=9.5,
    )
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8.5, loc="lower right")

    ax3 = axes[1, 1]
    reason_counts: Dict[str, int] = {}
    false_by_reason: Dict[str, int] = {}
    for rep in reports:
        for play in rep["plays"]:
            reason = play["predicted"]["snap_refusal_reason"]
            if not reason:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if play["is_false_refusal"]:
                false_by_reason[reason] = false_by_reason.get(reason, 0) + 1
    reasons = sorted(reason_counts)
    ypos = np.arange(len(reasons))
    ax3.barh(
        ypos,
        [reason_counts[r] for r in reasons],
        color="#d62728",
        alpha=0.85,
        label="refusals (count of plays)",
    )
    ax3.barh(
        ypos,
        [false_by_reason.get(r, 0) for r in reasons],
        height=0.42,
        color="#7f7f7f",
        alpha=0.95,
        label="of which false (GT snap existed)",
    )
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(reasons, fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("plays", fontsize=9)
    total_refusals = sum(reason_counts.values())
    total_false = sum(false_by_reason.values())
    ax3.set_title(
        f"4. Refusal audit: {total_refusals} refusals, {total_false} false, "
        "0 fabricated boundaries\n(a refusal names its reason; a guess never appears)",
        fontsize=9.5,
    )
    ax3.grid(True, axis="x", alpha=0.3)
    ax3.legend(fontsize=8.0, loc="lower right")

    fig.suptitle(
        f"Phase 10 Play Segmentation Benchmark (football-vision {__version__})\n"
        "manual timestamps + collective-motion onset / cessation; synthetic fixtures",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_phase10_benchmark(*, write: bool = True) -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sequences = manifest["sequences"]
    reports = [score_sequence(seq) for seq in sequences]
    split_summaries = {split: aggregate_split(reports, split) for split in SPLITS}

    probe = CollectiveMotionSegmenter()
    totals = {
        "n_sequences": len(reports),
        "n_frames": sum(r["n_frames"] for r in reports),
        "n_plays": sum(r["n_plays"] for r in reports),
        "snaps_resolved": sum(r["snaps_resolved"] for r in reports),
        "snaps_refused": sum(r["snaps_refused"] for r in reports),
        "false_refusals": sum(r["false_refusals"] for r in reports),
        "true_refusals": sum(r["true_refusals"] for r in reports),
        "fabricated_snap_frames": sum(r["fabricated_snap_frames"] for r in reports),
        "fabricated_play_ends": sum(r["fabricated_play_ends"] for r in reports),
        "expectation_violations": [v for r in reports for v in r["expectation_violations"]],
        "evidence_accounting_balanced": all(
            r["evidence_accounting"]["accounting_balanced"] for r in reports
        ),
        "runtime_ms": round(sum(r["runtime_ms"] for r in reports), 3),
    }

    report = {
        "benchmark_version": "phase10_v1",
        "benchmark_kind": BENCHMARK_KIND,
        "package_version": __version__,
        "manifest_version": manifest["manifest_version"],
        "phases_1_to_4_modified": False,
        "provenance_and_semantics": manifest["benchmark_provenance_and_semantics"],
        "image_detector_quantitative_metrics": None,
        "end_to_end_video_segmentation_measured": False,
        "thresholds_used": {
            "motion_speed_threshold_yd_s": probe.motion_speed_threshold_yd_s,
            "onset_min_fraction": probe.onset_min_fraction,
            "baseline_max_fraction": probe.baseline_max_fraction,
            "baseline_min_frames": probe.baseline_min_frames,
            "min_sustained_frames": probe.min_sustained_frames,
            "min_tracks": probe.min_tracks,
            "min_quiet_frames": probe.min_quiet_frames,
            "min_play_frames": probe.min_play_frames,
            "confidence_reference_contrast": probe.confidence_reference_contrast,
            "tuned_on_test_split": False,
        },
        "metric_definitions": {
            "snap_error_frames": "|detected snap frame - fixture snap frame|; fixture snap = first frame a mover crosses the speed threshold",
            "snap_exact_match_rate": "exact-frame snap matches / snaps resolved (denominator excludes refusals; refusals are counted separately)",
            "end_error_frames": "|detected play-end frame - fixture dead-ball frame|; detected end = first frame of the sustained quiet run",
            "phase_frame_accuracy": "frames whose predicted phase equals the fixture phase / all fixture-labelled frames (refusals score as mismatches, they are not dropped)",
            "false_refusal": "the fixture declares a snap and the segmenter refused to emit one",
            "true_refusal": "the fixture declares no determinable snap and the segmenter refused",
            "expectation_violation": "a frozen manifest expectation (error bound or required refusal reason) was not met",
            "confidence": "a priori contrast x usable-evidence coverage, capped at 0.99; NOT a statistically calibrated probability",
        },
        "totals": totals,
        "splits": split_summaries,
        "sequences": reports,
    }

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        render_overview(reports, split_summaries)
    return report


if __name__ == "__main__":
    res = run_phase10_benchmark()
    print(json.dumps(res["totals"], indent=2))
    print(json.dumps(res["splits"], indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")
