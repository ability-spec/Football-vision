"""Phase 4 Field-Space Trajectory Benchmark.

Evaluates the Phase 4 trajectory layer (persistent track state, multi-frame
field-space trajectories, camera-motion interaction, smoothing, velocity and
acceleration estimation, jump/outlier rejection, and uncertainty propagation)
on top of the unmodified Phase 2 calibration API and Phase 3 detector / tracker /
projector interfaces, across frozen TRAIN / VAL / TEST splits.

IMPORTANT PROVENANCE NOTES (audit remediation pass)
--------------------------------------------------
* Player motion is *synthetic deterministic field-space ground truth* projected
  through the real per-frame homography of each base image. These are engineering
  fixtures, NOT recorded NFL trajectories. This file evaluates a **synthetic
  trajectory benchmark**; it is not evidence of NFL/broadcast tracking accuracy.
* Real multi-frame trajectory accuracy is NOT measured (no real clip with
  frame-level ground-truth trajectories exists in this project). Real frames are
  used only for a single-frame integration smoke test, and are skipped cleanly
  when the third-party assets are absent (see ``FOOTBALL_VISION_NFL_FRAMES``).
* Detection boxes come from ``FixturePlayerDetector`` (``fixture_detector_v1``).
  Image-based player-detector accuracy remains UNMEASURED; every detection number
  in the output JSON is a fixture-harness pass-through statistic.
* Dominant-track selection uses ONLY runtime-observable counters (sample lifetime
  and ``track_id``); ground truth is used solely to attribute a track to the player
  it is scored against. See ``DOMINANT_TRACK_SELECTION`` in the output JSON.
* The reported covariance is NOT statistically calibrated and acceleration is
  EXPERIMENTAL; both are labelled as such in the output JSON.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_vision import (  # noqa: E402
    CalibrationResult,
    FieldProjector,
    FixturePlayerDetector,
    PlayerTracker,
    PlayerTrajectoryBuilder,
    TorsoTeamClassifier,
    TurfContrastPlayerDetector,
    calibrate_frame,
)
from football_vision.data_paths import require_nfl_frame, resolve_nfl_frame  # noqa: E402
from football_vision.trajectory.outlier import REASON_IMAGE_GATE  # noqa: E402
from football_vision.trajectory.uncertainty import (  # noqa: E402
    footpoint_pixel_covariance,
    mahalanobis_distance_sq,
)

MANIFEST_PATH = ROOT / "data" / "benchmarks" / "phase4_trajectory_manifest.json"


def _resolve_base_image(path_str: str) -> Path:
    """Resolve a manifest base image, honouring FOOTBALL_VISION_NFL_FRAMES.

    The real NFL stills are third-party assets that are deliberately not vendored;
    the manifest records the development-workstation path, and this resolver falls
    back to the environment-variable directory by file name.
    """
    direct = Path(path_str)
    if direct.is_file():
        return direct
    return require_nfl_frame(direct.name)

PLAYER_HEIGHT_YD = 2.0          # fixture player height used to size bounding boxes
TURN_TAU_S = 0.12               # heading-change blend time constant
SPEED_TAU_S = 0.10              # speed-change blend time constant
DENSE_DT_S = 1.0 / 240.0        # dense integration step for ground-truth routes
PROPAGATED_DRIFT_PX_PER_AGE = 2.5   # synthetic optical-flow propagation drift
COVERAGE_68_CHI2_2DOF = 2.278
COVERAGE_95_CHI2_2DOF = 5.991
TEAM_BY_GT = {1: "TEAM_A", 2: "TEAM_A", 3: "TEAM_A", 4: "TEAM_B", 5: "TEAM_B", 6: "TEAM_B"}


# ---------------------------------------------------------------------------
# Ground-truth route generation
# ---------------------------------------------------------------------------
def _route_arrays(player: Dict[str, Any], duration_s: float) -> Dict[str, np.ndarray]:
    """Integrate a player's field-space route on a dense grid (deterministic)."""
    n = int(np.ceil(duration_s / DENSE_DT_S)) + 2
    t = np.arange(n, dtype=np.float64) * DENSE_DT_S

    heading = np.full(n, np.deg2rad(float(player["heading_deg"])), dtype=np.float64)
    for t_k, d_deg in player.get("turns", []):
        heading = heading + np.deg2rad(float(d_deg)) * 0.5 * (
            1.0 + np.tanh((t - float(t_k)) / TURN_TAU_S)
        )

    speed = np.full(n, float(player["speed_yd_s"]), dtype=np.float64)
    for t_k, factor in player.get("speed_steps", []):
        speed = speed * (
            1.0 + (float(factor) - 1.0) * 0.5 * (1.0 + np.tanh((t - float(t_k)) / SPEED_TAU_S))
        )

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x = float(player["x0_yd"]) + np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * DENSE_DT_S)])
    y = float(player["y0_yd"]) + np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * DENSE_DT_S)])

    ax = np.gradient(vx, DENSE_DT_S)
    ay = np.gradient(vy, DENSE_DT_S)
    return {"t": t, "x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}


def _gt_at(route: Dict[str, np.ndarray], t_s: float) -> Dict[str, float]:
    return {
        "x_yd": float(np.interp(t_s, route["t"], route["x"])),
        "y_yd": float(np.interp(t_s, route["t"], route["y"])),
        "vx_yd_s": float(np.interp(t_s, route["t"], route["vx"])),
        "vy_yd_s": float(np.interp(t_s, route["t"], route["vy"])),
        "ax_yd_s2": float(np.interp(t_s, route["t"], route["ax"])),
        "ay_yd_s2": float(np.interp(t_s, route["t"], route["ay"])),
    }


# ---------------------------------------------------------------------------
# Fixture rendering (mirrors the Phase 3 benchmark's fixture painter so the
# torso appearance classifier sees plausible jerseys; never used for metrics)
# ---------------------------------------------------------------------------
def _paint_player(canvas: np.ndarray, bbox: Sequence[float], role: str) -> None:
    h_img, w_img = canvas.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(w_img - 2, x1))
    x2 = max(x1 + 2, min(w_img, x2))
    y1 = max(0, min(h_img - 2, y1))
    y2 = max(y1 + 2, min(h_img, y2))
    bw, bh = x2 - x1, y2 - y1

    tx1, tx2 = max(0, int(x1 + 0.15 * bw)), min(w_img, int(x2 - 0.15 * bw))
    ty1, ty2 = max(0, int(y1 + 0.12 * bh)), min(h_img, int(y1 + 0.58 * bh))
    if tx2 <= tx1 or ty2 <= ty1:
        return
    if role == "TEAM_A":
        canvas[ty1:ty2, tx1:tx2] = (65, 28, 18)
    elif role == "TEAM_B":
        canvas[ty1:ty2, tx1:tx2] = (238, 240, 238)


def _det_noise(gid: int, frame: int, channel: int) -> float:
    """Deterministic pseudo-random uniform noise in [-1, 1] (no RNG state)."""
    raw = np.sin(gid * 12.9898 + frame * 78.233 + channel * 37.719) * 43758.5453
    frac = raw - np.floor(raw)
    return float(2.0 * frac - 1.0)


# ---------------------------------------------------------------------------
# Per-frame calibration construction
# ---------------------------------------------------------------------------
def _camera_matrix(pan_dx: float, pan_dy: float, zoom: float, image_size: Tuple[int, int]) -> np.ndarray:
    """Image-space camera operator: scale about the image centre, then translate."""
    cx, cy = 0.5 * float(image_size[0]), 0.5 * float(image_size[1])
    S = np.array([[zoom, 0.0, cx * (1.0 - zoom)], [0.0, zoom, cy * (1.0 - zoom)], [0.0, 0.0, 1.0]])
    T = np.array([[1.0, 0.0, pan_dx], [0.0, 1.0, pan_dy], [0.0, 0.0, 1.0]])
    return T @ S


def _calibration_from_H(
    base_cal: CalibrationResult,
    H: np.ndarray,
    image_size: Tuple[int, int],
    *,
    is_propagated: bool = False,
    propagation_age: int = 0,
) -> CalibrationResult:
    return CalibrationResult(
        success=True,
        H=H,
        H_inv=np.linalg.inv(H),
        image_size=image_size,
        confidence=float(base_cal.confidence),
        confidence_components=dict(base_cal.confidence_components),
        x_coord_mode=base_cal.x_coord_mode,
        plausible_orientation_scale=True,
        detected_ten_yard_parity=base_cal.detected_ten_yard_parity,
        is_temporally_propagated=bool(is_propagated),
        propagation_age=int(propagation_age),
        notes=["phase4_fixture_geometry"],
    )


def _uncalibrated_result(image_size: Tuple[int, int], reason: str) -> CalibrationResult:
    return CalibrationResult(
        success=False,
        H=None,
        H_inv=None,
        image_size=image_size,
        confidence=0.0,
        x_coord_mode="uncalibrated",
        failure_reason=reason,
        notes=["phase4_fixture_geometry"],
    )


def _frame_geometry(
    seq: Dict[str, Any], base_cal: CalibrationResult, image_size: Tuple[int, int], frame: int
) -> Tuple[CalibrationResult, CalibrationResult]:
    """Return ``(projection_calibration, observation_calibration)`` for a frame.

    The observation calibration is the *true* geometry used to generate synthetic
    footpoint observations. The projection calibration is what the trajectory layer
    receives; for propagated frames it carries a deliberate Phase-2-style drift,
    and for unknown frames it is an explicit failure object.

    A ``camera_cut`` entry models a hard shot change: at (and after) the cut frame
    the camera framing jumps by a fixed pan/zoom transform, so image-space motion is
    discontinuous and calibration must be re-established. During the declared
    ``unknown`` cut frames the projection calibration is a failure object with
    reason ``camera_cut_uncalibrated``.
    """
    pan = seq.get("camera_pan_px_per_frame", [0.0, 0.0])
    zoom = float(seq.get("camera_zoom_per_frame", 0.0))
    cam = _camera_matrix(pan[0] * frame, pan[1] * frame, 1.0 + zoom * frame, image_size)

    cut = seq.get("camera_cut")
    if cut is not None and frame >= int(cut["frame"]):
        cut_params = cut.get("params", {})
        cut_pan = cut_params.get("pan_px", [0.0, 0.0])
        cut_zoom = float(cut_params.get("zoom", 1.0))
        cam = _camera_matrix(cut_pan[0], cut_pan[1], cut_zoom, image_size) @ cam

    H_true = base_cal.H @ cam
    observation_cal = _calibration_from_H(base_cal, H_true, image_size)

    state = str(seq.get("calibration_states", {}).get(str(frame), "calibrated"))
    if state.startswith("unknown"):
        reason = state.split(":", 1)[1] if ":" in state else "confidence_expired"
        return _uncalibrated_result(image_size, reason), observation_cal
    if state.startswith("propagated"):
        age = int(state.split(":")[1]) if ":" in state else 1
        drift = _camera_matrix(PROPAGATED_DRIFT_PX_PER_AGE * age, 0.0, 1.0, image_size)
        H_prop = H_true @ drift
        return (
            _calibration_from_H(base_cal, H_prop, image_size, is_propagated=True, propagation_age=age),
            observation_cal,
        )
    return observation_cal, observation_cal


def _build_sequence(
    seq: Dict[str, Any],
    base_img: np.ndarray,
    base_cal: CalibrationResult,
) -> Tuple[List[np.ndarray], List[CalibrationResult], Dict[int, List[Dict[str, Any]]], Dict[int, Dict[int, Dict[str, float]]], Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Render frames, per-frame projection calibrations, fixture boxes, GT tables.

    Also returns *jitter statistics*: the jitter actually injected into the fixture
    footpoints (harness truth) next to the pixel sigma the frozen Phase 4 uncertainty
    model assumes for the same boxes. Those two numbers are what the evaluator uses to
    explain — not to re-tune — the observed uncertainty coverage.
    """
    h, w = base_img.shape[:2]
    image_size = (w, h)
    n_frames = int(seq["num_frames"])
    fps = float(seq["fps"])
    jitter_px = float(seq.get("jitter_px", 0.8))
    duration_s = (n_frames - 1) / fps

    routes = {int(p["gt_id"]): _route_arrays(p, duration_s) for p in seq["players"]}
    outlier_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for ev in seq.get("outlier_events", []):
        outlier_map[(int(ev["gt_id"]), int(ev["frame"]))] = ev
    dropout_frames: Dict[int, set] = {}
    for ev in seq.get("dropout_events", []):
        dropout_frames.setdefault(int(ev["gt_id"]), set()).update(int(f) for f in ev["frames"])

    frames: List[np.ndarray] = []
    frame_cals: List[CalibrationResult] = []
    fixtures: Dict[int, List[Dict[str, Any]]] = {}
    gt_table: Dict[int, Dict[int, Dict[str, float]]] = {}
    jitter_u: List[float] = []
    jitter_v: List[float] = []
    model_sigma_u: List[float] = []
    model_sigma_v: List[float] = []

    for t in range(n_frames):
        proj_cal, obs_cal = _frame_geometry(seq, base_cal, image_size, t)
        frame_cals.append(proj_cal)
        canvas = base_img.copy()
        entries: List[Dict[str, Any]] = []
        gt_row: Dict[int, Dict[str, float]] = {}
        t_s = t / fps

        for gid, route in routes.items():
            gt = _gt_at(route, t_s)
            gt_row[gid] = gt
            if t in dropout_frames.get(gid, set()):
                continue

            uv = obs_cal.field_to_image([[gt["x_yd"], gt["y_yd"]]])
            if uv is None:
                raise RuntimeError(f"observation calibration cannot project GT for gt_id={gid} at t={t}")
            u, v = float(uv[0][0]), float(uv[0][1])

            uv_scale = obs_cal.field_to_image([[gt["x_yd"], gt["y_yd"] + 1.0]])
            px_per_yd = float(np.hypot(uv_scale[0][0] - u, uv_scale[0][1] - v))
            box_h = float(np.clip(PLAYER_HEIGHT_YD * px_per_yd, 26.0, 110.0))
            box_w = 0.42 * box_h

            du = jitter_px * _det_noise(gid, t, 1)
            dv = jitter_px * _det_noise(gid, t, 2)
            ev = outlier_map.get((gid, t))
            if ev is not None:
                du += float(ev["jump_px"]) * float(ev["direction"][0])
                dv += float(ev["jump_px"]) * float(ev["direction"][1])

            foot_u, foot_v = u + du, v + dv
            if not (
                6.0 <= foot_u <= w - 6.0 and 8.0 + box_h <= foot_v <= h - 3.0
            ):
                raise RuntimeError(
                    f"fixture footpoint out of frame in {seq['sequence_id']}: gt_id={gid} t={t} "
                    f"({foot_u:.1f},{foot_v:.1f}) box_h={box_h:.1f} image=({w},{h})"
                )

            bbox = (foot_u - 0.5 * box_w, foot_v - box_h, foot_u + 0.5 * box_w, foot_v)
            _paint_player(canvas, bbox, TEAM_BY_GT.get(gid, "TEAM_A"))
            confidence = 0.91
            entries.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "metadata": {"gt_id": gid, "scenario": seq["scenario"]},
                }
            )
            # Harness truth vs model assumption (evaluated through the real Phase 4
            # pixel-noise model so the two can never drift apart silently).
            # Record the *jitter* only: the injected outlier jump must not inflate the
            # harness-truth jitter statistics that explain the uncertainty coverage.
            jitter_u.append(float(du - (float(ev["jump_px"]) * float(ev["direction"][0]) if ev is not None else 0.0)))
            jitter_v.append(float(dv - (float(ev["jump_px"]) * float(ev["direction"][1]) if ev is not None else 0.0)))
            cov_px = footpoint_pixel_covariance(
                bbox, detection_confidence=float(confidence)
            )
            model_sigma_u.append(float(np.sqrt(cov_px[0, 0])))
            model_sigma_v.append(float(np.sqrt(cov_px[1, 1])))

        frames.append(canvas)
        fixtures[t] = entries
        gt_table[t] = gt_row

    jitter_stats = {
        "observed_jitter_std_u_px": float(np.std(jitter_u)) if jitter_u else 0.0,
        "observed_jitter_std_v_px": float(np.std(jitter_v)) if jitter_v else 0.0,
        "assumed_footpoint_sigma_u_px": float(np.mean(model_sigma_u)) if model_sigma_u else 0.0,
        "assumed_footpoint_sigma_v_px": float(np.mean(model_sigma_v)) if model_sigma_v else 0.0,
        "n_fixture_footpoints": int(len(jitter_u)),
        "note": (
            "observed_* = jitter actually injected into the fixture footpoints by this harness; "
            "assumed_* = pixel sigma the frozen Phase 4 uncertainty model uses for the same boxes."
        ),
    }
    return frames, frame_cals, fixtures, gt_table, routes, jitter_stats


# ---------------------------------------------------------------------------
# Image-space EMA comparison arm (baseline that smooths before projecting)
# ---------------------------------------------------------------------------
def _image_space_ema_field_error(
    seq: Dict[str, Any],
    fixtures: Dict[int, List[Dict[str, Any]]],
    frame_cals: List[CalibrationResult],
    gt_table: Dict[int, Dict[int, Dict[str, float]]],
    *,
    alpha: float = 0.5,
) -> Optional[float]:
    """Median field-position error (yd) of an EMA smoother applied in image space.

    This is a comparison arm only: it shows what happens when smoothing happens
    before projection, which is what camera motion makes dangerous.
    """
    ema: Dict[int, Tuple[float, float]] = {}
    errors: List[float] = []
    for t, entries in fixtures.items():
        cal = frame_cals[t]
        if not cal.can_project():
            continue
        for entry in entries:
            gid = int(entry["metadata"]["gt_id"])
            x1, y1, x2, y2 = entry["bbox"]
            uv = (0.5 * (x1 + x2), y2)
            prev = ema.get(gid)
            sm = uv if prev is None else (alpha * uv[0] + (1 - alpha) * prev[0], alpha * uv[1] + (1 - alpha) * prev[1])
            ema[gid] = sm
            xy = cal.image_to_field([list(sm)])
            if xy is None:
                continue
            gt = gt_table[t][gid]
            errors.append(float(np.hypot(xy[0][0] - gt["x_yd"], xy[0][1] - gt["y_yd"])))
    return float(np.median(errors)) if errors else None


# ---------------------------------------------------------------------------
# Sequence evaluation
# ---------------------------------------------------------------------------
def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


# ---------------------------------------------------------------------------
# Dominant-track selection: rule, documentation, and the non-oracle guarantee
# ---------------------------------------------------------------------------
DOMINANT_TRACK_RULE = "lifetime_sample_count_then_lowest_track_id"
ALTERNATIVE_DOMINANT_TRACK_RULES = (
    "positioned_sample_count_then_lowest_track_id",
    "longest_observed_run_then_lowest_track_id",
)
DOMINANT_TRACK_SELECTION = {
    "rule": DOMINANT_TRACK_RULE,
    "inputs": [
        "track_id",
        "sample lifetime (number of trajectory samples emitted for the track)",
    ],
    "tie_break": "lowest track_id (deterministic; independent of dictionary/iteration order)",
    "uses_ground_truth_trajectory": False,
    "uses_ground_truth_error": False,
    "uses_oracle_track_selection": False,
    "uses_future_information": False,
    "gt_dependency": (
        "Ground truth is used ONLY to attribute a track to the player it is scored against "
        "(an oracle association inherent to every labelled MOT benchmark: an error cannot be "
        "computed without knowing which player a track is compared to). That association is "
        "computed before, and never influences, the selection itself."
    ),
    "note": (
        "Selection is reproducible from runtime-observable counters alone. Fragmentation is "
        "additionally reported for every non-dominant track (see track_table / "
        "fragmentation_taxonomy), so a poor dominant track cannot hide fragment tracks."
    ),
}
FRAGMENTATION_CATEGORIES = (
    "legitimate_new_track",
    "active_association_swap",
    "post_expiration_reinitialization",
    "outlier_induced_spawn",
    "unmatched_spurious_track",
)
FRAGMENTATION_DEFINITIONS = {
    "legitimate_new_track": (
        "First track attributed to that ground-truth player in the sequence, and not created at "
        "an injected outlier event."
    ),
    "active_association_swap": (
        "A track for the player appears in a frame where the player's previous track is still "
        "present (both tracks exist in that frame at least momentarily: an active-association "
        "swap, the identity failure Phase 3 reports as 0)."
    ),
    "post_expiration_reinitialization": (
        "A track for the player appears after the previous track for that player had no sample "
        "in the re-acquisition frame (expired / missed / dropped out): a post-expiration "
        "re-initialization rather than a swap."
    ),
    "outlier_induced_spawn": (
        "The track's first sample coincides with an injected outlier event for that player "
        "(the corrupted measurement was strong enough to spawn a separate track)."
    ),
    "unmatched_spurious_track": (
        "A track that the harness cannot attribute to any ground-truth player. Expected to be 0 "
        "in this harness because every fixture detection carries a label; reported so that a "
        "non-zero value is visible rather than silently dropped."
    ),
}
REJECTION_ACCOUNTING_DEFINITION = {
    "clean_sample": (
        "A dominant-track sample whose (player, frame) pair is NOT an injected outlier event, "
        "and for which a raw field projection exists (so the innovation gate could act on it)."
    ),
    "clean_sample_rejected": (
        "A clean sample that the innovation gate rejected: the false-rejection event."
    ),
    "corrupted_sample": (
        "A dominant-track sample at an injected outlier event (the manifest injects a known "
        "pixel jump); 'accepted' means the corrupted measurement was folded into the track."
    ),
    "innovation_gate": (
        "Field-space chi-square test on the Mahalanobis distance between the measurement and the "
        "constant-velocity prediction. This is the only gate that can REJECT a measurement."
    ),
    "plausibility_gate": (
        "Image-space footpoint jump limit. It can only FLAG a sample (and only when geometry is "
        "unknown), never reject a measurement, so its count is reported separately and is not "
        "part of the false-rejection rate."
    ),
    "geometry_refusal": (
        "A sample for which no projectable field measurement existed at all (unknown geometry, "
        "unreliable footpoint, out-of-bounds projection, or coasting past the max-gap limit). It "
        "is accounted for in unpositioned_by_reason and is deliberately NOT counted as a "
        "measurement rejection: no measurement existed for any gate to accept or reject."
    ),
    "note": (
        "'Clean'/'corrupted' are harness labels from the manifest, not labels inferred from "
        "ground-truth error: this block is a harness metric, not a model-internal quantity."
    ),
}
ACCELERATION_LABEL = "experimental_not_validated"
ACCELERATION_NOTE = (
    "Acceleration is the finite difference of the smoothed filter velocity, bounded by a "
    "plausibility clip. It is EXPERIMENTAL AND NOT VALIDATED: no measured acceleration accuracy "
    "is claimed, and the reported error is dominated by the filter's own noise floor rather than "
    "by a validated acceleration model."
)
UNCERTAINTY_STATUS = "not_statistically_calibrated"
UNCERTAINTY_NOTE = (
    "The reported covariance is an a priori engineering model (footpoint pixel noise propagated "
    "through the homography Jacobian plus a propagation-drift allowance). It is NOT statistically "
    "calibrated against data: empirical coverage is reported per split, per geometry state, and "
    "before/after the smoothing stage, and the gap is explained rather than tuned away. Do not "
    "read the reported sigma as a calibrated confidence interval."
)
COVERAGE_68_MEDIAN_CHI2 = 1.386   # median of chi-square with 2 dof (theoretical reference)


def select_dominant_track(
    candidate_stats: Sequence[Dict[str, Any]],
    *,
    rule: str = DOMINANT_TRACK_RULE,
) -> Optional[int]:
    """Return the dominant ``track_id`` from per-track runtime-observable counters.

    Only ``track_id`` and counters computed from the emitted samples are consulted.
    No ground-truth coordinate, no ground-truth error, and no oracle information is
    available to this function, so a track can never be selected because it "scores
    better". Ties are broken by lowest ``track_id`` (deterministic, order-independent).
    """
    if not candidate_stats:
        return None
    if rule == "lifetime_sample_count_then_lowest_track_id":
        def key(c: Dict[str, Any]) -> Tuple[int, int]:
            return (int(c["n_samples"]), -int(c["track_id"]))
    elif rule == "positioned_sample_count_then_lowest_track_id":
        def key(c: Dict[str, Any]) -> Tuple[int, int]:
            return (int(c["n_positioned"]), -int(c["track_id"]))
    elif rule == "longest_observed_run_then_lowest_track_id":
        def key(c: Dict[str, Any]) -> Tuple[int, int]:
            return (int(c["longest_observed_run"]), -int(c["track_id"]))
    else:  # pragma: no cover - guarded by the tuple of supported rules
        raise ValueError(f"unknown dominant-track rule: {rule!r}")
    return int(max(candidate_stats, key=key)["track_id"])


def _track_runtime_stats(samples_all: Sequence[Any]) -> Dict[int, Dict[str, Any]]:
    """Per-track counters computed only from emitted samples (no ground truth)."""
    stats: Dict[int, Dict[str, Any]] = {}
    frames: Dict[int, List[int]] = {}
    for s in samples_all:
        tid = int(s.track_id)
        st = stats.get(tid)
        if st is None:
            st = {
                "track_id": tid,
                "n_samples": 0,
                "n_observed": 0,
                "n_positioned": 0,
                "n_dead_reckoned": 0,
                "n_measurements_used": 0,
                "n_outlier_rejected": 0,
                "first_frame": int(s.frame_id),
                "last_frame": int(s.frame_id),
                "longest_observed_run": 0,
                "geometry_states": {"calibrated": 0, "propagated": 0, "unknown": 0},
            }
            stats[tid] = st
            frames[tid] = []
        st["n_samples"] += 1
        if s.track_state == "observed":
            st["n_observed"] += 1
        if s.field_position is not None:
            st["n_positioned"] += 1
        if s.position_source == "predicted_dead_reckoning":
            st["n_dead_reckoned"] += 1
        if s.is_measurement_used:
            st["n_measurements_used"] += 1
        if s.is_outlier_rejected:
            st["n_outlier_rejected"] += 1
        st["first_frame"] = min(int(st["first_frame"]), int(s.frame_id))
        st["last_frame"] = max(int(st["last_frame"]), int(s.frame_id))
        st["geometry_states"][s.geometry_state] = st["geometry_states"].get(s.geometry_state, 0) + 1
        frames[tid].append(int(s.frame_id))

    for tid, st in stats.items():
        run = 0
        best = 0
        prev = None
        by_frame = {}
        for s in samples_all:
            if int(s.track_id) != tid:
                continue
            by_frame[int(s.frame_id)] = s.track_state == "observed"
        for frame in sorted(by_frame):
            if prev is not None and frame - prev > 1:
                run = 0
            if by_frame[frame]:
                run += 1
                best = max(best, run)
            else:
                run = 0
            prev = frame
        st["longest_observed_run"] = best
    return stats


def _sample_gt_id(sample: Any) -> Optional[int]:
    """Return the ground-truth player label attached to a sample by the harness.

    Attribution is **per sample**, matching the CLEAR-MOTA convention used by the Phase 3
    evaluator: a sample is scored against the player its detection claims at that frame.
    A track that is re-associated to a different player therefore loses the identity
    metric (IDSW) instead of silently re-defining "which player this track is". Track-level
    attribution would score such samples against the majority label and manufacture
    multi-yard "errors"; the contamination is reported explicitly instead
    (``identity_contamination``).
    """
    gid = sample.provenance.get("gt_id")
    return None if gid is None else int(gid)


def _track_label_map(samples_all: Sequence[Any]) -> Dict[int, Dict[int, int]]:
    """``track_id -> {gt_id: sample count}`` for contamination accounting."""
    labels: Dict[int, Dict[int, int]] = {}
    for s in samples_all:
        gid = _sample_gt_id(s)
        if gid is None:
            continue
        per = labels.setdefault(int(s.track_id), {})
        per[gid] = per.get(gid, 0) + 1
    return labels


def _fragmentation_taxonomy(
    samples_all: Sequence[Any],
    dominant_track: Dict[int, int],
    injected: set,
) -> Dict[str, Any]:
    """Classify every (player, track) episode into exactly one deterministic category.

    Categories are mutually exclusive and exhaustive. Accounting is asserted against the
    number of episodes, so a new failure mode cannot be silently dropped. A track that is
    re-associated to a second player (identity contamination) appears as two episodes and
    is additionally reported in ``tracks_with_multiple_gt_labels``.
    """
    stats = _track_runtime_stats(samples_all)
    labels = _track_label_map(samples_all)
    frames_at: Dict[int, set] = {}
    for s in samples_all:
        frames_at.setdefault(int(s.frame_id), set()).add(int(s.track_id))

    episodes: List[Dict[str, Any]] = []
    for tid in sorted(labels):
        for gid in sorted(labels[tid]):
            group = [
                s
                for s in samples_all
                if int(s.track_id) == tid and _sample_gt_id(s) == gid
            ]
            first_frame = min(int(s.frame_id) for s in group)
            episodes.append(
                {
                    "track_id": tid,
                    "gt_id": gid,
                    "first_frame": first_frame,
                    "last_frame": max(int(s.frame_id) for s in group),
                    "n_samples": len(group),
                    "n_observed": sum(1 for s in group if s.track_state == "observed"),
                    "n_positioned": sum(1 for s in group if s.field_position is not None),
                    "n_dead_reckoned": sum(
                        1 for s in group if s.position_source == "predicted_dead_reckoning"
                    ),
                }
            )

    unmatched_tracks = sorted(
        tid for tid, st in stats.items() if int(st["n_samples"]) > 0 and tid not in labels
    )
    episodes_by_gid: Dict[int, List[Dict[str, Any]]] = {}
    for ep in episodes:
        episodes_by_gid.setdefault(ep["gt_id"], []).append(ep)

    taxonomy = {category: 0 for category in FRAGMENTATION_CATEGORIES}
    details: List[Dict[str, Any]] = []
    for ep in sorted(episodes, key=lambda e: (e["first_frame"], e["track_id"])):
        gid = ep["gt_id"]
        first_frame = ep["first_frame"]
        if (gid, first_frame) in injected:
            category = "outlier_induced_spawn"
        else:
            prior = [
                other
                for other in episodes_by_gid[gid]
                if other is not ep and other["first_frame"] < first_frame
            ]
            if not prior:
                category = "legitimate_new_track"
            else:
                previous = max(prior, key=lambda e: (e["first_frame"], e["track_id"]))
                if previous["track_id"] in frames_at.get(first_frame, set()):
                    category = "active_association_swap"
                else:
                    category = "post_expiration_reinitialization"
        taxonomy[category] += 1
        details.append(
            {
                **ep,
                "origin_category": category,
                "excluded_from_dominant_metrics": dominant_track.get(gid) != ep["track_id"],
                "track_carries_multiple_labels": len(labels.get(ep["track_id"], {})) > 1,
            }
        )

    excluded = [d for d in details if d["excluded_from_dominant_metrics"]]
    excluded_by_origin = {category: 0 for category in FRAGMENTATION_CATEGORIES}
    for detail in excluded:
        excluded_by_origin[detail["origin_category"]] += 1
    return {
        "taxonomy": taxonomy,
        "fragment_excluded_from_dominant_total": int(len(excluded)),
        "fragment_excluded_from_dominant_by_origin": excluded_by_origin,
        "taxonomy_total": int(sum(taxonomy.values())),
        "n_episodes": int(len(episodes)),
        "n_tracks": int(len(stats)),
        "n_matched_tracks": int(len(labels)),
        "fragment_tracks_excluded_from_dominant": int(len(excluded)),
        "tracks_with_multiple_gt_labels": int(
            sum(1 for tid in labels if len(labels[tid]) > 1)
        ),
        "tracks_without_any_label": int(len(unmatched_tracks)),
        "tracks_per_gt_player": {
            str(gid): len({e["track_id"] for e in eps})
            for gid, eps in sorted(episodes_by_gid.items())
        },
        "max_tracks_per_gt_player": int(
            max((len({e["track_id"] for e in eps}) for eps in episodes_by_gid.values()), default=0)
        ),
        "samples_per_track": {str(tid): int(stats[tid]["n_samples"]) for tid in sorted(stats)},
        "track_details": details,
    }


def _rejection_accounting(
    samples_all: Sequence[Any],
    gt_table: Dict[int, Dict[int, Dict[str, float]]],
    dominant_track: Dict[int, int],
    injected: set,
    *,
    max_gap_frames: int,
) -> Dict[str, Any]:
    """Separate "the gate rejected bad data" from "the gate rejected good data"."""
    clean_gate_evaluated = 0
    clean_accepted = 0
    clean_accepted_warmup = 0
    clean_rejected = 0
    clean_in_warmup = 0
    corrupted_evaluated = 0
    corrupted_rejected = 0
    corrupted_accepted = 0
    dr_after_false_rejection = 0
    all_rejections_innovation = 0
    all_rejections_plausibility = 0
    image_flags = 0
    domain_rejections_innovation = 0
    domain_rejections_plausibility = 0
    false_rejection_gt_err: List[float] = []
    false_rejection_gate_stat: List[float] = []
    corrupted_gate_stat: List[float] = []
    unpositioned = {
        "gap_exceeded": 0,
        "geometry_refused": 0,
        "footpoint_refused": 0,
        "out_of_bounds": 0,
        "other": 0,
    }

    for s in samples_all:
        gid = _sample_gt_id(s)
        is_dominant = gid is not None and dominant_track.get(int(gid)) == int(s.track_id)
        is_injected = gid is not None and (int(gid), int(s.frame_id)) in injected

        if s.image_space_jump_flagged:
            image_flags += 1
        rejection_is_plausibility = s.rejection_reason == REASON_IMAGE_GATE
        if s.is_outlier_rejected:
            if rejection_is_plausibility:
                all_rejections_plausibility += 1
            else:
                # The field/innovation gate is the only mechanism that can currently reject a
                # measurement; an unlabelled rejection reason is charged to the innovation gate.
                all_rejections_innovation += 1
        if is_dominant and s.is_outlier_rejected:
            if rejection_is_plausibility:
                domain_rejections_plausibility += 1
            else:
                domain_rejections_innovation += 1

        if is_dominant and s.raw_field_position is not None:
            if is_injected:
                corrupted_evaluated += 1
                if s.is_outlier_rejected:
                    corrupted_rejected += 1
                    if s.gate_statistic is not None:
                        corrupted_gate_stat.append(float(s.gate_statistic))
                elif s.is_measurement_used and s.field_position is not None:
                    corrupted_accepted += 1
            else:
                if s.gate_statistic is None:
                    clean_in_warmup += 1
                else:
                    clean_gate_evaluated += 1
                if s.is_outlier_rejected:
                    clean_rejected += 1
                    if s.gate_statistic is not None:
                        false_rejection_gate_stat.append(float(s.gate_statistic))
                    if s.position_source == "predicted_dead_reckoning":
                        dr_after_false_rejection += 1
                    gt = gt_table.get(int(s.frame_id), {}).get(int(gid)) if gid is not None else None
                    if gt is not None:
                        false_rejection_gt_err.append(
                            float(
                                np.hypot(
                                    s.raw_field_position[0] - gt["x_yd"],
                                    s.raw_field_position[1] - gt["y_yd"],
                                )
                            )
                        )
                elif s.is_measurement_used and s.field_position is not None:
                    # Accepted samples are split by whether a gate was actually applied, so
                    # accepted + rejected == gate_evaluated holds exactly.
                    if s.gate_statistic is None:
                        clean_accepted_warmup += 1
                    else:
                        clean_accepted += 1

        if s.field_position is None and s.predicted_position is None:
            if int(s.frames_since_measurement) > int(max_gap_frames):
                unpositioned["gap_exceeded"] += 1
            elif s.projection_status.startswith("calibration_refused"):
                unpositioned["geometry_refused"] += 1
            elif s.projection_status.startswith("footpoint_unreliable"):
                unpositioned["footpoint_refused"] += 1
            elif s.projection_status.startswith("projection_refused"):
                unpositioned["out_of_bounds"] += 1
            else:
                unpositioned["other"] += 1

    if clean_accepted + clean_rejected != clean_gate_evaluated:
        raise AssertionError(
            "rejection accounting mismatch: accepted + rejected "
            f"({clean_accepted} + {clean_rejected}) != gated ({clean_gate_evaluated}); warm-up "
            f"samples ({clean_accepted_warmup}) must not be counted as gated"
        )

    if domain_rejections_innovation + domain_rejections_plausibility != (
        clean_rejected + corrupted_rejected
    ):
        raise AssertionError(
            "rejection accounting mismatch: dominant-track rejections "
            f"({domain_rejections_innovation} innovation + {domain_rejections_plausibility} "
            f"plausibility) != clean ({clean_rejected}) + corrupted ({corrupted_rejected})"
        )

    return {        "clean_samples_gate_evaluated": int(clean_gate_evaluated),
        "clean_samples_accepted": int(clean_accepted),
        "clean_samples_accepted_during_warmup": int(clean_accepted_warmup),
        "clean_samples_rejected": int(clean_rejected),
        "clean_samples_warmup_bypassed": int(clean_in_warmup),
        "false_rejection_rate": (
            round(clean_rejected / clean_gate_evaluated, 6) if clean_gate_evaluated else None
        ),
        "false_rejection_rate_including_warmup": (
            round(clean_rejected / (clean_gate_evaluated + clean_in_warmup), 6)
            if (clean_gate_evaluated + clean_in_warmup)
            else None
        ),
        "corrupted_samples_evaluated": int(corrupted_evaluated),
        "corrupted_samples_rejected": int(corrupted_rejected),
        "corrupted_samples_accepted": int(corrupted_accepted),
        "true_rejection_rate_on_corrupted": (
            round(corrupted_rejected / corrupted_evaluated, 6) if corrupted_evaluated else None
        ),
        "false_acceptance_rate_on_corrupted": (
            round(corrupted_accepted / corrupted_evaluated, 6) if corrupted_evaluated else None
        ),
        "denominators": {
            "false_rejection_rate": "clean_samples_gate_evaluated (clean samples the gate actually evaluated)",
            "false_rejection_rate_including_warmup": (
                "clean_samples_gate_evaluated + clean_samples_warmup_bypassed"
            ),
            "true_rejection_rate_on_corrupted": (
                "corrupted_samples_evaluated (dominant-track injected events with a raw projection)"
            ),
            "false_acceptance_rate_on_corrupted": (
                "corrupted_samples_evaluated (same denominator as true_rejection_rate_on_corrupted)"
            ),
            "geometry_refusal_rate": (
                "not reported as a rate: unpositioned_by_reason counts are absolute sample counts "
                "over all emitted samples of the split"
            ),
        },
        "dead_reckoned_samples_after_false_rejection": int(dr_after_false_rejection),
        # Gate attribution, reported in two explicit scopes: EVERY track (what the gate
        # actually did) and DOMINANT tracks only (the scope of the clean/corrupted sample
        # accounting above). Rejected non-dominant samples are excluded from the clean-sample
        # false-rejection rate by construction.
        "rejections_by_innovation_gate": int(all_rejections_innovation),
        "rejections_by_image_space_plausibility_gate": int(all_rejections_plausibility),
        "rejections_all_tracks": int(all_rejections_innovation + all_rejections_plausibility),
        "rejections_on_dominant_tracks": int(
            domain_rejections_innovation + domain_rejections_plausibility
        ),
        "rejections_by_innovation_gate_on_dominant_tracks": int(domain_rejections_innovation),
        "rejections_by_plausibility_gate_on_dominant_tracks": int(domain_rejections_plausibility),
        "image_space_plausibility_flags": int(image_flags),
        # True only if a sample actually carries the image-space rejection reason; the
        # builder is written as a flag-only plausibility gate, so this is expected False.
        "plausibility_gate_can_reject_measurements": bool(all_rejections_plausibility),
        "unpositioned_samples": int(sum(unpositioned.values())),
        "unpositioned_by_reason": unpositioned,
        "median_gt_error_of_falsely_rejected_samples_yd": _median(false_rejection_gt_err),
        "median_gate_statistic_of_false_rejections": _median(false_rejection_gate_stat),
        "median_gate_statistic_of_corrupted_rejections": _median(corrupted_gate_stat),
    }


def _uncertainty_diagnostics(
    mahal_smoothed: Sequence[float],
    mahal_raw_cov: Sequence[float],
    mahal_smoothed_injected: Sequence[float],
    mahal_raw_cov_injected: Sequence[float],
    mahal_by_state: Dict[str, List[float]],
    mahal_by_inflation: Dict[str, List[float]],
    jitter_stats: Dict[str, Any],
    err_values: Sequence[float],
    sigma_major: Sequence[float],
    gate_stat_clean: Sequence[float],
    gate_stat_corrupted: Sequence[float],
) -> Dict[str, Any]:
    """Coverage / normalized-innovation diagnostics for the *uncalibrated* covariance."""

    def _cov(values: Sequence[float], threshold: float) -> Optional[float]:
        return round(100.0 * float(np.mean([d <= threshold for d in values])), 2) if values else None

    def _k2(values: Sequence[float]) -> Optional[float]:
        """Scale factor between the empirical error and the claimed covariance."""
        if not values:
            return None
        return round(float(np.median(values)) / COVERAGE_68_MEDIAN_CHI2, 4)

    jitter_u = float(jitter_stats.get("observed_jitter_std_u_px") or 0.0)
    jitter_v = float(jitter_stats.get("observed_jitter_std_v_px") or 0.0)
    model_u = float(jitter_stats.get("assumed_footpoint_sigma_u_px") or 0.0)
    model_v = float(jitter_stats.get("assumed_footpoint_sigma_v_px") or 0.0)
    predicted = None
    explanation = None
    if model_u > 0.0 and model_v > 0.0 and (jitter_u > 0.0 or jitter_v > 0.0):
        k2_pred = 0.5 * ((jitter_u / model_u) ** 2 + (jitter_v / model_v) ** 2)
        predicted = {
            "assumed_footpoint_sigma_u_px": round(model_u, 4),
            "assumed_footpoint_sigma_v_px": round(model_v, 4),
            "observed_footpoint_jitter_std_u_px": round(jitter_u, 4),
            "observed_footpoint_jitter_std_v_px": round(jitter_v, 4),
            "predicted_covariance_scale_k2": round(k2_pred, 4),
            "predicted_coverage_68_pct": round(100.0 * (1.0 - float(np.exp(-COVERAGE_68_CHI2_2DOF / (2.0 * k2_pred)))), 2),
            "predicted_coverage_95_pct": round(100.0 * (1.0 - float(np.exp(-COVERAGE_95_CHI2_2DOF / (2.0 * k2_pred)))), 2),
        }
        explanation = (
            "First-order explanation of the coverage gap: the injected harness jitter is larger "
            "than the frozen 1 px footpoint-noise assumption, so the reported covariance is too "
            "small by roughly the ratio below. This is a diagnostic derivation, not a calibration."
        )
    return {
        "status": UNCERTAINTY_STATUS,
        "note": UNCERTAINTY_NOTE,
        "coverage_evaluated_samples": len(mahal_smoothed),
        "coverage_note": (
            "Coverage uses CLEAN accepted measurements only; accepted injected outliers are counted "
            "separately (injected_accepted_samples_excluded_from_coverage) and included in the "
            "*_including_injected_events variants so both views stay available."
        ),
        "coverage_excludes_injected_events": True,
        "injected_accepted_samples_excluded_from_coverage": int(len(mahal_smoothed_injected)),
        "median_mahalanobis_d2_of_injected_accepted_samples": _median(mahal_smoothed_injected),
        "coverage_68_pct_smoothed_covariance": _cov(mahal_smoothed, COVERAGE_68_CHI2_2DOF),
        "coverage_68_pct_smoothed_covariance_including_injected_events": _cov(
            list(mahal_smoothed) + list(mahal_smoothed_injected), COVERAGE_68_CHI2_2DOF
        ),
        "coverage_95_pct_smoothed_covariance_including_injected_events": _cov(
            list(mahal_smoothed) + list(mahal_smoothed_injected), COVERAGE_95_CHI2_2DOF
        ),
        "coverage_68_pct_raw_measurement_covariance_including_injected_events": _cov(
            list(mahal_raw_cov) + list(mahal_raw_cov_injected), COVERAGE_68_CHI2_2DOF
        ),
        "coverage_95_pct_smoothed_covariance": _cov(mahal_smoothed, COVERAGE_95_CHI2_2DOF),
        "coverage_68_pct_raw_measurement_covariance": _cov(mahal_raw_cov, COVERAGE_68_CHI2_2DOF),
        "coverage_95_pct_raw_measurement_covariance": _cov(mahal_raw_cov, COVERAGE_95_CHI2_2DOF),
        "coverage_68_pct_by_geometry_state": {
            state: _cov(values, COVERAGE_68_CHI2_2DOF) for state, values in mahal_by_state.items()
        },
        "coverage_95_pct_by_geometry_state": {
            state: _cov(values, COVERAGE_95_CHI2_2DOF) for state, values in mahal_by_state.items()
        },
        "coverage_samples_by_geometry_state": {state: len(v) for state, v in mahal_by_state.items()},
        # "before/after inflation": samples whose reported covariance was inflated by the
        # propagation/rejection path (after) vs samples whose covariance is the raw
        # footpoint-derived model (before).
        "coverage_68_pct_by_covariance_inflation": {
            key: _cov(values, COVERAGE_68_CHI2_2DOF) for key, values in mahal_by_inflation.items()
        },
        "coverage_95_pct_by_covariance_inflation": {
            key: _cov(values, COVERAGE_95_CHI2_2DOF) for key, values in mahal_by_inflation.items()
        },
        "coverage_samples_by_covariance_inflation": {
            key: len(values) for key, values in mahal_by_inflation.items()
        },
        "empirical_position_err_median_yd": _median(list(err_values)),
        "empirical_position_err_rmse_yd": (
            round(float(np.sqrt(np.mean(np.square(list(err_values))))), 4) if err_values else None
        ),
        "assumed_sigma_major_median_yd": _median(list(sigma_major)),
        "median_mahalanobis_distance_smoothed": (
            round(float(np.sqrt(np.median(mahal_smoothed))), 4) if mahal_smoothed else None
        ),
        "median_mahalanobis_distance_raw_measurement_covariance": (
            round(float(np.sqrt(np.median(mahal_raw_cov))), 4) if mahal_raw_cov else None
        ),
        "mean_gate_statistic_accepted_clean": (
            round(float(np.mean(gate_stat_clean)), 4) if gate_stat_clean else None
        ),
        "median_gate_statistic_accepted_clean": _median(gate_stat_clean),
        "mean_gate_statistic_corrupted_rejected": (
            round(float(np.mean(gate_stat_corrupted)), 4) if gate_stat_corrupted else None
        ),
        "covariance_scale_needed_k2_smoothed": _k2(mahal_smoothed),
        "covariance_scale_needed_k2_raw_measurement_covariance": _k2(mahal_raw_cov),
        "jitter_vs_model_explanation": explanation,
        "jitter_vs_model_prediction": predicted,
    }


def _score_samples(
    seq: Dict[str, Any],
    samples_all: List[Any],
    gt_table: Dict[int, Dict[int, Dict[str, float]]],
    trajectories: List[Any],
    builder: PlayerTrajectoryBuilder,
    *,
    n_frames: int,
    runtime_ms: float,
    jitter_stats: Dict[str, Any],
    max_gap_frames: int = 3,
) -> Dict[str, Any]:
    """Score one sequence's trajectory samples.

    Dominant-track selection is documented in :data:`DOMINANT_TRACK_SELECTION` and is
    performed by :func:`select_dominant_track`, which sees only runtime-observable
    counters. Primary accuracy metrics are computed over the dominant track per player;
    every other track is reported explicitly (``track_table``, ``fragmentation_taxonomy``)
    so fragmentation is never hidden by the dominant-track convention.
    """
    injected = {(int(e["gt_id"]), int(e["frame"])) for e in seq.get("outlier_events", [])}
    track_stats = _track_runtime_stats(samples_all)
    if len(track_stats) != len(trajectories):
        raise AssertionError(
            f"[{seq['sequence_id']}] track accounting mismatch: "
            f"{len(track_stats)} tracks with samples vs {len(trajectories)} trajectories"
        )
    labels_by_track = _track_label_map(samples_all)

    by_gid: Dict[int, List[Any]] = {}
    for s in samples_all:
        gid = _sample_gt_id(s)
        if gid is not None:
            by_gid.setdefault(gid, []).append(s)

    # Dominant-track candidates are (player, track) groups counted from emitted samples;
    # the only ground-truth dependency is the harness label that defines the group.
    group_stats: Dict[int, List[Dict[str, Any]]] = {}
    for gid, ss in by_gid.items():
        per_track: Dict[int, List[Any]] = {}
        for s in ss:
            per_track.setdefault(int(s.track_id), []).append(s)
        candidates: List[Dict[str, Any]] = []
        for tid in sorted(per_track):
            group = sorted(per_track[tid], key=lambda s: int(s.frame_id))
            run = best = 0
            prev_frame: Optional[int] = None
            prev_observed = False
            for s in group:
                frame = int(s.frame_id)
                observed = s.track_state == "observed"
                if observed and prev_observed and prev_frame is not None and frame == prev_frame + 1:
                    run += 1
                elif observed:
                    run = 1
                else:
                    run = 0
                best = max(best, run)
                prev_frame, prev_observed = frame, observed
            candidates.append(
                {
                    "track_id": tid,
                    "n_samples": len(group),
                    "n_positioned": sum(1 for s in group if s.field_position is not None),
                    "n_observed": sum(1 for s in group if s.track_state == "observed"),
                    "longest_observed_run": best,
                }
            )
        group_stats[gid] = candidates

    dominant_track: Dict[int, int] = {}
    alternative_dominants: Dict[str, Dict[int, int]] = {}
    for gid in sorted(group_stats):
        candidates = group_stats[gid]
        selected = select_dominant_track(candidates)
        if selected is not None:
            dominant_track[gid] = selected
        for rule in ALTERNATIVE_DOMINANT_TRACK_RULES:
            alt = select_dominant_track(candidates, rule=rule)
            if alt is not None:
                alternative_dominants.setdefault(rule, {})[gid] = alt

    raw_err: List[float] = []
    smooth_err: List[float] = []
    all_matched_err: List[float] = []
    non_dominant_err: List[float] = []
    non_dominant_total = 0
    dominant_total = 0
    unattributed_total = 0
    spurious_err: List[float] = []
    speed_err: List[float] = []
    vel_err: List[float] = []
    accel_err: List[float] = []
    dr_err: List[float] = []
    # Dead reckoning is split by CAUSE so a reader can see how much of it was forced by a
    # false rejection versus a genuinely missing measurement (occlusion / dropout / refusal).
    # Reporting only: no filter, gate or association behaviour depends on these lists.
    dr_rejection_err: List[float] = []
    dr_missing_err: List[float] = []
    # Trajectory error is split by estimation path: the normal accepted-measurement path versus
    # the explicit post-gap re-initialization path, reported separately instead of pooled.
    accepted_meas_err: List[float] = []
    post_reinit_meas_err: List[float] = []
    # Sub-arm of the accepted path: samples whose position comes from a filter that was
    # explicitly re-initialized after persistent rejections (a different recovery code path).
    reinit_rejection_meas_err: List[float] = []
    # Coverage is measured on CLEAN accepted measurements: an injected outlier that the gate
    # accepted is a harness-injected fault, not evidence about the noise model. Those samples are
    # counted and reported separately (``*_including_injected_events``) so nothing is hidden.
    mahal_smoothed: List[float] = []
    mahal_smoothed_injected: List[float] = []
    mahal_raw_cov: List[float] = []
    mahal_raw_cov_injected: List[float] = []
    mahal_by_state: Dict[str, List[float]] = {"calibrated": [], "propagated": []}
    mahal_by_inflation: Dict[str, List[float]] = {"inflated": [], "not_inflated": []}
    sigma_major: List[float] = []
    sigma_minor: List[float] = []
    gt_accel_mag: List[float] = []
    gt_speed: List[float] = []
    gate_stat_clean: List[float] = []
    gate_stat_corrupted: List[float] = []
    state_counts = {"calibrated": 0, "propagated": 0, "unknown": 0}
    positioned_by_state = {"calibrated": 0, "propagated": 0, "unknown": 0}
    positioned_samples = 0
    dominant_samples = 0
    measured_dominant_samples = 0
    unknown_state_positioned = 0
    reinit_gap = 0
    reinit_rejects = 0
    image_jumps = 0
    abs_violations = 0
    accel_clipped = 0
    speed_clipped = 0
    gate_evaluated = 0
    false_rejections = 0
    observed_dominant = 0
    inferred_dominant = 0
    longest_run = 0
    direction_err_deg: List[float] = []
    smoothed_len = 0.0
    raw_len = 0.0
    gt_len = 0.0

    for s in samples_all:
        gid = _sample_gt_id(s)
        is_dominant = gid is not None and s.track_id == dominant_track.get(gid)
        is_injected = gid is not None and (int(gid), int(s.frame_id)) in injected
        state_counts[s.geometry_state] = state_counts.get(s.geometry_state, 0) + 1
        if s.image_space_jump_flagged:
            image_jumps += 1
        if s.reinitialized_after_gap:
            reinit_gap += 1
        if s.reinitialized_after_rejections:
            reinit_rejects += 1
        if s.is_acceleration_clipped:
            accel_clipped += 1
        if s.is_speed_clipped:
            speed_clipped += 1
        if s.absolute_yardline is not None and s.x_coord_mode != "absolute":
            abs_violations += 1

        # Safety invariant, evaluated over ALL samples including spurious tracks.
        if s.field_position is not None:
            positioned_samples += 1
            if s.geometry_state == "unknown":
                unknown_state_positioned += 1

        if gid is None or s.frame_id not in gt_table:
            # No harness association for this sample (unlabelled track, or no ground-truth
            # row at this frame). Counted explicitly so the sample accounting closes.
            unattributed_total += 1
            continue
        gt = gt_table[s.frame_id][int(gid)]
        gt_err = None
        if s.field_position is not None:
            gt_err = float(
                np.hypot(s.field_position[0] - gt["x_yd"], s.field_position[1] - gt["y_yd"])
            )
        # All matched tracks (dominant + fragment) are accounted for; the dominant
        # track alone drives the primary metrics.
        if s.field_position is not None:
            all_matched_err.append(gt_err)
            if not is_dominant:
                non_dominant_err.append(gt_err)
        if not is_dominant:
            non_dominant_total += 1
            if gt_err is not None:
                spurious_err.append(gt_err)
            continue
        dominant_total += 1

        if s.raw_field_position is not None:
            gate_evaluated += 1 if not is_injected else 0
            if s.is_outlier_rejected and not is_injected:
                false_rejections += 1

        if s.predicted_position is not None and s.position_source == "predicted_dead_reckoning":
            dr_value = float(
                np.hypot(
                    s.predicted_position[0] - gt["x_yd"], s.predicted_position[1] - gt["y_yd"]
                )
            )
            dr_err.append(dr_value)
            (dr_rejection_err if s.is_outlier_rejected else dr_missing_err).append(dr_value)

        if s.field_position is None and s.predicted_position is None:
            continue

        dominant_samples += 1
        # Observed = the player was actually detected this frame (track not coasting
        # and footpoint reliable). Inferred = the field estimate is dead reckoning.
        if s.track_state == "observed":
            observed_dominant += 1
        if s.position_source == "predicted_dead_reckoning":
            inferred_dominant += 1
        if s.field_position is not None:
            measured_dominant_samples += 1

        if s.field_position is None:
            continue
        positioned_by_state[s.geometry_state] = positioned_by_state.get(s.geometry_state, 0) + 1
        assert gt_err is not None
        smooth_err.append(gt_err)
        if s.reinitialized_after_gap:
            post_reinit_meas_err.append(gt_err)
        else:
            accepted_meas_err.append(gt_err)
            if s.reinitialized_after_rejections:
                reinit_rejection_meas_err.append(gt_err)
        # Coverage/uncertainty diagnostics use ACCEPTED measurements only: rejected
        # outliers are intentionally far from the prediction and would make the
        # diagnostic meaningless. Two covariance models are compared:
        #   smoothed  = the reported (filter) covariance -> what a consumer is told
        #   raw_cov   = the raw measurement covariance (footpoint + drift) -> the
        #               model before the smoothing stage shrinks it.
        residual = [s.field_position[0] - gt["x_yd"], s.field_position[1] - gt["y_yd"]]
        if s.covariance_xy is not None and s.is_measurement_used:
            cov = np.array(
                [[s.covariance_xy[0], s.covariance_xy[1]], [s.covariance_xy[1], s.covariance_xy[2]]]
            )
            d2_value = mahalanobis_distance_sq(residual, cov)
            if is_injected:
                mahal_smoothed_injected.append(d2_value)
            else:
                mahal_smoothed.append(d2_value)
                mahal_by_state.setdefault(s.geometry_state, []).append(d2_value)
                mahal_by_inflation[
                    "inflated" if s.uncertainty_inflated else "not_inflated"
                ].append(d2_value)
            sigma_major.append(s.sigma_major_yd)
            sigma_minor.append(s.sigma_minor_yd)
        if s.measurement_covariance_xy is not None and s.is_measurement_used:
            cov_m = np.array(
                [
                    [s.measurement_covariance_xy[0], s.measurement_covariance_xy[1]],
                    [s.measurement_covariance_xy[1], s.measurement_covariance_xy[2]],
                ]
            )
            raw_d2_value = mahalanobis_distance_sq(residual, cov_m)
            if is_injected:
                mahal_raw_cov_injected.append(raw_d2_value)
            else:
                mahal_raw_cov.append(raw_d2_value)
        if s.gate_statistic is not None:
            (gate_stat_corrupted if is_injected else gate_stat_clean).append(float(s.gate_statistic))
        if s.raw_field_position is not None:
            raw_err.append(
                float(
                    np.hypot(
                        s.raw_field_position[0] - gt["x_yd"],
                        s.raw_field_position[1] - gt["y_yd"],
                    )
                )
            )
        gt_speed_now = float(np.hypot(gt["vx_yd_s"], gt["vy_yd_s"]))
        if gt_speed_now > 0.5 and s.speed_yd_s > 0.3:
            d_gt = float(np.arctan2(gt["vy_yd_s"], gt["vx_yd_s"]))
            d_rep = s.direction_rad
            diff = abs((d_rep - d_gt + np.pi) % (2.0 * np.pi) - np.pi)
            direction_err_deg.append(float(np.degrees(diff)))
        speed_err.append(abs(s.speed_yd_s - gt_speed_now))
        vel_err.append(
            float(np.hypot(s.velocity_yd_s[0] - gt["vx_yd_s"], s.velocity_yd_s[1] - gt["vy_yd_s"]))
        )
        accel_err.append(abs(s.accel_yd_s2 - float(np.hypot(gt["ax_yd_s2"], gt["ay_yd_s2"]))))
        gt_accel_mag.append(float(np.hypot(gt["ax_yd_s2"], gt["ay_yd_s2"])))
        gt_speed.append(float(np.hypot(gt["vx_yd_s"], gt["vy_yd_s"])))

    # Estimation-path partition must close: every positioned dominant-track sample was either
    # accepted as a measurement or produced by the post-gap re-initialization path.
    if len(accepted_meas_err) + len(post_reinit_meas_err) != measured_dominant_samples:
        raise AssertionError(
            f"[{seq['sequence_id']}] estimation-path mismatch: {len(accepted_meas_err)} accepted + "
            f"{len(post_reinit_meas_err)} post-reinit != {measured_dominant_samples} positioned "
            "dominant samples"
        )
    # The re-init-after-rejection arm is a SUBSET of the accepted arm, never a double count.
    if len(reinit_rejection_meas_err) > len(accepted_meas_err):
        raise AssertionError(
            f"[{seq['sequence_id']}] reinit-after-rejection arm ({len(reinit_rejection_meas_err)}) "
            f"exceeds the accepted-measurement arm ({len(accepted_meas_err)})"
        )
    # Dead-reckoning partition must close: a dead-reckoned sample is caused by a rejected
    # measurement or by a missing measurement (occlusion / dropout / refused projection).
    if len(dr_rejection_err) + len(dr_missing_err) != len(dr_err):
        raise AssertionError(
            f"[{seq['sequence_id']}] dead-reckoning mismatch: {len(dr_rejection_err)} from "
            f"rejection + {len(dr_missing_err)} from missing measurement != {len(dr_err)}"
        )

    # Sample accounting must close: every emitted sample is either in a dominant track, in a
    # non-dominant (fragment) track, or unattributed (no harness association). Fragmentation
    # therefore cannot hide samples from the report.
    if dominant_total + non_dominant_total + unattributed_total != len(samples_all):
        raise AssertionError(
            f"[{seq['sequence_id']}] sample accounting mismatch: {dominant_total} dominant + "
            f"{non_dominant_total} non-dominant + {unattributed_total} unattributed != "
            f"{len(samples_all)} emitted samples"
        )

    # ---- identity metrics (CLEAR-style, evaluated on OBSERVED samples) ---------
    # Same definitions as the Phase 3 evaluator so Phase 3 and Phase 4 numbers are
    # directly comparable:
    #   IDSW (lifetime): an observed sample whose track_id differs from the track_id
    #                    most recently observed for that ground-truth player.
    #   FRAG           : a ground-truth player that was NOT observed at the previous
    #                    frame and is observed again now.
    # The lifetime IDSW is additionally split into active association swaps (the
    # previous track still exists this frame) versus post-expiration reinitializations.
    observed_by_gid: Dict[int, Dict[int, int]] = {}
    for sample in samples_all:
        gid_raw = _sample_gt_id(sample)
        if gid_raw is None or sample.track_state != "observed":
            continue
        observed_by_gid.setdefault(gid_raw, {})[int(sample.frame_id)] = int(sample.track_id)

    tracks_per_frame: Dict[int, set] = {}
    for sample in samples_all:
        tracks_per_frame.setdefault(int(sample.frame_id), set()).add(int(sample.track_id))

    id_switches = 0
    id_switches_active_swap = 0
    id_switches_post_reinit = 0
    track_fragmentations = 0
    for gid, obs in sorted(observed_by_gid.items()):
        last_track: Optional[int] = None
        last_frame: Optional[int] = None
        for frame in sorted(obs):
            track = obs[frame]
            if last_track is not None:
                if frame - last_frame > 1:
                    track_fragmentations += 1
                if track != last_track:
                    id_switches += 1
                    if last_track in tracks_per_frame.get(frame, set()):
                        id_switches_active_swap += 1
                    else:
                        id_switches_post_reinit += 1
            last_track = track
            last_frame = frame

    # ---- motion preservation & travelled distance ------------------------------
    for gid, ss in sorted(by_gid.items()):
        dom = [x for x in ss if x.track_id == dominant_track.get(gid)]
        dom.sort(key=lambda x: x.frame_id)
        prev_sm = prev_raw = prev_gt = None
        run = 0
        for sample in dom:
            gt = gt_table.get(sample.frame_id, {}).get(gid)
            if gt is None:
                continue
            if sample.field_position is not None:
                if prev_sm is not None:
                    smoothed_len += float(
                        np.hypot(
                            sample.field_position[0] - prev_sm[0],
                            sample.field_position[1] - prev_sm[1],
                        )
                    )
                prev_sm = sample.field_position
            if sample.raw_field_position is not None:
                if prev_raw is not None:
                    raw_len += float(
                        np.hypot(
                            sample.raw_field_position[0] - prev_raw[0],
                            sample.raw_field_position[1] - prev_raw[1],
                        )
                    )
                prev_raw = sample.raw_field_position
            gt_xy = (gt["x_yd"], gt["y_yd"])
            if prev_gt is not None:
                gt_len += float(np.hypot(gt_xy[0] - prev_gt[0], gt_xy[1] - prev_gt[1]))
            prev_gt = gt_xy
            if sample.track_state == "observed":
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0

    # ---- event-level outlier accounting ---------------------------------------
    track_at: Dict[int, Dict[int, int]] = {}
    for s in samples_all:
        gid_raw = _sample_gt_id(s)
        if gid_raw is not None:
            track_at.setdefault(gid_raw, {})[int(s.frame_id)] = int(s.track_id)

    outcomes = {
        "rejected_by_field_gate": 0,
        "refused_by_projection_gate": 0,
        "spawning_spurious_track": 0,
        "absorbed_into_track": 0,
        "unknown": 0,
    }
    event_details: List[Dict[str, Any]] = []
    spurious_tracks: set = set()
    for gid, frame in sorted(injected):
        prev_track = track_at.get(gid, {}).get(frame - 1)
        at = [s for s in samples_all if _sample_gt_id(s) == gid and s.frame_id == frame]
        measured_at = [s for s in at if s.is_measurement_used and s.field_position is not None]
        rejected_at = any(s.is_outlier_rejected for s in at)
        flagged_at = any(s.image_space_jump_flagged for s in at)
        new_tracks = {s.track_id for s in at if prev_track is None or s.track_id != prev_track}
        spawned = bool(new_tracks) and any(s.track_id in new_tracks for s in measured_at)
        if spawned:
            outcome = "spawning_spurious_track"
            spurious_tracks.update(new_tracks)
        elif rejected_at:
            outcome = "rejected_by_field_gate"
        elif not measured_at:
            outcome = "refused_by_projection_gate"
        elif measured_at:
            outcome = "absorbed_into_track"
        else:  # pragma: no cover - defensive
            outcome = "unknown"
        outcomes[outcome] += 1
        event_details.append(
            {
                "gt_id": gid,
                "frame": frame,
                "outcome": outcome,
                "geometry_state": at[0].geometry_state if at else None,
                "n_samples_at_event": len(at),
                "new_track_ids": sorted(new_tracks),
                "image_space_jump_flagged": bool(flagged_at),
            }
        )

    n_injected = len(injected)
    handled = n_injected - outcomes["absorbed_into_track"]
    expected = int(len(seq["players"]) * n_frames)

    fragmentation = _fragmentation_taxonomy(samples_all, dominant_track, injected)
    rejection = _rejection_accounting(
        samples_all,
        gt_table,
        dominant_track,
        injected,
        max_gap_frames=max_gap_frames,
    )
    uncertainty = _uncertainty_diagnostics(
        mahal_smoothed,
        mahal_raw_cov,
        mahal_smoothed_injected,
        mahal_raw_cov_injected,
        mahal_by_state,
        mahal_by_inflation,
        jitter_stats,
        smooth_err,
        sigma_major,
        gate_stat_clean,
        gate_stat_corrupted,
    )

    # Alternative selection rules: reported so the reader can see how sensitive the
    # primary metric is to the (runtime-only) selection rule. These are diagnostics;
    # the shipped rule above is the one used for every primary metric.
    shipped_ids = {int(g): int(t) for g, t in dominant_track.items()}
    alternative_reports: Dict[str, Any] = {}
    for rule, mapping in alternative_dominants.items():
        agree = sum(
            1 for g in shipped_ids if int(mapping.get(g, -1)) == shipped_ids[g]
        )
        alt_err: List[float] = []
        for s in samples_all:
            gid_raw = _sample_gt_id(s)
            if gid_raw is None or int(mapping.get(gid_raw, -1)) != int(s.track_id):
                continue
            if s.field_position is None or s.frame_id not in gt_table:
                continue
            gt = gt_table[s.frame_id][gid_raw]
            alt_err.append(
                float(np.hypot(s.field_position[0] - gt["x_yd"], s.field_position[1] - gt["y_yd"]))
            )
        alternative_reports[rule] = {
            "agreeing_players": int(agree),
            "players": int(len(shipped_ids)),
            "field_pos_err_median_yd": _median(alt_err),
            "selection_rule_is_diagnostic_only": True,
        }

    return {
        "sequence_id": seq["sequence_id"],
        "title": seq["title"],
        "split": seq["split"],
        "scenario": seq["scenario"],
        "num_frames": n_frames,
        "fps": float(seq["fps"]),
        "jitter_px": float(seq.get("jitter_px", 0.8)),
        "base_image_path": seq["base_image_path"],
        "detector_implementation": "fixture_detector_v1",
        "n_players": len(seq["players"]),
        "n_tracks": len(trajectories),
        "n_samples": len(samples_all),
        "expected_samples": expected,
        "dominant_track_rule": DOMINANT_TRACK_RULE,
        "dominant_track_id_by_gt": {str(g): int(t) for g, t in sorted(dominant_track.items())},
        "dominant_track_alternatives": alternative_reports,
        "identity_contamination": {
            "tracks_with_multiple_gt_labels": fragmentation["tracks_with_multiple_gt_labels"],
            "tracks_without_any_label": fragmentation["tracks_without_any_label"],
            "tracks": {
                str(tid): {str(g): int(c) for g, c in sorted(per.items())}
                for tid, per in sorted(labels_by_track.items())
                if len(per) > 1
            },
            "note": (
                "A track whose samples carry two different ground-truth labels means the tracker "
                "kept one track_id alive across a player change (observed after the camera cut). "
                "Samples are still scored against the player they claim (CLEAR-MOTA per-frame "
                "convention); this block makes the contamination visible instead of hiding it."
            ),
        },
        "dominant_samples": dominant_samples,
        "measured_dominant_samples": measured_dominant_samples,
        "observed_dominant_samples": observed_dominant,
        "inferred_dominant_samples": inferred_dominant,
        "observed_sample_ratio": round(observed_dominant / max(1, dominant_samples), 4),
        "inferred_sample_ratio": round(
            inferred_dominant / max(1, dominant_samples), 4
        ),
        "longest_observed_run_frames": longest_run,
        "direction_err_median_deg": _median(direction_err_deg),
        "smoothed_path_length_yd": round(smoothed_len, 4),
        "raw_projection_path_length_yd": round(raw_len, 4),
        "gt_path_length_yd": round(gt_len, 4),
        "motion_preservation_ratio_vs_raw": round(smoothed_len / raw_len, 4) if raw_len > 0 else None,
        "motion_preservation_ratio_vs_gt": round(smoothed_len / gt_len, 4) if gt_len > 0 else None,
        "track_completeness": round(dominant_samples / max(1, expected), 4),
        "measured_completeness": round(measured_dominant_samples / max(1, expected), 4),
        "positioned_samples": positioned_samples,
        "positioned_by_geometry_state": positioned_by_state,
        "raw_field_pos_err_median_yd": _median(raw_err),
        "raw_field_pos_err_rmse_yd": round(float(np.sqrt(np.mean(np.square(raw_err)))), 4) if raw_err else None,
        "field_pos_err_median_yd": _median(smooth_err),
        "field_pos_err_p90_yd": _percentile(smooth_err, 90.0),
        "field_pos_err_rmse_yd": round(float(np.sqrt(np.mean(np.square(smooth_err)))), 4) if smooth_err else None,
        "smoothing_error_reduction_pct": (
            round(100.0 * (1.0 - float(np.median(smooth_err)) / max(1e-9, float(np.median(raw_err)))), 2)
            if (smooth_err and raw_err)
            else None
        ),
        "all_matched_track_field_err_median_yd": _median(all_matched_err),
        "all_matched_track_samples": len(all_matched_err),
        "non_dominant_track_field_err_median_yd": _median(non_dominant_err),
        "non_dominant_track_samples": len(non_dominant_err),
        "non_dominant_track_samples_total": int(non_dominant_total),
        "dominant_track_samples_total": int(dominant_total),
        "unattributed_samples_total": int(unattributed_total),
        "dominant_track_samples_without_position": int(dominant_total - dominant_samples),
        "spurious_track_field_err_median_yd": _median(spurious_err),
        "spurious_track_samples": len(spurious_err),
        "excluded_gt_player_frames": int(max(0, expected - dominant_samples)),
        "dominant_fraction_of_gt_player_frames": round(dominant_samples / max(1, expected), 4),
        "fragmentation_taxonomy": fragmentation["taxonomy"],
        "fragmentation_taxonomy_total": fragmentation["taxonomy_total"],
        "fragmentation_n_matched_tracks": fragmentation["n_matched_tracks"],
        "fragmentation_dominant_tracks_total": int(
            len(
                {
                    row["track_id"]
                    for row in fragmentation["track_details"]
                    if not row["excluded_from_dominant_metrics"]
                }
            )
        ),
        "fragmentation_excluded_from_dominant_total": fragmentation[
            "fragment_excluded_from_dominant_total"
        ],
        "fragmentation_excluded_by_origin": fragmentation[
            "fragment_excluded_from_dominant_by_origin"
        ],
        "fragmentation_fragment_tracks_excluded_from_dominant": fragmentation[
            "fragment_tracks_excluded_from_dominant"
        ],
        "fragmentation_tracks_per_gt_player": fragmentation["tracks_per_gt_player"],
        "fragmentation_max_tracks_per_gt_player": fragmentation["max_tracks_per_gt_player"],
        "fragmentation_samples_per_track": fragmentation["samples_per_track"],
        "track_table": fragmentation["track_details"],
        "rejection_accounting": rejection,
        "uncertainty_diagnostics": uncertainty,
        "footpoint_jitter_stats": {
            key: jitter_stats[key]
            for key in (
                "observed_jitter_std_u_px",
                "observed_jitter_std_v_px",
                "assumed_footpoint_sigma_u_px",
                "assumed_footpoint_sigma_v_px",
                "n_fixture_footpoints",
            )
        },
        "speed_err_median_yd_s": _median(speed_err),
        "velocity_rmse_yd_s": round(float(np.sqrt(np.mean(np.square(vel_err)))), 4) if vel_err else None,
        "accel_mag_err_median_yd_s2": _median(accel_err),
        "acceleration_status": ACCELERATION_LABEL,
        "gt_accel_mag_median_yd_s2": _median(gt_accel_mag),
        "gt_speed_median_yd_s": _median(gt_speed),
        "dead_reckoning_samples": len(dr_err),
        "dead_reckoning_err_median_yd": _median(dr_err),
        "dead_reckoning_err_p90_yd": _percentile(dr_err, 90.0),
        "accepted_measurement_samples": len(accepted_meas_err),
        "accepted_measurement_err_median_yd": _median(accepted_meas_err),
        "accepted_measurement_err_p90_yd": _percentile(accepted_meas_err, 90.0),
        "accepted_measurement_err_rmse_yd": (
            round(float(np.sqrt(np.mean(np.square(accepted_meas_err)))), 4)
            if accepted_meas_err
            else None
        ),
        "reinit_after_rejection_measurement_samples": len(reinit_rejection_meas_err),
        "reinit_after_rejection_measurement_err_median_yd": _median(reinit_rejection_meas_err),
        "reinit_after_rejection_measurement_err_rmse_yd": (
            round(float(np.sqrt(np.mean(np.square(reinit_rejection_meas_err)))), 4)
            if reinit_rejection_meas_err
            else None
        ),
        "post_reinit_measurement_samples": len(post_reinit_meas_err),
        "post_reinit_measurement_err_median_yd": _median(post_reinit_meas_err),
        "post_reinit_measurement_err_p90_yd": _percentile(post_reinit_meas_err, 90.0),
        "post_reinit_measurement_err_rmse_yd": (
            round(float(np.sqrt(np.mean(np.square(post_reinit_meas_err)))), 4)
            if post_reinit_meas_err
            else None
        ),
        "dead_reckoning_samples_from_rejection": len(dr_rejection_err),
        "dead_reckoning_err_from_rejection_median_yd": _median(dr_rejection_err),
        "dead_reckoning_err_from_rejection_p90_yd": _percentile(dr_rejection_err, 90.0),
        "dead_reckoning_err_from_rejection_rmse_yd": (
            round(float(np.sqrt(np.mean(np.square(dr_rejection_err)))), 4)
            if dr_rejection_err
            else None
        ),
        "dead_reckoning_samples_from_missing_measurement": len(dr_missing_err),
        "dead_reckoning_err_from_missing_measurement_median_yd": _median(dr_missing_err),
        "dead_reckoning_err_from_missing_measurement_p90_yd": _percentile(dr_missing_err, 90.0),
        "dead_reckoning_err_from_missing_measurement_rmse_yd": (
            round(float(np.sqrt(np.mean(np.square(dr_missing_err)))), 4)
            if dr_missing_err
            else None
        ),
        "outliers_injected": n_injected,
        "outliers_rejected_by_field_gate": outcomes["rejected_by_field_gate"],
        "outlier_rejection_rate": round(outcomes["rejected_by_field_gate"] / n_injected, 4) if n_injected else None,
        "outlier_events_refused_by_projection_gate": outcomes["refused_by_projection_gate"],
        "outlier_events_spawning_spurious_track": outcomes["spawning_spurious_track"],
        "outlier_events_absorbed_into_track": outcomes["absorbed_into_track"],
        "outlier_handling_rate": round(handled / n_injected, 4) if n_injected else None,
        "spurious_tracks_from_outliers": int(len(spurious_tracks)),
        "outlier_event_details": event_details,
        "gate_evaluated_samples": gate_evaluated,
        "false_rejections": false_rejections,
        "false_rejection_rate": round(false_rejections / gate_evaluated, 6) if gate_evaluated else None,
        "gate_warmup_measurements": int(builder.warmup_measurements),
        "geometry_state_sample_counts": state_counts,
        "samples_with_position_in_unknown_geometry": unknown_state_positioned,
        "image_space_jumps_flagged": image_jumps,
        "image_space_plausibility_flags_total": int(builder.image_jump_flags),
        "reinitialized_after_gap": reinit_gap,
        "reinitialized_after_rejections": reinit_rejects,
        "filter_reinitializations": int(builder.filter_reinitializations),
        "acceleration_clipped_samples": accel_clipped,
        "speed_clipped_samples": speed_clipped,
        "coverage_68_pct": uncertainty["coverage_68_pct_smoothed_covariance"],
        "coverage_95_pct": uncertainty["coverage_95_pct_smoothed_covariance"],
        "mean_mahalanobis_d2": round(float(np.mean(mahal_smoothed)), 3) if mahal_smoothed else None,
        "mean_sigma_major_yd": round(float(np.mean(sigma_major)), 4) if sigma_major else None,
        "mean_sigma_minor_yd": round(float(np.mean(sigma_minor)), 4) if sigma_minor else None,
        "id_switches": id_switches,
        "id_switches_active_swap": id_switches_active_swap,
        "id_switches_post_reinit": id_switches_post_reinit,
        "track_fragmentations": track_fragmentations,
        "field_identity_changes": int(
            sum(len({s.track_id for s in ss}) - 1 for ss in by_gid.values())
        ),
        "tracks_per_gt_max": int(max((len({s.track_id for s in ss}) for ss in by_gid.values()), default=0)),
        "fabricated_field_positions": int(builder.fabricated_field_positions),
        "fabricated_trajectory_samples": int(builder.fabricated_field_positions),
        "recovery_events_observed_gap": len(builder.recovery_latencies_frames),
        "recovery_latency_frames_median": _median([float(v) for v in builder.recovery_latencies_frames]),
        "recovery_latency_frames_p90": _percentile([float(v) for v in builder.recovery_latencies_frames], 90.0),
        "recovery_latency_frames_max": (
            int(max(builder.recovery_latencies_frames)) if builder.recovery_latencies_frames else 0
        ),
        "projection_recovery_events": len(builder.projection_recovery_latencies_frames),
        "projection_recovery_latency_frames_max": (
            int(max(builder.projection_recovery_latencies_frames))
            if builder.projection_recovery_latencies_frames
            else 0
        ),
        "absolute_yardline_violations": int(abs_violations),
        "measurements_rejected_total": int(builder.measurements_rejected),
        "mean_runtime_ms_per_frame": round(runtime_ms / max(1, n_frames), 3),
        "mean_runtime_ms_per_sample": round(runtime_ms / max(1, len(samples_all)), 3),
        "image_space_ema_field_err_median_yd": None,  # filled by caller (needs fixture tables)
        "_uncertainty_d2": {
            "smoothed": mahal_smoothed,
            "raw_covariance": mahal_raw_cov,
        },
    }


def _run_sequence(
    seq: Dict[str, Any],
    base_img: np.ndarray,
    base_cal: CalibrationResult,
    *,
    builder_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one sequence end-to-end and score it.

    ``builder_kwargs`` exists so A/B ablations construct their variant through the
    same code path as the shipped configuration (no monkeypatching of module globals).
    """
    frames, frame_cals, fixtures, gt_table, routes, jitter_stats = _build_sequence(
        seq, base_img, base_cal
    )
    n_frames = int(seq["num_frames"])
    fps = float(seq["fps"])

    detector = FixturePlayerDetector(fixtures)
    classifier = TorsoTeamClassifier()
    tracker = PlayerTracker(max_missed_frames=3, projector=FieldProjector())
    # The harness opts in to label passthrough here. The trajectory layer's default is
    # empty, so production call sites cannot leak fixture labels into a trajectory.
    builder = PlayerTrajectoryBuilder(
        fps=fps,
        provenance_passthrough_keys=("gt_id", "scenario", "note"),
        **dict(builder_kwargs or {}),
    )

    samples_all: List[Any] = []
    t_wall0 = time.perf_counter()
    for t in range(n_frames):
        dets = detector.detect(frames[t], frame_id=t)
        teams = classifier.classify_detections(frames[t], dets)
        tracks = tracker.update(
            dets, frame_id=t, team_assignments=teams, calibration=frame_cals[t]
        )
        samples_all.extend(builder.update(tracks, frame_id=t, calibration=frame_cals[t]))
    runtime_ms = (time.perf_counter() - t_wall0) * 1000.0

    trajectories = builder.finalize()
    scores = _score_samples(
        seq,
        samples_all,
        gt_table,
        trajectories,
        builder,
        n_frames=n_frames,
        runtime_ms=runtime_ms,
        jitter_stats=jitter_stats,
        max_gap_frames=int(builder.max_gap_frames),
    )
    scores["image_space_ema_field_err_median_yd"] = _image_space_ema_field_error(
        seq, fixtures, frame_cals, gt_table
    )
    # Raw Mahalanobis values stay in memory (used for the TRAIN/VAL-fitted uncertainty
    # counterfactual); they are not written into the benchmark JSON.
    d2 = scores.pop("_uncertainty_d2", {"smoothed": [], "raw_covariance": []})
    extras = {
        "trajectories": trajectories,
        "gt_table": gt_table,
        "frame_cals": frame_cals,
        "routes": routes,
        "seq": seq,
        "jitter_stats": jitter_stats,
        "uncertainty_d2": d2,
    }
    return scores, extras


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_SUM_KEYS = (
    "n_players",
    "n_tracks",
    "n_samples",
    "all_matched_track_samples",
    "non_dominant_track_samples",
    "non_dominant_track_samples_total",
    "dominant_track_samples_total",
    "unattributed_samples_total",
    "dominant_track_samples_without_position",
    "excluded_gt_player_frames",
    "fragmentation_taxonomy_total",
    "fragmentation_n_matched_tracks",
    "fragmentation_dominant_tracks_total",
    "fragmentation_fragment_tracks_excluded_from_dominant",
    "image_space_plausibility_flags_total",
    "expected_samples",
    "dominant_samples",
    "measured_dominant_samples",
    "inferred_dominant_samples",
    "positioned_samples",
    "dead_reckoning_samples",
    "accepted_measurement_samples",
    "post_reinit_measurement_samples",
    "dead_reckoning_samples_from_rejection",
    "dead_reckoning_samples_from_missing_measurement",
    "reinit_after_rejection_measurement_samples",
    "outliers_injected",
    "outliers_rejected_by_field_gate",
    "outlier_events_refused_by_projection_gate",
    "outlier_events_spawning_spurious_track",
    "outlier_events_absorbed_into_track",
    "spurious_tracks_from_outliers",
    "gate_evaluated_samples",
    "false_rejections",
    "gate_warmup_measurements",
    "filter_reinitializations",
    "samples_with_position_in_unknown_geometry",
    "image_space_jumps_flagged",
    "reinitialized_after_gap",
    "reinitialized_after_rejections",
    "acceleration_clipped_samples",
    "speed_clipped_samples",
    "field_identity_changes",
    "id_switches",
    "id_switches_active_swap",
    "id_switches_post_reinit",
    "track_fragmentations",
    "observed_dominant_samples",
    "inferred_dominant_samples",
    "recovery_events_observed_gap",
    "projection_recovery_events",
    "fabricated_field_positions",
    "fabricated_trajectory_samples",
    "absolute_yardline_violations",)

# Float-valued sums (yards). These were previously integer-truncated by the count loop above,
# which is a reporting defect: path lengths are not counts. Kept separate so the count loop can
# stay strictly integral.
_SUM_FLOAT_KEYS = (
    "smoothed_path_length_yd",
    "raw_projection_path_length_yd",
    "gt_path_length_yd",
)


def _aggregate_split(seq_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"num_sequences": len(seq_scores), "num_frames": 0}
    for key in _SUM_KEYS:
        agg[key] = int(sum(int(s.get(key) or 0) for s in seq_scores))
    for key in _SUM_FLOAT_KEYS:
        agg[key] = round(float(sum(float(s.get(key) or 0.0) for s in seq_scores)), 4)
    agg["num_frames"] = int(sum(int(s["num_frames"]) for s in seq_scores))
    agg["track_completeness"] = round(
        agg["dominant_samples"] / max(1, agg["expected_samples"]), 4
    )
    agg["measured_completeness"] = round(
        agg["measured_dominant_samples"] / max(1, agg["expected_samples"]), 4
    )

    def _pool(field: str) -> List[float]:
        vals: List[float] = []
        for s in seq_scores:
            if s.get(field) is not None:
                vals.append(float(s[field]))
        return vals

    for field in ("field_pos_err_median_yd", "raw_field_pos_err_median_yd", "field_pos_err_p90_yd"):
        vals = _pool(field)
        agg[field] = round(float(np.median(vals)), 4) if vals else None
    for field in (
        "field_pos_err_rmse_yd",
        "velocity_rmse_yd_s",
        "accepted_measurement_err_rmse_yd",
        "post_reinit_measurement_err_rmse_yd",
        "reinit_after_rejection_measurement_err_rmse_yd",
        "dead_reckoning_err_from_rejection_rmse_yd",
        "dead_reckoning_err_from_missing_measurement_rmse_yd",
    ):
        vals = _pool(field)
        agg[field] = round(float(np.sqrt(np.mean(np.square(vals)))), 4) if vals else None
    for field in (
        "direction_err_median_deg",
        "recovery_latency_frames_median",
        "recovery_latency_frames_p90",
        "motion_preservation_ratio_vs_raw",
        "motion_preservation_ratio_vs_gt",
        "gt_accel_mag_median_yd_s2",
        "gt_speed_median_yd_s",
        "speed_err_median_yd_s",
        "accel_mag_err_median_yd_s2",
        "dead_reckoning_err_median_yd",
        "dead_reckoning_err_p90_yd",
        "accepted_measurement_err_median_yd",
        "accepted_measurement_err_p90_yd",
        "post_reinit_measurement_err_median_yd",
        "post_reinit_measurement_err_p90_yd",
        "reinit_after_rejection_measurement_err_median_yd",
        "dead_reckoning_err_from_rejection_median_yd",
        "dead_reckoning_err_from_rejection_p90_yd",
        "dead_reckoning_err_from_missing_measurement_median_yd",
        "dead_reckoning_err_from_missing_measurement_p90_yd",
        "spurious_track_field_err_median_yd",
        "all_matched_track_field_err_median_yd",
        "non_dominant_track_field_err_median_yd",
        "mean_sigma_major_yd",
        "mean_sigma_minor_yd",
        "image_space_ema_field_err_median_yd",
    ):
        vals = _pool(field)
        agg[field] = round(float(np.median(vals)), 4) if vals else None

    if agg["raw_field_pos_err_median_yd"] and agg["field_pos_err_median_yd"] is not None:
        agg["smoothing_error_reduction_pct"] = round(
            100.0
            * (
                1.0
                - float(agg["field_pos_err_median_yd"])
                / max(1e-9, float(agg["raw_field_pos_err_median_yd"]))
            ),
            2,
        )
    else:
        agg["smoothing_error_reduction_pct"] = None

    agg["motion_preservation_ratio_vs_raw"] = (
        round(agg["smoothed_path_length_yd"] / agg["raw_projection_path_length_yd"], 4)
        if agg["raw_projection_path_length_yd"] > 0
        else None
    )
    agg["motion_preservation_ratio_vs_gt"] = (
        round(agg["smoothed_path_length_yd"] / agg["gt_path_length_yd"], 4)
        if agg["gt_path_length_yd"] > 0
        else None
    )
    agg["observed_sample_ratio"] = round(
        agg["observed_dominant_samples"] / max(1, agg["dominant_samples"]), 4
    )
    agg["inferred_sample_ratio"] = round(
        agg["inferred_dominant_samples"] / max(1, agg["dominant_samples"]), 4
    )
    agg["recovery_latency_frames_max"] = int(
        max((int(s["recovery_latency_frames_max"] or 0) for s in seq_scores), default=0)
    )
    agg["projection_recovery_latency_frames_max"] = int(
        max((int(s["projection_recovery_latency_frames_max"] or 0) for s in seq_scores), default=0)
    )
    agg["outlier_rejection_rate"] = (
        round(agg["outliers_rejected_by_field_gate"] / agg["outliers_injected"], 4)
        if agg["outliers_injected"]
        else None
    )
    agg["outlier_handling_rate"] = (
        round(
            (agg["outliers_injected"] - agg["outlier_events_absorbed_into_track"])
            / agg["outliers_injected"],
            4,
        )
        if agg["outliers_injected"]
        else None
    )
    agg["false_rejection_rate"] = (
        round(agg["false_rejections"] / max(1, agg["gate_evaluated_samples"]), 6)
        if agg["gate_evaluated_samples"]
        else None
    )
    state_counts = {"calibrated": 0, "propagated": 0, "unknown": 0}
    positioned_by_state = {"calibrated": 0, "propagated": 0, "unknown": 0}
    mahal: List[float] = []
    cov68: List[float] = []
    cov95: List[float] = []
    for s in seq_scores:
        for k, v in s["geometry_state_sample_counts"].items():
            state_counts[k] = state_counts.get(k, 0) + int(v)
        for k, v in s["positioned_by_geometry_state"].items():
            positioned_by_state[k] = positioned_by_state.get(k, 0) + int(v)
        if s.get("mean_mahalanobis_d2") is not None:
            mahal.append(float(s["mean_mahalanobis_d2"]))
        if s.get("coverage_68_pct") is not None:
            cov68.append(float(s["coverage_68_pct"]))
        if s.get("coverage_95_pct") is not None:
            cov95.append(float(s["coverage_95_pct"]))
    agg["geometry_state_sample_counts"] = state_counts
    agg["positioned_by_geometry_state"] = positioned_by_state
    agg["mean_mahalanobis_d2"] = round(float(np.median(mahal)), 3) if mahal else None
    agg["coverage_68_pct"] = round(float(np.median(cov68)), 2) if cov68 else None
    agg["coverage_95_pct"] = round(float(np.median(cov95)), 2) if cov95 else None
    agg["mean_runtime_ms_per_frame"] = round(
        float(np.mean([float(s["mean_runtime_ms_per_frame"]) for s in seq_scores])), 3
    )

    # ---- nested accounting blocks ------------------------------------------
    # Split-level values are sums of counted events, medians of per-sequence values,
    # or ratios recomputed from the sums. Nothing is re-fit or re-tuned here.
    agg["fragmentation_taxonomy"] = {
        category: int(sum(int(s["fragmentation_taxonomy"].get(category, 0)) for s in seq_scores))
        for category in FRAGMENTATION_CATEGORIES
    }
    agg["identity_contamination"] = {
        "tracks_with_multiple_gt_labels": int(
            sum(int(s["identity_contamination"]["tracks_with_multiple_gt_labels"]) for s in seq_scores)
        ),
        "tracks_without_any_label": int(
            sum(int(s["identity_contamination"]["tracks_without_any_label"]) for s in seq_scores)
        ),
        "contaminated_track_ids": sorted(
            tid
            for s in seq_scores
            for tid in (int(k) for k in s["identity_contamination"]["tracks"])
        ),
    }
    agg["fragmentation_max_tracks_per_gt_player"] = int(
        max((int(s["fragmentation_max_tracks_per_gt_player"]) for s in seq_scores), default=0)
    )
    agg["dominant_fraction_of_gt_player_frames"] = round(
        agg["dominant_samples"] / max(1, agg["expected_samples"]), 4
    )

    rej_int_keys = (
        "clean_samples_gate_evaluated",
        "clean_samples_accepted",
        "clean_samples_accepted_during_warmup",
        "clean_samples_rejected",
        "clean_samples_warmup_bypassed",
        "corrupted_samples_evaluated",
        "corrupted_samples_rejected",
        "corrupted_samples_accepted",
        "dead_reckoned_samples_after_false_rejection",
        "rejections_by_innovation_gate",
        "rejections_by_image_space_plausibility_gate",
        "rejections_all_tracks",
        "rejections_on_dominant_tracks",
        "rejections_by_innovation_gate_on_dominant_tracks",
        "rejections_by_plausibility_gate_on_dominant_tracks",
        "image_space_plausibility_flags",
        "unpositioned_samples",
    )
    rej = {
        key: int(sum(int(s["rejection_accounting"].get(key, 0) or 0) for s in seq_scores))
        for key in rej_int_keys
    }
    unpositioned_reasons: Dict[str, int] = {}
    for s in seq_scores:
        for reason, count in s["rejection_accounting"]["unpositioned_by_reason"].items():
            unpositioned_reasons[reason] = unpositioned_reasons.get(reason, 0) + int(count)
    rej["unpositioned_by_reason"] = unpositioned_reasons
    rej["false_rejection_rate"] = (
        round(rej["clean_samples_rejected"] / rej["clean_samples_gate_evaluated"], 6)
        if rej["clean_samples_gate_evaluated"]
        else None
    )
    rej["false_rejection_rate_including_warmup"] = (
        round(
            rej["clean_samples_rejected"]
            / max(1, rej["clean_samples_gate_evaluated"] + rej["clean_samples_warmup_bypassed"]),
            6,
        )
        if (rej["clean_samples_gate_evaluated"] + rej["clean_samples_warmup_bypassed"])
        else None
    )
    rej["true_rejection_rate_on_corrupted"] = (
        round(rej["corrupted_samples_rejected"] / rej["corrupted_samples_evaluated"], 6)
        if rej["corrupted_samples_evaluated"]
        else None
    )
    rej["false_acceptance_rate_on_corrupted"] = (
        round(rej["corrupted_samples_accepted"] / rej["corrupted_samples_evaluated"], 6)
        if rej["corrupted_samples_evaluated"]
        else None
    )
    # Denominator provenance travels with the split-level rates, not only with the per-sequence
    # block, so a reader of the split table cannot quote a rate without its base.
    rej["denominators"] = {
        "false_rejection_rate": "clean_samples_gate_evaluated (clean samples the gate actually evaluated)",
        "false_rejection_rate_including_warmup": (
            "clean_samples_gate_evaluated + clean_samples_warmup_bypassed"
        ),
        "true_rejection_rate_on_corrupted": (
            "corrupted_samples_evaluated (dominant-track injected events with a raw projection)"
        ),
        "false_acceptance_rate_on_corrupted": (
            "corrupted_samples_evaluated (same denominator as true_rejection_rate_on_corrupted)"
        ),
        "geometry_refusal": (
            "not reported as a rate: unpositioned_by_reason counts are absolute sample counts "
            "over the split's emitted samples"
        ),
    }
    rej["plausibility_gate_can_reject_measurements"] = bool(
        rej["rejections_by_image_space_plausibility_gate"]
    )
    for key in (
        "median_gt_error_of_falsely_rejected_samples_yd",
        "median_gate_statistic_of_false_rejections",
        "median_gate_statistic_of_corrupted_rejections",
    ):
        vals = [
            float(s["rejection_accounting"][key])
            for s in seq_scores
            if s["rejection_accounting"].get(key) is not None
        ]
        rej[key] = round(float(np.median(vals)), 4) if vals else None
    rej["note"] = (
        "Split-level values are sums over sequences; the three median_* fields are medians "
        "of per-sequence medians (documented approximation, not a pooled median)."
    )
    agg["rejection_accounting"] = rej

    unc = {
        "status": UNCERTAINTY_STATUS,
        "note": UNCERTAINTY_NOTE,
        "coverage_evaluated_samples": int(
            sum(int(s["uncertainty_diagnostics"].get("coverage_evaluated_samples", 0) or 0) for s in seq_scores)
        ),
    }
    unc["coverage_excludes_injected_events"] = True
    unc["injected_accepted_samples_excluded_from_coverage"] = int(
        sum(
            int(s["uncertainty_diagnostics"].get("injected_accepted_samples_excluded_from_coverage") or 0)
            for s in seq_scores
        )
    )
    for key in (
        "coverage_68_pct_smoothed_covariance",
        "coverage_95_pct_smoothed_covariance",
        "coverage_68_pct_smoothed_covariance_including_injected_events",
        "coverage_95_pct_smoothed_covariance_including_injected_events",
        "coverage_68_pct_raw_measurement_covariance_including_injected_events",
        "median_mahalanobis_d2_of_injected_accepted_samples",
        "empirical_position_err_median_yd",
        "empirical_position_err_rmse_yd",
        "assumed_sigma_major_median_yd",
        "median_mahalanobis_distance_smoothed",
        "median_mahalanobis_distance_raw_measurement_covariance",
        "mean_gate_statistic_accepted_clean",
        "median_gate_statistic_accepted_clean",
        "mean_gate_statistic_corrupted_rejected",
        "covariance_scale_needed_k2_smoothed",
        "covariance_scale_needed_k2_raw_measurement_covariance",
        "coverage_68_pct_raw_measurement_covariance",
        "coverage_95_pct_raw_measurement_covariance",
    ):
        vals = [
            float(s["uncertainty_diagnostics"][key])
            for s in seq_scores
            if s["uncertainty_diagnostics"].get(key) is not None
        ]
        unc[key] = round(float(np.median(vals)), 4) if vals else None
    for pct in ("68", "95"):
        key = f"coverage_{pct}_pct_smoothed_covariance"
        vals = [
            float(s["uncertainty_diagnostics"][key])
            for s in seq_scores
            if s["uncertainty_diagnostics"].get(key) is not None
        ]
        unc[key] = round(float(np.median(vals)), 2) if vals else None
    for pct in ("68", "95"):
        key = f"predicted_coverage_from_jitter_vs_model_{pct}_pct_median_over_sequences"
        vals = [
            float(s["uncertainty_diagnostics"]["jitter_vs_model_prediction"][
                f"predicted_coverage_{pct}_pct"
            ])
            for s in seq_scores
            if s["uncertainty_diagnostics"].get("jitter_vs_model_prediction")
        ]
        unc[key] = round(float(np.median(vals)), 2) if vals else None
    for state in ("calibrated", "propagated"):
        for pct in ("68", "95"):
            key = f"coverage_{pct}_pct_by_geometry_state"
            vals = [
                float(s["uncertainty_diagnostics"][key][state])
                for s in seq_scores
                if s["uncertainty_diagnostics"].get(key, {}).get(state) is not None
            ]
            unc.setdefault(key, {})[state] = round(float(np.median(vals)), 2) if vals else None
    for pct in ("68", "95"):
        key = f"coverage_{pct}_pct_smoothed_covariance"
        vals = [
            float(s["uncertainty_diagnostics"][key])
            for s in seq_scores
            if s["uncertainty_diagnostics"].get(key) is not None
        ]
        unc[key] = round(float(np.median(vals)), 2) if vals else None
    for pct in ("68", "95"):
        key = f"coverage_{pct}_pct_by_covariance_inflation"
        for group in ("inflated", "not_inflated"):
            vals = [
                float(s["uncertainty_diagnostics"][key][group])
                for s in seq_scores
                if s["uncertainty_diagnostics"].get(key, {}).get(group) is not None
            ]
            unc.setdefault(key, {})[group] = round(float(np.median(vals)), 2) if vals else None
    inflation_samples: Dict[str, int] = {}
    for s in seq_scores:
        for group, count in s["uncertainty_diagnostics"].get(
            "coverage_samples_by_covariance_inflation", {}
        ).items():
            inflation_samples[group] = inflation_samples.get(group, 0) + int(count)
    unc["coverage_samples_by_covariance_inflation"] = inflation_samples
    state_samples: Dict[str, int] = {}
    for s in seq_scores:
        for state, count in s["uncertainty_diagnostics"].get("coverage_samples_by_geometry_state", {}).items():
            state_samples[state] = state_samples.get(state, 0) + int(count)
    unc["coverage_samples_by_geometry_state"] = state_samples
    predicted_values: Dict[str, List[float]] = {"68": [], "95": []}
    for s in seq_scores:
        prediction = s["uncertainty_diagnostics"].get("jitter_vs_model_prediction")
        if prediction:
            predicted_values["68"].append(float(prediction["predicted_coverage_68_pct"]))
            predicted_values["95"].append(float(prediction["predicted_coverage_95_pct"]))
    unc["predicted_coverage_from_jitter_vs_model_68_pct_median_over_sequences"] = (
        round(float(np.median(predicted_values["68"])), 2) if predicted_values["68"] else None
    )
    unc["predicted_coverage_from_jitter_vs_model_95_pct_median_over_sequences"] = (
        round(float(np.median(predicted_values["95"])), 2) if predicted_values["95"] else None
    )
    unc["note"] = (
        "Split-level coverage/uncertainty values are medians of per-sequence values (documented "
        "approximation); jitter/model pixel sigmas are weighted by fixture footpoint count. "
        "Coverage is measured on CLEAN accepted measurements; the *_including_injected_events "
        "variants keep the pre-remediation view (corrupted-but-accepted samples included) visible."
    )
    jitter_pool = {
        key: int(sum(int(s.get("footpoint_jitter_stats", {}).get("n_fixture_footpoints", 0) or 0) for s in seq_scores))
        for key in ("n_fixture_footpoints",)
    }
    n_fp = jitter_pool["n_fixture_footpoints"]
    if n_fp:
        for key in (
            "observed_jitter_std_u_px",
            "observed_jitter_std_v_px",
            "assumed_footpoint_sigma_u_px",
            "assumed_footpoint_sigma_v_px",
        ):
            total = sum(
                float(s["footpoint_jitter_stats"].get(key, 0.0))
                * int(s["footpoint_jitter_stats"].get("n_fixture_footpoints", 0))
                for s in seq_scores
            )
            unc[key] = round(total / n_fp, 4)
    unc["footpoint_jitter_stats_note"] = (
        "observed_* and assumed_* pool the per-sequence values weighted by the number of "
        "fixture footpoints; see uncertainty_diagnostics per sequence for the full derivation."
    )
    agg["uncertainty_diagnostics"] = unc
    # Identity contamination is a track-level property; the split total is the sum of the
    # per-sequence counts of tracks that carry more than one player label.
    agg["identity_contamination"] = {
        "tracks_with_multiple_gt_labels": int(
            sum(int(s["identity_contamination"]["tracks_with_multiple_gt_labels"]) for s in seq_scores)
        ),
        "track_ids": sorted(
            {int(tid) for s in seq_scores for tid in s["identity_contamination"]["tracks"]}
        ),
        "note": (
            "Tracks whose samples carry more than one ground-truth label. Per-sample attribution "
            "(CLEAR convention) keeps the identity metrics honest; these rows exist so the "
            "contamination itself stays visible."
        ),
    }
    # Fragmentation visibility: excluded (fragment) tracks broken down by origin category, so
    # "not the dominant track" and "how it came to exist" are both reported.
    excluded_by_origin: Dict[str, int] = {}
    counts_by_origin: Dict[str, int] = {}
    for s in seq_scores:
        for category, count in s.get("fragmentation_excluded_by_origin", {}).items():
            excluded_by_origin[category] = excluded_by_origin.get(category, 0) + int(count)
        for category, count in s.get("fragmentation_taxonomy", {}).items():
            counts_by_origin[category] = counts_by_origin.get(category, 0) + int(count)
    agg["fragmentation_counts_by_origin"] = counts_by_origin
    agg["fragmentation_excluded_by_origin"] = excluded_by_origin
    agg["fragmentation_excluded_from_dominant_total"] = int(sum(excluded_by_origin.values()))
    agg["fragmentation_matched_tracks_total"] = int(
        sum(int(s.get("fragmentation_n_matched_tracks") or 0) for s in seq_scores)
    )
    return agg


def _determinism_check(seq_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-run one sequence per split and confirm metric reproducibility.

    Wall-clock runtime fields are excluded; every other score key (including the new
    track/fragmentation/rejection/uncertainty blocks) must reproduce bit-for-bit.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # One sequence from EVERY split, so reproducibility is demonstrated on TRAIN and VAL as
    # well as on the frozen TEST split.
    targets = [
        next(s for s in seq_scores if s["scenario"] == "linear_mixed_speeds"),
        next(s for s in seq_scores if s["scenario"] == "jump_outliers"),
        next(s for s in seq_scores if s["scenario"] == "camera_cut_refusal_and_recovery"),
    ]
    ignore = {"mean_runtime_ms_per_frame", "mean_runtime_ms_per_sample"}
    per_sequence: List[Dict[str, Any]] = []
    all_ok = True
    for target in targets:
        seq_cfg = next(
            s
            for s in manifest["trajectory_sequences"]
            if s["sequence_id"] == target["sequence_id"]
        )
        base_img = cv2.imread(str(_resolve_base_image(seq_cfg["base_image_path"])))
        base_cal = calibrate_frame(base_img, x_start_yd=float(seq_cfg["x_start_yd"]))
        again, _ = _run_sequence(seq_cfg, base_img, base_cal)
        diff_keys = [
            k
            for k in sorted(set(target) | set(again))
            if k not in ignore and target.get(k) != again.get(k)
        ]
        all_ok = all_ok and not diff_keys
        per_sequence.append(
            {
                "sequence_id": target["sequence_id"],
                "split": target["split"],
                "deterministic": len(diff_keys) == 0,
                "differing_keys": diff_keys,
            }
        )
    return {
        "sequence_id": targets[0]["sequence_id"],
        "deterministic": all_ok,
        "differing_keys": [k for r in per_sequence for k in r["differing_keys"]],
        "per_sequence": per_sequence,
        "note": (
            "wall-clock runtime fields excluded; one sequence from each split (TRAIN, VAL, "
            "frozen TEST) re-run through the identical code path"
        ),
    }


# ---------------------------------------------------------------------------
# Real-frame single-frame integration smoke test (no ground truth)
# ---------------------------------------------------------------------------
def _builder_ablation(
    manifest: Dict[str, Any],
    img_cache: Dict[str, Any],
    cal_cache: Dict[str, Any],
    *,
    variants: Dict[str, Dict[str, Any]],
    metrics: Sequence[str],
    sum_metrics: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run builder-parameter variants through the *same* code path, TRAIN + VAL only.

    TEST is deliberately never consulted here: ablation is a tuning/diagnostic
    activity and the frozen split must not inform any constant. Metrics named in
    ``sum_metrics`` are summed across the split's sequences (event counts); the rest
    are medians of the per-sequence values.
    """
    results: Dict[str, Any] = {}
    for label, kwargs in variants.items():
        per_split: Dict[str, Any] = {}
        for split in ("train", "val"):
            pooled: Dict[str, List[float]] = {m: [] for m in metrics}
            for seq in [s for s in manifest["trajectory_sequences"] if s["split"] == split]:
                path = seq["base_image_path"]
                scores, _ = _run_sequence(
                    seq, img_cache[path], cal_cache[path], builder_kwargs=kwargs
                )
                for metric in metrics:
                    value = scores.get(metric)
                    if value is not None:
                        pooled[metric].append(float(value))
            per_split[split] = {}
            for metric, values in pooled.items():
                if metric in sum_metrics:
                    # Sums keep the metric's own units: counts stay integers, distances stay
                    # floats (integer-truncating a path length would corrupt the ratio below).
                    total = float(sum(values))
                    per_split[split][metric] = (
                        int(round(total)) if all(float(v).is_integer() for v in values) else round(total, 4)
                    )
                else:
                    per_split[split][metric] = (
                        round(float(np.median(values)), 4) if values else None
                    )
        results[label] = per_split
    return results


def _gate_warmup_ablation(
    manifest: Dict[str, Any],
    img_cache: Dict[str, Any],
    cal_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure the track-initiation warm-up A/B from the implementation (TRAIN + VAL only).

    The shipped rule bypasses the innovation gate for the first two accepted updates of a
    track (standard track initiation: the velocity state is not yet estimable). This block
    *computes* that comparison instead of quoting numbers from an earlier manual run, so the
    reported values can never drift away from the code. The gate THRESHOLD is unchanged.
    """
    variants = {
        "shipped_warmup_2_updates": {"gate_warmup_updates": 2},
        "gate_from_2nd_update": {"gate_warmup_updates": 0},
    }
    metrics = ("false_rejections", "gate_evaluated_samples")
    raw = _builder_ablation(
        manifest,
        img_cache,
        cal_cache,
        variants=variants,
        metrics=metrics,
        sum_metrics=metrics,
    )
    results: Dict[str, Any] = {}
    for split in ("train", "val"):
        shipped = raw["shipped_warmup_2_updates"][split]
        ungated = raw["gate_from_2nd_update"][split]
        results[split] = {
            "gate_from_2nd_update_false_rejections": ungated["false_rejections"],
            "gate_from_2nd_update_evaluated": ungated["gate_evaluated_samples"],
            "shipped_warmup_false_rejections": shipped["false_rejections"],
            "shipped_warmup_evaluated": shipped["gate_evaluated_samples"],
        }
    return {
        "note": (
            "Computed by run_phase4_benchmark() from the shipped implementation, TRAIN/VAL only "
            "(TEST is not consulted, so warm-up selection cannot leak into the frozen split). The "
            "innovation-gate threshold is unchanged; the shipped rule delays the gate until a track "
            "has two accepted updates (M-of-N track initiation)."
        ),
        "computed_by": "benchmarks/evaluate_phase4_trajectories.py::_gate_warmup_ablation",
        "test_split_consulted": False,
        "results": results,
    }


def _process_noise_ablation(
    manifest: Dict[str, Any],
    img_cache: Dict[str, Any],
    cal_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """A/B the only constant that could hide real motion, on TRAIN + VAL only.

    The constant-velocity filter's process-noise scale sets how strongly a
    measurement can move the state. A too-stiff setting would visibly shorten
    travelled distance relative to ground truth. TEST is deliberately excluded
    so this comparison cannot be used to tune on the frozen split.
    """
    metrics = ("field_pos_err_median_yd", "smoothed_path_length_yd", "gt_path_length_yd", "mean_sigma_major_yd")
    raw = _builder_ablation(
        manifest,
        img_cache,
        cal_cache,
        variants={"process_accel_std_yd_s2=0.5": {"process_accel_std_yd_s2": 0.5}},
        metrics=metrics,
        # Path lengths are SPLIT TOTALS here too (same convention as the split tables), so the
        # preservation ratio is a ratio of totals and not of two per-sequence medians.
        sum_metrics=("smoothed_path_length_yd", "gt_path_length_yd"),
    )
    results = raw["process_accel_std_yd_s2=0.5"]
    for split, values in results.items():
        values["motion_preservation_ratio_vs_gt"] = round(
            float(values["smoothed_path_length_yd"]) / max(1e-9, float(values["gt_path_length_yd"])), 4
        )
    return {
        "note": (
            "Comparison of the only constant that could attenuate real motion (the filter's process-noise "
            "acceleration scale). Measured on TRAIN + VAL only; TEST is excluded so this cannot be used to "
            "tune the frozen split. Shipped value remains 5.0."
        ),
        "variants": {"process_accel_std_yd_s2=0.5": results},
        "test_split_consulted": False,
    }


def _real_frame_smoke() -> Dict[str, Any]:
    frame_path = resolve_nfl_frame("nfl-game-broadcast-screenshot-1st-and-10-5.jpg")
    if frame_path is None:
        return {
            "frame_id": "rf_01_sea_sf_smoke",
            "skipped": True,
            "skip_reason": (
                "real NFL frame asset unavailable; set FOOTBALL_VISION_NFL_FRAMES to the "
                "directory containing it (assets are not vendored in the repository)"
            ),
            "evaluation_scope": "single_frame_integration_smoke_test_no_ground_truth_trajectory",
            "quantitative_ground_truth_available": False,
        }
    img = cv2.imread(str(frame_path))
    cal = calibrate_frame(img, x_start_yd=15.0)
    detector = TurfContrastPlayerDetector()
    dets = detector.detect(img, frame_id=0)
    teams = TorsoTeamClassifier().classify_detections(img, dets)
    tracker = PlayerTracker()
    tracks = tracker.update(dets, frame_id=0, team_assignments=teams, calibration=cal)
    builder = PlayerTrajectoryBuilder(fps=30.0)
    samples = builder.update(tracks, frame_id=0, calibration=cal)

    return {
        "frame_id": "rf_01_sea_sf_smoke",
        "skipped": False,
        "frame_path": str(frame_path),
        "detector_implementation": detector.detector_name,
        "evaluation_scope": "single_frame_integration_smoke_test_no_ground_truth_trajectory",
        "quantitative_ground_truth_available": False,
        "calibration_success": bool(cal.success),
        "x_coord_mode": cal.x_coord_mode,
        "detections": len(dets),
        "tracks": len(tracks),
        "trajectory_samples": len(samples),
        "geometry_state_counts": {
            "calibrated": sum(1 for s in samples if s.geometry_state == "calibrated"),
            "propagated": sum(1 for s in samples if s.geometry_state == "propagated"),
            "unknown": sum(1 for s in samples if s.geometry_state == "unknown"),
        },
        "samples_with_field_position": sum(1 for s in samples if s.field_position is not None),
        "samples_with_absolute_yardline": sum(1 for s in samples if s.absolute_yardline is not None),
        "fabricated_field_positions": int(builder.fabricated_field_positions),
        "fabricated_trajectory_samples": int(builder.fabricated_field_positions),
        "recovery_events_observed_gap": len(builder.recovery_latencies_frames),
        "recovery_latency_frames_median": _median([float(v) for v in builder.recovery_latencies_frames]),
        "recovery_latency_frames_p90": _percentile([float(v) for v in builder.recovery_latencies_frames], 90.0),
        "recovery_latency_frames_max": (
            int(max(builder.recovery_latencies_frames)) if builder.recovery_latencies_frames else 0
        ),
        "projection_recovery_events": len(builder.projection_recovery_latencies_frames),
        "projection_recovery_latency_frames_max": (
            int(max(builder.projection_recovery_latencies_frames))
            if builder.projection_recovery_latencies_frames
            else 0
        ),
        "note": (
            "Single-frame smoke test only: one still frame cannot exercise multi-frame "
            "trajectory, smoothing, camera motion, or occlusion behaviour."
        ),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def _uncertainty_calibration_analysis(
    seq_scores: List[Dict[str, Any]],
    extras_by_seq: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """TRAIN/VAL-fitted covariance-scale analysis, recorded but **not applied**.

    Procedure: (1) fit a single scalar covariance-scale ``k`` from the median Mahalanobis
    statistic of the TRAIN + VAL sequences only, (2) record the k and what TEST coverage
    would be under it, (3) change nothing. The shipped covariance stays exactly as
    published, so the frozen TEST numbers above remain the reference. This exists so the
    under-coverage is documented with a number instead of an excuse - not so it can be
    hidden by inflating sigma until the intervals cover.
    """
    def _k2(split: str) -> Optional[float]:
        values: List[float] = []
        for s in seq_scores:
            if s["split"] != split:
                continue
            values.extend(extras_by_seq[s["sequence_id"]]["uncertainty_d2"]["smoothed"])
        if not values:
            return None
        return float(np.median(values)) / COVERAGE_68_MEDIAN_CHI2

    def _coverage_at_k2(split: str, scale: Optional[float]) -> Optional[float]:
        if not scale or scale <= 0.0:
            return None
        values: List[float] = []
        for s in seq_scores:
            if s["split"] != split:
                continue
            values.extend(extras_by_seq[s["sequence_id"]]["uncertainty_d2"]["smoothed"])
        if not values:
            return None
        corrected = [v / scale for v in values]
        return round(
            100.0 * float(np.mean([d <= COVERAGE_68_CHI2_2DOF for d in corrected])), 2
        )

    k2_train = _k2("train")
    k2_val = _k2("val")
    k2_fitted = float(np.median([v for v in (k2_train, k2_val) if v is not None])) if (k2_train or k2_val) else None
    return {
        "status": UNCERTAINTY_STATUS,
        "reporting_only": True,
        "applied_to_shipped_covariance": False,
        "shipped_covariance_unchanged": True,
        "fit_split": "train+val",
        "test_split_used_for_fitting": False,
        "covariance_scale_k2_train": round(k2_train, 4) if k2_train is not None else None,
        "covariance_scale_k2_val": round(k2_val, 4) if k2_val is not None else None,
        "covariance_scale_k2_fitted_on_train_val": (
            round(k2_fitted, 4) if k2_fitted is not None else None
        ),
        "coverage_note": (
            "A value of k2 > 1 means the empirical error is larger than the reported "
            "covariance (intervals too narrow). Fitting an inflation factor and shipping it "
            "would change the reported uncertainty and could change gate behaviour; that is a "
            "Phase-5 decision, so nothing was changed here."
        ),
        "test_coverage_68_pct_shipped": None,
        "test_coverage_68_pct_counterfactual_under_fitted_scale": _coverage_at_k2("test", k2_fitted),
        "counterfactual_note": (
            "TEST is evaluated with the TRAIN/VAL-fitted scale exactly once, for documentation. "
            "No threshold, constant, or reported value in this benchmark uses it."
        ),
    }


METRIC_SCOPES: Dict[str, List[str]] = {
    "model_observable_no_ground_truth": [
        "n_tracks",
        "n_samples",
        "geometry_state_sample_counts",
        "positioned_by_geometry_state",
        "samples_with_position_in_unknown_geometry",
        "fabricated_field_positions",
        "absolute_yardline_violations",
        "measurements_rejected_total",
        "filter_reinitializations",
        "image_space_plausibility_flags_total",
        "recovery_events_observed_gap",
        "projection_recovery_events",
        "gate_warmup_measurements",
        "track_table",
        "samples_per_track",
        "mean_runtime_ms_per_frame",
        "mean_runtime_ms_per_sample",
        "projection_status accounting (unpositioned_by_reason)",
        "accepted_measurement_samples",
        "post_reinit_measurement_samples",
        "dead_reckoning_samples_from_rejection",
        "dead_reckoning_samples_from_missing_measurement",
        "reinit_after_rejection_measurement_samples",
    ],
    "harness_gt_dependent": [
        "field_pos_err_median_yd",
        "field_pos_err_p90_yd",
        "field_pos_err_rmse_yd",
        "raw_field_pos_err_median_yd",
        "all_matched_track_field_err_median_yd",
        "non_dominant_track_field_err_median_yd",
        "speed_err_median_yd_s",
        "velocity_rmse_yd_s",
        "accel_mag_err_median_yd_s2",
        "direction_err_median_deg",
        "dead_reckoning_err_median_yd",
        "accepted_measurement_err_median_yd",
        "post_reinit_measurement_err_median_yd",
        "dead_reckoning_err_from_rejection_median_yd",
        "dead_reckoning_err_from_missing_measurement_median_yd",
        "coverage_68_pct",
        "coverage_95_pct",
        "id_switches",
        "id_switches_active_swap",
        "id_switches_post_reinit",
        "track_fragmentations",
        "track_completeness",
        "measured_completeness",
        "dominant_fraction_of_gt_player_frames",
        "smoothed_path_length_yd",
        "raw_projection_path_length_yd",
        "gt_path_length_yd",
        "motion_preservation_ratio_vs_raw",
        "motion_preservation_ratio_vs_gt",
        "fragmentation_taxonomy",
        "image_space_ema_field_err_median_yd",
    ],
    "harness_label_dependent": [
        "outliers_injected",
        "outliers_rejected_by_field_gate",
        "outlier_events_absorbed_into_track",
        "outlier_events_spawning_spurious_track",
        "outlier_events_refused_by_projection_gate",
        "false_rejections",
        "false_rejection_rate",
        "false_rejection_rate_including_warmup",
        "clean_samples_gate_evaluated",
        "clean_samples_accepted",
        "clean_samples_rejected",
        "corrupted_samples_evaluated",
        "corrupted_samples_rejected",
        "corrupted_samples_accepted",
        "true_rejection_rate_on_corrupted",
        "false_acceptance_rate_on_corrupted",
        "dead_reckoned_samples_after_false_rejection",
        "recovery_latency_frames_median",
        "recovery_latency_frames_max",
        "projection_recovery_latency_frames_max",
        "gate_warmup_ablation",
        "outlier_event_details",
    ],
    "note": [
        "harness_gt_dependent = requires ground-truth player coordinates to exist at all",
        "harness_label_dependent = requires harness event labels (injected outliers, dropouts) "
        "or map-side GT association, not model internals",
        "model_observable_no_ground_truth = computable from the emitted samples alone",
    ],
}


def run_phase4_benchmark() -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    img_cache: Dict[str, np.ndarray] = {}
    cal_cache: Dict[str, CalibrationResult] = {}

    seq_scores: List[Dict[str, Any]] = []
    extras_by_seq: Dict[str, Dict[str, Any]] = {}

    for seq in manifest["trajectory_sequences"]:
        path = seq["base_image_path"]
        if path not in img_cache:
            resolved = _resolve_base_image(path)
            img = cv2.imread(str(resolved))
            if img is None:
                raise FileNotFoundError(str(resolved))
            img_cache[path] = img
            cal_cache[path] = calibrate_frame(img, x_start_yd=float(seq["x_start_yd"]))
        scores, extras = _run_sequence(seq, img_cache[path], cal_cache[path])
        seq_scores.append(scores)
        extras_by_seq[seq["sequence_id"]] = extras

    splits = {
        sp: _aggregate_split([s for s in seq_scores if s["split"] == sp])
        for sp in ("train", "val", "test")
    }

    determinism = _determinism_check(seq_scores)
    warmup_ablation = _gate_warmup_ablation(manifest, img_cache, cal_cache)
    noise_ablation = _process_noise_ablation(manifest, img_cache, cal_cache)
    uncertainty_analysis = _uncertainty_calibration_analysis(seq_scores, extras_by_seq)
    uncertainty_analysis["test_coverage_68_pct_shipped"] = splits["test"][
        "coverage_68_pct"
    ]
    smoke = _real_frame_smoke()

    report = {
        "benchmark_version": "phase4_v1",
        "benchmark_kind": "synthetic_trajectory_benchmark",
        "benchmark_kind_note": (
            "Field-space trajectories are scored against SYNTHETIC deterministic ground-truth routes "
            "projected through real per-frame homographies. This is an engineering benchmark of the "
            "trajectory layer, not a measurement of NFL or broadcast tracking accuracy."
        ),
        "real_multiframe_trajectory_accuracy": "not_measured",
        "real_multiframe_trajectory_accuracy_note": (
            "Real multi-frame trajectory accuracy is not yet measured: no real broadcast clip with "
            "frame-level ground-truth trajectories is available in this project. Real frames are used "
            "for single-frame smoke/integration and geometry-refusal tests only."
        ),
        "calibration_layer_modified": False,
        "gate_warmup_ablation": warmup_ablation,
        "detector_implementation": "FixturePlayerDetector (fixture_detector_v1)",
        "image_detector_accuracy_status": "unmeasured",
        "image_detector_quantitative_metrics": None,
        "detector_metric_separation_note": (
            "Detection counts/precision/recall in this file are FIXTURE HARNESS pass-through statistics "
            "(the boxes are scheduled by the harness, not predicted from pixels). They must never be "
            "reported as player-image-detector accuracy; image-based detector accuracy is unmeasured."
        ),
        "acceleration_status": ACCELERATION_LABEL,
        "acceleration_note": ACCELERATION_NOTE,
        "uncertainty_status": UNCERTAINTY_STATUS,
        "uncertainty_note": UNCERTAINTY_NOTE,
        "uncertainty_calibration_analysis": uncertainty_analysis,
        "dominant_track_selection": DOMINANT_TRACK_SELECTION,
        "fragmentation_definitions": FRAGMENTATION_DEFINITIONS,
        "rejection_accounting_definition": REJECTION_ACCOUNTING_DEFINITION,
        "metric_scopes": METRIC_SCOPES,
        "benchmark_provenance_and_semantics": manifest["benchmark_provenance_and_semantics"],
        "frozen_parameters": manifest["frozen_parameters"],
        "split_protocol": manifest["split_protocol"],
        "splits": splits,
        "sequences": seq_scores,
        "determinism_check": determinism,
        "process_noise_ablation": noise_ablation,
        "real_frame_smoke_test": smoke,
        "limitations": [
            "Player motion is synthetic deterministic field-space ground truth projected through real per-frame homographies: an engineering fixture, not recorded NFL trajectories. Real multi-frame trajectory accuracy is NOT measured.",
            "Detection boxes are fixture boxes (fixture_detector_v1); image-based player detection accuracy remains unmeasured and no detection-accuracy number in this file is a detector metric.",
            "The reported covariance is an a priori engineering model and is NOT statistically calibrated; empirical coverage is reported per split and per geometry state, and the gap is explained (jitter vs assumed pixel noise) rather than tuned away.",
            "Acceleration is experimental: it is a clipped finite difference of the smoothed velocity, not a validated acceleration estimator.",
            "Fragmentation is split by a deterministic taxonomy (active association swap / post-expiration reinitialization / outlier-induced spawn / legitimate new track / unmatched). The taxonomy is exhaustive and asserted against the track count, but the categories are harness-visible states, not inferred internal tracker modes.",
            "Camera pan/zoom is applied as an exact synthetic image-space transform; real broadcast camera motion also changes perspective, rolling shutter, and motion blur.",
            "One single-frame real-frame smoke test (no ground truth) plus geometry-refusal/integration tests only: no real multi-frame broadcast clip was available in this phase.",
        ],
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase4_trajectory_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _render_overview(manifest, extras_by_seq, seq_scores, splits, out_dir)
    return report


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _render_overview(
    manifest: Dict[str, Any],
    extras_by_seq: Dict[str, Dict[str, Any]],
    seq_scores: List[Dict[str, Any]],
    splits: Dict[str, Any],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.6))

    # Panel 1: field-space trajectories, GT vs smoothed vs raw (camera pan sequence)
    ax = axes[0, 0]
    pan_seq = "traj_seq_03_val_camera_pan"
    ex = extras_by_seq[pan_seq]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for gid in sorted(ex["routes"]):
        route = ex["routes"][gid]
        ax.plot(route["x"], route["y"], ls="--", lw=3.4, color="black", alpha=0.35, zorder=1)
    for traj in ex["trajectories"]:
        rxs = [s.raw_field_position[0] for s in traj.samples if s.raw_field_position is not None]
        rys = [s.raw_field_position[1] for s in traj.samples if s.raw_field_position is not None]
        if rxs:
            ax.scatter(rxs, rys, s=9, color=colors[traj.track_id % 10], alpha=0.45, zorder=2)
        xs = [s.field_position[0] for s in traj.samples if s.field_position is not None]
        ys = [s.field_position[1] for s in traj.samples if s.field_position is not None]
        if xs:
            ax.plot(xs, ys, lw=1.8, color=colors[traj.track_id % 10], marker="o", ms=2.2, zorder=3)
    ax.set_xlabel("Field X (yd, relative_10yd frame)")
    ax.set_ylabel("Field Y (yd)")
    ax.set_title(
        "1. Camera-Pan Sequence: Field-Space Trajectories\n(dashed = GT route, line = smoothed, dots = raw projections)",
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.3)

    # Panel 2: camera-cut refusal, track re-acquisition, and projection recovery
    ax = axes[0, 1]
    seq_id2 = "traj_seq_08_test_camera_cut_refusal_and_recovery"
    ex2 = extras_by_seq[seq_id2]
    cut_cfg = ex2["seq"]["camera_cut"]
    cut_frames = sorted(int(k) for k, v in ex2["seq"]["calibration_states"].items() if str(v).startswith("unknown"))
    ax.axvspan(min(cut_frames) - 0.5, max(cut_frames) + 0.5, color="#fdecea", zorder=0)
    ax.text(
        float(np.mean(cut_frames)),
        ax.get_ylim()[1],
        "camera cut\n(calibration refused)",
        ha="center",
        va="top",
        fontsize=8,
        color="#b71c1c",
    )
    for traj in ex2["trajectories"]:
        frames_, errs_ = [], []
        for sample in traj.samples:
            gid = sample.provenance.get("gt_id")
            if gid is None or sample.field_position is None:
                continue
            gt = ex2["gt_table"][sample.frame_id][int(gid)]
            frames_.append(sample.frame_id)
            errs_.append(
                float(np.hypot(sample.field_position[0] - gt["x_yd"], sample.field_position[1] - gt["y_yd"]))
            )
        if frames_:
            ax.plot(frames_, errs_, lw=1.6, marker="o", ms=2.6, color=colors[traj.track_id % 10])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Field position error (yd)")
    ax.set_title("2. Camera Cut (Test 8): Refusal During Cut, Recovery After", fontsize=9.5)
    rec_lat = next(
        x["projection_recovery_latency_frames_max"] for x in seq_scores if x["sequence_id"] == seq_id2
    )
    ax.text(
        0.99,
        0.02,
        f"shot change @ f{cut_cfg['frame']} | refused: {cut_frames}\n"
        f"no field position claimed while refused\nprojection-recovery latency <= {rec_lat} frames",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    # Panel 3: innovation gate statistic with injected outliers
    ax = axes[1, 0]
    seq_id3 = "traj_seq_04_val_jump_outliers"
    ex3 = extras_by_seq[seq_id3]
    gate = float(manifest["frozen_parameters"]["gate_chi2_2dof"])
    injected = {
        (int(e["gt_id"]), int(e["frame"])) for e in ex3["seq"]["outlier_events"]
    }
    for traj in ex3["trajectories"]:
        xs, ds = [], []
        for s in traj.samples:
            gid = s.provenance.get("gt_id")
            if gid is None or s.raw_field_position is None or s.covariance_xy is None:
                continue
            xs.append(s.frame_id)
            ds.append(1.0 if s.is_outlier_rejected else 0.05)
        if xs:
            ax.plot(xs, ds, lw=1.4, marker="o", ms=3, color=colors[traj.track_id % 10])
    for gid, frame in sorted(injected):
        ax.axvline(frame, color="#b71c1c", ls=":", lw=1.1, alpha=0.7)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Measurement rejected (1 = rejected)")
    ax.set_title(
        f"3. Jump Gate Decisions (Val 4, injected outliers marked)\ngate: chi2(2 dof) <= {gate}",
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.3)

    # Panel 4: split-level summary
    ax = axes[1, 1]
    labels = ["Train", "Val", "Frozen Test"]
    keys = ("train", "val", "test")
    med = [splits[k]["field_pos_err_median_yd"] or 0.0 for k in keys]
    raw = [splits[k]["raw_field_pos_err_median_yd"] or 0.0 for k in keys]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, raw, width=0.34, color="#9e9e9e", label="Raw projection (median yd)")
    ax.bar(x + 0.18, med, width=0.34, color="#1f77b4", label="Smoothed trajectory (median yd)")
    for i, (r_, m_) in enumerate(zip(raw, med)):
        ax.text(i - 0.18, r_ + 0.002, f"{r_:.3f}", ha="center", fontsize=8.5)
        ax.text(i + 0.18, m_ + 0.002, f"{m_:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Field position error (yd)")
    notes = []
    for k, lbl in zip(keys, ("Train", "Val", "Test")):
        agg = splits[k]
        notes.append(
            f"{lbl}: observed {100.0 * (agg['observed_sample_ratio'] or 0.0):.0f}% / inferred "
            f"{100.0 * (agg['inferred_sample_ratio'] or 0.0):.0f}%, "
            f"path {100.0 * (agg['motion_preservation_ratio_vs_gt'] or 0.0):.0f}% of GT, "
            f"tracks {agg['n_tracks']} (dominant {agg['n_players'] * agg['num_sequences']}, "
            f"fragments excluded {agg['fragmentation_fragment_tracks_excluded_from_dominant']})"
        )
    ax.set_title("4. Position Error: Raw vs Smoothed (0 fabricated, 0 absolute-yardline violations)", fontsize=9.5)
    ax.text(
        0.99,
        0.02,
        " | ".join(notes),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 4 Field-Space Trajectory Benchmark (fixture/synthetic motion; detector accuracy unmeasured)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(h_pad=2.4, rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(out_dir / "phase4_trajectory_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    started = time.perf_counter()
    res = run_phase4_benchmark()
    print(json.dumps(res["splits"], indent=2))
    print(json.dumps(res["sequences"], indent=2)[:1200])
    print("determinism:", json.dumps(res["determinism_check"]))
    print("smoke:", json.dumps(res["real_frame_smoke_test"]))
    print(f"total wall time: {time.perf_counter() - started:.1f}s")
