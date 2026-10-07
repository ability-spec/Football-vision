"""Run the untrained CPU baseline on a local video, without annotation access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from football_vision import __version__, calibrate_frame
from football_vision.calibration.tracker import CalibrationTracker
from football_vision.detection.detector import TurfContrastPlayerDetector
from football_vision.tracking.tracker import PlayerTracker
from football_vision.trajectory.builder import PlayerTrajectoryBuilder
from football_vision.analytics import PlaySegmenter, analyze_play
from football_vision.schema import PlayTimestampLabel
from football_vision.evaluation.labels import load_labels


def run_video(path: Path, *, source_kind: str, max_frames: int = 300,
              play_labels: list[PlayTimestampLabel] | None = None,
              game_id: str = "", two_stage_tracking: bool = False) -> dict:
    if source_kind not in ("real", "synthetic") or max_frames < 1:
        raise ValueError("source_kind must be real/synthetic; max_frames must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        advertised_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        # Weak recovery needs boxes that the default 0.4 detector filter discards.
        detector = (TurfContrastPlayerDetector(min_confidence=0.1) if two_stage_tracking
                    else TurfContrastPlayerDetector())
        tracker = PlayerTracker(high_confidence_threshold=0.4 if two_stage_tracking else None)
        geometry = CalibrationTracker()
        builder = PlayerTrajectoryBuilder(fps=fps)
        frames = []
        start = time.perf_counter()
        for fid in range(max_frames):
            ok, image = capture.read()
            if not ok:
                if not frames or (advertised_frames > 0 and fid < advertised_frames):
                    raise ValueError(f"Video decode failed before expected end at frame {fid}")
                break
            calibration = geometry.update_from_result(calibrate_frame(image), image)
            detections = detector.detect(image, frame_id=fid)
            tracks = tracker.update(detections, frame_id=fid, calibration=calibration)
            samples = builder.update(tracks, frame_id=fid, calibration=calibration)
            by_id = {s.track_id: s for s in samples}
            players = []
            for t in tracks:
                s = by_id[t.track_id]
                players.append(
                    {
                        "track_id": t.track_id,
                        "detection_confidence": t.detection_confidence,
                        "bbox": list(t.bbox),
                        "observed": t.missed_frames == 0,
                        "field_position": s.field_position,
                        "predicted_position": s.predicted_position,
                        "x_coord_mode": s.x_coord_mode,
                        "coordinate_frame_id": s.coordinate_frame_id,
                        "coordinate_segment": s.coordinate_segment,
                    }
                )
            frames.append(
                {"frame_id": fid, "players": players,
                 "calibration_projectable": calibration.can_project(),
                 "camera_cut_detected": calibration.camera_cut_detected}
            )
        elapsed = time.perf_counter() - start
        code_hash = hashlib.sha256()
        package_root = Path(__file__).resolve().parents[1]
        for source in sorted(package_root.rglob("*.py")):
            code_hash.update(str(source.relative_to(package_root)).encode())
            code_hash.update(source.read_bytes())
        result = {
            "schema_version": 1,
            "package_version": __version__,
            "source_kind": source_kind,
            "video_sha256": digest.hexdigest(),
            "pipeline_sha256": code_hash.hexdigest(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "detector_settings": vars(detector),
            "tracking_settings": {
                "association": "strong_then_weak" if two_stage_tracking else "single_stage",
                "high_confidence_threshold": tracker.high_confidence_threshold,
                "weak_confidence_floor": 0.1 if two_stage_tracking else None,
                "recovery_min_hits": tracker.recovery_min_hits,
                "weak_min_iou": 0.3 if two_stage_tracking else None,
                "max_missed_frames": tracker.max_missed_frames,
            },
            "fps": fps,
            "advertised_frames": advertised_frames,
            "frames_processed": len(frames),
            "max_frames": max_frames,
            "wall_time_s": elapsed,
            "processing_fps": len(frames) / elapsed,
            "detector": "TurfContrastPlayerDetector (untrained CPU baseline)",
            "coordinate_policy": "relative; automatic epoch boundaries are heuristic",
            "frames": frames,
        }
        if play_labels is not None:
            for label in play_labels:
                for name in ("start_frame", "snap_frame", "end_frame"):
                    value = getattr(label, name)
                    if value is not None and (type(value) is not int or not 0 <= value < len(frames)):
                        raise ValueError(f"{name} must be an integer within decoded frames")
                if label.end_frame is None:
                    raise ValueError("manual end_frame is required for bounded video analysis")
            trajectories = builder.finalize()
            segmentation = PlaySegmenter(fps=fps).segment(play_labels, trajectories, game_id=game_id)
            result["segmentation"] = segmentation.to_dict()
            result["plays"] = [analyze_play(segment, trajectories).to_dict()
                               for segment in segmentation.segments]
            result["trajectories"] = [trajectory.to_dict() for trajectory in trajectories]
            phases = segmentation.frame_phases()
            for record in frames:
                record["play_phase"] = phases.get(record["frame_id"], "outside_play")
            result["analytics_policy"] = "manual play boundaries; teams and offense unknown"
        return result
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-kind", choices=["real", "synthetic"], required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--two-stage-tracking", action="store_true",
                        help="Experimental: strong detections first, weak boxes recover mature tracks")
    parser.add_argument("--play-labels", type=Path,
                        help="JSON with game_id and plays (play_id/start_frame/end_frame/snap_frame)")
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise FileExistsError(args.out)
        labels = None
        game_id = ""
        labels_hash = None
        if args.play_labels is not None:
            game_id, labels, labels_hash = load_labels(args.play_labels)
        result = run_video(args.video, source_kind=args.source_kind, max_frames=args.max_frames,
                           play_labels=labels, game_id=game_id,
                           two_stage_tracking=args.two_stage_tracking)
        if labels_hash is not None:
            result["play_labels_sha256"] = labels_hash
        with args.out.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    except (ValueError, TypeError, OSError, cv2.error) as exc:
        parser.exit(2, f"Video run failed: {exc}\n")


if __name__ == "__main__":
    main()
