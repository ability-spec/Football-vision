"""Render a synchronized review video from a saved video-runner artifact."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2

from football_vision.visualization import render_frame


def render_video(video: Path, predictions: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    artifact = json.loads(predictions.read_text())
    digest = hashlib.sha256()
    with video.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if artifact.get("video_sha256") != digest.hexdigest():
        raise ValueError("predictions belong to a different video")
    frames = artifact["frames"]
    if not frames or any(frame["frame_id"] != i for i, frame in enumerate(frames)):
        raise ValueError("prediction frames must be nonempty and consecutive from zero")
    windows = {}
    for record in frames:
        for player in record["players"]:
            xy = player.get("field_position")
            mode = player.get("x_coord_mode")
            if (player.get("observed", True) and xy is not None and len(xy) == 2
                    and all(math.isfinite(v) for v in xy)
                    and player.get("coordinate_frame_id") is not None
                    and mode in ("relative_5yd", "relative_10yd")):
                key = (player["coordinate_frame_id"], player.get("coordinate_segment", 0), mode)
                lo, hi = windows.get(key, (xy[0] - 5., xy[0] + 5.))
                windows[key] = min(lo, xy[0] - 5.), max(hi, xy[0] + 5.)
    capture = cv2.VideoCapture(str(video))
    writer = None
    created = False
    completed = False
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps <= 0 or fps != artifact["fps"]:
            raise ValueError("video frame rate does not match predictions")
        # Reserve exclusively before OpenCV opens it; preserve existing files.
        with output.open("xb"):
            pass
        created = True
        for record in frames:
            ok, image = capture.read()
            if not ok:
                raise ValueError("video ended before prediction frames")
            keys = {(p.get("coordinate_frame_id"), p.get("coordinate_segment", 0), p.get("x_coord_mode"))
                    for p in record["players"] if p.get("observed", True) and p.get("field_position") is not None}
            relative_keys = keys & windows.keys()
            bounds = windows[next(iter(relative_keys))] if len(relative_keys) == 1 else None
            rendered = render_frame(image, record, x_bounds=bounds)
            # MJPEG requires even dimensions; pad rather than lose edge pixels.
            rendered = cv2.copyMakeBorder(rendered, 0, rendered.shape[0] % 2,
                                           0, rendered.shape[1] % 2, cv2.BORDER_CONSTANT)
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                                         fps, (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise ValueError("MJPEG video output could not be opened; use .avi")
            writer.write(rendered)
        writer.release()
        writer = None
        # Check the actual encoder output, not merely its startup status.
        check = cv2.VideoCapture(str(output))
        try:
            count = 0
            while True:
                ok, decoded = check.read()
                if not ok:
                    break
                if decoded.shape != rendered.shape:
                    raise ValueError("rendered video failed dimension verification")
                count += 1
            if count != len(frames):
                raise ValueError("rendered video failed frame-count verification")
        finally:
            check.release()
        completed = True
        return count
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if created and not completed:
            output.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = render_video(args.video, args.predictions, args.out)
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
        parser.exit(2, f"Rendering failed: {exc}\n")
    print(f"Rendered {count} frames")


if __name__ == "__main__":
    main()
