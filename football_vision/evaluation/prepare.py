"""Extract exact decoded frames and a pending manual-annotation template."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

import cv2
import numpy as np


def prepare_annotations(video: Path, output: Path, frame_ids: Sequence[int], *,
                        source_kind: str, split: str) -> dict:
    """Decode sequentially so frame IDs agree with run_video; never infer labels."""
    if source_kind not in ("real", "synthetic") or split not in ("dev", "test"):
        raise ValueError("source_kind must be real/synthetic and split must be dev/test")
    if not frame_ids or any(type(fid) is not int or fid < 0 for fid in frame_ids):
        raise ValueError("frame IDs must be nonempty nonnegative integers")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame IDs must be unique")
    if output.exists():
        raise FileExistsError(output)
    selected = set(frame_ids)
    digest = hashlib.sha256()
    with video.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        output.mkdir()  # Exclusive ownership; failed work removes only this directory.
        try:
            frames = []
            for fid in range(max(selected) + 1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Video ended or decode failed before requested frame {fid}")
                if fid not in selected:
                    continue
                filename = f"frame_{fid:08d}.png"
                if not cv2.imwrite(str(output / filename), image):
                    raise ValueError(f"Could not write {filename}")
                frames.append(dict(frame_id=fid, image=filename,
                                   image_size=[image.shape[1], image.shape[0]],
                                   exhaustive=False, players=[]))
            document = dict(schema_version=1, video_sha256=digest.hexdigest(),
                            source_kind=source_kind, split=split,
                            annotation_method="pending_manual", fps=fps,
                            opencv_version=cv2.__version__, frames=frames)
            (output / "annotations.json").write_text(json.dumps(document, indent=2, allow_nan=False))
            (output / "README.txt").write_text(
                "These are decoded source frames, not model predictions or completed labels.\n"
                "Edit annotations.json after manually reviewing each PNG at original resolution.\n"
                "Add every visible player's pixel box [x1,y1,x2,y2] and stable manual track_id.\n"
                "Set exhaustive to true only after checking all players in that frame; an empty\n"
                "players list is valid only for a manually confirmed negative frame.\n"
                "Do not copy detector predictions into ground truth. Leave field_position absent\n"
                "unless independent landmarks establish its units, axes and coordinate-frame ID.\n"
                "After completing all frames, set annotation_method to independent_manual.\n"
                "The evaluator refuses this pending template. Frame IDs match sequential decoding\n"
                "in the video runner. FPS is metadata, not a variable-rate timestamp guarantee.\n"
            )
        except BaseException:
            shutil.rmtree(output)
            raise
        return document
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new annotation directory")
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="zero-based decoded frame IDs")
    parser.add_argument("--source-kind", choices=["real", "synthetic"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args()
    try:
        result = prepare_annotations(args.video, args.out, args.frames,
                                     source_kind=args.source_kind, split=args.split)
    except (ValueError, TypeError, OSError, cv2.error) as exc:
        parser.exit(2, f"Annotation preparation failed: {exc}\n")
    print(f"Prepared {len(result['frames'])} frames for manual annotation: {args.out}")


if __name__ == "__main__":
    main()
