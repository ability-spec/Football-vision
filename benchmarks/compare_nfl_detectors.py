"""Compare fixed CPU person modes using native helmet-center diagnostics only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from football_vision.detection.yolox import create_player_detector
from football_vision.evaluation.nfl import read_helmet_labels, helmet_coverage_diagnostic


def compare(videos: list[Path], labels: Path, weights: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not videos or len({v.name for v in videos}) != len(videos):
        raise ValueError("Video basenames must be nonempty and unique")
    inventory = {}
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or count < 1 or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid video: {video}")
            inventory[video.name] = dict(frame_count=count, fps=fps,
                                        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
        finally:
            capture.release()
    native = read_helmet_labels(labels, inventory)
    modes = {mode: create_player_detector(mode, weights) for mode in ("yolox", "yolox-tiled")}
    result = dict(schema_version=1, source_kind="real", split="dev",
                  diagnostic="one-to-one on-field helmet-center containment; NOT player recall, precision or AP",
                  helmet_labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
                  opencv_version=cv2.__version__, opencv_threads=cv2.getNumThreads(),
                  model_modes={mode: detector.metadata() for mode, detector in modes.items()}, videos=[])
    output.mkdir()
    try:
        previews = []
        for video in videos:
            metadata = inventory[video.name]
            count = metadata["frame_count"]
            selected = sorted({0, min(10, count - 1), count // 4, count // 2, 3 * count // 4, count - 1})
            record = dict(video=video.name, **metadata, samples=[])
            capture = cv2.VideoCapture(str(video))
            try:
                for fid in range(count):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f"Premature video decode failure: {video}:{fid}")
                    if fid not in selected:
                        continue
                    helmets = native[video.name].get(fid)
                    if not helmets:
                        raise ValueError(f"Missing sampled helmet labels: {video}:{fid}")
                    sample = dict(frame_id=fid, kaggle_frame=fid + 1, modes={})
                    tiles = []
                    for mode, detector in modes.items():
                        start = time.perf_counter()
                        detections = detector.detect(image, fid)
                        elapsed = time.perf_counter() - start
                        boxes = [list(d.bbox) for d in detections]
                        diagnostic = helmet_coverage_diagnostic(helmets, boxes)
                        sample["modes"][mode] = dict(**diagnostic, inference_seconds=elapsed,
                                                    boxes=boxes, scores=[d.confidence for d in detections])
                        if fid == min(10, count - 1):
                            overlay = image.copy()
                            for h in helmets:
                                if not h["is_sideline_player"]:
                                    a, b, c, d = map(round, h["helmet_bbox"])
                                    cv2.rectangle(overlay, (a, b), (c, d), (255, 170, 0), 2)
                            for box in boxes:
                                a, b, c, d = map(round, box)
                                cv2.rectangle(overlay, (a, b), (c, d), (0, 220, 255), 2)
                            tile = cv2.resize(overlay, (640, 360))
                            band = np.full((42, 640, 3), 20, dtype=np.uint8)
                            title = (f"{video.stem} | {mode}: "
                                     f"{diagnostic['matched_helmet_centers']}/{diagnostic['labeled_on_field_helmets']}")
                            cv2.putText(band, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                            tiles.append(np.vstack([band, tile]))
                    record["samples"].append(sample)
                    if tiles:
                        previews.append(np.hstack(tiles))
            finally:
                capture.release()
            result["videos"].append(record)
        (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        if not cv2.imwrite(str(output / "preview.jpg"), np.vstack(previews)):
            raise ValueError("Could not write comparison preview")
        lines = ["# CPU detector comparison", "", result["diagnostic"], "",
                 "Yellow: person candidates. Blue: native on-field helmet boxes. All clips are development data.", "",
                 "| Video | Baseline matches / helmets | Tiled matches / helmets |", "| --- | ---: | ---: |"]
        for record in result["videos"]:
            counts = {mode: sum(s["modes"][mode]["matched_helmet_centers"] for s in record["samples"])
                      for mode in modes}
            total = sum(s["modes"]["yolox"]["labeled_on_field_helmets"] for s in record["samples"])
            lines.append(f"| {record['video']} | {counts['yolox']} / {total} | {counts['yolox-tiled']} / {total} |")
        lines += ["", "Tiled mode makes up to five inference calls per image. Timings are local CPU diagnostics,",
                  "not a general speed benchmark. Coverage does not measure false positives, player box quality,",
                  "tracking identity or calibration. No thresholds were tuned using a held-out test."]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--helmets", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--opencv-threads", type=int, default=2)
    args = parser.parse_args()
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    cv2.setNumThreads(args.opencv_threads)
    try:
        result = compare(args.videos, args.helmets, args.weights, args.out)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")
    print(f"Compared {len(result['videos'])} videos. Results: {args.out}")


if __name__ == "__main__":
    main()
