"""One-command CPU MVP: video, play analytics, exports and synchronized review."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from football_vision.evaluation.video import run_video
from football_vision.evaluation.labels import load_labels
from football_vision.visualization.__main__ import render_video


def write_contact_sheet(video: Path, output: Path, frame_count: int) -> None:
    """Sample up to six decoded review frames, preserving their frame labels."""
    selected = set(np.linspace(0, frame_count - 1, min(6, frame_count), dtype=int))
    capture = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for index in range(frame_count):
            ok, image = capture.read()
            if not ok:
                raise ValueError("review video ended during contact-sheet generation")
            if index in selected:
                tiles.append(image)
    finally:
        capture.release()
    if not tiles:
        raise ValueError("review video contains no preview frames")
    height, width = tiles[0].shape[:2]
    columns = min(2, len(tiles))
    sheet = np.zeros((((len(tiles) + columns - 1) // columns) * height,
                      columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = image
    if not cv2.imwrite(str(output), sheet):
        raise ValueError("contact-sheet PNG could not be written")


def _write_report(result: dict, output: Path) -> None:
    rows = []
    fields = ["game_id", "play_id", "metric", "value", "units", "definition", "source", "confidence", "limitations"]
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for play in result["plays"]:
            for name, metric in play["metrics"].items():
                row = dict(game_id=play["game_id"], play_id=play["play_id"], metric=name,
                           **{key: metric[key] for key in fields[3:]})
                # Blank CSV values preserve unavailable metrics; never replace by zero.
                # User labels must remain text when opened by spreadsheet software.
                writer.writerow({key: "'" + value if isinstance(value, str)
                                 and value.lstrip().startswith(("=", "+", "-", "@")) else value
                                 for key, value in row.items()})
                rows.append("<tr>" + "".join(f"<td>{html.escape(str(row[key])) if row[key] is not None else 'Unavailable'}</td>"
                                               for key in fields[:6]) + "</tr>")
    refusals = html.escape(json.dumps(result["segmentation"]["refusals"], indent=2))
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Vision review</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1200px}}td,th{{padding:.6rem;border-bottom:1px solid #ccc;text-align:left}}table{{border-collapse:collapse}}pre{{white-space:pre-wrap}}</style>
<h1>Football Vision CPU MVP</h1>
<p>{result['frames_processed']} frames processed. Source: {html.escape(result['source_kind'])}.
Untrained detector baseline; real-video accuracy unmeasured. Unknown teams and missing field geometry remain unavailable.</p>
<p><a href="review.avi">Download synchronized review video (AVI)</a> ·
<a href="analysis.json">Full analysis JSON</a> · <a href="metrics.csv">Metrics CSV</a></p>
<h2>Sampled review frames</h2><p>Still images sampled across the processed window. Open the AVI in a video player for motion.</p>
<a href="preview.png"><img src="preview.png" alt="Sampled synchronized review frames" style="max-width:100%;height:auto"></a>
<h2>Play metrics</h2><table><thead><tr>{''.join('<th>'+key+'</th>' for key in fields[:6])}</tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>Boundary refusals</h2><pre>{refusals}</pre>
<p>Video SHA-256: {result['video_sha256']}</p></html>"""
    (output / "report.html").write_text(report)


def run_workflow(video: Path, labels: Path, output: Path, *, source_kind: str,
                 max_frames: int = 300) -> dict:
    """Create a new result directory; remove only our own partial output on failure."""
    game, plays, label_hash = load_labels(labels)
    if output.exists():
        raise FileExistsError(output)
    result = run_video(video, source_kind=source_kind, max_frames=max_frames,
                       play_labels=plays, game_id=game)
    result["play_labels_sha256"] = label_hash
    output.mkdir()  # Exclusive creation; never overwrite a previous run.
    try:
        analysis = output / "analysis.json"
        analysis.write_text(json.dumps(result, indent=2, allow_nan=False))
        render_video(video, analysis, output / "review.avi")
        write_contact_sheet(output / "review.avi", output / "preview.png", result["frames_processed"])
        _write_report(result, output)
        manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.iterdir())}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def create_demo(directory: Path) -> tuple[Path, Path]:
    """Deterministic synthetic clip, explicitly not a detector accuracy benchmark."""
    video, labels = directory / "synthetic.avi", directory / "plays.json"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (320, 240))
    try:
        if not writer.isOpened():
            raise ValueError("MJPEG encoder unavailable")
        for frame in range(20):
            image = np.full((240, 320, 3), (30, 140, 30), dtype=np.uint8)
            offset = max(0, frame - 5) * 2
            for x, y in ((65, 75), (140, 120), (220, 90)):
                cv2.rectangle(image, (x + offset, y), (x + offset + 10, y + 30), (20, 20, 20), -1)
            writer.write(image)
    finally:
        writer.release()
    labels.write_text(json.dumps(dict(game_id="synthetic-demo", plays=[
        dict(play_id="demo-play", start_frame=0, snap_frame=5, end_frame=19)])))
    return video, labels
