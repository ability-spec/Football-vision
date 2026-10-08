"""One-command CPU MVP: video, play analytics, exports and synchronized review."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from football_vision.evaluation.video import run_video
from football_vision.evaluation.labels import load_labels
from football_vision.visualization.__main__ import render_video


def summarize_review(result: dict) -> dict:
    """Expose observation/coordinate counts without claiming perception quality."""
    frames = result["frames"]
    observed = [p for frame in frames for p in frame["players"] if p["observed"]]
    coordinates = {p["coordinate_frame_id"] for p in observed
                   if p["field_position"] is not None and p["coordinate_frame_id"] is not None}
    return dict(observed_image_track_ids=len({p["track_id"] for p in observed}),
                max_observed_tracks_per_frame=max((sum(p["observed"] for p in f["players"])
                                                   for f in frames), default=0),
                detected_camera_cuts=sum(f["camera_cut_detected"] for f in frames),
                coordinate_frame_count=len(coordinates),
                projectable_frame_count=sum(f["calibration_projectable"] for f in frames),
                boundary_refusal_count=len(result["segmentation"]["refusals"]),
                scope="Observation counts only; IDs are not unique athletes; projectability is not calibration accuracy")


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
    text_rows = []
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
                text_rows.append("| " + " | ".join(
                    html.escape(str(row[key])).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
                    if row[key] is not None else "Unavailable" for key in fields[:6]) + " |")
    diagnostic_text = json.dumps(result.get("review_diagnostics", {"status": "Unavailable in supplied artifact"}), indent=2)
    refusal_text = json.dumps(result["segmentation"]["refusals"], indent=2).replace("`", "\\u0060")
    text_report = (
        "# Football Vision review\n\n"
        f"Frames processed: {result['frames_processed']}. Source: {result['source_kind']}.\n\n"
        f"Detector: {result.get('detector', 'Unavailable')}. Real-video accuracy unmeasured.\n\n"
        "Open preview.png for sampled frames, or review.avi in a video player for motion.\n"
        "Unknown teams and missing field geometry remain unavailable.\n\n"
        "## Observation diagnostics\n\n```json\n" + diagnostic_text + "\n```\n\n"
        "## Boundary refusals\n\n```json\n" + refusal_text + "\n```\n\n"
        "## Play metrics\n\n| " + " | ".join(fields[:6]) + " |\n| "
        + " | ".join(["---"] * 6) + " |\n" + "\n".join(text_rows) + "\n\n"
        + "Full observations: analysis.json. Spreadsheet export: metrics.csv.\n"
    )
    (output / "report.md").write_text(text_report)
    refusals = html.escape(json.dumps(result["segmentation"]["refusals"], indent=2))
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Vision review</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1200px}}td,th{{padding:.6rem;border-bottom:1px solid #ccc;text-align:left}}table{{border-collapse:collapse}}pre{{white-space:pre-wrap}}</style>
<h1>Football Vision CPU MVP</h1>
<p>{result['frames_processed']} frames processed. Source: {html.escape(result['source_kind'])}.
Detector: {html.escape(result.get('detector', 'Unavailable'))} ({html.escape(result.get('detector_source', 'unknown'))}); real-video accuracy unmeasured.
Unknown teams and missing field geometry remain unavailable.</p>
<p><a href="review.avi">Download synchronized review video (AVI)</a> ·
<a href="analysis.json">Full analysis JSON</a> · <a href="metrics.csv">Metrics CSV</a> ·
<a href="bundle.zip">Download complete report (ZIP)</a></p>
<h2>Sampled review frames</h2><p>Still images sampled across the processed window. Open the AVI in a video player for motion.</p>
<a href="preview.png"><img src="preview.png" alt="Sampled synchronized review frames" style="max-width:100%;height:auto"></a>
<h2>Play metrics</h2><table><thead><tr>{''.join('<th>'+key+'</th>' for key in fields[:6])}</tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>Boundary refusals</h2><pre>{refusals}</pre>
<h2>Observation diagnostics</h2><pre>{html.escape(diagnostic_text)}</pre>
<p>Video SHA-256: {result['video_sha256']}</p></html>"""
    (output / "report.html").write_text(report)


def run_workflow(video: Path, labels: Path, output: Path, *, source_kind: str,
                 max_frames: int = 300, detector_kind: str = "turf", weights: Path | None = None) -> dict:
    """Create a new result directory; remove only our own partial output on failure."""
    game, plays, label_hash = load_labels(labels)
    if output.exists():
        raise FileExistsError(output)
    result = run_video(video, source_kind=source_kind, max_frames=max_frames,
                       play_labels=plays, game_id=game, detector_kind=detector_kind, weights=weights)
    result["play_labels_sha256"] = label_hash
    result["review_diagnostics"] = summarize_review(result)
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
        # Preserve relative report links when results are downloaded from a viewer.
        artifacts = sorted(output.iterdir())
        with ZipFile(output / "bundle.zip", "x", compression=ZIP_DEFLATED) as archive:
            for path in artifacts:
                archive.write(path, arcname=path.name)
            archive.writestr("OPEN_ME.txt",
                             "Extract all files into one folder, then open report.html in a browser.\n"
                             "Do not open the HTML directly inside the ZIP.\n"
                             "preview.png can be opened directly as an image.\n"
                             "For motion, open review.avi in VLC or another AVI-capable video player.\n"
                             "analysis.json, metrics.csv and manifest.json are included.\n")
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
