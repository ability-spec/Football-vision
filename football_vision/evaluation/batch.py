"""Review a manifest of local clips with the CPU pipeline and optional helmets."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from football_vision.evaluation.nfl import helmet_coverage_diagnostic, read_helmet_labels
from football_vision.evaluation.video import run_video
from football_vision.pipeline import write_contact_sheet
from football_vision.visualization.__main__ import render_video


DIAGNOSTIC_SCOPE = "one-to-one helmet-center containment; not player recall, AP or precision"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[dict, list[Path], str]:
    content = path.read_bytes()
    document = json.loads(content)
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1):
        raise ValueError("Batch manifest needs schema_version 1")
    if document.get("source_kind") not in ("real", "synthetic") or document.get("split") not in ("dev", "test"):
        raise ValueError("Manifest needs source_kind real/synthetic and split dev/test")
    entries = document.get("videos")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest videos must be a nonempty list")
    videos = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise ValueError("Each video needs a nonempty path")
        video = Path(entry["path"])
        videos.append((path.parent / video).resolve())
    if len({video.name for video in videos}) != len(videos) or len(set(videos)) != len(videos):
        raise ValueError("Video paths and basenames must be unique")
    return document, videos, hashlib.sha256(content).hexdigest()


def _inspect_video(path: Path) -> dict:
    """Decode the full source to distinguish its bounds from an inference prefix."""
    digest = _sha256(path)
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        advertised = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if (not capture.isOpened() or not np.isfinite(fps) or fps <= 0
                or not np.isfinite(advertised) or advertised < 0):
            raise ValueError(f"Invalid video metadata: {path}")
        count = 0
        shape = None
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if shape is None:
                shape = image.shape
                if shape[0] < 96:
                    raise ValueError("Review video requires frame height >=96")
            elif image.shape != shape:
                raise ValueError(f"Video dimensions change during decoding: {path}")
            count += 1
        if count == 0 or count < int(advertised):
            raise ValueError(f"Video decode failed before expected end: {path}:{count}")
        return dict(frame_count=count, advertised_frames=int(advertised), fps=fps,
                    width=shape[1], height=shape[0], video_sha256=digest)
    finally:
        capture.release()


def _summarize(result: dict, native: dict[int, list[dict]] | None) -> dict:
    labeled_frames = 0
    matched = helmets = 0
    ids = set()
    for frame in result["frames"]:
        ids.update(p["track_id"] for p in frame["players"] if p["observed"])
        labels = None if native is None else native.get(frame["frame_id"])
        diagnostic = None if labels is None else helmet_coverage_diagnostic(
            labels, [d["bbox"] for d in frame["detections"]])
        frame["helmet_diagnostic"] = diagnostic
        if diagnostic is not None:
            labeled_frames += 1
            matched += diagnostic["matched_helmet_centers"]
            helmets += diagnostic["labeled_on_field_helmets"]
    return dict(frames_processed=result["frames_processed"],
                observed_image_track_ids=len(ids),
                detected_camera_cuts=sum(frame["camera_cut_detected"] for frame in result["frames"]),
                labeled_frames=labeled_frames,
                unlabeled_frames=result["frames_processed"] - labeled_frames,
                matched_helmet_centers=matched if labeled_frames else None,
                labeled_on_field_helmets=helmets if labeled_frames else None,
                helmet_center_containment_fraction=matched / helmets if helmets else None,
                metric_scope=DIAGNOSTIC_SCOPE,
                identity_scope="Observed image IDs are not unique athletes or tracking accuracy")


def _write_reports(summary: dict, output: Path) -> None:
    lines = ["# Football Vision batch review", "",
             f"Source: {summary['source_kind']}. Declared split: {summary['split']}.", "",
             "Declared split is metadata, not proof that footage is unseen or independently labeled.", "",
             DIAGNOSTIC_SCOPE + ". Missing labels are unavailable, not confirmed negative frames.", "",
             "Tracks use the CPU pipeline's heuristic calibration and camera-cut handling.",
             "Image IDs are not unique athletes; field positions are not independently validated.",
             "Play analytics are unavailable here: manual play boundaries were not supplied.", "",
             "| Video | Frames processed / source | Labeled frames | Helmet matches / labels | Observed image IDs |",
             "| --- | ---: | ---: | ---: | ---: |"]
    cards = []
    for clip in summary["videos"]:
        name = html.escape(clip["video"]).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        d = clip["diagnostics"]
        coverage = ("Unavailable" if d["labeled_frames"] == 0 else
                    f"{d['matched_helmet_centers']} / {d['labeled_on_field_helmets']}")
        window = f"{d['frames_processed']} / {clip['source']['frame_count']}"
        lines.append(f"| {name} | {window} | {d['labeled_frames']} | {coverage} | {d['observed_image_track_ids']} |")
        directory = clip["directory"]
        cards.append(f"<section><h2>{html.escape(clip['video'])}</h2>"
                     f"<p>Processed/source frames: {window}. Labeled frames: {d['labeled_frames']}. "
                     f"Helmet centers: {coverage}. Observed image IDs: {d['observed_image_track_ids']}.</p>"
                     f'<p><a href="{directory}/analysis.json">Analysis JSON</a> · '
                     f'<a href="{directory}/review.avi">Review video (AVI)</a></p>'
                     f'<a href="{directory}/preview.png"><img src="{directory}/preview.png" '
                     'alt="Sampled review frames" style="max-width:100%;height:auto"></a></section>')
    lines += ["", "Each clip folder contains analysis.json, preview.png and review.avi.",
              "Open a PNG directly for still frames; open AVI in a video player for motion.",
              "If the embedded HTML viewer is blocked, download bundle.zip, extract everything",
              "and open report.html in a local browser. No web server is needed.", "",
              "Source counts are fully decoded before bounded inference. Per-frame diagnostics",
              "use fresh detector boxes only, without coasted tracks or label input to inference.",
              "Frames without native CSV rows are excluded from the containment denominator."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Football Vision batch review</title>'
            '<style>body{font:16px system-ui;margin:2rem;max-width:1200px}section{margin:2rem 0}</style>'
            '<h1>Football Vision batch review</h1><p>' + html.escape(DIAGNOSTIC_SCOPE) + '.</p>'
            '<p>Missing labels are unavailable. Image IDs are not unique athletes. '
            'Automatic field geometry is unvalidated. Manual play boundaries were not supplied.</p>'
            '<p><a href="summary.json">Summary JSON</a> · <a href="report.md">Text report</a> · '
            '<a href="bundle.zip">Download report bundle</a></p>' + ''.join(cards) + '</html>')
    (output / "report.html").write_text(page, encoding="utf-8")


def run_batch(manifest: Path, output: Path, *, max_frames: int = 300,
              detector_kind: str = "turf", weights: Path | None = None,
              helmet_labels: Path | None = None) -> dict:
    """Create a new batch and clean partial outputs; inference never receives labels."""
    if output.exists():
        raise FileExistsError(output)
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    document, videos, manifest_hash = _load_manifest(manifest)
    inventory = {video.name: _inspect_video(video) for video in videos}
    label_hash = None if helmet_labels is None else _sha256(helmet_labels)
    native = None if helmet_labels is None else read_helmet_labels(
        helmet_labels, inventory, require_selected_labels=False)
    if helmet_labels is not None and _sha256(helmet_labels) != label_hash:
        raise ValueError("Helmet labels changed during loading")
    summary = dict(schema_version=1, source_kind=document["source_kind"], split=document["split"],
                   manifest_sha256=manifest_hash, helmet_labels_sha256=label_hash,
                   input_manifest=document,
                   opencv_threads=cv2.getNumThreads(), max_frames=max_frames,
                   metric_scope=DIAGNOSTIC_SCOPE, videos=[])
    output.mkdir()  # Exclusive ownership before cleanup is allowed.
    try:
        for index, video in enumerate(videos, start=1):
            result = run_video(video, source_kind=document["source_kind"], max_frames=max_frames,
                               detector_kind=detector_kind, weights=weights, include_detections=True)
            source = inventory[video.name]
            if result["video_sha256"] != source["video_sha256"]:
                raise ValueError(f"Video changed after source inspection: {video}")
            if result["frames_processed"] != min(max_frames, source["frame_count"]):
                raise ValueError(f"Processed window disagrees with decoded source: {video}")
            result["split"] = document["split"]
            result["batch_manifest_sha256"] = manifest_hash
            result["source_inventory"] = source
            result["helmet_labels_sha256"] = label_hash
            result["batch_diagnostics"] = _summarize(result, None if native is None else native[video.name])
            directory = f"clip-{index:04d}"
            clip_output = output / directory
            clip_output.mkdir()
            analysis = clip_output / "analysis.json"
            analysis.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            render_video(video, analysis, clip_output / "review.avi")
            write_contact_sheet(clip_output / "review.avi", clip_output / "preview.png", result["frames_processed"])
            summary["videos"].append(dict(video=video.name, directory=directory, source=source,
                                           diagnostics=result["batch_diagnostics"]))
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        _write_reports(summary, output)
        artifacts = sorted(path for path in output.rglob("*") if path.is_file())
        hashes = {path.relative_to(output).as_posix(): _sha256(path) for path in artifacts}
        (output / "manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        with ZipFile(output / "bundle.zip", "x", compression=ZIP_DEFLATED) as archive:
            for path in [*artifacts, output / "manifest.json"]:
                archive.write(path, arcname=path.relative_to(output).as_posix())
            archive.writestr("OPEN_ME.txt", "Extract all files, then open report.html or report.md.\n"
                             "Open each clip's preview.png directly, or review.avi in a video player.\n"
                             "Missing helmet annotations are unavailable; containment is not player accuracy.\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new result directory")
    parser.add_argument("--max-frames", type=int, default=300, help="inference prefix per clip (default: 300)")
    parser.add_argument("--detector", choices=["turf", "yolox", "yolox-tiled"], default="turf")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--helmets", type=Path, help="native ground-truth train_labels.csv, not baseline predictions")
    parser.add_argument("--opencv-threads", type=int, default=2)
    args = parser.parse_args()
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    cv2.setNumThreads(args.opencv_threads)
    try:
        result = run_batch(args.manifest, args.out, max_frames=args.max_frames,
                           detector_kind=args.detector, weights=args.weights, helmet_labels=args.helmets)
    except (ValueError, TypeError, KeyError, OSError, cv2.error) as exc:
        parser.exit(2, f"Batch review failed: {exc}\n")
    print(f"Reviewed {len(result['videos'])} clips: {args.out}")


if __name__ == "__main__":
    main()
