"""Import CVAT image boxes and measure sampled detection, without inventing identities."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from football_vision.detection.yolox import create_player_detector
from football_vision.tracking.tracker import _bbox_iou


def load_cvat(archive: Path, images: Path) -> dict:
    """Validate all XML and source images before inference; never extract ZIP paths."""
    raw = Path(archive).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as package:
        entries = [i for i in package.infolist() if i.filename == "annotations.xml"]
        if len(entries) != 1 or entries[0].file_size > 10_000_000:
            raise ValueError("ZIP must contain one annotations.xml smaller than 10 MB")
        xml = package.read(entries[0]).decode("utf-8-sig")
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise ValueError("XML entity declarations are unsupported")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("Invalid CVAT XML") from exc
    if root.tag != "annotations" or root.findtext("version") != "1.1" or root.findall("track"):
        raise ValueError("Expected CVAT for images 1.1, without tracks")
    records, names, ids = [], set(), set()
    for node in root.findall("image"):
        name = node.get("name", "")
        if not name or Path(name).name != name or "\\" in name or name in names:
            raise ValueError("Image names must be unique local basenames")
        try:
            image_id = int(node.attrib["id"])
            width, height = int(node.attrib["width"]), int(node.attrib["height"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid CVAT image dimensions or id") from exc
        if image_id < 0 or image_id in ids or width <= 0 or height <= 0:
            raise ValueError("Invalid or duplicate CVAT image id/dimensions")
        path = (Path(images) / name).resolve()
        if path.parent != Path(images).resolve():
            raise ValueError("Source image must remain inside images directory")
        image_bytes = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"Source image size differs from annotation: {name}")
        boxes = []
        for box in node:
            if box.tag != "box" or box.get("label") != "player" or box.get("source") != "manual":
                raise ValueError("Only independently drawn manual player boxes are supported")
            try:
                bounds = [float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
            except (KeyError, ValueError) as exc:
                raise ValueError("Invalid player box") from exc
            x1, y1, x2, y2 = bounds
            if not np.isfinite(bounds).all() or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Player box must be finite, nonempty and inside image")
            boxes.append(bounds)
        records.append(dict(image=name, cvat_image_id=image_id, image_size=[width, height],
                            image_sha256=hashlib.sha256(image_bytes).hexdigest(), boxes=boxes))
        names.add(name)
        ids.add(image_id)
    if not records:
        raise ValueError("CVAT export contains no images")
    return dict(schema_version=1, annotation_zip_sha256=hashlib.sha256(raw).hexdigest(),
                annotation_method="independent_manual_declared", identity_status="not_annotated",
                frames=records)


def score_boxes(labels: list, predictions: list, *, iou_threshold: float = 0.5) -> dict:
    """One-to-one maximum-cardinality matching; duplicate detections count as FP."""
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be finite and in (0, 1]")
    for box in labels + predictions:
        a = np.asarray(box, dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all() or np.any(a[2:] <= a[:2]):
            raise ValueError("Boxes must be finite xyxy with positive area")
    matrix = np.asarray([[_bbox_iou(g, p) for p in predictions] for g in labels],
                        dtype=float).reshape(len(labels), len(predictions))
    matches = []
    if matrix.size:
        valid = matrix >= iou_threshold
        bonus = min(matrix.shape) + 1
        rows, cols = linear_sum_assignment(-(valid * bonus + np.where(valid, matrix, 0)))
        matches = [dict(annotation_box_index=int(r), prediction_index=int(c), iou=float(matrix[r, c]))
                   for r, c in zip(rows, cols) if valid[r, c]]
    tp = len(matches)
    return dict(true_positives=tp, false_positives=len(predictions) - tp,
                false_negatives=len(labels) - tp, matches=matches)


def compare_cvat(archive: Path, images: Path, weights: Path, output: Path, *,
                 confirm_exhaustive: bool = False, source_kind: str = "real", split: str = "dev") -> dict:
    """Score submitted player boxes only after explicit owner completeness confirmation."""
    if confirm_exhaustive is not True:
        raise ValueError("Confirm exhaustive visible-player annotations before computing precision/recall")
    if source_kind not in ("real", "synthetic") or split not in ("dev", "test"):
        raise ValueError("Invalid source_kind/split")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    annotations = load_cvat(archive, images)
    result = dict(schema_version=1, source_kind=source_kind, split=split,
                  accuracy_status="owner_confirmed_manual_sample_not_independently_audited",
                  annotation_zip_sha256=annotations["annotation_zip_sha256"],
                  completeness_confirmation="owner_declared_exhaustive_visible_players",
                  annotated_frames=len(annotations["frames"]), iou_threshold=0.5,
                  tracking=None, field=None, opencv_version=cv2.__version__,
                  scope="Sampled player-box detection only; no identity, field or generalization measurement. "
                        "COCO person detector also detects referees and bystanders, counted as false positives.",
                  modes={})
    output.mkdir(parents=True)
    try:
        for mode in ("yolox", "yolox-tiled"):
            detector = create_player_detector(mode, weights)
            records = []
            for frame in annotations["frames"]:
                raw = (Path(images) / frame["image"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != frame["image_sha256"]:
                    raise ValueError("Source image changed after validation")
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                # This id is local to the CVAT image collection, not a source-video frame or identity.
                predictions = detector.detect(image, frame["cvat_image_id"])
                boxes = [list(p.bbox) for p in predictions]
                record = dict(image=frame["image"], image_sha256=frame["image_sha256"],
                              **score_boxes(frame["boxes"], boxes),
                              detections=[dict(bbox=list(p.bbox), confidence=float(p.confidence))
                                          for p in predictions])
                records.append(record)
            counts = {k: sum(r[k] for r in records)
                      for k in ("true_positives", "false_positives", "false_negatives")}
            tp, fp, fn = (counts[k] for k in ("true_positives", "false_positives", "false_negatives"))
            result["modes"][mode] = dict(**counts, precision=tp / (tp + fp) if tp + fp else None,
                                        recall=tp / (tp + fn) if tp + fn else None,
                                        detector=detector.metadata(), frames=records)
        annotations["exhaustive"] = True
        annotations["completeness_confirmation"] = result["completeness_confirmation"]
        (output / "annotations.json").write_text(json.dumps(annotations, indent=2, allow_nan=False))
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        lines = ["# Independent manual detection pilot", "", result["scope"], "",
                 f"{result['annotated_frames']} images; IoU >= 0.5. Completeness confirmed by owner; no second-annotator audit.",
                 "", "| Mode | TP | FP | FN | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for mode, metrics in result["modes"].items():
            ratio = lambda k: f"{metrics[k]:.1%}" if metrics[k] is not None else "undefined"
            lines.append(f"| {mode} | {metrics['true_positives']} | {metrics['false_positives']} | "
                         f"{metrics['false_negatives']} | {ratio('precision')} | {ratio('recall')} |")
        lines += ["", f"Split: {split}. Four isolated development frames do not measure full-clip or held-out accuracy."]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confirm-exhaustive", action="store_true")
    parser.add_argument("--source-kind", choices=("real", "synthetic"), default="real")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--opencv-threads", type=int, default=2)
    args = parser.parse_args()
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    cv2.setNumThreads(args.opencv_threads)
    try:
        result = compare_cvat(args.archive, args.images, args.weights, args.out,
                              confirm_exhaustive=args.confirm_exhaustive,
                              source_kind=args.source_kind, split=args.split)
    except (ValueError, OSError, zipfile.BadZipFile, cv2.error) as exc:
        parser.exit(2, f"CVAT evaluation failed: {exc}\n")
    print(f"Evaluated {result['annotated_frames']} manually annotated images: {args.out}")


if __name__ == "__main__":
    main()
