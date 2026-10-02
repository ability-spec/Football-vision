"""Score independently annotated frames against saved predictions.

Run: python -m benchmarks.evaluate_real_video annotations.json predictions.json --out metrics.json
Missing predictions count as misses. Field error is reported with coverage, never alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from football_vision.tracking.tracker import _bbox_iou


def _frames(document: dict, *, ground_truth: bool) -> dict:
    if document.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(document.get("frames"), list) or (ground_truth and not document["frames"]):
        raise ValueError("frames must be a list; ground truth must be nonempty")
    output = {}
    for frame in document["frames"]:
        fid = frame["frame_id"]
        if type(fid) is not int or fid < 0 or fid in output:
            raise ValueError("frame_id must be a unique nonnegative integer")
        if ground_truth and frame.get("exhaustive") is not True:
            raise ValueError("Each GT frame must exhaustively label visible players")
        players = frame["players"]
        if not isinstance(players, list):
            raise ValueError("players must be a list (empty for a confirmed negative frame)")
        seen = set()
        for player in players:
            tid = player["track_id"]
            if not isinstance(tid, (str, int)) or isinstance(tid, bool) or tid in seen:
                raise ValueError("track_id must be unique within each frame")
            seen.add(tid)
            if "observed" in player and type(player["observed"]) is not bool:
                raise ValueError("observed must be boolean")
            bbox = np.asarray(player["bbox"], dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
                raise ValueError("bbox must be finite [x1,y1,x2,y2] with positive area")
            xy = player.get("field_position")
            if xy is not None:
                xy = np.asarray(xy, dtype=float)
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    raise ValueError("field_position must be two finite yard coordinates")
                if player.get("x_coord_mode") not in ("absolute", "relative_10yd", "relative_5yd"):
                    raise ValueError("field_position requires a projectable x_coord_mode")
                if (
                    not isinstance(player.get("coordinate_frame_id"), str)
                    or not player["coordinate_frame_id"]
                ):
                    raise ValueError("field_position requires explicit coordinate_frame_id")
        output[fid] = players
    return output


def evaluate(
    annotations: dict, predictions: dict, *, iou_threshold: float = 0.5, field_threshold_yd: float = 1.0
) -> dict:
    if not 0 < iou_threshold <= 1 or not np.isfinite(field_threshold_yd) or field_threshold_yd <= 0:
        raise ValueError("Invalid metric thresholds")
    gt = _frames(annotations, ground_truth=True)
    pred = _frames(predictions, ground_truth=False)
    video_hash = annotations.get("video_sha256", "")
    if len(video_hash) != 64 or any(c not in "0123456789abcdef" for c in video_hash):
        raise ValueError("annotations require the video's SHA-256")
    if video_hash != predictions.get("video_sha256"):
        raise ValueError("Annotation/prediction video hashes differ")
    kind = annotations.get("source_kind")
    if kind not in ("real", "synthetic") or kind != predictions.get("source_kind"):
        raise ValueError("source_kind must agree and be real or synthetic")
    if annotations.get("split") not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    if annotations.get("annotation_method") != "independent_manual":
        raise ValueError("Labels must be independently annotated, not generated from predictions")

    tp = fp = fn = switches = fragments = n_field_gt = incompatible = 0
    previous_identity, ever_matched, currently_missing = {}, set(), set()
    errors = []
    for fid, labels in sorted(gt.items()):
        detections = pred.get(fid, [])
        # Coasting boxes are predictions of an absent observation, not detections.
        detections = [p for p in detections if p.get("observed", True)]
        matrix = np.array(
            [[_bbox_iou(g["bbox"], p["bbox"]) for p in detections] for g in labels], dtype=float
        ).reshape(len(labels), len(detections))
        matches = []
        if matrix.size:
            # Maximize cardinality among valid matches, then IoU; an invalid pair
            # must not displace two valid pairs in the assignment.
            valid = matrix >= iou_threshold
            bonus = min(matrix.shape) + 1
            rows, cols = linear_sum_assignment(-(valid * bonus + np.where(valid, matrix, 0.0)))
            matches = [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]
        tp += len(matches)
        fp += len(detections) - len(matches)
        fn += len(labels) - len(matches)
        matched = {r for r, _ in matches}
        n_field_gt += sum(g.get("field_position") is not None for g in labels)
        for i, g in enumerate(labels):
            if i not in matched and g["track_id"] in ever_matched:
                currently_missing.add(g["track_id"])
        for r, c in matches:
            g, p = labels[r], detections[c]
            gid, pid = g["track_id"], p["track_id"]
            switches += int(gid in previous_identity and previous_identity[gid] != pid)
            fragments += int(gid in currently_missing)
            currently_missing.discard(gid)
            ever_matched.add(gid)
            previous_identity[gid] = pid
            if g.get("field_position") is not None and p.get("field_position") is not None:
                compatible = (
                    g["x_coord_mode"] == p["x_coord_mode"]
                    and g["coordinate_frame_id"] == p["coordinate_frame_id"]
                )
                if compatible:
                    errors.append(
                        float(np.linalg.norm(np.asarray(g["field_position"]) - p["field_position"]))
                    )
                else:
                    incompatible += 1
    ratio = lambda n, d: n / d if d else None
    return {
        "schema_version": 1,
        "source_kind": kind,
        "split": annotations["split"],
        "accuracy_status": "independent_labels_declared_not_audited",
        "annotated_frames": len(gt),
        "unannotated_prediction_frames": len(set(pred) - set(gt)),
        "missing_prediction_frames": len(set(gt) - set(pred)),
        "thresholds": {"iou": iou_threshold, "field_error_yd": field_threshold_yd},
        "detection": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
        },
        "tracking": {
            "identity_switches": switches,
            "fragments": fragments,
            "definition": "IoU-associated identities over annotated frames; not full MOT/IDF1",
        },
        "field": {
            "ground_truth_positions": n_field_gt,
            "scored_positions": len(errors),
            "coverage": ratio(len(errors), n_field_gt),
            "incompatible_coordinate_frames": incompatible,
            "conditional_median_error_yd": float(np.median(errors)) if errors else None,
            "conditional_p95_error_yd": float(np.percentile(errors, 95)) if errors else None,
            "fraction_all_gt_within_threshold": ratio(
                sum(e <= field_threshold_yd for e in errors), n_field_gt
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--field-threshold-yd", type=float, default=1.0)
    args = parser.parse_args()
    try:
        a, p = args.annotations.read_bytes(), args.predictions.read_bytes()
        result = evaluate(
            json.loads(a),
            json.loads(p),
            iou_threshold=args.iou_threshold,
            field_threshold_yd=args.field_threshold_yd,
        )
        result["annotation_sha256"] = hashlib.sha256(a).hexdigest()
        result["prediction_sha256"] = hashlib.sha256(p).hexdigest()
        # Exclusive creation prevents accidentally replacing a previous benchmark.
        with args.out.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Evaluation refused: {exc}\n")


if __name__ == "__main__":
    main()
