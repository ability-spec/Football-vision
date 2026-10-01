"""Footpoint extraction and ground-contact reliability analysis (Phase 3C).

Extracts the bottom-center ground-contact estimate ``(u_px, v_px)`` from player
bounding boxes while explicitly flagging and refusing unreliable footpoints:
  1. Partial truncation at frame borders (especially bottom-border clipping)
  2. Sideline proximity / bench-apron congestion
  3. Players crossing each other (multi-box lower-body occlusion)
  4. Temporary lower-body occlusion (squat/truncated upper-torso-only boxes)

Never claims a reliable ground-contact position when the footpoint is unreliable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from football_vision.schema import FootpointEstimate


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def _lower_body_overlap_ratio(
    target_box: Sequence[float],
    other_box: Sequence[float],
) -> Tuple[float, float]:
    """Return (horizontal_overlap_ratio, lower_third_area_overlap_ratio)."""
    tx1, ty1, tx2, ty2 = [float(v) for v in target_box]
    ox1, oy1, ox2, oy2 = [float(v) for v in other_box]
    tw = max(1e-6, tx2 - tx1)
    th = max(1e-6, ty2 - ty1)

    horiz_ov = max(0.0, min(tx2, ox2) - max(tx1, ox1)) / tw
    # Lower third of target box (where knees/ankles/cleats reside)
    low_y1 = ty1 + (2.0 / 3.0) * th
    low_y2 = ty2
    low_area = max(1e-6, tw * (low_y2 - low_y1))

    ix1, iy1 = max(tx1, ox1), max(low_y1, oy1)
    ix2, iy2 = min(tx2, ox2), min(low_y2, oy2)
    inter_low = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(horiz_ov), float(inter_low / low_area)


def extract_footpoint(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    *,
    detection_confidence: float = 1.0,
    other_bboxes: Optional[Sequence[Sequence[float]]] = None,
    sideline_v_bounds_px: Optional[Tuple[float, float]] = None,
    border_margin_px: float = 4.0,
    min_aspect_ratio: float = 1.05,
    crossing_lower_overlap_thresh: float = 0.45,
    metadata: Optional[Dict[str, object]] = None,
) -> FootpointEstimate:
    """Compute bottom-center footpoint and evaluate ground-contact reliability.

    Parameters
    ----------
    bbox:
        ``(x1, y1, x2, y2)`` in pixel coordinates.
    image_size:
        ``(width, height)`` of the frame in pixels.
    detection_confidence:
        Detector confidence in ``[0, 1]``.
    other_bboxes:
        Optional list of other player bounding boxes in the same frame to detect
        player crossings and lower-body occlusions.
    sideline_v_bounds_px:
        Optional ``(top_sideline_v_px, bottom_sideline_v_px)`` in image space.
        If omitted, defaults to ``(0.14 * H, 0.94 * H)`` as a conservative
        image-space sideline proximity band.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w_img, h_img = int(image_size[0]), int(image_size[1])
    bw = max(1e-3, x2 - x1)
    bh = max(1e-3, y2 - y1)

    u_px = 0.5 * (x1 + x2)
    v_px = float(y2)

    is_partially_truncated = False
    is_near_sideline = False
    is_player_crossing = False
    is_temporarily_occluded = False
    unreliable_reason: Optional[str] = None
    penalty = 0.0

    # 1. Partial truncation at image borders
    bottom_truncated = y2 >= (h_img - border_margin_px)
    top_truncated = y1 <= border_margin_px
    side_truncated = (x1 <= border_margin_px) or (x2 >= (w_img - border_margin_px))

    if bottom_truncated or top_truncated or side_truncated:
        is_partially_truncated = True
        if bottom_truncated:
            unreliable_reason = "bottom_border_truncated"
            penalty += 0.65
        elif side_truncated and (bw / max(1.0, bh)) < 0.25:
            unreliable_reason = "lateral_border_truncated"
            penalty += 0.55
        else:
            penalty += 0.12

    # 2. Sideline proximity in image space
    if sideline_v_bounds_px is not None:
        top_v, bot_v = float(sideline_v_bounds_px[0]), float(sideline_v_bounds_px[1])
    else:
        top_v, bot_v = 0.14 * h_img, 0.94 * h_img

    if v_px <= top_v or v_px >= bot_v:
        is_near_sideline = True
        penalty += 0.08
        if metadata and bool(metadata.get("sideline_occluded", False)):
            if unreliable_reason is None:
                unreliable_reason = "sideline_bench_occlusion"
            penalty += 0.55

    # 3. Players crossing each other
    if other_bboxes:
        for ob in other_bboxes:
            if tuple(float(v) for v in ob) == (x1, y1, x2, y2):
                continue
            iou = _bbox_iou((x1, y1, x2, y2), ob)
            horiz_ov, low_ov = _lower_body_overlap_ratio((x1, y1, x2, y2), ob)
            _, oy1, _, oy2 = [float(v) for v in ob]
            if iou >= 0.15 or (horiz_ov >= 0.35 and low_ov >= 0.25):
                is_player_crossing = True
                # If the other player's box covers our lower third and reaches at least as low as ours
                if low_ov >= crossing_lower_overlap_thresh and oy2 >= (y2 - 2.0):
                    is_temporarily_occluded = True
                    if unreliable_reason is None:
                        unreliable_reason = "player_crossing_lower_body_occluded"
                    penalty += 0.60
                else:
                    penalty += 0.10

    # 4. Temporary lower-body occlusion (aspect ratio or explicit flag)
    aspect_ratio = bh / bw
    if aspect_ratio < min_aspect_ratio:
        is_temporarily_occluded = True
        if unreliable_reason is None:
            unreliable_reason = "lower_body_occluded_aspect_ratio"
        penalty += 0.60

    if metadata:
        if bool(metadata.get("lower_body_occluded", False)):
            is_temporarily_occluded = True
            if unreliable_reason is None:
                unreliable_reason = "lower_body_occluded"
            penalty += 0.60
        if bool(metadata.get("player_crossing", False)):
            is_player_crossing = True

    is_reliable = unreliable_reason is None
    raw_conf = float(np.clip(detection_confidence * (1.0 - penalty), 0.0, 1.0))
    if not is_reliable:
        raw_conf = min(raw_conf, 0.35)

    return FootpointEstimate(
        u_px=round(u_px, 3),
        v_px=round(v_px, 3),
        is_reliable=is_reliable,
        confidence=round(raw_conf, 4),
        is_partially_truncated=is_partially_truncated,
        is_near_sideline=is_near_sideline,
        is_player_crossing=is_player_crossing,
        is_temporarily_occluded=is_temporarily_occluded,
        unreliable_reason=unreliable_reason,
    )


def extract_footpoints_for_frame(
    bboxes: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
    *,
    confidences: Optional[Sequence[float]] = None,
    sideline_v_bounds_px: Optional[Tuple[float, float]] = None,
    metadatas: Optional[Sequence[Optional[Dict[str, object]]]] = None,
) -> List[FootpointEstimate]:
    """Extract footpoints for all detections in a frame with mutual crossing checks."""
    n = len(bboxes)
    confs = list(confidences) if confidences is not None else [1.0] * n
    metas = list(metadatas) if metadatas is not None else [None] * n
    estimates: List[FootpointEstimate] = []
    for i in range(n):
        others = [bboxes[j] for j in range(n) if j != i]
        est = extract_footpoint(
            bboxes[i],
            image_size,
            detection_confidence=float(confs[i]),
            other_bboxes=others,
            sideline_v_bounds_px=sideline_v_bounds_px,
            metadata=metas[i],
        )
        estimates.append(est)
    return estimates
