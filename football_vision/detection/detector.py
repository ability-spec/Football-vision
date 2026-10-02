"""Modular player detection interface and reproducible baseline detectors (Phase 3B).

Operates strictly in image space without any dependency on field calibration
or experimental CPU stand-in heuristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from football_vision.detection.footpoint import extract_footpoints_for_frame
from football_vision.schema import PlayerDetection


class BasePlayerDetector(ABC):
    """Abstract interface for Phase 3 player detectors.

    Implementations must operate purely in image space (independent of
    ``CalibrationResult``) and return ``List[PlayerDetection]``.
    """

    detector_name: str = "base_detector"
    source_type: str = "abstract"

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        *,
        sideline_v_bounds_px: Optional[Tuple[float, float]] = None,
    ) -> List[PlayerDetection]:
        """Detect players in ``frame_bgr`` and populate ``FootpointEstimate``."""
        raise NotImplementedError

    def build_detections_from_boxes(
        self,
        bboxes: Sequence[Sequence[float]],
        confidences: Sequence[float],
        image_size: Tuple[int, int],
        frame_id: int = 0,
        *,
        sideline_v_bounds_px: Optional[Tuple[float, float]] = None,
        extra_metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[PlayerDetection]:
        """Convert raw bounding boxes into canonical ``PlayerDetection`` objects."""
        footpoints = extract_footpoints_for_frame(
            bboxes,
            image_size,
            confidences=confidences,
            sideline_v_bounds_px=sideline_v_bounds_px,
            metadatas=extra_metadata,
        )
        detections: List[PlayerDetection] = []
        for idx, (box, conf, fp_est) in enumerate(zip(bboxes, confidences, footpoints)):
            x1, y1, x2, y2 = [round(float(v), 3) for v in box]
            meta: Dict[str, Any] = {
                "detector_name": self.detector_name,
                "source": self.source_type,
            }
            if extra_metadata and idx < len(extra_metadata) and extra_metadata[idx]:
                meta.update(extra_metadata[idx])
            detections.append(
                PlayerDetection(
                    frame_id=int(frame_id),
                    detection_id=f"f{int(frame_id):04d}_det{idx:03d}",
                    bbox=(x1, y1, x2, y2),
                    confidence=round(float(conf), 4),
                    footpoint=fp_est.xy,
                    footpoint_estimate=fp_est,
                    detector_metadata=meta,
                )
            )
        return detections


class TurfContrastPlayerDetector(BasePlayerDetector):
    """Reproducible baseline player detector using turf-masked foreground + NMS.

    Designed for deterministic CPU execution without external weights or
    calibration dependency. Finds non-turf vertical player figures inside the
    playing field region, suppresses long horizontal/diagonal field lines, and
    applies greedy IoU Non-Maximum Suppression.
    """

    detector_name = "turf_contrast_baseline_v1"
    source_type = "image_space_baseline"

    def __init__(
        self,
        *,
        min_confidence: float = 0.40,
        nms_iou_threshold: float = 0.35,
        min_height_frac: float = 0.026,
        max_height_frac: float = 0.24,
        min_width_frac: float = 0.007,
        max_width_frac: float = 0.085,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.min_height_frac = float(min_height_frac)
        self.max_height_frac = float(max_height_frac)
        self.min_width_frac = float(min_width_frac)
        self.max_width_frac = float(max_width_frac)

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        *,
        sideline_v_bounds_px: Optional[Tuple[float, float]] = None,
    ) -> List[PlayerDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Estimate dominant turf green mask and vertical playfield envelope
        turf_raw = cv2.inRange(hsv, (28, 35, 35), (88, 255, 255))
        row_green_frac = np.mean(turf_raw > 0, axis=1)
        green_rows = np.where(row_green_frac > 0.30)[0]
        if len(green_rows) < 10:
            return []

        v_top = int(max(8, green_rows[0]))
        v_bot = int(min(int(0.84 * h), green_rows[-1]))
        if v_bot <= v_top + 20:
            return []

        # 2. Local turf background subtraction: players (both dark and white jerseys)
        # have strong dark components (helmet/visor/pads/skin/cleat shadow) relative
        # to local turf, whereas painted white yard lines and yard numbers do not.
        bg_gray = cv2.medianBlur(gray, 31)
        diff_dark = np.clip(bg_gray.astype(np.float32) - gray.astype(np.float32), 0.0, 255.0)
        dark_mask = (diff_dark > 24.0).astype(np.uint8) * 255

        playfield_mask = np.zeros_like(dark_mask)
        playfield_mask[v_top:v_bot, int(0.02 * w):int(0.98 * w)] = 255
        fg = cv2.bitwise_and(dark_mask, playfield_mask)

        # Vertical morphological closing to connect helmet, jersey, and pants/cleats
        vert_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
        fg_closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, vert_kern)
        fg_clean = cv2.morphologyEx(
            fg_closed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
        )

        # Sobel edge energy for confidence scoring
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg_clean, connectivity=8)
        min_h = max(12.0, self.min_height_frac * h)
        max_h = self.max_height_frac * h
        min_w = max(6.0, self.min_width_frac * w)
        max_w = self.max_width_frac * w

        candidate_boxes: List[Tuple[float, float, float, float]] = []
        candidate_confs: List[float] = []

        for lbl in range(1, num_labels):
            x, y, bw, bh, area = [float(v) for v in stats[lbl]]
            if bh < min_h or bh > max_h or bw < min_w or bw > max_w:
                continue

            # Expand anchor to nominal upright player bounding box so white jerseys
            # above dark pants/cleat shadows are enclosed in the torso crop
            nom_h = max(bh, 0.078 * h)
            nom_w = max(bw, 0.024 * w)
            cx = x + 0.5 * bw
            by2 = min(float(h), y + bh + 0.12 * nom_h)
            by1 = max(0.0, by2 - nom_h)
            bx1 = max(0.0, cx - 0.5 * nom_w)
            bx2 = min(float(w), cx + 0.5 * nom_w)

            ebw = max(1.0, bx2 - bx1)
            ebh = max(1.0, by2 - by1)
            aspect = ebh / ebw

            # Require local turf context around the lower half of the candidate box
            pad_x = int(max(4, 0.35 * ebw))
            ry1 = int(max(0, by1 + 0.4 * ebh))
            ry2 = int(min(h, by2 + 6))
            rx1 = int(max(0, bx1 - pad_x))
            rx2 = int(min(w, bx2 + pad_x))
            ring_turf = float(np.mean(turf_raw[ry1:ry2, rx1:rx2] > 0)) if ry2 > ry1 and rx2 > rx1 else 0.0
            if ring_turf < 0.20:
                continue

            fill_ratio = min(1.0, area / max(1.0, bw * bh))
            patch_grad = float(np.mean(grad_mag[int(by1):int(by2), int(bx1):int(bx2)]))
            grad_score = float(np.clip(patch_grad / 55.0, 0.0, 1.0))
            shape_score = float(np.clip(1.0 - abs(aspect - 2.2) / 2.5, 0.25, 1.0))
            conf = float(np.clip(0.35 * fill_ratio + 0.40 * grad_score + 0.25 * shape_score, 0.0, 0.98))
            if conf >= self.min_confidence:
                candidate_boxes.append((bx1, by1, bx2, by2))
                candidate_confs.append(conf)

        # Deterministic NMS sorted by confidence descending, then x1 ascending
        order = sorted(
            range(len(candidate_boxes)),
            key=lambda i: (-round(candidate_confs[i], 6), candidate_boxes[i][0], candidate_boxes[i][1]),
        )
        kept_boxes: List[Tuple[float, float, float, float]] = []
        kept_confs: List[float] = []
        for idx in order:
            box = candidate_boxes[idx]
            if any(self._iou(box, kbox) > self.nms_iou_threshold for kbox in kept_boxes):
                continue
            kept_boxes.append(box)
            kept_confs.append(candidate_confs[idx])

        # Sort kept boxes left-to-right for deterministic detection_id assignment
        lr_order = sorted(range(len(kept_boxes)), key=lambda i: (kept_boxes[i][0], kept_boxes[i][1]))
        kept_boxes = [kept_boxes[i] for i in lr_order]
        kept_confs = [kept_confs[i] for i in lr_order]

        return self.build_detections_from_boxes(
            kept_boxes,
            kept_confs,
            (w, h),
            frame_id=frame_id,
            sideline_v_bounds_px=sideline_v_bounds_px,
        )

    @staticmethod
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return float(inter / (area_a + area_b - inter))


class FixturePlayerDetector(BasePlayerDetector):
    """Deterministic detector backed by pre-recorded frame box fixtures.

    Used in controlled multi-frame tracking and occlusion benchmarks so detector
    and tracker behaviors can be evaluated both jointly and independently.
    """

    detector_name = "fixture_detector_v1"
    source_type = "benchmark_fixture"

    def __init__(
        self,
        frame_fixtures: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        self.frame_fixtures = {int(k): list(v) for k, v in frame_fixtures.items()}

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        *,
        sideline_v_bounds_px: Optional[Tuple[float, float]] = None,
    ) -> List[PlayerDetection]:
        h, w = frame_bgr.shape[:2]
        entries = self.frame_fixtures.get(int(frame_id), [])
        bboxes = [e["bbox"] for e in entries]
        confs = [float(e.get("confidence", 0.90)) for e in entries]
        metas = [e.get("metadata") for e in entries]
        return self.build_detections_from_boxes(
            bboxes,
            confs,
            (w, h),
            frame_id=frame_id,
            sideline_v_bounds_px=sideline_v_bounds_px,
            extra_metadata=metas,
        )

