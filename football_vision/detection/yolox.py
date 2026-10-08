"""Official YOLOX-Tiny 0.1.1rc0 COCO person inference using OpenCV DNN.

This fixed artifact profile uses unnormalized BGR, top-left padding and raw grid
outputs. The old tagged demo's RGB/ImageNet preparation is incompatible with the
currently published file; bind preprocessing to the model hash, not just its tag.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from football_vision.detection.detector import BasePlayerDetector

YOLOX_TINY_SHA256 = "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
YOLOX_TINY_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"


class YoloXTinyPlayerDetector(BasePlayerDetector):
    """Pretrained person candidates; does not establish player/team identity."""

    detector_name = "yolox_tiny_coco_0.1.1rc0"
    source_type = "pretrained_coco_person"
    input_size = 416

    def __init__(self, weights: Path, *, min_confidence: float = 0.3,
                 nms_iou_threshold: float = 0.45) -> None:
        if not np.isfinite(min_confidence) or not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must be finite and in (0, 1]")
        if not np.isfinite(nms_iou_threshold) or not 0 < nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be finite and in (0, 1]")
        weights = Path(weights)
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != YOLOX_TINY_SHA256:
            raise ValueError("Weights do not match the supported YOLOX-Tiny 0.1.1rc0 profile; "
                             "check the official download and SHA-256")
        self.min_confidence = float(min_confidence)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.weights_sha256 = digest
        self._net = cv2.dnn.readNetFromONNX(str(weights))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        grids, strides = [], []
        for stride in (8, 16, 32):
            side = self.input_size // stride
            x, y = np.meshgrid(np.arange(side), np.arange(side))
            grids.append(np.column_stack((x.ravel(), y.ravel())))
            strides.append(np.full((side * side, 1), stride))
        self._grid = np.concatenate(grids).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)

    def metadata(self) -> dict:
        return dict(detector=self.detector_name, detector_source=self.source_type,
                    detector_accuracy_status="real_video_unmeasured",
                    detector_settings=dict(input_size=self.input_size, min_confidence=self.min_confidence,
                                           nms_iou_threshold=self.nms_iou_threshold, device="cpu",
                                           opencv_threads=cv2.getNumThreads()),
                    model=dict(sha256=self.weights_sha256, source_url=YOLOX_TINY_URL,
                               profile="yolox-tiny-bgr-0-255-raw-grid",
                               training_dataset="COCO", selected_class="person", selected_class_id=0,
                               scope="person candidates; players/referees/spectators not distinguished"))

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        ratio = min(self.input_size / height, self.input_size / width)
        resized = cv2.resize(image, (max(1, int(width * ratio)), max(1, int(height * ratio))))
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.float32)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), ratio

    def detect(self, frame_bgr: np.ndarray, frame_id: int = 0, *,
               sideline_v_bounds_px=None):
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("YOLOX input must be a uint8 BGR image")
        blob, ratio = self._prepare(frame_bgr)
        self._net.setInput(blob)
        raw = np.asarray(self._net.forward())
        if raw.shape != (1, len(self._grid), 85) or not np.isfinite(raw).all():
            raise ValueError("YOLOX returned incompatible or non-finite raw predictions")
        prediction = raw[0]
        probabilities = prediction[:, 4:]
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("YOLOX objectness/class scores must be probabilities")
        scores = prediction[:, 4] * prediction[:, 5]  # objectness × COCO person
        selected = np.flatnonzero(scores >= self.min_confidence)
        if not len(selected):
            return []
        center = (prediction[selected, :2] + self._grid[selected]) * self._strides[selected]
        with np.errstate(over="ignore", invalid="ignore"):
            size = np.exp(prediction[selected, 2:4]) * self._strides[selected]
        boxes = np.column_stack((center - size / 2, center + size / 2)) / ratio
        height, width = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = np.isfinite(size).all(axis=1) & np.isfinite(boxes).all(axis=1)
        valid &= np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
        boxes, scores = boxes[valid], scores[selected][valid]
        if not len(boxes):
            return []
        xywh = np.column_stack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]))
        # Scores have already been gated; threshold zero keeps exact-boundary scores.
        kept = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0,
                                         self.nms_iou_threshold)).reshape(-1).astype(int)
        return self.build_detections_from_boxes(
            boxes[kept], scores[kept], (width, height), frame_id,
            sideline_v_bounds_px=sideline_v_bounds_px,
            extra_metadata=[dict(model_sha256=self.weights_sha256, class_id=0, class_name="person")
                            for _ in kept],
        )


class YoloXTiledPlayerDetector(YoloXTinyPlayerDetector):
    """Full image plus four overlapping crops, with global image-space NMS.

    Keeps fixed verified model input while increasing the pixel size of small
    people. Costs up to five inference calls; not a football-trained model.
    """

    detector_name = "yolox_tiny_coco_0.1.1rc0_tiled"

    def metadata(self) -> dict:
        result = super().metadata()
        result["detector_settings"].update(
            tiling="full image plus overlapping 2x2 crops", crop_overlap_fraction=0.2,
            max_inference_calls=5, internal_edge_margin_px=2,
            merge="global NMS in original image coordinates",
        )
        return result

    def detect(self, frame_bgr: np.ndarray, frame_id: int = 0, *, sideline_v_bounds_px=None):
        full = super().detect(frame_bgr, frame_id, sideline_v_bounds_px=sideline_v_bounds_px)
        if frame_bgr is None or frame_bgr.size == 0:
            return full
        height, width = frame_bgr.shape[:2]
        if max(height, width) <= self.input_size:
            return full
        boxes = [list(d.bbox) for d in full]
        scores = [d.confidence for d in full]
        for x1, x2 in ((0, (3 * width + 4) // 5), (2 * width // 5, width)):
            for y1, y2 in ((0, (3 * height + 4) // 5), (2 * height // 5, height)):
                # Footpoints must be recomputed after translation using the full
                # image and full-image sideline bounds, never local crop bounds.
                for detection in super().detect(frame_bgr[y1:y2, x1:x2], frame_id):
                    a, b, c, d = detection.bbox
                    if ((x1 > 0 and a <= 2) or (y1 > 0 and b <= 2)
                            or (x2 < width and c >= x2 - x1 - 2)
                            or (y2 < height and d >= y2 - y1 - 2)):
                        continue  # Refuse truncated boxes at internal crop edges.
                    boxes.append([a + x1, b + y1, c + x1, d + y1])
                    scores.append(detection.confidence)
        if not boxes:
            return []
        xywh = [[a, b, c - a, d - b] for a, b, c, d in boxes]
        kept = np.asarray(cv2.dnn.NMSBoxes(xywh, scores, 0.0, self.nms_iou_threshold)).reshape(-1)
        return self.build_detections_from_boxes(
            [boxes[int(i)] for i in kept], [scores[int(i)] for i in kept],
            (width, height), frame_id, sideline_v_bounds_px=sideline_v_bounds_px,
            extra_metadata=[dict(model_sha256=self.weights_sha256, class_id=0, class_name="person",
                                 inference_mode="full_and_overlapping_crops") for _ in kept],
        )


def create_player_detector(kind: str = "turf", weights: Path | None = None, *,
                           min_confidence: float | None = None) -> BasePlayerDetector:
    """Explicit model choice with no automatic downloads or silent fallback."""
    if kind == "turf":
        if weights is not None:
            raise ValueError("--weights requires --detector yolox or yolox-tiled")
        from football_vision.detection.detector import TurfContrastPlayerDetector
        return TurfContrastPlayerDetector(**({"min_confidence": min_confidence} if min_confidence is not None else {}))
    if kind in ("yolox", "yolox-tiled"):
        if weights is None:
            raise ValueError(f"--detector {kind} requires a local --weights ONNX file")
        detector = YoloXTiledPlayerDetector if kind == "yolox-tiled" else YoloXTinyPlayerDetector
        return detector(weights, **({"min_confidence": min_confidence} if min_confidence is not None else {}))
    raise ValueError("detector must be turf, yolox or yolox-tiled")
