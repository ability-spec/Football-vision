"""Temporal homography smoothing and camera-cut reset tracker (Phase 1 / Phase 2 module boundary)."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from football_vision.field_spec import FIELD_WIDTH_YD
from football_vision.calibration.homography import plausible_homography


class CalibrationTracker:
    """Smooths frame-to-frame field homographies and resets on camera cuts or large jumps.

    Ported to NFL field coordinates (yards) from Alex R. Haigh's HomographyTracker in Hockey-Vision.
    """

    def __init__(self, smoothing: float = 0.5, max_age: int = 5, max_jump_yd: float = 4.0):
        self.smoothing = smoothing
        self.max_age = max_age
        self.max_jump_yd = max_jump_yd
        self.H: Optional[np.ndarray] = None
        self.age: int = 0

    def reset(self) -> None:
        self.H = None
        self.age = 0

    def update(self, H_new: Optional[np.ndarray], frame_size: Tuple[int, int]) -> Optional[np.ndarray]:
        if H_new is not None:
            H_new = H_new / H_new[2, 2]
            if self.H is None or self._jump_yd(H_new, frame_size) > self.max_jump_yd:
                self.H = H_new
            else:
                blended = self.smoothing * self.H + (1.0 - self.smoothing) * H_new
                self.H = blended if plausible_homography(blended, frame_size) else H_new
            self.age = 0
        elif self.H is not None:
            self.age += 1
            if self.age > self.max_age:
                self.H = None
        return self.H

    def _jump_yd(self, H_new: np.ndarray, frame_size: Tuple[int, int]) -> float:
        import cv2
        w, h = frame_size
        probes = np.float32([
            [[w * 0.5, h * 0.5]],
            [[w * 0.25, h * 0.5]],
            [[w * 0.75, h * 0.5]],
        ])
        old_xy = cv2.perspectiveTransform(probes, self.H)[:, 0]
        new_xy = cv2.perspectiveTransform(probes, H_new)[:, 0]
        return float(np.max(np.linalg.norm(old_xy - new_xy, axis=1)))
