"""Phase 1 Temporal Homography Tracker (`CalibrationTracker`).

Handles:
  - Camera-cut detection via HSV histogram correlation + field-space jump threshold
  - Inter-frame background-turf homography propagation (Shi-Tomasi + pyramidal LK optical flow + RANSAC)
  - Confidence-weighted temporal homography filtering
  - Multiplicative confidence decay when direct calibration fails on a frame
  - Strict refusal to project coordinates once confidence drops below `min_confidence` or `age > max_age`
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from football_vision.field_spec import MIN_CALIBRATION_CONFIDENCE
from football_vision.schema import CalibrationResult, XCoordMode
from football_vision.calibration.homography import plausible_homography


def _compute_field_hist(frame: np.ndarray) -> np.ndarray:
    """Compute normalized 2D H-S histogram over the main playing-field band."""
    h, w = frame.shape[:2]
    crop = frame[int(0.10 * h):int(0.82 * h), int(0.05 * w):int(0.95 * w)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _estimate_turf_interframe_homography(
    prev_gray: np.ndarray,
    prev_turf: np.ndarray,
    curr_gray: np.ndarray,
    curr_turf: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate inter-frame pixel homography dH (prev_uv -> curr_uv) on green turf background."""
    h, w = prev_gray.shape[:2]
    mask = prev_turf.copy()
    mask[:int(0.08 * h), :] = 0
    mask[int(0.84 * h):, :] = 0

    pts0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=250, qualityLevel=0.01, minDistance=12, mask=mask
    )
    if pts0 is None or len(pts0) < 12:
        return None

    pts1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts0, None, winSize=(21, 21), maxLevel=3
    )
    if pts1 is None or st_fwd is None:
        return None

    pts0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, pts1, None, winSize=(21, 21), maxLevel=3
    )
    if pts0_back is None or st_bwd is None:
        return None

    fb_err = np.linalg.norm((pts0 - pts0_back).reshape(-1, 2), axis=1)
    good = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1) & (fb_err < 1.5)

    p0 = pts0.reshape(-1, 2)[good]
    p1 = pts1.reshape(-1, 2)[good]
    if len(p0) < 10:
        return None

    # Keep points that remain inside the current turf mask
    in_bounds = (
        (p1[:, 0] >= 4)
        & (p1[:, 0] < w - 4)
        & (p1[:, 1] >= int(0.08 * h))
        & (p1[:, 1] < int(0.84 * h))
    )
    p0, p1 = p0[in_bounds], p1[in_bounds]
    if len(p0) < 10:
        return None

    turf_ok = np.array([curr_turf[int(round(y)), int(round(x))] > 0 for x, y in p1], dtype=bool)
    p0, p1 = p0[turf_ok], p1[turf_ok]
    if len(p0) < 8:
        return None

    dH, inliers = cv2.findHomography(p0, p1, cv2.RANSAC, 2.5)
    if dH is None or inliers is None or int(np.sum(inliers)) < 6 or abs(np.linalg.det(dH)) < 1e-9:
        return None
    return dH / dH[2, 2]


class CalibrationTracker:
    """Temporal field-calibration tracker with camera-cut reset, turf propagation, and confidence decay."""

    def __init__(
        self,
        smoothing: float = 0.5,
        max_age: int = 5,
        max_jump_yd: float = 4.0,
        decay_factor: float = 0.80,
        min_confidence: float = MIN_CALIBRATION_CONFIDENCE,
        cut_hist_corr_thresh: float = 0.70,
    ):
        self.smoothing = smoothing
        self.max_age = max_age
        self.max_jump_yd = max_jump_yd
        self.decay_factor = decay_factor
        self.min_confidence = min_confidence
        self.cut_hist_corr_thresh = cut_hist_corr_thresh

        self._coordinate_epoch = 0
        self._input_coordinate_id: Optional[str] = None
        self.H: Optional[np.ndarray] = None
        self.age: int = 0
        self.confidence: float = 0.0
        self.x_coord_mode: XCoordMode = "uncalibrated"
        self.last_result: Optional[CalibrationResult] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_turf: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear all temporal state (called on camera cuts or explicit reset)."""
        self._coordinate_epoch += 1
        self._input_coordinate_id = None
        self.H = None
        self.age = 0
        self.confidence = 0.0
        self.x_coord_mode = "uncalibrated"
        self.last_result = None
        self._prev_hist = None
        self._prev_gray = None
        self._prev_turf = None

    def update(
        self,
        H_new: Optional[np.ndarray],
        frame_size: Tuple[int, int],
        confidence: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Matrix-level update for backward compatibility; returns None if confidence < min_confidence."""
        if H_new is not None:
            obs_conf = 1.0 if confidence is None else float(confidence)
            if obs_conf < self.min_confidence or not plausible_homography(H_new, frame_size):
                return self._decay_matrix_only(frame_size)
            H_new = H_new / H_new[2, 2]
            if self.H is None or self._jump_yd(self.H, H_new, frame_size) > self.max_jump_yd:
                self.H = H_new
                self.confidence = obs_conf
            else:
                blended = self.smoothing * self.H + (1.0 - self.smoothing) * H_new
                self.H = blended if plausible_homography(blended, frame_size) else H_new
                self.confidence = obs_conf
            self.age = 0
            self.x_coord_mode = "relative_10yd"
            return self.H
        return self._decay_matrix_only(frame_size)

    def _decay_matrix_only(self, frame_size: Tuple[int, int]) -> Optional[np.ndarray]:
        if self.H is None:
            return None
        self.age += 1
        self.confidence = round(self.confidence * self.decay_factor, 4)
        if self.age > self.max_age or self.confidence < self.min_confidence:
            self.H = None
            self.confidence = 0.0
            self.x_coord_mode = "uncalibrated"
        return self.H

    def update_from_result(
        self,
        cal: CalibrationResult,
        frame: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Update temporal tracker with a single-frame CalibrationResult and optional BGR frame."""
        frame_size = cal.image_size
        if frame_size is None and frame is not None:
            frame_size = (frame.shape[1], frame.shape[0])
        if frame_size is None:
            frame_size = (900, 506)

        is_cut = False
        curr_hist = None
        curr_gray = None
        curr_turf = None
        dH_interframe = None

        if frame is not None and frame.ndim == 3:
            curr_hist = _compute_field_hist(frame)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            curr_turf = cv2.inRange(hsv, (28, 25, 40), (88, 255, 255))

            if self._prev_hist is not None:
                corr = float(cv2.compareHist(self._prev_hist, curr_hist, cv2.HISTCMP_CORREL))
                if corr < self.cut_hist_corr_thresh:
                    is_cut = True

            if not is_cut and self.H is not None and self._prev_gray is not None and self._prev_turf is not None:
                dH_interframe = _estimate_turf_interframe_homography(
                    self._prev_gray, self._prev_turf, curr_gray, curr_turf
                )

        # Emit a one-frame lifecycle signal even when direct calibration succeeds.
        cal = replace(cal, camera_cut_detected=is_cut)
        if is_cut:
            self.H = None
            self.age = 0
            self.confidence = 0.0
            self.x_coord_mode = "uncalibrated"
            self.last_result = None

        # Update cached frame features for next step
        if curr_hist is not None:
            self._prev_hist = curr_hist
            self._prev_gray = curr_gray
            self._prev_turf = curr_turf

        # Case 1: Direct single-frame calibration succeeded with sufficient confidence
        if cal.can_project(min_confidence=self.min_confidence):
            H_obs = cal.H / cal.H[2, 2]
            H_prior = self._propagate_prior(dH_interframe, frame_size)

            discontinuity = (
                H_prior is None
                or self.x_coord_mode != cal.x_coord_mode
                or cal.coordinate_frame_id != self._input_coordinate_id
                or self._jump_yd(H_prior, H_obs, frame_size) > self.max_jump_yd
            )
            if discontinuity:
                self._coordinate_epoch += 1
                H_filt = H_obs
            else:
                w_prior = self.smoothing * (self.confidence / max(1e-6, self.confidence + cal.confidence))
                blended = w_prior * H_prior + (1.0 - w_prior) * H_obs
                blended = blended / blended[2, 2]
                H_filt = blended if plausible_homography(blended, frame_size) else H_obs

            self._input_coordinate_id = cal.coordinate_frame_id
            self.H = H_filt
            self.age = 0
            self.confidence = cal.confidence
            self.x_coord_mode = cal.x_coord_mode

            out = replace(
                cal,
                success=True,
                coordinate_frame_id=f"calibration:{self._coordinate_epoch}",
                H=H_filt,
                H_inv=np.linalg.inv(H_filt),
                confidence=self.confidence,
                x_coord_mode=self.x_coord_mode,
                failure_reason=None,
                is_temporally_propagated=False,
                propagation_age=0,
            )
            self.last_result = out
            return out

        # Case 2: Direct calibration failed on a camera cut -> immediately report camera_cut_uncalibrated
        if is_cut:
            out = replace(
                cal,
                success=False,
                H=None,
                H_inv=None,
                confidence=0.0,
                x_coord_mode="uncalibrated",
                failure_reason="camera_cut_uncalibrated",
                is_temporally_propagated=False,
                propagation_age=0,
                notes=list(cal.notes) + ["Camera cut detected and new frame failed calibration"],
            )
            self.last_result = out
            return out

        # Case 3: Direct calibration failed, no prior valid calibration -> return failure as-is
        if self.H is None:
            out = replace(
                cal,
                success=False,
                H=None,
                H_inv=None,
                confidence=0.0,
                x_coord_mode="uncalibrated",
                is_temporally_propagated=False,
                propagation_age=0,
            )
            self.last_result = out
            return out

        # Case 4: Direct calibration failed, prior calibration exists -> propagate & decay confidence
        self.age += 1
        self.confidence = float(np.round(self.confidence * self.decay_factor, 4))
        H_prop = self._propagate_prior(dH_interframe, frame_size)

        if (
            H_prop is None
            or self.age > self.max_age
            or self.confidence < self.min_confidence
        ):
            self.H = None
            self.x_coord_mode = "uncalibrated"
            out = replace(
                cal,
                success=False,
                H=None,
                H_inv=None,
                confidence=self.confidence,
                x_coord_mode="uncalibrated",
                failure_reason="confidence_expired",
                is_temporally_propagated=False,
                propagation_age=self.age,
                notes=list(cal.notes) + [
                    f"Temporal confidence expired (age={self.age}, confidence={self.confidence:.3f} < {self.min_confidence:.3f})"
                ],
            )
            self.last_result = out
            return out

        self.H = H_prop
        out = replace(
            cal,
            success=True,
            coordinate_frame_id=f"calibration:{self._coordinate_epoch}",
            H=H_prop,
            H_inv=np.linalg.inv(H_prop),
            plausible_orientation_scale=True,
            confidence=self.confidence,
            x_coord_mode=self.x_coord_mode,
            failure_reason=None,
            is_temporally_propagated=True,
            propagation_age=self.age,
            notes=list(cal.notes) + [
                f"Temporally propagated (age={self.age}, confidence={self.confidence:.3f}, flow_used={dH_interframe is not None})"
            ],
        )
        self.last_result = out
        return out

    def _propagate_prior(
        self,
        dH_interframe: Optional[np.ndarray],
        frame_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if self.H is None:
            return None
        if dH_interframe is not None:
            try:
                dH_inv = np.linalg.inv(dH_interframe)
                H_prop = self.H @ dH_inv
                if abs(H_prop[2, 2]) > 1e-9:
                    H_prop = H_prop / H_prop[2, 2]
                    if plausible_homography(H_prop, frame_size) and self._jump_yd(self.H, H_prop, frame_size) <= self.max_jump_yd * 2.0:
                        return H_prop
            except np.linalg.LinAlgError:
                pass
        return self.H

    @staticmethod
    def _jump_yd(H_old: np.ndarray, H_new: np.ndarray, frame_size: Tuple[int, int]) -> float:
        w, h = frame_size
        probes = np.float32([
            [[w * 0.5, h * 0.5]],
            [[w * 0.25, h * 0.5]],
            [[w * 0.75, h * 0.5]],
        ])
        old_xy = cv2.perspectiveTransform(probes, H_old)[:, 0]
        new_xy = cv2.perspectiveTransform(probes, H_new)[:, 0]
        return float(np.max(np.linalg.norm(old_xy - new_xy, axis=1)))

    def can_project(self) -> bool:
        """Return True iff the tracker currently holds a valid, above-threshold calibration."""
        return bool(self.last_result is not None and self.last_result.can_project(self.min_confidence))

    def project_image_points(
        self,
        pts_uv: Union[Sequence[Sequence[float]], np.ndarray],
    ) -> Optional[np.ndarray]:
        """Project image pixel points (N, 2) -> field yards (N, 2), or return None if uncertain."""
        if self.last_result is None:
            return None
        return self.last_result.image_to_field(pts_uv, min_confidence=self.min_confidence)

    def reproject_field_points(
        self,
        pts_xy_yd: Union[Sequence[Sequence[float]], np.ndarray],
    ) -> Optional[np.ndarray]:
        """Reproject field points (N, 2) in yards -> image pixels (N, 2), or return None if uncertain."""
        if self.last_result is None:
            return None
        return self.last_result.field_to_image(pts_xy_yd, min_confidence=self.min_confidence)

