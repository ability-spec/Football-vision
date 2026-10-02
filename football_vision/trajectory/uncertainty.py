"""Uncertainty propagation for Phase 4 field-space trajectories.

Two models are used, both frozen engineering assumptions (NOT measured NFL
accuracy claims) and both documented explicitly:

1. **Footpoint pixel covariance** :func:`footpoint_pixel_covariance`
   The footpoint is the bottom-centre of a detection box. Its horizontal error is
   assumed to be half the box-side localization noise and its vertical error the
   full box-edge noise (the bottom edge is the least reliable because of contact
   shadows, clipping and merges). A hard floor guarantees the reported
   uncertainty is never exactly zero, so a sample can never claim infinite
   precision.

2. **First-order homography propagation** :func:`propagate_to_field_covariance`
   The 2x2 image covariance is pushed through the analytic Jacobian of the
   homography ``H`` at that pixel. An additive term covers calibration drift
   (larger when geometry was propagated by the Phase 2 optical-flow tracker, and
   when calibration confidence is low).

The resulting field covariance is always symmetrized and projected onto the
positive semi-definite cone.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from football_vision.field_spec import MIN_CALIBRATION_CONFIDENCE
from football_vision.schema import FootpointEstimate

# --- Frozen model constants (engineering assumptions, a priori) -------------
FOOTPOINT_EDGE_SIGMA_PX = 1.00        # per-edge box localization noise (px)
FOOTPOINT_SIGMA_FLOOR_PX = 0.30       # never claim sub-0.3 px precision
FOOTPOINT_HORIZONTAL_FACTOR = 0.70    # bottom-centre: horizontal error < vertical
FOOTPOINT_VERTICAL_FACTOR = 1.00
FOOTPOINT_CORRELATION = 0.20
FOOTPOINT_REFERENCE_BOX_HEIGHT_PX = 50.0
COV_FLOOR_YD2 = 1.0e-4                # 1 cm^2 floor: covariance is never zero


def _symmetrize_psd(cov: np.ndarray, floor: float = COV_FLOOR_YD2) -> np.ndarray:
    """Symmetrize a 2x2 covariance and clamp eigenvalues to a positive floor."""
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, floor, None)
    return (vecs * vals) @ vecs.T


def footpoint_pixel_covariance(
    bbox: Sequence[float],
    footpoint_estimate: Optional[FootpointEstimate] = None,
    *,
    detection_confidence: float = 0.90,
    edge_sigma_px: float = FOOTPOINT_EDGE_SIGMA_PX,
    extra_sigma_px: float = 0.0,
    reference_box_height_px: float = FOOTPOINT_REFERENCE_BOX_HEIGHT_PX,
) -> np.ndarray:
    """Return the assumed 2x2 pixel covariance of a bottom-centre footpoint estimate.

    Larger boxes are assumed to carry proportionally larger footpoint error; boxes
    smaller than ``0.5 * reference_box_height_px`` are treated as the reference
    scale (perspective far-field boxes still have at least a one-pixel edge noise).
    Low detection confidence inflates the covariance.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_h = max(1.0, y2 - y1)
    scale = max(0.5, box_h / max(1.0, float(reference_box_height_px)))

    sigma = (float(edge_sigma_px) * scale) + float(extra_sigma_px)
    sigma *= 1.0 + 1.5 * max(0.0, 1.0 - float(detection_confidence))
    sigma = max(float(FOOTPOINT_SIGMA_FLOOR_PX), sigma)

    sigma_u = FOOTPOINT_HORIZONTAL_FACTOR * sigma
    sigma_v = FOOTPOINT_VERTICAL_FACTOR * sigma
    rho = float(FOOTPOINT_CORRELATION)

    cov = np.array(
        [
            [sigma_u * sigma_u, rho * sigma_u * sigma_v],
            [rho * sigma_u * sigma_v, sigma_v * sigma_v],
        ],
        dtype=np.float64,
    )
    return _symmetrize_psd(cov, floor=FOOTPOINT_SIGMA_FLOOR_PX**2 * 0.25)


def homography_jacobian(H: np.ndarray, uv: Sequence[float]) -> np.ndarray:
    """Analytic 2x2 Jacobian d(X_yd, Y_yd) / d(u_px, v_px) of a homography at ``uv``."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {H.shape}")
    if not np.all(np.isfinite(H)) or np.max(np.abs(H)) == 0:
        raise ValueError("homography must be finite and nonzero")
    H = H / np.max(np.abs(H))
    u, v = float(uv[0]), float(uv[1])
    w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    scale = abs(H[2, 0] * u) + abs(H[2, 1] * v) + abs(H[2, 2])
    if not np.isfinite(w) or abs(w) <= 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("homography maps point to infinity (w ~ 0)")
    X = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w
    Y = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w
    return np.array(
        [
            [(H[0, 0] - X * H[2, 0]) / w, (H[0, 1] - X * H[2, 1]) / w],
            [(H[1, 0] - Y * H[2, 0]) / w, (H[1, 1] - Y * H[2, 1]) / w],
        ],
        dtype=np.float64,
    )


def propagate_to_field_covariance(
    H: np.ndarray,
    uv: Sequence[float],
    cov_px: np.ndarray,
    *,
    calibration_confidence: float = 1.0,
    extra_field_sigma_yd: float = 0.0,
    min_confidence: float = MIN_CALIBRATION_CONFIDENCE,
) -> np.ndarray:
    """Propagate an image-space covariance to field space (yards^2).

    Adds a calibration-confidence inflation term and an optional additive drift
    allowance ``extra_field_sigma_yd`` (used for propagated geometry).
    """
    J = homography_jacobian(H, uv)
    cov_field = J @ np.asarray(cov_px, dtype=np.float64) @ J.T

    conf = float(calibration_confidence)
    # Engineering sigma multiplier: 1 at confidence 1, rising throughout the
    # accepted range. This is not an empirically calibrated probability.
    if not np.isfinite(conf):
        raise ValueError("calibration_confidence must be finite")
    inflation = 1.0 + 2.5 * (1.0 - np.clip(conf, min_confidence, 1.0))
    cov_field = cov_field * (inflation * inflation)

    if extra_field_sigma_yd > 0.0:
        cov_field = cov_field + (float(extra_field_sigma_yd) ** 2) * np.eye(2, dtype=np.float64)

    return _symmetrize_psd(cov_field)


def sigma_ellipse(cov: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(sigma_major_yd, sigma_minor_yd, angle_rad)`` of a 2x2 covariance."""
    sym = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(sym)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    major = float(np.sqrt(vals[0]))
    minor = float(np.sqrt(vals[1]))
    angle = float(np.arctan2(vecs[1, order[0]], vecs[0, order[0]]))
    return major, minor, angle


def inflate_covariance(cov: np.ndarray, factor: float) -> np.ndarray:
    """Return ``cov * factor**2`` (uncertainty inflation), keeping PSD validity."""
    f = float(factor)
    return _symmetrize_psd(np.asarray(cov, dtype=np.float64) * (f * f))


def mahalanobis_distance_sq(residual: Sequence[float], cov: np.ndarray) -> float:
    """Return ``r^T Sigma^-1 r`` for a 2-vector residual and 2x2 covariance."""
    r = np.asarray(residual, dtype=np.float64).reshape(2)
    cov = _symmetrize_psd(np.asarray(cov, dtype=np.float64))
    try:
        solved = np.linalg.solve(cov, r)
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(cov, r, rcond=None)[0]
    return float(r @ solved)

