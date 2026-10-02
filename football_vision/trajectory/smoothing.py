"""Deterministic constant-velocity field-space trajectory smoothing (Phase 4).

The filter is a 4-state (X, Y, VX, VY) constant-velocity Kalman filter with a
discrete white-noise-acceleration process model, running in **field space** (yards
and yards/second) so camera pan/zoom never leaks into the smoothed trajectory.

All operations are deterministic (no randomness, fixed floating-point order), and
the filter exposes the innovation test statistic used by the Phase 4 jump/outlier
gate. The process-noise acceleration scale is a *plausibility assumption* about
human locomotion, not a measured NFL quantity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from football_vision.trajectory.uncertainty import _symmetrize_psd

# Frozen model constants (a priori engineering assumptions).
DEFAULT_PROCESS_ACCEL_STD_YD_S2 = 5.0   # ~0.5 g elite-athlete scale
DEFAULT_GATE_CHI2_2DOF = 9.21           # chi-square 2-dof, 99% quantile
DEFAULT_MAX_SPEED_YD_S = 12.0           # ~24.5 mph physical plausibility clamp


def _transition(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _process_noise(dt: float, accel_std_yd_s2: float) -> np.ndarray:
    q = float(accel_std_yd_s2) ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    per_axis = np.array(
        [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]], dtype=np.float64
    )
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = per_axis * q
    Q[np.ix_([1, 3], [1, 3])] = per_axis * q
    return Q


class ConstantVelocityFieldFilter:
    """Deterministic constant-velocity Kalman filter over field coordinates."""

    def __init__(
        self,
        *,
        process_accel_std_yd_s2: float = DEFAULT_PROCESS_ACCEL_STD_YD_S2,
        gate_chi2_2dof: float = DEFAULT_GATE_CHI2_2DOF,
        max_speed_yd_s: float = DEFAULT_MAX_SPEED_YD_S,
        position_cov_init_yd2: float = 0.05,
        velocity_cov_init_yd2_s2: float = 4.0,
    ) -> None:
        self.process_accel_std_yd_s2 = float(process_accel_std_yd_s2)
        self.gate_chi2_2dof = float(gate_chi2_2dof)
        self.max_speed_yd_s = float(max_speed_yd_s)
        self.position_cov_init_yd2 = float(position_cov_init_yd2)
        self.velocity_cov_init_yd2_s2 = float(velocity_cov_init_yd2_s2)

        self.initialized: bool = False
        self.mean: np.ndarray = np.zeros(4, dtype=np.float64)
        self.cov: np.ndarray = np.eye(4, dtype=np.float64)
        self.timestamp_s: float = 0.0
        self.n_updates: int = 0
        self.n_rejections: int = 0
        self.last_speed_clipped = False

    # -- lifecycle ---------------------------------------------------------
    def initialize(
        self,
        xy_yd: Tuple[float, float],
        cov_xy_yd2: np.ndarray,
        timestamp_s: float,
        *,
        velocity_yd_s: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.mean = np.array(
            [float(xy_yd[0]), float(xy_yd[1]), float(velocity_yd_s[0]), float(velocity_yd_s[1])],
            dtype=np.float64,
        )
        self.cov = np.diag(
            [
                max(float(cov_xy_yd2[0, 0]), 1e-6),
                max(float(cov_xy_yd2[1, 1]), 1e-6),
                self.velocity_cov_init_yd2_s2,
                self.velocity_cov_init_yd2_s2,
            ]
        ).astype(np.float64)
        self.cov[0, 1] = self.cov[1, 0] = float(cov_xy_yd2[0, 1])
        self.timestamp_s = float(timestamp_s)
        self.initialized = True
        self.n_updates = 1
        self.n_rejections = 0
        self._clamp_speed()

    # -- predict / update --------------------------------------------------
    def predict(self, timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the state forward to ``timestamp_s`` (no measurement)."""
        if not self.initialized:
            raise RuntimeError("filter not initialized")
        self.last_speed_clipped = False
        dt = float(timestamp_s) - self.timestamp_s
        if dt < 0.0:
            raise ValueError("timestamps must be non-decreasing")
        F = _transition(dt)
        Q = _process_noise(dt, self.process_accel_std_yd_s2)
        self.mean = F @ self.mean
        self.cov = _symmetrize_psd(F @ self.cov @ F.T + Q, floor=1e-8)
        self.timestamp_s = float(timestamp_s)
        return self.mean, self.cov

    def innovation_d2(self, xy_yd: Tuple[float, float], cov_xy_yd2: np.ndarray) -> float:
        """Mahalanobis distance^2 of a measurement against the current prediction."""
        R = np.asarray(cov_xy_yd2, dtype=np.float64)
        Hmat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        S = Hmat @ self.cov @ Hmat.T + R
        nu = np.array([float(xy_yd[0]) - self.mean[0], float(xy_yd[1]) - self.mean[1]])
        try:
            solved = np.linalg.solve(_symmetrize_psd(S), nu)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(_symmetrize_psd(S), nu, rcond=None)[0]
        return float(nu @ solved)

    def update(
        self,
        xy_yd: Tuple[float, float],
        cov_xy_yd2: np.ndarray,
        *,
        reject_if_gated: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Kalman update with the innovation gate.

        Returns ``(mean, cov, innovation_d2, accepted)``. When ``accepted`` is
        False the filter state is left at the prediction (the measurement is
        treated as an outlier and is never folded into the track).
        """
        self.last_speed_clipped = False
        d2 = self.innovation_d2(xy_yd, cov_xy_yd2)
        accepted = (not reject_if_gated) or (d2 <= self.gate_chi2_2dof)
        if not accepted:
            self.n_rejections += 1
            return self.mean, self.cov, d2, False

        R = np.asarray(cov_xy_yd2, dtype=np.float64)
        Hmat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        nu = np.array([float(xy_yd[0]) - self.mean[0], float(xy_yd[1]) - self.mean[1]])
        S = Hmat @ self.cov @ Hmat.T + R
        try:
            K = self.cov @ Hmat.T @ np.linalg.inv(_symmetrize_psd(S))
        except np.linalg.LinAlgError:
            K = np.zeros((4, 2), dtype=np.float64)

        self.mean = self.mean + K @ nu
        I = np.eye(4, dtype=np.float64)
        # Joseph form keeps the covariance symmetric positive semi-definite.
        A = I - K @ Hmat
        self.cov = _symmetrize_psd(A @ self.cov @ A.T + K @ R @ K.T, floor=1e-8)

        self._clamp_speed()
        self.n_updates += 1
        return self.mean, self.cov, d2, True

    def _clamp_speed(self) -> None:
        speed = float(np.hypot(self.mean[2], self.mean[3]))
        self.last_speed_clipped = speed > self.max_speed_yd_s and speed > 0.0
        if self.last_speed_clipped:
            scale = self.max_speed_yd_s / speed
            self.mean[2] *= scale
            self.mean[3] *= scale

    # -- accessors ---------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float]:
        return (float(self.mean[0]), float(self.mean[1]))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.mean[2]), float(self.mean[3]))

    @property
    def position_covariance(self) -> np.ndarray:
        return self.cov[np.ix_([0, 1], [0, 1])].copy()

    def velocity_covariance(self) -> np.ndarray:
        return self.cov[np.ix_([2, 3], [2, 3])].copy()


class RawProjectionBaseline:
    """Non-smoothing reference model: the raw per-frame projected footpoint.

    Used as the comparison arm when measuring smoothing benefit; acceleration is
    undefined for this baseline (reported as ``None``).
    """

    name = "raw_projection"

    def predict(self, timestamp_s: float) -> Optional[Tuple[float, float]]:  # pragma: no cover
        return None

