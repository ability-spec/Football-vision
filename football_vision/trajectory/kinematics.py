"""Velocity / acceleration kinematics for Phase 4 field-space trajectories.

Kinematics are derived from the smoothed constant-velocity filter state:

* velocity comes from the filter state (already in yards/second),
* acceleration is the finite difference of smoothed velocity, clipped to a
  physical plausibility bound,
* speed and heading are derived quantities.

The plausibility bounds are explicit engineering assumptions documented in the
Phase 4 report; they are clamping guards against numerical blow-ups and are never
presented as measured NFL biomechanics.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

DEFAULT_MAX_SPEED_YD_S = 12.0        # ~24.5 mph
DEFAULT_MAX_ACCEL_YD_S2 = 25.0       # hard cut guard (~2.6 g)


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def speed_yd_s(velocity_yd_s: Sequence[float]) -> float:
    return float(np.hypot(float(velocity_yd_s[0]), float(velocity_yd_s[1])))


def direction_rad(velocity_yd_s: Sequence[float]) -> float:
    """Heading in radians (0 rad = +X / downfield, counter-clockwise toward +Y)."""
    if abs(float(velocity_yd_s[0])) < 1e-9 and abs(float(velocity_yd_s[1])) < 1e-9:
        return 0.0
    return float(np.arctan2(float(velocity_yd_s[1]), float(velocity_yd_s[0])))


def clip_speed(
    velocity_yd_s: Sequence[float],
    *,
    max_speed_yd_s: float = DEFAULT_MAX_SPEED_YD_S,
) -> Tuple[Tuple[float, float], bool]:
    """Clamp a velocity vector to ``max_speed_yd_s``; returns ``(velocity, clipped)``."""
    limit = _positive_finite("max_speed_yd_s", max_speed_yd_s)
    vx, vy = float(velocity_yd_s[0]), float(velocity_yd_s[1])
    speed = float(np.hypot(vx, vy))
    if speed <= limit or speed <= 0.0:
        return (vx, vy), False
    scale = limit / speed
    return (vx * scale, vy * scale), True


def acceleration_from_velocity(
    previous_velocity_yd_s: Sequence[float],
    current_velocity_yd_s: Sequence[float],
    dt_s: float,
    *,
    max_accel_yd_s2: float = DEFAULT_MAX_ACCEL_YD_S2,
) -> Tuple[Tuple[float, float], float, bool]:
    """Finite-difference acceleration; returns ``(accel_xy, accel_magnitude, clipped)``."""
    limit = _positive_finite("max_accel_yd_s2", max_accel_yd_s2)
    if isinstance(dt_s, (bool, str, bytes)):
        raise ValueError("dt_s must be a positive finite number")
    try:
        dt = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("dt_s must be a positive finite number") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be a positive finite number")
    ax = (float(current_velocity_yd_s[0]) - float(previous_velocity_yd_s[0])) / dt
    ay = (float(current_velocity_yd_s[1]) - float(previous_velocity_yd_s[1])) / dt
    mag = float(np.hypot(ax, ay))
    if mag <= limit or mag <= 0.0:
        return (ax, ay), mag, False
    scale = limit / mag
    return (ax * scale, ay * scale), limit, True
