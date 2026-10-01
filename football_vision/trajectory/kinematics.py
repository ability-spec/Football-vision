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
    vx, vy = float(velocity_yd_s[0]), float(velocity_yd_s[1])
    speed = float(np.hypot(vx, vy))
    if speed <= max_speed_yd_s or speed <= 0.0:
        return (vx, vy), False
    scale = max_speed_yd_s / speed
    return (vx * scale, vy * scale), True


def acceleration_from_velocity(
    previous_velocity_yd_s: Sequence[float],
    current_velocity_yd_s: Sequence[float],
    dt_s: float,
    *,
    max_accel_yd_s2: float = DEFAULT_MAX_ACCEL_YD_S2,
) -> Tuple[Tuple[float, float], float, bool]:
    """Finite-difference acceleration; returns ``(accel_xy, accel_magnitude, clipped)``."""
    dt = max(1e-6, float(dt_s))
    ax = (float(current_velocity_yd_s[0]) - float(previous_velocity_yd_s[0])) / dt
    ay = (float(current_velocity_yd_s[1]) - float(previous_velocity_yd_s[1])) / dt
    mag = float(np.hypot(ax, ay))
    if mag <= max_accel_yd_s2 or mag <= 0.0:
        return (ax, ay), mag, False
    scale = max_accel_yd_s2 / mag
    return (ax * scale, ay * scale), max_accel_yd_s2, True
