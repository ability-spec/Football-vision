"""Phase 4 Field-Space Trajectory module.

Persistent track state, multi-frame field-space trajectories, camera-motion-safe
smoothing, kinematics (velocity / acceleration), jump & outlier rejection,
uncertainty propagation, and explicit calibrated / propagated / unknown geometry
states. Builds on the Phase 3 detector, tracker, and projector interfaces without
modifying any Phase 2/3 algorithm.
"""

from football_vision.trajectory.builder import PlayerTrajectoryBuilder
from football_vision.trajectory.geometry_state import (
    GEOMETRY_STATES,
    geometry_state_is_projectable,
    geometry_state_propagation_age,
    resolve_geometry_state,
)
from football_vision.trajectory.kinematics import (
    acceleration_from_velocity,
    clip_speed,
    direction_rad,
    speed_yd_s,
)
from football_vision.trajectory.outlier import (
    FieldJumpGate,
    ImageSpaceJumpGate,
    RejectionTracker,
)
from football_vision.trajectory.smoothing import (
    ConstantVelocityFieldFilter,
    RawProjectionBaseline,
)
from football_vision.trajectory.uncertainty import (
    footpoint_pixel_covariance,
    homography_jacobian,
    inflate_covariance,
    mahalanobis_distance_sq,
    propagate_to_field_covariance,
    sigma_ellipse,
)

__all__ = [
    "PlayerTrajectoryBuilder",
    "GEOMETRY_STATES",
    "resolve_geometry_state",
    "geometry_state_is_projectable",
    "geometry_state_propagation_age",
    "ConstantVelocityFieldFilter",
    "RawProjectionBaseline",
    "FieldJumpGate",
    "ImageSpaceJumpGate",
    "RejectionTracker",
    "footpoint_pixel_covariance",
    "homography_jacobian",
    "propagate_to_field_covariance",
    "sigma_ellipse",
    "inflate_covariance",
    "mahalanobis_distance_sq",
    "speed_yd_s",
    "direction_rad",
    "clip_speed",
    "acceleration_from_velocity",
]
