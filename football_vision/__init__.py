"""Football-Vision: Modular CV field calibration, tracking, and play analytics for NFL video."""

from football_vision.field_spec import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    FIELD_WIDTH_FT,
    Y_NEAR_HASH_YD,
    Y_FAR_HASH_YD,
    Y_NEAR_HASH_INNER_BUG_YD,
    Y_FAR_HASH_INNER_BUG_YD,
    Y_NEAR_NUM_OUTER_YD,
    Y_NEAR_NUM_INNER_YD,
    Y_FAR_NUM_INNER_YD,
    Y_FAR_NUM_OUTER_YD,
    OUTLIER_YD,
    MIN_PLAYER_SPREAD_YD,
    MIN_PLAYERS_FOR_SPREAD,
    PLAYER_HEIGHT_RANGE_YD,
    PLAYER_HEIGHT_RANGE_FT,
    MIN_PLAYERS_FOR_HEIGHT,
    MIN_CALIBRATION_CONFIDENCE,
    PARITY_MIN_SCORE,
    PARITY_MIN_RATIO,
    VALID_METRICS_BY_X_COORD_MODE,
    yards_to_feet,
    feet_to_yards,
)
from football_vision.schema import (
    CalibrationResult,
    CalibrationFailureMode,
    XCoordMode,
    Detection,
    PlayerIdentity,
    TrackState,
    FieldProjection,
    TrajectoryPoint,
    PlayMetric,
    PlayRecord,
    TeamLabel,
    FootpointEstimate,
    PlayerDetection,
    TeamAssignment,
    ProjectedFieldPosition,
    PlayerTrack,
)
from football_vision.calibration import (
    naive_canny_hough,
    extract_white_paint_ridge,
    detect_yard_lines_and_vp,
    detect_hash_rows_guided,
    detect_far_sideline,
    calibrate_frame,
    compute_calibration_confidence,
    resolve_x_coord_mode_and_parity,
    image_to_field,
    field_to_image,
    plausible_homography,
    players_clustered,
    implied_player_height_ft,
    CalibrationTracker,
)
from football_vision.detection import (
    BasePlayerDetector,
    TurfContrastPlayerDetector,
    FixturePlayerDetector,
    extract_footpoint,
    extract_footpoints_for_frame,
)
from football_vision.identity import TorsoTeamClassifier
from football_vision.tracking import PlayerTracker
from football_vision.projection import FieldProjector

__version__ = "0.3.0"
