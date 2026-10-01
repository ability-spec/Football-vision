"""Field Calibration Engine public API."""

from football_vision.calibration.white_ridge import (
    naive_canny_hough,
    extract_white_paint_ridge,
)
from football_vision.calibration.yard_lines import detect_yard_lines_and_vp
from football_vision.calibration.hash_marks import detect_hash_rows_guided
from football_vision.calibration.sidelines import detect_far_sideline
from football_vision.calibration.homography import (
    calibrate_frame,
    compute_calibration_confidence,
    resolve_x_coord_mode_and_parity,
    image_to_field,
    field_to_image,
    plausible_homography,
    players_clustered,
    implied_player_height_ft,
)
from football_vision.calibration.tracker import CalibrationTracker

__all__ = [
    "naive_canny_hough",
    "extract_white_paint_ridge",
    "detect_yard_lines_and_vp",
    "detect_hash_rows_guided",
    "detect_far_sideline",
    "calibrate_frame",
    "compute_calibration_confidence",
    "resolve_x_coord_mode_and_parity",
    "image_to_field",
    "field_to_image",
    "plausible_homography",
    "players_clustered",
    "implied_player_height_ft",
    "CalibrationTracker",
]
