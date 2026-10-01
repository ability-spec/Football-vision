"""NFL Rulebook field geometry constants and coordinate conversion helpers.

Coordinate convention (NFL field in yards, converted to feet where needed):
  - X in [0, 100] yards (goal line to goal line; end zones are [-10, 0] and [100, 110])
  - Y in [0, 53.3333] yards (0 = near sideline, 53.3333 = far sideline = 160.0 feet)
  - NFL Rulebook (Rule 1, Sec 2):
      Inbound lines (inner edges of 2-ft hash marks) are 70 ft 9 in (70.75 ft) from each sideline,
      leaving 18 ft 6 in (18.50 ft) between inner edges. Because each 1-yard hash mark extends
      2.0 ft toward the sideline ([68.75, 70.75] ft near; [89.25, 91.25] ft far), the CENTROID of
      each hash mark row sits at:
          Y_NEAR_HASH_YD = 69.75 / 3.0 = 23.2500 yd  (69.75 ft)
          Y_FAR_HASH_YD  = 90.25 / 3.0 = 30.0833 yd  (90.25 ft)
      Center-to-center spacing = 20.50 ft (6.8333 yd).
  - Painted yard-line numbers (10, 20, 30, 40, 50) are 6 ft (2.0 yd) tall:
      Near numbers: Y in [12.0, 14.0] yd (36.0 to 42.0 ft from near sideline)
      Far numbers:  Y in [39.3333, 41.3333] yd (36.0 to 42.0 ft from far sideline)
"""

from __future__ import annotations

from typing import Tuple

YARDS_TO_FEET: float = 3.0
FEET_TO_YARDS: float = 1.0 / 3.0

FIELD_LENGTH_YD: float = 100.0
ENDZONE_DEPTH_YD: float = 10.0
FIELD_WIDTH_FT: float = 160.0
FIELD_WIDTH_YD: float = FIELD_WIDTH_FT / YARDS_TO_FEET  # 53.333333333333336 yd

# True centroids of the 2-foot (0.6667 yd) NFL hash marks:
# Near hash spans [68.75, 70.75] ft -> centroid 69.75 ft (23.2500 yd)
# Far hash spans  [89.25, 91.25] ft -> centroid 90.25 ft (30.0833 yd)
Y_NEAR_HASH_YD: float = 69.75 / YARDS_TO_FEET
Y_FAR_HASH_YD: float = 90.25 / YARDS_TO_FEET
HASH_SPACING_CENTER_FT: float = (Y_FAR_HASH_YD - Y_NEAR_HASH_YD) * YARDS_TO_FEET  # 20.50 ft

# Common 18.5-ft inner-edge rulebook misinterpretation constants (kept for benchmark comparison)
Y_NEAR_HASH_INNER_BUG_YD: float = 70.75 / YARDS_TO_FEET  # 23.5833 yd
Y_FAR_HASH_INNER_BUG_YD: float = 89.25 / YARDS_TO_FEET   # 29.7500 yd

# Painted yard-line numbers (6 ft = 2.0 yd tall; outer edge 12.0 yd from sideline)
Y_NEAR_NUM_OUTER_YD: float = 12.0
Y_NEAR_NUM_INNER_YD: float = 14.0
Y_FAR_NUM_INNER_YD: float = FIELD_WIDTH_YD - 14.0  # 39.3333 yd
Y_FAR_NUM_OUTER_YD: float = FIELD_WIDTH_YD - 12.0  # 41.3333 yd

# Plausibility constants (ported from Alex R. Haigh's Hockey-Vision homography.py, in yards)
OUTLIER_YD: float = 1.5                             # 4.5 ft max control-point reprojection residual
MIN_PLAYER_SPREAD_YD: float = 3.0                   # 9.0 ft (Alex's MIN_PLAYER_SPREAD_FT = 9.0)
MIN_PLAYERS_FOR_SPREAD: int = 4
PLAYER_HEIGHT_RANGE_YD: Tuple[float, float] = (1.0, 3.0)  # 3.0 to 9.0 ft
PLAYER_HEIGHT_RANGE_FT: Tuple[float, float] = (3.0, 9.0)
MIN_PLAYERS_FOR_HEIGHT: int = 3


def yards_to_feet(yd: float) -> float:
    """Convert yards to feet."""
    return float(yd * YARDS_TO_FEET)


def feet_to_yards(ft: float) -> float:
    """Convert feet to yards."""
    return float(ft * FEET_TO_YARDS)
