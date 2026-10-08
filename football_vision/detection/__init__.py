"""Phase 3 Player Detection and Footpoint Estimation module."""

from football_vision.detection.detector import (
    BasePlayerDetector,
    FixturePlayerDetector,
    TurfContrastPlayerDetector,
)
from football_vision.detection.footpoint import (
    extract_footpoint,
    extract_footpoints_for_frame,
)
from football_vision.detection.yolox import YoloXTinyPlayerDetector, YoloXTiledPlayerDetector, create_player_detector

__all__ = [
    "BasePlayerDetector",
    "FixturePlayerDetector",
    "TurfContrastPlayerDetector",
    "YoloXTinyPlayerDetector",
    "YoloXTiledPlayerDetector",
    "create_player_detector",
    "extract_footpoint",
    "extract_footpoints_for_frame",
]
