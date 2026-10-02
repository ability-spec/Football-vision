"""Compatibility checks that need no third-party media or GPU packages."""

from importlib.metadata import version

import numpy as np
import pytest

from football_vision import __version__
from football_vision.calibration import yard_lines


def test_installed_version_matches_runtime():
    assert version("football-vision") == __version__


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 3)])
def test_hough_accumulator_layouts_reach_explicit_refusal(monkeypatch, shape):
    # One line cannot calibrate a field, but both binding layouts must reach
    # that explicit refusal rather than fail during tuple unpacking.
    lines = np.array([100, 0.2, 60], dtype=np.float32).reshape(shape)
    monkeypatch.setattr(yard_lines.cv2, "HoughLinesWithAccumulator", lambda *a, **k: lines)
    result = yard_lines.detect_yard_lines_and_vp(np.zeros((100, 200), np.uint8), 100)
    assert result[0] is None
    assert result[-1] == "Fewer than 3 yard lines in VP pencil"
