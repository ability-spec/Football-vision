"""Location resolution for the optional real-frame assets used by tests/benchmarks.

Audit remediation note (see ``docs/PHASE4_TRAJECTORY_REPORT.md`` section 8):
the real NFL broadcast / All-22 stills used for smoke and calibration-regression
tests are third-party images and are deliberately **not** vendored into this
repository. Historical tests and benchmarks hard-coded a single absolute path
(``/home/user/image-search/...``), which made the suite unrunnable anywhere else
and impossible to run in CI.

Resolution order:

1. ``FOOTBALL_VISION_NFL_FRAMES`` environment variable (directory), else
2. the development default ``/home/user/image-search``.

Callers that can tolerate absence use :func:`resolve_nfl_frame` and skip with an
explicit reason; callers that cannot use :func:`require_nfl_frame`, which raises a
``FileNotFoundError`` naming the missing asset and the environment variable.

This module contains no algorithmic content and is imported by no calibration,
detection, tracking, projection or trajectory code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

NFL_FRAMES_ENV = "FOOTBALL_VISION_NFL_FRAMES"
DEFAULT_NFL_FRAMES_DIR = "/home/user/image-search"

# Real frames referenced by tests and benchmarks (documentation only: the
# resolver does not require this list to be exhaustive).
KNOWN_NFL_FRAMES: tuple[str, ...] = (
    "nfl-game-broadcast-screenshot-1st-and-10-5.jpg",
    "nfl-game-broadcast-screenshot-1st-and-10-4.jpg",
    "nfl-all-22-film-pre-snap-formation-offen-5.png",
    "nfl-all-22-film-pre-snap-formation-offen-4.jpg",
)


def nfl_frames_dir() -> Path:
    """Return the directory that holds the optional real NFL still frames."""
    override = os.environ.get(NFL_FRAMES_ENV)
    if override:
        return Path(override).expanduser()
    return Path(DEFAULT_NFL_FRAMES_DIR)


def resolve_nfl_frame(name: str) -> Optional[Path]:
    """Return the path of ``name`` inside :func:`nfl_frames_dir`, or ``None``.

    ``name`` may be a bare filename or a path whose file name is used as a
    fallback when the original location does not exist.
    """
    candidate = Path(name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    fallback = nfl_frames_dir() / candidate.name
    if fallback.is_file():
        return fallback
    return None


def require_nfl_frame(name: str) -> Path:
    """Return the path of ``name`` or raise a ``FileNotFoundError`` explaining how to fix it."""
    resolved = resolve_nfl_frame(name)
    if resolved is None:
        raise FileNotFoundError(
            f"real NFL frame {Path(name).name!r} not found in {str(nfl_frames_dir())!r}. "
            f"These third-party images are not vendored in the repository; set the "
            f"{NFL_FRAMES_ENV} environment variable to a directory containing them, or run "
            f"only the tests that do not require real frames."
        )
    return resolved


def missing_nfl_frames() -> List[str]:
    """Return the names of the known real frames that are currently unavailable."""
    return [name for name in KNOWN_NFL_FRAMES if resolve_nfl_frame(name) is None]


def real_frames_skip_reason() -> str:
    """Return a pytest.skip-style reason describing why real frames are unavailable."""
    missing = missing_nfl_frames()
    listed = ", ".join(missing) if missing else "all known frames"
    return (
        f"real NFL frame assets are not available ({listed}); set {NFL_FRAMES_ENV} to a "
        f"directory containing them (current: {str(nfl_frames_dir())})"
    )
