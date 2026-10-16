"""Yard-line detection must refuse degenerate geometry instead of crashing, and the vectorised
candidate suppression / pencil RANSAC must stay bit-identical to the original scalar algorithm."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_vision import calibrate_frame
from football_vision.calibration import yard_lines
from football_vision.calibration.yard_lines import (
    _pencil_ransac,
    _suppress_near_duplicates,
    detect_yard_lines_and_vp,
)


def parallel_vertical_ridge(height=360, width=640, xs=(100, 220, 340, 460, 580)):
    ridge = np.zeros((height, width), dtype=np.uint8)
    for x in xs:
        ridge[int(0.1 * height):int(0.9 * height), x] = 255
    return ridge


def test_exactly_parallel_yard_lines_are_refused_not_a_crash():
    """Regression: parallel lines give a zero-slope fit, 1/alpha0 = inf, and scipy raised
    ``ValueError: Initial guess is outside of provided bounds`` which aborted a whole video run."""
    lines, vp, n_pixels, median, mean, note = detect_yard_lines_and_vp(parallel_vertical_ridge(), 360)
    assert lines is None and vp is None and n_pixels == 0
    assert np.isnan(median) and np.isnan(mean)
    assert "Degenerate vanishing point" in note


def test_calibrate_frame_reports_a_specific_failure_reason_for_parallel_lines():
    image = np.full((480, 640, 3), (40, 140, 45), np.uint8)
    for i in range(1, 7):
        x = int(i * 640 / 7)
        cv2.line(image, (x, int(0.12 * 480)), (x, int(0.8 * 480)), (235, 235, 235), 3)
    result = calibrate_frame(image)
    assert not result.success and not result.can_project()
    assert result.failure_reason == "degenerate_vanishing_point"
    assert result.H is None and result.confidence == 0.0


# --- equivalence with the original scalar implementation -----------------------------------------
def _reference_candidates_and_pencil(lines, h):
    """Verbatim copy of the pre-vectorisation algorithm (the oracle)."""
    kept = []
    for rho, theta, votes in np.asarray(lines).reshape(-1, 3):
        if abs(np.cos(theta)) < 0.55:
            continue
        m = -np.sin(theta) / np.cos(theta)
        b = rho / np.cos(theta)
        x_mid = m * (h / 2.0) + b
        if any(abs(x_mid - k["x_mid"]) < 25 and abs(m - k["m"]) < 0.12 for k in kept):
            continue
        kept.append({"x_mid": float(x_mid), "m": float(m), "b": float(b), "votes": float(votes)})
    best_inliers = []
    best_score = -1.0
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            dxm = kept[j]["x_mid"] - kept[i]["x_mid"]
            if abs(dxm) < 80:
                continue
            alpha = (kept[j]["m"] - kept[i]["m"]) / dxm
            beta = kept[i]["m"] - alpha * kept[i]["x_mid"]
            if not (0.0 <= alpha <= 0.005):
                continue
            inliers = [k for k in kept if abs(k["m"] - (alpha * k["x_mid"] + beta)) < 0.025]
            dedup = []
            for k in sorted(inliers, key=lambda x: -x["votes"]):
                if all(abs(k["x_mid"] - d["x_mid"]) > 45 for d in dedup):
                    dedup.append(k)
            score = sum(d["votes"] for d in dedup)
            if score > best_score:
                best_score = score
                best_inliers = sorted(dedup, key=lambda x: x["x_mid"])
    return kept, best_inliers


def _random_hough_lines(rng, n, h, *, integer_votes=True, ties=False):
    """Hough-like (rho, theta, votes) rows: a vanishing-point pencil of near-vertical lines plus clutter."""
    rows = []
    n_pencil = int(rng.integers(0, 14))
    x_vp, y_vp = rng.uniform(-400, 1800), -rng.uniform(300, 6000)
    for xmid in np.sort(rng.uniform(30, 1800, n_pencil)):
        slope = (xmid - x_vp) / ((h / 2.0) - y_vp)
        theta = np.arctan(-slope)                      # line x = m*y + b  <->  theta = atan(-m)
        b = xmid - slope * (h / 2.0)
        rows.append((b * np.cos(theta), theta, rng.integers(60, 900)))
    for _ in range(n):
        theta = rng.uniform(-1.0, 1.0)
        rows.append((rng.uniform(-300, 2200) * np.cos(theta), theta, rng.integers(45, 400)))
    arr = np.array(rows, dtype=np.float32)
    if not integer_votes:
        arr[:, 2] = (arr[:, 2] + rng.uniform(0, 0.99, len(arr))).astype(np.float32)
    if ties:
        arr[:, 2] = np.minimum(arr[:, 2], 120.0)       # many equal vote counts -> stable-sort tie paths
    order = np.argsort(-arr[:, 2], kind="stable")      # OpenCV returns strongest first
    arr = arr[order]
    return arr.reshape(-1, 1, 3) if rng.random() < 0.5 else arr   # OpenCV 4 vs OpenCV 5 layout


@pytest.mark.parametrize("variant", ["integer", "float", "ties"])
def test_vectorised_candidate_selection_matches_original_algorithm_exactly(variant):
    rng = np.random.default_rng({"integer": 1, "float": 2, "ties": 3}[variant])
    for _ in range(25):
        h = int(rng.choice([360, 540, 720, 1080]))
        lines = _random_hough_lines(rng, int(rng.integers(0, 110)), h,
                                    integer_votes=variant != "float", ties=variant == "ties")
        ref_kept, ref_best = _reference_candidates_and_pencil(lines, h)
        kept = _suppress_near_duplicates(lines, h)
        assert kept == ref_kept                                   # same dicts, same order, same floats
        best = _pencil_ransac(kept)
        assert best == ref_best                                   # same winner, tie rule included
        assert [id(b) in {id(k) for k in kept} for b in best] == [True] * len(best)


def test_candidate_selection_handles_empty_and_tiny_inputs():
    assert _suppress_near_duplicates(np.zeros((0, 3), dtype=np.float32), 360) == []
    assert _pencil_ransac([]) == []
    one = np.array([[100.0, 0.0, 80.0]], dtype=np.float32)
    kept = _suppress_near_duplicates(one, 360)
    assert len(kept) == 1 and _pencil_ransac(kept) == []


def test_module_exposes_the_documented_refusal_message():
    assert "Degenerate vanishing point" in yard_lines._DEGENERATE_VP


def _lines_from_x_and_slope(pairs, h, votes=100):
    """Hough rows for lines x = m*y + b given (x_mid, m) at y = h/2."""
    rows = []
    for k, (x_mid, m) in enumerate(pairs):
        theta = np.arctan(-m)
        b = x_mid - m * (h / 2.0)
        rows.append((b * np.cos(theta), theta, votes + 7 * k))
    return np.array(rows, dtype=np.float32)


@pytest.mark.parametrize("spacing", [44.0, 45.5, 79.5, 80.5, 150.0])
@pytest.mark.parametrize("alpha", [-1e-4, 0.0, 0.0049, 0.0051])
@pytest.mark.parametrize("n_lines", [2, 3, 5])
def test_exact_decision_boundaries_match_original_algorithm(spacing, alpha, n_lines):
    """Pairs at the 80 px spacing limit, the 45 px dedup limit and the alpha in [0, 0.005] window."""
    h = 720
    pairs = [(100.0 + i * spacing, 0.02 + alpha * (100.0 + i * spacing)) for i in range(n_lines)]
    lines = _lines_from_x_and_slope(pairs, h)
    ref_kept, ref_best = _reference_candidates_and_pencil(lines, h)
    kept = _suppress_near_duplicates(lines, h)
    assert kept == ref_kept
    assert _pencil_ransac(kept) == ref_best


@pytest.mark.parametrize("dx, dm", [(24.0, 0.119), (25.5, 0.119), (24.0, 0.121), (25.5, 0.121)])
def test_near_duplicate_suppression_boundaries_match_original(dx, dm):
    h = 720
    lines = _lines_from_x_and_slope([(300.0, 0.01), (300.0 + dx, 0.01 + dm), (900.0, 0.02)], h)
    ref_kept, _ = _reference_candidates_and_pencil(lines, h)
    assert _suppress_near_duplicates(lines, h) == ref_kept
