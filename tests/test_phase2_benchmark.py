"""Phase 2 Calibration Robustness & Temporal Benchmark Tests (7 tests).

Together with Phase 0/1 tests (12 tests), brings the Phase 2 test suite to 19 tests.
Verifies:
  - Train / Val / Frozen Test split separation
  - Exact Phase 0 / Phase 1 regression parity (0.275, 0.350, 0.057 yd)
  - Zero fabricated calibrations on single-frame and temporal benchmarks
  - Mutual consistency of 34 temporal refusals (0 + 0 + 1 + 10 + 6 + 7 + 3 + 7 = 34)
    across manifest, benchmark JSON, sequence aggregates, and markdown report
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "benchmarks" / "phase2_calibration_manifest.json"
BENCHMARK_JSON_PATH = ROOT / "outputs" / "phase2_calibration_benchmark.json"
REPORT_MD_PATH = ROOT / "docs" / "PHASE2_CALIBRATION_ROBUSTNESS_REPORT.md"


def test_phase2_manifest_split_separation():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sf = manifest["single_frame_dataset"]
    assert len(sf) == 12
    splits = [item["split"] for item in sf]
    assert splits.count("train") == 3
    assert splits.count("val") == 3
    assert splits.count("test") == 6

    seqs = manifest["temporal_sequences"]
    assert len(seqs) == 8
    assert sum(s["num_frames"] for s in seqs) == 96
    exp_refusals = [s["expected_refused_frames"] for s in seqs]
    assert exp_refusals == [0, 0, 1, 10, 6, 7, 3, 7]
    assert sum(exp_refusals) == 34


def test_phase2_benchmark_json_matches_phase1_regression_gate():
    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    gate = data["phase1_regression_gate"]
    assert gate["frame1_median_err_yd_hash_only"] == 0.275
    assert gate["frame2_median_err_yd_hash_only"] == 0.350
    assert gate["frame3_median_err_yd_hash_only"] == 0.057
    assert gate["matches_phase1_baseline_exactly"] is True


def test_phase2_single_frame_zero_fabricated_calibrations():
    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    sf_agg = data["single_frame_aggregate"]
    assert sf_agg["total_frames"] == 12
    assert sf_agg["calibrated_frames"] == 9
    assert sf_agg["refused_frames"] == 3
    assert sf_agg["fabricated_frames"] == 0

    for rec in data["single_frame_records"]:
        assert rec["fabricated_calibration"] is False
        assert rec["success"] == rec["expected_calibratable"]
        if not rec["success"]:
            assert rec["confidence"] == 0.0
            assert rec["x_coord_mode"] == "uncalibrated"
            assert rec["failure_reason"] is not None


def test_phase2_temporal_sequence_refusals_equal_34():
    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    t_agg = data["temporal_aggregate"]
    assert t_agg["num_sequences"] == 8
    assert t_agg["total_frames"] == 96
    assert t_agg["per_sequence_refusals"] == [0, 0, 1, 10, 6, 7, 3, 7]
    assert sum(t_agg["per_sequence_refusals"]) == 34
    assert t_agg["total_refused_frames"] == 34
    assert t_agg["total_calibrated_frames"] == 62
    assert t_agg["direct_calibrated_frames"] + t_agg["propagated_calibrated_frames"] == 62
    assert t_agg["fabricated_frames"] == 0
    assert sum(t_agg["failure_reason_counts"].values()) == 34

    seqs = data["temporal_sequences"]
    assert [s["refused_frames"] for s in seqs] == [0, 0, 1, 10, 6, 7, 3, 7]
    assert sum(s["refused_frames"] for s in seqs) == 34


def test_phase2_markdown_report_and_json_consistency():
    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    md_text = REPORT_MD_PATH.read_text(encoding="utf-8")

    # Verify exact refusal sum expression and totals appear in markdown report
    assert "0 + 0 + 1 + 10 + 6 + 7 + 3 + 7 = 34" in md_text
    assert "`34 / 96`" in md_text
    assert "`62 / 96`" in md_text
    assert "44 temporal sequence refusals" not in md_text
    assert data["temporal_aggregate"]["total_refused_frames"] == 34
    assert data["temporal_aggregate"]["total_calibrated_frames"] == 62
    assert data["temporal_aggregate"]["refusal_rate_of_total_frames_pct"] == 35.42

    pct_34 = data["temporal_aggregate"]["failure_reason_percentages_of_34_refusals"]
    assert pct_34 == {
        "no_hough_lines": 64.71,
        "missing_hash_rows": 17.65,
        "camera_cut_uncalibrated": 14.71,
        "confidence_expired": 2.94,
    }
    for pct_str in ("64.71%", "17.65%", "14.71%", "2.94%", "35.42%"):
        assert pct_str in md_text


def test_phase2_temporal_short_dropout_and_recovery():
    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    seq3 = next(s for s in data["temporal_sequences"] if s["sequence_id"] == "seq_03_short_line_dropout")
    assert seq3["calibrated_frames"] == 11
    assert seq3["refused_frames"] == 1
    assert seq3["propagated_frames"] == 4
    assert seq3["recoveries_count"] == 1
    frames = seq3["frames"]
    # t=4,5 and t=8,9 are propagated; t=10 expires; t=11 recovers
    assert frames[4]["state"] == "propagated"
    assert frames[5]["state"] == "propagated"
    assert frames[6]["state"] == "direct"
    assert frames[8]["state"] == "propagated"
    assert frames[9]["state"] == "propagated"
    assert frames[10]["state"] == "refused"
    assert frames[10]["failure_reason"] == "confidence_expired"
    assert frames[11]["state"] == "direct"


def test_phase2_temporal_camera_cut_and_prolonged_dropout_refusal():
    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    seq4 = next(s for s in data["temporal_sequences"] if s["sequence_id"] == "seq_04_prolonged_dropout_expiration")
    assert seq4["calibrated_frames"] == 2
    assert seq4["refused_frames"] == 10
    for f in seq4["frames"][2:]:
        assert f["success"] is False
        assert f["fabricated"] is False

    seq5 = next(s for s in data["temporal_sequences"] if s["sequence_id"] == "seq_05_camera_cut_and_recovery")
    assert seq5["calibrated_frames"] == 6
    assert seq5["refused_frames"] == 6
    assert seq5["recoveries_count"] == 1
    for f in seq5["frames"][3:9]:
        assert f["success"] is False
        assert f["fabricated"] is False
    for f in seq5["frames"][9:12]:
        assert f["success"] is True
        assert f["state"] == "direct"
