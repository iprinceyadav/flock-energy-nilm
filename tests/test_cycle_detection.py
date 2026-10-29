"""
test_cycle_detection.py — Unit tests for cycle detection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.cycle_detection import detect_cycles, cycle_summary_stats


def _make_wm_series(n_minutes=1440):
    """Synthetic WM power: mostly off with a few cycles."""
    idx = pd.date_range("2024-01-01", periods=n_minutes, freq="1min")
    power = pd.Series(np.zeros(n_minutes), index=idx)

    # Cycle 1: minutes 60-120, ~500W
    power.iloc[60:120] = np.random.RandomState(42).uniform(400, 600, 60)

    # Cycle 2: minutes 300-380, ~600W with brief interruption
    power.iloc[300:340] = 550
    power.iloc[340:342] = 0    # 2-min interruption (should be bridged)
    power.iloc[342:380] = 500

    # Too-short event: minutes 500-505 (5 min < 10 min threshold)
    power.iloc[500:505] = 200

    return power


class TestDetectCycles:
    def test_finds_two_cycles(self):
        wm = _make_wm_series()
        cycles, active = detect_cycles(wm, power_threshold=20, min_duration_min=10, merge_gap_min=3)
        # Should find 2 cycles (Cycle 1 and Cycle 2 merged), not the short event
        assert len(cycles) == 2

    def test_respects_min_duration(self):
        wm = _make_wm_series()
        cycles, _ = detect_cycles(wm, power_threshold=20, min_duration_min=10, merge_gap_min=0)
        # The 5-minute event at minute 500 should be excluded
        for _, row in cycles.iterrows():
            assert row["duration_min"] >= 10

    def test_bridges_short_gaps(self):
        wm = _make_wm_series()
        # With merge_gap=3, the 2-min interruption in Cycle 2 should be bridged
        cycles, _ = detect_cycles(wm, power_threshold=20, min_duration_min=10, merge_gap_min=3)
        # Cycle 2 should be one continuous cycle spanning ~80 minutes
        cycle2 = cycles.iloc[1]
        assert cycle2["duration_min"] >= 70

    def test_no_merge_without_gap_bridging(self):
        wm = _make_wm_series()
        cycles, _ = detect_cycles(wm, power_threshold=20, min_duration_min=10, merge_gap_min=0)
        # Without bridging, Cycle 2 splits into two parts (each ≥ 10 min)
        assert len(cycles) >= 2

    def test_energy_calculation(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        # 60 minutes at exactly 600 W = 600 × 60 / 60000 = 0.6 kWh
        power = pd.Series([600.0] * 60, index=idx)
        cycles, _ = detect_cycles(power, power_threshold=20, min_duration_min=10)
        assert len(cycles) == 1
        assert abs(cycles.iloc[0]["energy_kwh"] - 0.6) < 0.01

    def test_handles_all_zeros(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1min")
        power = pd.Series(np.zeros(100), index=idx)
        cycles, _ = detect_cycles(power)
        assert len(cycles) == 0

    def test_handles_all_nan(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1min")
        power = pd.Series([np.nan] * 100, index=idx)
        cycles, _ = detect_cycles(power)
        assert len(cycles) == 0

    def test_missing_data_separation(self):
        """Cycles separated by missing data should not merge."""
        idx = pd.date_range("2024-01-01", periods=200, freq="1min")
        power = pd.Series(np.zeros(200), index=idx)
        power.iloc[10:30] = 500   # Cycle A
        power.iloc[30:33] = np.nan  # Missing (not just off)
        power.iloc[33:55] = 500   # Cycle B
        cycles, _ = detect_cycles(power, merge_gap_min=5)
        # Should NOT bridge across NaN gap
        assert len(cycles) >= 2


class TestCycleSummaryStats:
    def test_summary(self):
        wm = _make_wm_series()
        cycles, _ = detect_cycles(wm)
        stats = cycle_summary_stats(cycles)
        assert stats["n_cycles"] >= 1
        assert "duration_mean_min" in stats
        assert "energy_mean_kwh" in stats

    def test_empty_cycles(self):
        stats = cycle_summary_stats(pd.DataFrame())
        assert stats["n_cycles"] == 0
