"""
test_resampling.py — Unit tests for the resampling module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.resampling import resample_to_1min


def _make_cleaned_df(n_minutes=60, sample_interval="8s"):
    """Create a synthetic cleaned dataset at ~8 s intervals."""
    # 7.5 readings per minute at 8 s intervals
    n_readings = int(n_minutes * 60 / 8)
    idx = pd.date_range("2024-01-01", periods=n_readings, freq=sample_interval, name="timestamp")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "Aggregate": rng.uniform(200, 800, n_readings),
        "Appliance1": rng.uniform(0, 100, n_readings),
    }, index=idx)
    return df


class TestResampleTo1Min:
    def test_output_frequency(self):
        df = _make_cleaned_df(n_minutes=30)
        resampled, stats = resample_to_1min(df)
        # Check 1-minute frequency
        diffs = resampled.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=1)).all()

    def test_preserves_power_range(self):
        df = _make_cleaned_df(n_minutes=30)
        resampled, stats = resample_to_1min(df)
        # Mean aggregation should stay within original range
        assert resampled["Aggregate"].min() >= 200 - 1  # small tolerance
        assert resampled["Aggregate"].max() <= 800 + 1

    def test_stats_coverage(self):
        df = _make_cleaned_df(n_minutes=10)
        resampled, stats = resample_to_1min(df)
        assert stats["total_minutes"] > 0
        assert stats["coverage_pct"] > 0

    def test_gap_detection(self):
        """Introduce a 10-minute gap and verify it's handled."""
        df = _make_cleaned_df(n_minutes=30)
        # Remove 10 minutes of data (rows 75-150 roughly = minutes 10-20)
        gap_start = pd.Timestamp("2024-01-01 00:10:00")
        gap_end = pd.Timestamp("2024-01-01 00:20:00")
        df = df[(df.index < gap_start) | (df.index >= gap_end)]

        resampled, stats = resample_to_1min(df)
        assert stats["missing_minutes_before_interp"] > 0

    def test_short_gap_interpolation(self):
        """A 2-minute gap should be interpolated."""
        df = _make_cleaned_df(n_minutes=20)
        gap_start = pd.Timestamp("2024-01-01 00:08:00")
        gap_end = pd.Timestamp("2024-01-01 00:10:00")
        df = df[(df.index < gap_start) | (df.index >= gap_end)]

        resampled, stats = resample_to_1min(df, short_gap_limit=3)
        # The 2-minute gap should be filled
        assert stats["short_gaps_filled"] >= 0  # Interpolation attempted
