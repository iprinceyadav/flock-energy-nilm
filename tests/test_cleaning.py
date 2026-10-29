"""
test_cleaning.py — Unit tests for the cleaning module.
Uses small synthetic fixtures (no REFIT download required).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.cleaning import inspect_raw_data, clean_raw_data


def _make_raw_df(n=100, freq="8s", start="2024-01-01"):
    """Create a synthetic raw REFIT-like DataFrame."""
    idx = pd.date_range(start, periods=n, freq=freq, name="timestamp")
    rng = np.random.RandomState(42)
    data = {
        "Aggregate": rng.uniform(100, 2000, n).astype(float),
        "Appliance1": rng.uniform(0, 200, n).astype(float),
        "Appliance2": rng.uniform(0, 100, n).astype(float),
    }
    df = pd.DataFrame(data, index=idx)
    return df


class TestInspectRawData:
    def test_basic_report(self):
        df = _make_raw_df()
        rpt = inspect_raw_data(df, house_id=99)
        assert rpt.total_rows == 100
        assert rpt.house_id == 99
        assert rpt.duplicate_timestamps == 0
        assert rpt.non_monotonic_timestamps == 0

    def test_detects_duplicates(self):
        df = _make_raw_df()
        # Introduce 3 duplicate timestamps
        dup_rows = df.iloc[:3].copy()
        df = pd.concat([df, dup_rows])
        df.sort_index(inplace=True)
        rpt = inspect_raw_data(df)
        assert rpt.duplicate_timestamps == 3

    def test_detects_negative_readings(self):
        df = _make_raw_df()
        df.iloc[0, df.columns.get_loc("Appliance1")] = -50
        df.iloc[1, df.columns.get_loc("Appliance1")] = -10
        rpt = inspect_raw_data(df)
        assert rpt.negative_readings_per_col["Appliance1"] == 2

    def test_detects_extreme_readings(self):
        df = _make_raw_df()
        df.iloc[0, df.columns.get_loc("Aggregate")] = 5000  # > 4000 W
        rpt = inspect_raw_data(df)
        assert rpt.extreme_readings_per_col["Aggregate"] == 1

    def test_missing_values(self):
        df = _make_raw_df()
        df.iloc[5, df.columns.get_loc("Appliance2")] = np.nan
        df.iloc[6, df.columns.get_loc("Appliance2")] = np.nan
        rpt = inspect_raw_data(df)
        assert rpt.missing_values_per_col["Appliance2"] == 2


class TestCleanRawData:
    def test_removes_duplicates(self):
        df = _make_raw_df(n=50)
        dup = df.iloc[:2].copy()
        df = pd.concat([df, dup])
        cleaned, stats = clean_raw_data(df)
        assert stats["duplicate_timestamps_removed"] == 2
        assert not cleaned.index.duplicated().any()

    def test_clips_negatives(self):
        df = _make_raw_df(n=20)
        df.iloc[0, df.columns.get_loc("Appliance1")] = -100
        cleaned, stats = clean_raw_data(df)
        assert cleaned["Appliance1"].min() >= 0
        assert stats["negative_readings_clipped"]["Appliance1"] == 1

    def test_replaces_extremes_with_nan(self):
        df = _make_raw_df(n=20)
        df.iloc[0, df.columns.get_loc("Aggregate")] = 9999
        cleaned, stats = clean_raw_data(df, max_valid_power=4000)
        assert pd.isna(cleaned["Aggregate"].iloc[0])
        assert stats["extreme_readings_replaced_nan"]["Aggregate"] == 1

    def test_handles_empty_df(self):
        df = _make_raw_df(n=0)
        cleaned, stats = clean_raw_data(df)
        assert len(cleaned) == 0
