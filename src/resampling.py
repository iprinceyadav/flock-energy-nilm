"""
resampling.py — Resample REFIT power data to 1-minute intervals.

Strategy:
- Mean aggregation for power channels (appropriate for instantaneous power
  sampled at ~8 s → produces 1-min average power in watts).
- Short gaps (≤ SHORT_GAP_LIMIT_MIN) are linearly interpolated.
- Long outages are left as NaN and flagged in 'is_outage' column.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import RESAMPLE_FREQ, SHORT_GAP_LIMIT_MIN, LONG_OUTAGE_THRESHOLD_MIN

logger = logging.getLogger(__name__)


def resample_to_1min(
    df: pd.DataFrame,
    freq: str = RESAMPLE_FREQ,
    short_gap_limit: int = SHORT_GAP_LIMIT_MIN,
    long_outage_threshold: int = LONG_OUTAGE_THRESHOLD_MIN,
) -> Tuple[pd.DataFrame, dict]:
    """Resample power data to fixed 1-minute intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with DatetimeIndex and power columns.
    freq : str
        Pandas frequency string (default '1min').
    short_gap_limit : int
        Maximum gap in minutes to interpolate.
    long_outage_threshold : int
        Gaps longer than this are flagged as outages.

    Returns
    -------
    (resampled_df, stats) tuple.
    """
    power_cols = [c for c in df.columns if c.startswith(("Aggregate", "Appliance"))]

    # Resample using mean — appropriate for instantaneous power → average power
    resampled = df[power_cols].resample(freq).mean()

    # Identify which minutes had zero source readings (pure gaps)
    counts = df[power_cols[0]].resample(freq).count()
    resampled["source_count"] = counts

    # Detect outage stretches: consecutive NaN minutes
    is_missing = resampled[power_cols[0]].isna()
    # Compute gap lengths
    gap_groups = is_missing.ne(is_missing.shift()).cumsum()
    gap_lengths = is_missing.groupby(gap_groups).transform("sum")

    # Short gap interpolation
    short_gap_mask = is_missing & (gap_lengths <= short_gap_limit)
    for col in power_cols:
        resampled[col] = resampled[col].interpolate(
            method="linear", limit=short_gap_limit, limit_direction="forward"
        )

    # Flag long outages
    resampled["is_outage"] = (is_missing & (gap_lengths > long_outage_threshold)).astype(int)

    # Statistics
    total_minutes = len(resampled)
    missing_before_interp = int(is_missing.sum())
    missing_after = int(resampled[power_cols[0]].isna().sum())
    outage_minutes = int(resampled["is_outage"].sum())
    short_gaps_filled = missing_before_interp - missing_after

    stats = {
        "total_minutes": total_minutes,
        "missing_minutes_before_interp": missing_before_interp,
        "short_gaps_filled": short_gaps_filled,
        "missing_minutes_after": missing_after,
        "outage_minutes_flagged": outage_minutes,
        "valid_minutes": total_minutes - missing_after,
        "coverage_pct": round((total_minutes - missing_after) / total_minutes * 100, 2),
    }

    logger.info("Resampling stats: %s", stats)
    return resampled, stats
