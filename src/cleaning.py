"""
cleaning.py — Raw REFIT data cleaning pipeline.

Operations (in order):
1. Parse and validate timestamps
2. Sort chronologically
3. Resolve duplicate timestamps (keep last reading)
4. Detect and flag irregular sampling intervals
5. Handle negative / extreme power readings
6. Produce data-quality report
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import MAX_VALID_POWER_W

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data-quality findings."""
    house_id: int = 0
    total_rows: int = 0
    columns: List[str] = field(default_factory=list)
    date_range: Tuple[str, str] = ("", "")
    timestamp_parse_failures: int = 0
    duplicate_timestamps: int = 0
    non_monotonic_timestamps: int = 0
    missing_values_per_col: Dict[str, int] = field(default_factory=dict)
    missing_values_pct_per_col: Dict[str, float] = field(default_factory=dict)
    negative_readings_per_col: Dict[str, int] = field(default_factory=dict)
    extreme_readings_per_col: Dict[str, int] = field(default_factory=dict)
    sampling_interval_stats: Dict[str, float] = field(default_factory=dict)
    timestamp_gaps: List[Dict] = field(default_factory=list)
    rows_after_dedup: int = 0
    rows_after_cleaning: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            f"=== Data Quality Report — House {self.house_id} ===",
            f"Total rows: {self.total_rows:,}",
            f"Columns: {self.columns}",
            f"Date range: {self.date_range[0]} → {self.date_range[1]}",
            f"Timestamp parse failures: {self.timestamp_parse_failures:,}",
            f"Duplicate timestamps: {self.duplicate_timestamps:,}",
            f"Non-monotonic timestamps: {self.non_monotonic_timestamps:,}",
            "",
            "Missing values:",
        ]
        for col, cnt in self.missing_values_per_col.items():
            pct = self.missing_values_pct_per_col.get(col, 0)
            lines.append(f"  {col}: {cnt:,} ({pct:.2f}%)")
        lines.append("")
        lines.append("Negative readings:")
        for col, cnt in self.negative_readings_per_col.items():
            if cnt > 0:
                lines.append(f"  {col}: {cnt:,}")
        lines.append("")
        lines.append("Extreme readings (> {} W):".format(MAX_VALID_POWER_W))
        for col, cnt in self.extreme_readings_per_col.items():
            if cnt > 0:
                lines.append(f"  {col}: {cnt:,}")
        lines.append("")
        lines.append("Sampling interval (seconds):")
        for k, v in self.sampling_interval_stats.items():
            lines.append(f"  {k}: {v:.2f}")
        lines.append("")
        lines.append(f"Significant timestamp gaps (> 60 s): {len(self.timestamp_gaps)}")
        if self.timestamp_gaps:
            for g in self.timestamp_gaps[:10]:
                lines.append(
                    f"  {g['start']} → {g['end']} ({g['gap_minutes']:.1f} min)"
                )
            if len(self.timestamp_gaps) > 10:
                lines.append(f"  ... and {len(self.timestamp_gaps) - 10} more")
        lines.append(f"\nRows after dedup: {self.rows_after_dedup:,}")
        lines.append(f"Rows after cleaning: {self.rows_after_cleaning:,}")
        return "\n".join(lines)


def inspect_raw_data(df: pd.DataFrame, house_id: int = 0) -> DataQualityReport:
    """Produce a structured data-quality report for raw REFIT data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with DatetimeIndex named 'timestamp'.
    house_id : int

    Returns
    -------
    DataQualityReport
    """
    rpt = DataQualityReport(house_id=house_id)
    rpt.total_rows = len(df)
    rpt.columns = list(df.columns)
    rpt.date_range = (str(df.index.min()), str(df.index.max()))

    # Timestamp parse failures (NaT in index)
    rpt.timestamp_parse_failures = int(df.index.isna().sum())

    # Duplicates
    rpt.duplicate_timestamps = int(df.index.duplicated().sum())

    # Non-monotonic
    diffs = df.index.to_series().diff()
    rpt.non_monotonic_timestamps = int((diffs < pd.Timedelta(0)).sum())

    # Missing values
    power_cols = [c for c in df.columns if c.startswith(("Aggregate", "Appliance"))]
    for col in power_cols:
        cnt = int(df[col].isna().sum())
        rpt.missing_values_per_col[col] = cnt
        rpt.missing_values_pct_per_col[col] = round(cnt / len(df) * 100, 4)

    # Negative readings
    for col in power_cols:
        neg = int((df[col] < 0).sum())
        rpt.negative_readings_per_col[col] = neg

    # Extreme readings
    for col in power_cols:
        ext = int((df[col] > MAX_VALID_POWER_W).sum())
        rpt.extreme_readings_per_col[col] = ext

    # Sampling interval statistics
    valid_diffs = diffs.dropna().dt.total_seconds()
    if len(valid_diffs) > 0:
        rpt.sampling_interval_stats = {
            "mean": float(valid_diffs.mean()),
            "median": float(valid_diffs.median()),
            "min": float(valid_diffs.min()),
            "max": float(valid_diffs.max()),
            "std": float(valid_diffs.std()),
            "mode": float(valid_diffs.mode().iloc[0]) if len(valid_diffs.mode()) > 0 else float("nan"),
        }

    # Significant timestamp gaps (> 60 seconds)
    big_gaps = valid_diffs[valid_diffs > 60]
    gap_records = []
    for idx, gap_sec in big_gaps.items():
        gap_records.append({
            "start": str(idx - pd.Timedelta(seconds=gap_sec)),
            "end": str(idx),
            "gap_seconds": float(gap_sec),
            "gap_minutes": float(gap_sec / 60),
        })
    # Sort by gap size descending
    gap_records.sort(key=lambda g: g["gap_seconds"], reverse=True)
    rpt.timestamp_gaps = gap_records

    return rpt


def clean_raw_data(
    df: pd.DataFrame,
    max_valid_power: float = MAX_VALID_POWER_W,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean raw REFIT data.

    Steps:
    1. Drop rows with unparseable timestamps.
    2. Sort by timestamp.
    3. Resolve duplicate timestamps (keep last occurrence).
    4. Clip negative power values to 0.
    5. Replace extreme readings (> max_valid_power) with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with DatetimeIndex.
    max_valid_power : float
        Maximum valid power reading in watts.

    Returns
    -------
    (cleaned_df, cleaning_stats) tuple.
    """
    stats = {}
    power_cols = [c for c in df.columns if c.startswith(("Aggregate", "Appliance"))]

    # 1. Drop NaT timestamps
    nat_count = int(df.index.isna().sum())
    df = df[~df.index.isna()].copy()
    stats["nat_timestamps_dropped"] = nat_count

    # 2. Sort
    df.sort_index(inplace=True)

    # 3. Deduplicate — keep last reading at each timestamp
    dup_count = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")]
    stats["duplicate_timestamps_removed"] = dup_count

    # 4. Handle negative readings — clip to 0
    neg_counts = {}
    for col in power_cols:
        mask = df[col] < 0
        neg_counts[col] = int(mask.sum())
        df.loc[mask, col] = 0
    stats["negative_readings_clipped"] = neg_counts

    # 5. Handle extreme readings — replace with NaN
    extreme_counts = {}
    for col in power_cols:
        mask = df[col] > max_valid_power
        extreme_counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    stats["extreme_readings_replaced_nan"] = extreme_counts

    stats["rows_before"] = nat_count + dup_count + len(df)
    stats["rows_after"] = len(df)

    logger.info("Cleaning stats: %s", stats)
    return df, stats
