"""
features_enhanced.py — Enhanced feature engineering for NILM disaggregation.

Extends the base feature set with additional discriminative features that
help distinguish washing machine power signatures from other appliances.

All features are strictly causal (no future information leakage).
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_enhanced_features(
    df: pd.DataFrame,
    aggregate_col: str = "Aggregate",
    lag_minutes: list = None,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """Build an enhanced feature set from aggregate power.

    Improvements over base features.py:
    - More lag offsets (including 3, 45 for capturing wash cycle phases)
    - Rolling range (max - min) to capture power swings
    - Second-order differences (acceleration of power change)
    - Power-level binary indicators (is aggregate in WM-typical range?)
    - Rolling coefficient of variation
    - Ratio features (current vs rolling mean)
    - Month cyclical encoding
    - is_night flag (23:00-06:00)

    Parameters
    ----------
    df : pd.DataFrame
        Must have DatetimeIndex and at least the aggregate_col column.
    aggregate_col : str
        Name of the aggregate power column.
    lag_minutes : list of int
        Lag offsets in minutes.
    rolling_windows : list of int
        Rolling window sizes in minutes.

    Returns
    -------
    pd.DataFrame with feature columns.
    """
    if lag_minutes is None:
        lag_minutes = [1, 2, 3, 5, 10, 15, 30, 45, 60]
    if rolling_windows is None:
        rolling_windows = [5, 15, 30, 60]

    features = pd.DataFrame(index=df.index)
    agg = df[aggregate_col]

    # ── 1. Current aggregate power ────────────────────────────────────
    features["agg_power"] = agg

    # ── 2. Lagged aggregate values (causal) ───────────────────────────
    for lag in lag_minutes:
        features[f"agg_lag_{lag}m"] = agg.shift(lag)

    # ── 3. First-order differences ────────────────────────────────────
    for d in [1, 2, 5]:
        features[f"agg_diff_{d}m"] = agg.diff(d)

    # ── 4. Second-order differences (acceleration) ────────────────────
    diff1 = agg.diff(1)
    features["agg_diff2_1m"] = diff1.diff(1)  # d²P/dt²
    features["agg_diff2_5m"] = agg.diff(5).diff(5)

    # ── 5. Rolling statistics (causal windows) ────────────────────────
    for win in rolling_windows:
        roll = agg.rolling(window=win, min_periods=1)
        features[f"agg_rmean_{win}m"] = roll.mean()
        features[f"agg_rmax_{win}m"] = roll.max()
        features[f"agg_rmin_{win}m"] = roll.min()
        features[f"agg_rstd_{win}m"] = roll.std()
        # Rolling range: captures the spread of power over the window
        features[f"agg_rrange_{win}m"] = features[f"agg_rmax_{win}m"] - features[f"agg_rmin_{win}m"]

    # ── 6. Ratio features ─────────────────────────────────────────────
    # Current power relative to recent rolling mean
    for win in [15, 60]:
        rmean = features[f"agg_rmean_{win}m"]
        features[f"agg_ratio_{win}m"] = agg / rmean.replace(0, np.nan)

    # ── 7. Power-level indicators ─────────────────────────────────────
    # Washing machines typically draw 100-2500W; these flags help identify
    # when aggregate is in ranges consistent with WM activity
    features["agg_above_500w"] = (agg > 500).astype(np.float32)
    features["agg_above_1000w"] = (agg > 1000).astype(np.float32)
    features["agg_above_1500w"] = (agg > 1500).astype(np.float32)
    features["agg_above_2000w"] = (agg > 2000).astype(np.float32)

    # ── 8. Step/surge detection ───────────────────────────────────────
    # Large jumps in aggregate power often indicate WM turning on
    abs_diff_1m = agg.diff(1).abs()
    features["agg_abs_diff_1m"] = abs_diff_1m
    features["agg_surge_500w"] = (abs_diff_1m > 500).astype(np.float32)

    # ── 9. Time-of-day features (cyclical encoding) ───────────────────
    hour = df.index.hour + df.index.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # ── 10. Day-of-week features (cyclical) ───────────────────────────
    dow = df.index.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ── 11. Month features (cyclical) — seasonal patterns ─────────────
    month = df.index.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # ── 12. Binary time flags ─────────────────────────────────────────
    features["is_weekend"] = (dow >= 5).astype(int)
    features["is_night"] = ((df.index.hour >= 23) | (df.index.hour < 6)).astype(int)

    logger.info("Built %d enhanced features for %d rows", features.shape[1], len(features))
    return features

