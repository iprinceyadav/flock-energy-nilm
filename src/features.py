"""
features.py — Feature engineering for the NILM disaggregation model.

All features are derived from aggregate power and time information only.
No appliance-level target information is used as input features.
Rolling/lagged features use only current and past observations (causal).
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    aggregate_col: str = "Aggregate",
    lag_minutes: list = None,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """Build leakage-free features from aggregate power.

    Parameters
    ----------
    df : pd.DataFrame
        Must have DatetimeIndex and at least the aggregate_col column.
    aggregate_col : str
        Name of the aggregate power column.
    lag_minutes : list of int
        Lag offsets in minutes (default [1, 2, 5, 10, 15, 30, 60]).
    rolling_windows : list of int
        Rolling window sizes in minutes (default [5, 15, 30, 60]).

    Returns
    -------
    pd.DataFrame with feature columns (NaN rows from lags are preserved).
    """
    if lag_minutes is None:
        lag_minutes = [1, 2, 5, 10, 15, 30, 60]
    if rolling_windows is None:
        rolling_windows = [5, 15, 30, 60]

    features = pd.DataFrame(index=df.index)
    agg = df[aggregate_col]

    # Current aggregate power
    features["agg_power"] = agg

    # Lagged aggregate values (causal: only past values)
    for lag in lag_minutes:
        features[f"agg_lag_{lag}m"] = agg.shift(lag)

    # Difference features
    features["agg_diff_1m"] = agg.diff(1)
    features["agg_diff_5m"] = agg.diff(5)

    # Rolling statistics (causal: window includes current + past)
    for win in rolling_windows:
        roll = agg.rolling(window=win, min_periods=1)
        features[f"agg_rmean_{win}m"] = roll.mean()
        features[f"agg_rmax_{win}m"] = roll.max()
        features[f"agg_rmin_{win}m"] = roll.min()
        features[f"agg_rstd_{win}m"] = roll.std()

    # Time-of-day features (cyclical encoding)
    hour = df.index.hour + df.index.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day-of-week features (cyclical)
    dow = df.index.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Is weekend
    features["is_weekend"] = (dow >= 5).astype(int)

    logger.info("Built %d features for %d rows", features.shape[1], len(features))
    return features
