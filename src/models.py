"""
models.py — Baseline and primary NILM disaggregation models.

All models predict continuous washing-machine power (watts) from
aggregate household power and derived features.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

from src.config import RANDOM_SEED, METRICS_DIR, PREDICTIONS_DIR

logger = logging.getLogger(__name__)


class AlwaysOffBaseline:
    """Baseline A: Predict zero washing-machine power for every minute.

    Rationale: The washing machine is inactive for ~95%+ of all minutes.
    This baseline can achieve misleadingly good MAE / RMSE because the
    target distribution is heavily skewed toward zero.
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def __repr__(self):
        return "AlwaysOffBaseline()"


class UsageProfileBaseline:
    """Baseline B: Predict WM power from hour-of-day average profile.

    Computed from training data only. Provides a time-varying baseline
    that captures typical diurnal washing patterns but cannot detect
    individual cycle events.
    """

    def __init__(self):
        self.hourly_profile = None

    def fit(self, X, y):
        """Fit using training data to learn hourly average WM power.

        X must have a DatetimeIndex or 'hour_sin'/'hour_cos' features.
        y is the target washing-machine power series.
        """
        if hasattr(X, "index") and isinstance(X.index, pd.DatetimeIndex):
            hours = X.index.hour
        else:
            # Fallback: try to decode from cyclical features
            hours = np.zeros(len(X))

        train_df = pd.DataFrame({"hour": hours, "target": y})
        self.hourly_profile = train_df.groupby("hour")["target"].mean().to_dict()
        return self

    def predict(self, X):
        if hasattr(X, "index") and isinstance(X.index, pd.DatetimeIndex):
            hours = X.index.hour
        else:
            hours = np.zeros(len(X), dtype=int)

        default = np.mean(list(self.hourly_profile.values())) if self.hourly_profile else 0
        preds = np.array([self.hourly_profile.get(h, default) for h in hours])
        return preds

    def __repr__(self):
        return "UsageProfileBaseline()"


def train_primary_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    random_seed: int = RANDOM_SEED,
) -> HistGradientBoostingRegressor:
    """Train a HistGradientBoostingRegressor for WM power disaggregation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (WM power in watts).
    X_val, y_val : optional
        Validation data for early stopping.
    random_seed : int

    Returns
    -------
    Trained HistGradientBoostingRegressor.
    """
    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        max_leaf_nodes=64,
        l2_regularization=1.0,
        random_state=random_seed,
        early_stopping=X_val is not None,
        validation_fraction=None if X_val is not None else 0.1,
        n_iter_no_change=20,
        verbose=0,
    )

    if X_val is not None and y_val is not None:
        # Manual early stopping: sklearn HistGBR accepts validation_fraction
        # but not external validation sets directly. We use internal early stopping.
        model.set_params(early_stopping=True, validation_fraction=0.15)

    logger.info("Training HistGradientBoostingRegressor with %d samples, %d features",
                len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)
    logger.info("Training complete. Iterations used: %d", model.n_iter_)
    return model


def save_model(model, filepath: Path):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)


def load_model(filepath: Path):
    """Load trained model from disk."""
    return joblib.load(filepath)
