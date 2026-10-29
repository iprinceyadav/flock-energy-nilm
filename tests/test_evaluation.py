"""
test_evaluation.py — Unit tests for evaluation metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluation import compute_regression_metrics, compute_active_period_metrics


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 100, 200, 0, 50])
        m = compute_regression_metrics(y, y, "perfect")
        assert m["MAE_W"] == 0
        assert m["RMSE_W"] == 0
        assert m["R2"] == 1.0

    def test_always_off_vs_mostly_off(self):
        """AlwaysOff baseline gets good MAE when target is mostly zero."""
        y_true = np.zeros(1000)
        y_true[0:10] = 500  # Only 1% active
        y_pred = np.zeros(1000)  # Always off
        m = compute_regression_metrics(y_true, y_pred, "always_off")
        # MAE should be low because 99% of predictions are correct
        assert m["MAE_W"] < 10  # 500 * 10 / 1000 = 5

    def test_energy_error(self):
        y_true = np.array([600.0] * 60)  # 60 min at 600W = 0.6 kWh
        y_pred = np.array([500.0] * 60)  # Underpredicts
        m = compute_regression_metrics(y_true, y_pred, "under")
        assert m["actual_energy_kWh"] == 0.6
        assert m["predicted_energy_kWh"] == 0.5
        assert m["energy_error_kWh"] == -0.1

    def test_handles_nan(self):
        y_true = np.array([100, np.nan, 200])
        y_pred = np.array([100, 150, np.nan])
        m = compute_regression_metrics(y_true, y_pred, "nan_test")
        assert m["n_samples"] == 1  # Only first pair is valid

    def test_negative_r2(self):
        """A bad model can have R² < 0."""
        y_true = np.array([0, 0, 0, 0, 500])
        y_pred = np.array([500, 500, 500, 500, 0])
        m = compute_regression_metrics(y_true, y_pred, "bad")
        assert m["R2"] < 0


class TestActivePeriodMetrics:
    def test_active_only(self):
        y_true = np.array([0, 0, 0, 500, 600, 400])
        y_pred = np.array([0, 0, 0, 480, 550, 380])
        m = compute_active_period_metrics(y_true, y_pred, active_threshold=20, label="test")
        assert m["n_active_samples"] == 3
        assert m["MAE_W"] > 0

    def test_no_active_periods(self):
        y_true = np.zeros(100)
        y_pred = np.zeros(100)
        m = compute_active_period_metrics(y_true, y_pred, label="none")
        assert m["n_active_samples"] == 0
