"""
evaluation.py — Evaluation metrics for NILM disaggregation.
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : array-like
        Actual washing-machine power (watts).
    y_pred : array-like
        Predicted washing-machine power (watts).
    label : str
        Description for logging.

    Returns
    -------
    Dict with MAE, RMSE, R², and energy estimation error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Energy estimation error (kWh)
    # Each minute: P(W) × 1min / 60000 = kWh
    actual_energy_kwh = y_true.sum() / 60000.0
    predicted_energy_kwh = y_pred.sum() / 60000.0
    energy_error_kwh = predicted_energy_kwh - actual_energy_kwh
    energy_error_pct = (
        (energy_error_kwh / actual_energy_kwh * 100)
        if actual_energy_kwh != 0 else float("nan")
    )

    metrics = {
        "label": label,
        "n_samples": int(len(y_true)),
        "MAE_W": round(mae, 2),
        "RMSE_W": round(rmse, 2),
        "R2": round(r2, 4),
        "actual_energy_kWh": round(actual_energy_kwh, 2),
        "predicted_energy_kWh": round(predicted_energy_kwh, 2),
        "energy_error_kWh": round(energy_error_kwh, 2),
        "energy_error_pct": round(energy_error_pct, 2),
    }

    logger.info("Metrics [%s]: MAE=%.2f W, RMSE=%.2f W, R²=%.4f",
                label, mae, rmse, r2)
    return metrics


def compute_active_period_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    active_threshold: float = 20.0,
    label: str = "",
) -> Dict[str, float]:
    """Compute metrics only for minutes when WM is actually active.

    This conditional metric is useful but can be misleading—it excludes
    the dominant off-period where the baseline excels.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true >= active_threshold
    if mask.sum() == 0:
        return {"label": label + " (active)", "n_active_samples": 0}

    y_t_active = y_true[mask]
    y_p_active = y_pred[mask]

    mae = mean_absolute_error(y_t_active, y_p_active)
    rmse = np.sqrt(mean_squared_error(y_t_active, y_p_active))
    r2 = r2_score(y_t_active, y_p_active)

    return {
        "label": label + " (active only)",
        "n_active_samples": int(mask.sum()),
        "active_pct": round(mask.sum() / len(y_true) * 100, 2),
        "MAE_W": round(mae, 2),
        "RMSE_W": round(rmse, 2),
        "R2": round(r2, 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """Create comparison DataFrame from a list of metric dicts."""
    return pd.DataFrame(results)
