"""
cycle_detection.py — Rule-based washing-machine cycle detection.

This module detects approximate operating events from power readings.
The detected events are NOT ground-truth cycles; they are estimates
based on configurable power thresholds and duration rules.

Algorithm:
1. Mark minutes where WM power ≥ threshold as "active".
2. Bridge very short inactive gaps (≤ merge_gap minutes) between active periods.
3. Group consecutive active minutes into candidate cycles.
4. Discard cycles shorter than min_duration minutes.
5. Compute cycle statistics (duration, peak power, energy).
"""
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CYCLE_POWER_THRESHOLD_W,
    CYCLE_MIN_DURATION_MIN,
    CYCLE_MERGE_GAP_MIN,
)

logger = logging.getLogger(__name__)


def detect_cycles(
    wm_power: pd.Series,
    power_threshold: float = CYCLE_POWER_THRESHOLD_W,
    min_duration_min: int = CYCLE_MIN_DURATION_MIN,
    merge_gap_min: int = CYCLE_MERGE_GAP_MIN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Detect washing-machine operating cycles.

    Parameters
    ----------
    wm_power : pd.Series
        1-minute resampled washing-machine power (watts), DatetimeIndex.
    power_threshold : float
        Minimum power to consider the machine "active" (watts).
    min_duration_min : int
        Minimum cycle duration (minutes).
    merge_gap_min : int
        Bridge inactive gaps ≤ this many minutes.

    Returns
    -------
    (cycles_df, is_active) tuple.
        cycles_df : DataFrame with columns [start, end, duration_min,
                     peak_power_w, mean_power_w, energy_kwh].
        is_active : boolean Series aligned with wm_power index.
    """
    # Step 1: Mark active minutes (exclude NaN — they are missing, not off)
    is_active = (wm_power >= power_threshold) & wm_power.notna()
    active = is_active.copy()

    # Step 2: Bridge short gaps
    if merge_gap_min > 0:
        active_int = active.astype(int)
        # Find inactive stretches
        groups = active_int.ne(active_int.shift()).cumsum()
        for grp_id, grp in active_int.groupby(groups):
            if grp.iloc[0] == 0 and len(grp) <= merge_gap_min:
                # Check if flanked by active periods (not by NaN/missing)
                idx_start = grp.index[0]
                idx_end = grp.index[-1]
                pos_start = active_int.index.get_loc(idx_start)
                pos_end = active_int.index.get_loc(idx_end)
                before_active = pos_start > 0 and active_int.iloc[pos_start - 1] == 1
                after_active = pos_end < len(active_int) - 1 and active_int.iloc[pos_end + 1] == 1
                # Also check that the gap isn't during a data outage
                gap_has_data = wm_power.loc[grp.index].notna().all()
                if before_active and after_active and gap_has_data:
                    active.loc[grp.index] = True

    # Step 3: Group consecutive active minutes
    groups = active.ne(active.shift()).cumsum()
    active_groups = active[active].groupby(groups[active])

    cycles: List[Dict] = []
    for grp_id, grp in active_groups:
        start = grp.index[0]
        end = grp.index[-1]
        duration = len(grp)

        # Step 4: Filter by minimum duration
        if duration < min_duration_min:
            continue

        segment = wm_power.loc[start:end]
        valid_segment = segment.dropna()

        if len(valid_segment) == 0:
            continue

        peak_power = float(valid_segment.max())
        mean_power = float(valid_segment.mean())
        # Energy: P(W) × 1 min / 60000 = kWh per minute
        energy_kwh = float(valid_segment.sum() / 60000.0)

        cycles.append({
            "start": start,
            "end": end,
            "duration_min": duration,
            "peak_power_w": round(peak_power, 1),
            "mean_power_w": round(mean_power, 1),
            "energy_kwh": round(energy_kwh, 4),
        })

    cycles_df = pd.DataFrame(cycles)
    if not cycles_df.empty:
        cycles_df["start"] = pd.to_datetime(cycles_df["start"])
        cycles_df["end"] = pd.to_datetime(cycles_df["end"])

    logger.info("Detected %d cycles (threshold=%.0f W, min_dur=%d min, merge_gap=%d min)",
                len(cycles_df), power_threshold, min_duration_min, merge_gap_min)

    return cycles_df, active


def cycle_summary_stats(cycles_df: pd.DataFrame) -> Dict:
    """Compute summary statistics for detected cycles."""
    if cycles_df.empty:
        return {"n_cycles": 0}

    return {
        "n_cycles": len(cycles_df),
        "duration_mean_min": round(cycles_df["duration_min"].mean(), 1),
        "duration_median_min": round(cycles_df["duration_min"].median(), 1),
        "duration_std_min": round(cycles_df["duration_min"].std(), 1),
        "peak_power_mean_w": round(cycles_df["peak_power_w"].mean(), 1),
        "peak_power_median_w": round(cycles_df["peak_power_w"].median(), 1),
        "energy_mean_kwh": round(cycles_df["energy_kwh"].mean(), 4),
        "energy_median_kwh": round(cycles_df["energy_kwh"].median(), 4),
        "energy_total_kwh": round(cycles_df["energy_kwh"].sum(), 2),
    }
