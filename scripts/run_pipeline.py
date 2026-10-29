#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end Flock Energy REFIT pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --raw-dir data/raw --cleaned-dir data/cleaned

Phases:
    1. Raw data inspection & cleaning (House 1)
    2. Washing-machine EDA (cleaned data, multiple households)
    3. NILM model training & evaluation
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DIR, CLEANED_DIR, PROCESSED_DIR, FIGURES_DIR,
    METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR,
    WM_CHANNEL_MAP, EDA_HOUSES, MODEL_HOUSES, HELD_OUT_HOUSE,
    CYCLE_POWER_THRESHOLD_W, CYCLE_MIN_DURATION_MIN, CYCLE_MERGE_GAP_MIN,
    SHORT_GAP_LIMIT_MIN, LONG_OUTAGE_THRESHOLD_MIN,
    MAX_VALID_POWER_W, RANDOM_SEED,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from src.data_loader import load_raw_house, load_cleaned_house
from src.cleaning import inspect_raw_data, clean_raw_data
from src.resampling import resample_to_1min
from src.cycle_detection import detect_cycles, cycle_summary_stats
from src.features import build_features
from src.models import (
    AlwaysOffBaseline, UsageProfileBaseline,
    train_primary_model, save_model,
)
from src.evaluation import (
    compute_regression_metrics,
    compute_active_period_metrics,
    compare_models,
)
from src import visualization as viz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Flock Energy REFIT Pipeline")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--cleaned-dir", type=Path, default=CLEANED_DIR)
    p.add_argument("--skip-section1", action="store_true", help="Skip raw cleaning")
    p.add_argument("--skip-section2", action="store_true", help="Skip WM EDA")
    p.add_argument("--skip-section3", action="store_true", help="Skip NILM model")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — RAW DATA INSPECTION & CLEANING (House 1)
# ═══════════════════════════════════════════════════════════════════════

def run_section1(raw_dir: Path):
    logger.info("=" * 60)
    logger.info("SECTION 1 — RAW DATA INSPECTION & CLEANING (House 1)")
    logger.info("=" * 60)

    # 1.1 Load raw House 1
    t0 = time.time()
    raw = load_raw_house(1, data_dir=raw_dir)
    logger.info("Loaded House 1 raw in %.1f s", time.time() - t0)

    # 1.2 Inspect
    rpt = inspect_raw_data(raw, house_id=1)
    rpt_text = rpt.summary()
    logger.info("\n%s", rpt_text)

    # Save report
    rpt_path = REPORTS_DIR / "house1_data_quality_report.txt"
    rpt_path.write_text(rpt_text, encoding="utf-8")
    rpt_json = METRICS_DIR / "house1_data_quality.json"
    rpt_json.write_text(json.dumps(rpt.to_dict(), indent=2, default=str), encoding="utf-8")

    # 1.3 Clean
    cleaned, stats = clean_raw_data(raw, max_valid_power=MAX_VALID_POWER_W)
    stats_path = METRICS_DIR / "house1_cleaning_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    rpt.rows_after_dedup = stats["rows_after"]

    # 1.4 Resample to 1 min
    resampled, rs_stats = resample_to_1min(
        cleaned,
        short_gap_limit=SHORT_GAP_LIMIT_MIN,
        long_outage_threshold=LONG_OUTAGE_THRESHOLD_MIN,
    )
    rs_stats_path = METRICS_DIR / "house1_resampling_stats.json"
    rs_stats_path.write_text(json.dumps(rs_stats, indent=2), encoding="utf-8")

    rpt.rows_after_cleaning = len(resampled)

    # Save processed data
    proc_path = PROCESSED_DIR / "house1_cleaned_1min.parquet"
    resampled.to_parquet(proc_path)
    logger.info("Saved processed House 1 to %s", proc_path)

    # 1.5 Visualizations
    # Pick a representative 2-day window with decent coverage
    # Find a period without large gaps
    power_cols = [c for c in resampled.columns if c.startswith(("Aggregate", "Appliance"))]

    # Use a window from the first month of data
    start_date = str(raw.index.min().date() + pd.Timedelta(days=7))
    end_date = str(raw.index.min().date() + pd.Timedelta(days=9))
    logger.info("Visualization window: %s to %s", start_date, end_date)

    # Raw aggregate
    viz.plot_raw_vs_cleaned(
        raw["Aggregate"], resampled["Aggregate"],
        "Aggregate Power", start_date, end_date,
        save_name="house1_aggregate_raw_vs_cleaned"
    )

    # Appliance channels (Appliance1=Fridge, Appliance5=Washing Machine)
    for app, label in [("Appliance1", "Fridge"), ("Appliance5", "Washing Machine")]:
        if app in raw.columns and app in resampled.columns:
            viz.plot_raw_vs_cleaned(
                raw[app], resampled[app],
                f"{label} ({app})", start_date, end_date,
                save_name=f"house1_{app.lower()}_raw_vs_cleaned"
            )

    # Sampling interval distribution
    diffs = raw.index.to_series().diff().dt.total_seconds().dropna()
    viz.plot_sampling_interval_distribution(
        diffs,
        title="House 1 — Raw Sampling Interval Distribution",
        save_name="house1_sampling_intervals"
    )

    # Missing data heatmap
    viz.plot_missing_data(
        resampled, power_cols[:5],
        title="House 1 — Missing Data After Resampling (Hourly)",
        save_name="house1_missing_data_heatmap"
    )

    logger.info("Section 1 complete.")
    return resampled


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — WASHING MACHINE EDA
# ═══════════════════════════════════════════════════════════════════════

def run_section2(cleaned_dir: Path):
    logger.info("=" * 60)
    logger.info("SECTION 2 — WASHING MACHINE EDA")
    logger.info("=" * 60)

    houses_to_try = EDA_HOUSES
    all_cycles = {}          # house_id → cycles_df
    all_stats = {}           # house_id → summary dict
    availability = {}        # house_id → coverage info
    household_data = {}      # house_id → (agg_1min, wm_1min) for model reuse

    for house_id in houses_to_try:
        wm_col = WM_CHANNEL_MAP.get(house_id)
        if wm_col is None:
            logger.warning("House %d: no WM channel mapped, skipping", house_id)
            continue

        logger.info("─── House %d (WM = %s) ───", house_id, wm_col)
        try:
            t0 = time.time()
            df = load_cleaned_house(house_id, data_dir=cleaned_dir)
            logger.info("Loaded in %.1f s", time.time() - t0)
        except FileNotFoundError:
            logger.warning("House %d: cleaned file not found, skipping", house_id)
            continue

        if wm_col not in df.columns:
            logger.warning("House %d: column %s not in data, skipping", house_id, wm_col)
            continue

        # Drop Issues column if present
        if "Issues" in df.columns:
            df = df.drop(columns=["Issues"])

        # Resample to 1 min
        power_cols = [c for c in df.columns if c.startswith(("Aggregate", "Appliance"))]
        resampled, rs_stats = resample_to_1min(df)

        agg_1min = resampled["Aggregate"]
        wm_1min = resampled[wm_col]

        # Coverage stats
        total_min = len(resampled)
        valid_min = int(wm_1min.notna().sum())
        missing_min = total_min - valid_min
        zero_min = int((wm_1min == 0).sum())
        valid_nonzero = valid_min - zero_min

        avail = {
            "house_id": house_id,
            "wm_channel": wm_col,
            "date_range_start": str(resampled.index.min()),
            "date_range_end": str(resampled.index.max()),
            "total_minutes": total_min,
            "valid_minutes": valid_min,
            "missing_minutes": missing_min,
            "missing_pct": round(missing_min / total_min * 100, 2),
            "valid_pct": round(valid_min / total_min * 100, 2),
            "zero_off_minutes": zero_min,
            "zero_off_pct": round(zero_min / valid_min * 100, 2) if valid_min > 0 else 0,
        }
        availability[house_id] = avail
        logger.info("House %d coverage: %.1f%% valid, %d zero/off", house_id, avail["valid_pct"], zero_min)

        # Cycle detection
        cycles_df, is_active = detect_cycles(
            wm_1min,
            power_threshold=CYCLE_POWER_THRESHOLD_W,
            min_duration_min=CYCLE_MIN_DURATION_MIN,
            merge_gap_min=CYCLE_MERGE_GAP_MIN,
        )
        all_cycles[house_id] = cycles_df
        summary = cycle_summary_stats(cycles_df)
        summary.update(avail)
        all_stats[house_id] = summary
        household_data[house_id] = (agg_1min, wm_1min, resampled)

        # Per-household plots
        # Find a date with WM activity
        if not cycles_df.empty:
            active_date = str(cycles_df.iloc[len(cycles_df)//2]["start"].date())
        else:
            active_date = str(resampled.index[len(resampled)//2].date())

        viz.plot_24h_trace(
            agg_1min, wm_1min, active_date, house_id,
            save_name=f"house{house_id}_24h_trace"
        )
        viz.plot_detected_cycles(
            wm_1min, cycles_df, house_id, n_cycles=3,
            save_name=f"house{house_id}_detected_cycles"
        )
        viz.plot_hourly_usage(
            cycles_df, house_id,
            save_name=f"house{house_id}_hourly_usage"
        )
        viz.plot_dow_usage(
            cycles_df, house_id,
            save_name=f"house{house_id}_dow_usage"
        )

    # Cross-household plots
    viz.plot_cross_household_comparison(
        all_cycles, metric="duration_min",
        title="Cycle Duration Distribution by Household",
        save_name="cross_household_duration"
    )
    viz.plot_cross_household_comparison(
        all_cycles, metric="energy_kwh",
        title="Energy per Cycle by Household",
        save_name="cross_household_energy"
    )
    viz.plot_cross_household_comparison(
        all_cycles, metric="peak_power_w",
        title="Peak Power per Cycle by Household",
        save_name="cross_household_peak_power"
    )
    viz.plot_data_availability(
        availability,
        save_name="data_availability_comparison"
    )

    # Save stats
    stats_path = METRICS_DIR / "eda_household_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, default=str), encoding="utf-8")

    avail_path = METRICS_DIR / "eda_data_availability.json"
    avail_path.write_text(json.dumps(availability, indent=2, default=str), encoding="utf-8")

    # Save cycle details per house
    for hid, cdf in all_cycles.items():
        if not cdf.empty:
            cdf.to_csv(METRICS_DIR / f"house{hid}_cycles.csv", index=False)

    logger.info("Section 2 complete. Analyzed %d households.", len(all_stats))
    return household_data, all_cycles, all_stats, availability


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — NILM MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def run_section3(cleaned_dir: Path, household_data: dict = None):
    logger.info("=" * 60)
    logger.info("SECTION 3 — NILM MODEL (Washing Machine Disaggregation)")
    logger.info("=" * 60)

    # Load data for model houses if not already available
    model_houses = MODEL_HOUSES
    held_out = HELD_OUT_HOUSE

    if household_data is None:
        household_data = {}

    for house_id in model_houses + [held_out]:
        if house_id not in household_data:
            wm_col = WM_CHANNEL_MAP[house_id]
            logger.info("Loading House %d for modelling...", house_id)
            df = load_cleaned_house(house_id, data_dir=cleaned_dir)
            if "Issues" in df.columns:
                df = df.drop(columns=["Issues"])
            resampled, _ = resample_to_1min(df)
            agg_1min = resampled["Aggregate"]
            wm_1min = resampled[wm_col]
            household_data[house_id] = (agg_1min, wm_1min, resampled)

    # ── Prepare combined training dataset ──────────────────────────────
    logger.info("Preparing combined dataset from houses: %s", model_houses)

    train_frames = []
    val_frames = []
    test_frames = []
    split_info = {}

    for house_id in model_houses:
        agg_1min, wm_1min, resampled = household_data[house_id]

        # Build features
        feat_df = build_features(resampled)
        feat_df["wm_power"] = wm_1min

        # Drop rows with NaN target
        feat_df = feat_df.dropna(subset=["wm_power"])

        # Also drop rows where too many features are NaN (beginning of series)
        # Keep rows where aggregate power is available
        feat_df = feat_df.dropna(subset=["agg_power"])

        n = len(feat_df)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        train_part = feat_df.iloc[:n_train].copy()
        val_part = feat_df.iloc[n_train:n_train + n_val].copy()
        test_part = feat_df.iloc[n_train + n_val:].copy()

        train_part["house_id"] = house_id
        val_part["house_id"] = house_id
        test_part["house_id"] = house_id

        train_frames.append(train_part)
        val_frames.append(val_part)
        test_frames.append(test_part)

        split_info[house_id] = {
            "total_valid": n,
            "train": n_train,
            "val": len(val_part),
            "test": len(test_part),
            "train_end": str(train_part.index[-1]) if len(train_part) > 0 else "",
            "test_start": str(test_part.index[0]) if len(test_part) > 0 else "",
        }

    train_df = pd.concat(train_frames)
    val_df = pd.concat(val_frames)
    test_df = pd.concat(test_frames)

    feature_cols = [c for c in train_df.columns if c not in ["wm_power", "house_id", "source_count", "is_outage"]]

    X_train = train_df[feature_cols]
    y_train = train_df["wm_power"]
    X_val = val_df[feature_cols]
    y_val = val_df["wm_power"]
    X_test = test_df[feature_cols]
    y_test = test_df["wm_power"]

    logger.info("Train: %d, Val: %d, Test: %d samples", len(X_train), len(X_val), len(X_test))

    # ── Prepare held-out household ─────────────────────────────────────
    agg_h, wm_h, res_h = household_data[held_out]
    feat_held = build_features(res_h)
    feat_held["wm_power"] = wm_h
    feat_held = feat_held.dropna(subset=["wm_power", "agg_power"])
    X_held = feat_held[feature_cols]
    y_held = feat_held["wm_power"]
    logger.info("Held-out House %d: %d samples", held_out, len(X_held))

    # ── Baselines ──────────────────────────────────────────────────────
    baseline_off = AlwaysOffBaseline()
    baseline_off.fit(X_train, y_train)
    pred_off_test = baseline_off.predict(X_test)
    pred_off_held = baseline_off.predict(X_held)

    baseline_prof = UsageProfileBaseline()
    baseline_prof.fit(X_train, y_train)
    pred_prof_test = baseline_prof.predict(X_test)
    pred_prof_held = baseline_prof.predict(X_held)

    # ── Primary model ──────────────────────────────────────────────────
    logger.info("Training primary model (HistGradientBoostingRegressor)...")
    t0 = time.time()
    model = train_primary_model(X_train, y_train, X_val, y_val, random_seed=RANDOM_SEED)
    train_time = time.time() - t0
    logger.info("Training took %.1f s", train_time)

    pred_model_test = model.predict(X_test)
    pred_model_held = model.predict(X_held)

    # Clip negative predictions
    pred_model_test = np.clip(pred_model_test, 0, None)
    pred_model_held = np.clip(pred_model_held, 0, None)

    # ── Evaluation ─────────────────────────────────────────────────────
    all_metrics = []

    # Same-household test
    m_off = compute_regression_metrics(y_test, pred_off_test, "AlwaysOff (test)")
    m_prof = compute_regression_metrics(y_test, pred_prof_test, "UsageProfile (test)")
    m_model = compute_regression_metrics(y_test, pred_model_test, "HistGBR (test)")
    all_metrics.extend([m_off, m_prof, m_model])

    # Active-period metrics
    m_model_active = compute_active_period_metrics(
        y_test, pred_model_test, label="HistGBR (test)"
    )
    m_off_active = compute_active_period_metrics(
        y_test, pred_off_test, label="AlwaysOff (test)"
    )

    # Held-out household
    m_off_h = compute_regression_metrics(y_held, pred_off_held, f"AlwaysOff (House {held_out})")
    m_prof_h = compute_regression_metrics(y_held, pred_prof_held, f"UsageProfile (House {held_out})")
    m_model_h = compute_regression_metrics(y_held, pred_model_held, f"HistGBR (House {held_out})")
    all_metrics.extend([m_off_h, m_prof_h, m_model_h])

    # Save metrics
    metrics_df = compare_models(all_metrics)
    metrics_df.to_csv(METRICS_DIR / "model_comparison_metrics.csv", index=False)
    metrics_path = METRICS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")

    # Active-period metrics
    active_metrics = [m_off_active, m_model_active]
    active_path = METRICS_DIR / "active_period_metrics.json"
    active_path.write_text(json.dumps(active_metrics, indent=2, default=str), encoding="utf-8")

    # Save split info
    split_path = METRICS_DIR / "train_test_split_info.json"
    split_path.write_text(json.dumps(split_info, indent=2, default=str), encoding="utf-8")

    # ── Save predictions ───────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "timestamp": test_df.index,
        "house_id": test_df["house_id"].values,
        "actual_wm_power_w": y_test.values,
        "pred_always_off": pred_off_test,
        "pred_usage_profile": pred_prof_test,
        "pred_histgbr": pred_model_test,
    })
    pred_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    pred_held_df = pd.DataFrame({
        "timestamp": feat_held.index,
        "house_id": held_out,
        "actual_wm_power_w": y_held.values,
        "pred_always_off": pred_off_held,
        "pred_usage_profile": pred_prof_held,
        "pred_histgbr": pred_model_held,
    })
    pred_held_df.to_csv(PREDICTIONS_DIR / f"held_out_house{held_out}_predictions.csv", index=False)

    # Save model
    save_model(model, PROCESSED_DIR / "histgbr_model.joblib")

    # ── Visualizations ─────────────────────────────────────────────────
    # Time-series comparison (pick a 2-day window from test set)
    test_ts = y_test.copy()
    test_pred_series = pd.Series(pred_model_test, index=y_test.index)

    # Find a period with WM activity in the test set
    active_test = y_test[y_test > CYCLE_POWER_THRESHOLD_W]
    if len(active_test) > 0:
        mid_active = active_test.index[len(active_test) // 2]
        ts_start = str(mid_active - pd.Timedelta(days=1))
        ts_end = str(mid_active + pd.Timedelta(days=1))
    else:
        ts_start = str(y_test.index[0])
        ts_end = str(y_test.index[min(2880, len(y_test) - 1)])

    viz.plot_actual_vs_predicted_ts(
        y_test, pred_model_test,
        title="HistGBR — Actual vs Predicted WM Power (Test Set)",
        start=ts_start, end=ts_end,
        save_name="actual_vs_predicted_timeseries"
    )

    # Metrics comparison bar chart
    viz.plot_metrics_comparison(
        [m_off, m_prof, m_model],
        save_name="model_comparison_test"
    )
    viz.plot_metrics_comparison(
        [m_off_h, m_prof_h, m_model_h],
        save_name=f"model_comparison_house{held_out}"
    )

    # Residuals
    viz.plot_residuals(
        y_test, pred_model_test,
        title="HistGBR Residual Distribution (Test Set)",
        save_name="residuals_test"
    )

    # Scatter
    viz.plot_scatter_actual_vs_predicted(
        y_test.values, pred_model_test,
        title="Actual vs Predicted WM Power (Test Set)",
        save_name="scatter_actual_vs_predicted"
    )

    # Feature importance
    try:
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        feat_imp.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = feat_imp.head(20)
        ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#42A5F5")
        ax.set_title("Top 20 Feature Importances (HistGBR)")
        ax.set_xlabel("Importance")
        viz.save_fig(fig, "feature_importance", subdir="section3_model")
    except Exception as e:
        logger.warning("Could not plot feature importances: %s", e)

    # ── Failure analysis ───────────────────────────────────────────────
    failure_analysis = {}

    # Case 1: High-error minutes — when does the model fail?
    residuals = np.abs(y_test.values - pred_model_test)
    high_err_mask = residuals > np.percentile(residuals, 99)
    high_err_times = y_test.index[high_err_mask]
    failure_analysis["high_error_minutes"] = {
        "count": int(high_err_mask.sum()),
        "threshold_w": round(float(np.percentile(residuals, 99)), 1),
        "mean_actual_w": round(float(y_test.values[high_err_mask].mean()), 1),
        "mean_predicted_w": round(float(pred_model_test[high_err_mask].mean()), 1),
    }

    # Case 2: False positives — model predicts high WM when it's off
    off_mask = y_test.values < 5
    false_pos_pred = pred_model_test[off_mask]
    false_pos_high = false_pos_pred > 50
    failure_analysis["false_positives_when_off"] = {
        "total_off_minutes": int(off_mask.sum()),
        "predictions_above_50w": int(false_pos_high.sum()),
        "false_positive_rate_pct": round(false_pos_high.sum() / off_mask.sum() * 100, 3) if off_mask.sum() > 0 else 0,
    }

    # Case 3: Missed WM events — model predicts near-zero when WM is active
    active_mask = y_test.values > CYCLE_POWER_THRESHOLD_W
    if active_mask.sum() > 0:
        active_preds = pred_model_test[active_mask]
        missed = active_preds < 10  # Model says < 10 W when WM is active
        failure_analysis["missed_active_events"] = {
            "total_active_minutes": int(active_mask.sum()),
            "missed_count": int(missed.sum()),
            "miss_rate_pct": round(missed.sum() / active_mask.sum() * 100, 2),
        }

    failure_path = METRICS_DIR / "failure_analysis.json"
    failure_path.write_text(json.dumps(failure_analysis, indent=2), encoding="utf-8")

    logger.info("Section 3 complete.")
    logger.info("Test metrics: %s", m_model)
    logger.info("Held-out House %d metrics: %s", held_out, m_model_h)

    return all_metrics, failure_analysis


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    logger.info("Flock Energy REFIT Pipeline — Starting")
    logger.info("Raw dir: %s", args.raw_dir)
    logger.info("Cleaned dir: %s", args.cleaned_dir)

    household_data = None

    if not args.skip_section1:
        run_section1(args.raw_dir)
    else:
        logger.info("Skipping Section 1")

    if not args.skip_section2:
        household_data, all_cycles, all_stats, availability = run_section2(args.cleaned_dir)
    else:
        logger.info("Skipping Section 2")

    if not args.skip_section3:
        all_metrics, failure_analysis = run_section3(
            args.cleaned_dir, household_data=household_data
        )
    else:
        logger.info("Skipping Section 3")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("Outputs: %s", FIGURES_DIR.parent)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
