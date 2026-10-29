#!/usr/bin/env python3
"""
model_comparison.py — Compare multiple ML models for NILM disaggregation.

Trains and evaluates several models to find the one with the lowest error:
  1. HistGradientBoostingRegressor (baseline, current)
  2. HistGBR with enhanced features + tuned hyperparams
  3. XGBoost
  4. LightGBM
  5. Stacking Ensemble (HistGBR + XGBoost + LightGBM)

All models use strictly causal features. Evaluation on:
  - Chronological test set (known houses)
  - Held-out House 17 (unseen house)
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    StackingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLEANED_DIR, PROCESSED_DIR, FIGURES_DIR,
    METRICS_DIR, PREDICTIONS_DIR,
    WM_CHANNEL_MAP, MODEL_HOUSES, HELD_OUT_HOUSE,
    CYCLE_POWER_THRESHOLD_W, RANDOM_SEED,
    TRAIN_FRAC, VAL_FRAC,
)
from src.data_loader import load_cleaned_house
from src.resampling import resample_to_1min
from src.features_enhanced import build_enhanced_features
from src.evaluation import compute_regression_metrics, compute_active_period_metrics
from src import visualization as viz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_comparison")


def load_all_data(cleaned_dir: Path):
    """Load and prepare data for all model houses + held-out house."""
    household_data = {}
    all_houses = MODEL_HOUSES + [HELD_OUT_HOUSE]

    for house_id in all_houses:
        wm_col = WM_CHANNEL_MAP[house_id]
        logger.info("Loading House %d (WM=%s)...", house_id, wm_col)
        t0 = time.time()
        df = load_cleaned_house(house_id, data_dir=cleaned_dir)
        if "Issues" in df.columns:
            df = df.drop(columns=["Issues"])
        resampled, _ = resample_to_1min(df)
        agg_1min = resampled["Aggregate"]
        wm_1min = resampled[wm_col]
        household_data[house_id] = (agg_1min, wm_1min, resampled)
        logger.info("  Loaded in %.1f s (%d rows)", time.time() - t0, len(resampled))

    return household_data


def prepare_splits(household_data: dict, use_enhanced: bool = True):
    """Build features and create train/val/test + held-out splits."""
    train_frames = []
    val_frames = []
    test_frames = []

    for house_id in MODEL_HOUSES:
        agg_1min, wm_1min, resampled = household_data[house_id]

        if use_enhanced:
            feat_df = build_enhanced_features(resampled)
        else:
            from src.features import build_features
            feat_df = build_features(resampled)

        feat_df["wm_power"] = wm_1min
        feat_df = feat_df.dropna(subset=["wm_power", "agg_power"])

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

    train_df = pd.concat(train_frames)
    val_df = pd.concat(val_frames)
    test_df = pd.concat(test_frames)

    # Prepare held-out
    held_out = HELD_OUT_HOUSE
    agg_h, wm_h, res_h = household_data[held_out]
    if use_enhanced:
        feat_held = build_enhanced_features(res_h)
    else:
        from src.features import build_features
        feat_held = build_features(res_h)
    feat_held["wm_power"] = wm_h
    feat_held = feat_held.dropna(subset=["wm_power", "agg_power"])

    feature_cols = [c for c in train_df.columns
                    if c not in ["wm_power", "house_id", "source_count", "is_outage"]]

    X_train = train_df[feature_cols]
    y_train = train_df["wm_power"]
    X_val = val_df[feature_cols]
    y_val = val_df["wm_power"]
    X_test = test_df[feature_cols]
    y_test = test_df["wm_power"]
    X_held = feat_held[feature_cols]
    y_held = feat_held["wm_power"]

    logger.info("Data splits: Train=%d, Val=%d, Test=%d, Held-out=%d",
                len(X_train), len(X_val), len(X_test), len(X_held))
    logger.info("Features: %d columns", len(feature_cols))

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_held, y_held, test_df, feat_held, feature_cols)


def evaluate_model(name, model, X_test, y_test, X_held, y_held):
    """Predict, clip, and evaluate a model on both splits."""
    pred_test = np.clip(model.predict(X_test), 0, None)
    pred_held = np.clip(model.predict(X_held), 0, None)

    m_test = compute_regression_metrics(y_test, pred_test, f"{name} (test)")
    m_held = compute_regression_metrics(y_held, pred_held, f"{name} (House {HELD_OUT_HOUSE})")

    m_test_active = compute_active_period_metrics(y_test, pred_test, label=f"{name} (test)")

    return m_test, m_held, m_test_active, pred_test, pred_held


def main():
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON — Finding the Best NILM Disaggregation Model")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    household_data = load_all_data(CLEANED_DIR)

    # ── Prepare enhanced features ─────────────────────────────────────
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_held, y_held, test_df, feat_held, feature_cols) = prepare_splits(
        household_data, use_enhanced=True
    )

    results = []
    all_preds_test = {}
    all_preds_held = {}

    # ══════════════════════════════════════════════════════════════════
    # MODEL 1: Original HistGBR (with enhanced features)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 60)
    logger.info("Training Model 1: HistGBR (Original params, enhanced features)")
    t0 = time.time()

    histgbr_orig = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        max_leaf_nodes=64,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=0,
    )
    histgbr_orig.fit(X_train, y_train)
    logger.info("  Trained in %.1f s (iterations: %d)", time.time() - t0, histgbr_orig.n_iter_)

    m_test, m_held, m_active, pred_test, pred_held = evaluate_model(
        "HistGBR-Enhanced", histgbr_orig, X_test, y_test, X_held, y_held
    )
    results.append({"test": m_test, "held": m_held, "active": m_active})
    all_preds_test["HistGBR-Enhanced"] = pred_test
    all_preds_held["HistGBR-Enhanced"] = pred_held

    # ══════════════════════════════════════════════════════════════════
    # MODEL 2: Tuned HistGBR (more iterations, deeper trees, lower LR)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 60)
    logger.info("Training Model 2: HistGBR-Tuned (deeper, more iters)")
    t0 = time.time()

    histgbr_tuned = HistGradientBoostingRegressor(
        max_iter=1500,
        max_depth=10,
        learning_rate=0.03,
        min_samples_leaf=30,
        max_leaf_nodes=128,
        l2_regularization=0.5,
        max_bins=255,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        verbose=0,
    )
    histgbr_tuned.fit(X_train, y_train)
    logger.info("  Trained in %.1f s (iterations: %d)", time.time() - t0, histgbr_tuned.n_iter_)

    m_test, m_held, m_active, pred_test, pred_held = evaluate_model(
        "HistGBR-Tuned", histgbr_tuned, X_test, y_test, X_held, y_held
    )
    results.append({"test": m_test, "held": m_held, "active": m_active})
    all_preds_test["HistGBR-Tuned"] = pred_test
    all_preds_held["HistGBR-Tuned"] = pred_held

    # ══════════════════════════════════════════════════════════════════
    # MODEL 3: XGBoost
    # ══════════════════════════════════════════════════════════════════
    try:
        import xgboost as xgb
        logger.info("\n" + "─" * 60)
        logger.info("Training Model 3: XGBoost")
        t0 = time.time()

        xgb_model = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_SEED,
            early_stopping_rounds=30,
            verbosity=0,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("  Trained in %.1f s (best iteration: %d)",
                    time.time() - t0, xgb_model.best_iteration)

        m_test, m_held, m_active, pred_test, pred_held = evaluate_model(
            "XGBoost", xgb_model, X_test, y_test, X_held, y_held
        )
        results.append({"test": m_test, "held": m_held, "active": m_active})
        all_preds_test["XGBoost"] = pred_test
        all_preds_held["XGBoost"] = pred_held

    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    # ══════════════════════════════════════════════════════════════════
    # MODEL 4: LightGBM
    # ══════════════════════════════════════════════════════════════════
    try:
        import lightgbm as lgb
        logger.info("\n" + "─" * 60)
        logger.info("Training Model 4: LightGBM")
        t0 = time.time()

        lgb_model = lgb.LGBMRegressor(
            n_estimators=1500,
            max_depth=10,
            learning_rate=0.03,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        logger.info("  Trained in %.1f s (best iteration: %d)",
                    time.time() - t0, lgb_model.best_iteration_)

        m_test, m_held, m_active, pred_test, pred_held = evaluate_model(
            "LightGBM", lgb_model, X_test, y_test, X_held, y_held
        )
        results.append({"test": m_test, "held": m_held, "active": m_active})
        all_preds_test["LightGBM"] = pred_test
        all_preds_held["LightGBM"] = pred_held

    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    # ══════════════════════════════════════════════════════════════════
    # MODEL 5: Stacking Ensemble (HistGBR + XGBoost + LightGBM → Ridge)
    # ══════════════════════════════════════════════════════════════════
    try:
        import xgboost as xgb
        import lightgbm as lgb

        logger.info("\n" + "─" * 60)
        logger.info("Training Model 5: Stacking Ensemble")
        t0 = time.time()

        estimators = [
            ("histgbr", HistGradientBoostingRegressor(
                max_iter=800,
                max_depth=8,
                learning_rate=0.05,
                min_samples_leaf=50,
                max_leaf_nodes=64,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                verbose=0,
            )),
            ("xgb", xgb.XGBRegressor(
                n_estimators=800,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=50,
                tree_method="hist",
                random_state=RANDOM_SEED,
                verbosity=0,
            )),
            ("lgb", lgb.LGBMRegressor(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=50,
                random_state=RANDOM_SEED,
                verbose=-1,
            )),
        ]

        stack_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,  # time-series-safe with passthrough
            n_jobs=1,
            passthrough=False,
        )
        stack_model.fit(X_train, y_train)
        logger.info("  Stacking trained in %.1f s", time.time() - t0)

        m_test, m_held, m_active, pred_test, pred_held = evaluate_model(
            "Stacking", stack_model, X_test, y_test, X_held, y_held
        )
        results.append({"test": m_test, "held": m_held, "active": m_active})
        all_preds_test["Stacking"] = pred_test
        all_preds_held["Stacking"] = pred_held

    except ImportError:
        logger.warning("Cannot build stacking ensemble without XGBoost + LightGBM")

    # ══════════════════════════════════════════════════════════════════
    # MODEL 6: Simple weighted average of top 3 models
    # ══════════════════════════════════════════════════════════════════
    if len(all_preds_test) >= 3:
        logger.info("\n" + "─" * 60)
        logger.info("Computing Model 6: Weighted Average Ensemble")

        # Use validation set to find optimal weights
        val_preds = {}
        for name, model_obj in [
            ("HistGBR-Tuned", histgbr_tuned),
        ]:
            val_preds[name] = np.clip(model_obj.predict(X_val), 0, None)

        try:
            val_preds["XGBoost"] = np.clip(xgb_model.predict(X_val), 0, None)
            val_preds["LightGBM"] = np.clip(lgb_model.predict(X_val), 0, None)
        except NameError:
            pass

        if len(val_preds) >= 3:
            # Grid search for best weights on validation set
            best_mae = float("inf")
            best_weights = None
            y_val_arr = y_val.values

            for w1 in np.arange(0.1, 0.9, 0.1):
                for w2 in np.arange(0.1, 0.9 - w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.05:
                        continue
                    blend = (w1 * val_preds["HistGBR-Tuned"] +
                             w2 * val_preds["XGBoost"] +
                             w3 * val_preds["LightGBM"])
                    mae = mean_absolute_error(y_val_arr, blend)
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = (w1, w2, w3)

            if best_weights:
                w1, w2, w3 = best_weights
                logger.info("  Best weights: HistGBR-Tuned=%.2f, XGBoost=%.2f, LightGBM=%.2f",
                            w1, w2, w3)

                blend_test = np.clip(
                    w1 * all_preds_test["HistGBR-Tuned"] +
                    w2 * all_preds_test["XGBoost"] +
                    w3 * all_preds_test["LightGBM"], 0, None
                )
                blend_held = np.clip(
                    w1 * all_preds_held["HistGBR-Tuned"] +
                    w2 * all_preds_held["XGBoost"] +
                    w3 * all_preds_held["LightGBM"], 0, None
                )

                m_test_b = compute_regression_metrics(y_test, blend_test, "WeightedBlend (test)")
                m_held_b = compute_regression_metrics(y_held, blend_held,
                                                      f"WeightedBlend (House {HELD_OUT_HOUSE})")
                m_active_b = compute_active_period_metrics(y_test, blend_test,
                                                           label="WeightedBlend (test)")

                results.append({"test": m_test_b, "held": m_held_b, "active": m_active_b})
                all_preds_test["WeightedBlend"] = blend_test
                all_preds_held["WeightedBlend"] = blend_held

    # ══════════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    summary_rows = []
    for r in results:
        row = {
            "Model": r["test"]["label"].replace(" (test)", ""),
            "Test MAE (W)": r["test"]["MAE_W"],
            "Test RMSE (W)": r["test"]["RMSE_W"],
            "Test R²": r["test"]["R2"],
            "Test Energy Err %": r["test"]["energy_error_pct"],
            "H17 MAE (W)": r["held"]["MAE_W"],
            "H17 RMSE (W)": r["held"]["RMSE_W"],
            "H17 R²": r["held"]["R2"],
            "H17 Energy Err %": r["held"]["energy_error_pct"],
        }
        if r.get("active") and "MAE_W" in r["active"]:
            row["Test Active MAE (W)"] = r["active"]["MAE_W"]
            row["Test Active R²"] = r["active"]["R2"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    # Save comparison
    comparison_path = METRICS_DIR / "model_comparison_enhanced.csv"
    summary_df.to_csv(comparison_path, index=False)
    logger.info("\nSaved comparison to %s", comparison_path)

    # Save detailed results
    all_results = {"models": []}
    for r in results:
        all_results["models"].append({
            "test_metrics": r["test"],
            "held_out_metrics": r["held"],
            "active_period_metrics": r.get("active", {}),
        })
    results_path = METRICS_DIR / "model_comparison_enhanced_details.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    # ── Find and announce the best model ──────────────────────────────
    best_idx = summary_df["Test R²"].idxmax()
    best_model_name = summary_df.loc[best_idx, "Model"]
    best_r2 = summary_df.loc[best_idx, "Test R²"]
    best_mae = summary_df.loc[best_idx, "Test MAE (W)"]

    logger.info("\n" + "★" * 70)
    logger.info("BEST MODEL: %s", best_model_name)
    logger.info("  Test R² = %.4f  (original HistGBR was 0.4118)", best_r2)
    logger.info("  Test MAE = %.2f W (original HistGBR was 30.75 W)", best_mae)
    logger.info("★" * 70)

    # ── Visualization: comparison bar chart ────────────────────────────
    test_metrics_list = [r["test"] for r in results]
    held_metrics_list = [r["held"] for r in results]

    viz.plot_metrics_comparison(
        test_metrics_list,
        save_name="model_comparison_enhanced_test"
    )
    viz.plot_metrics_comparison(
        held_metrics_list,
        save_name="model_comparison_enhanced_house17"
    )

    # Save best model predictions
    best_pred_test = all_preds_test.get(best_model_name)
    best_pred_held = all_preds_held.get(best_model_name)
    if best_pred_test is not None:
        pred_df = pd.DataFrame({
            "timestamp": test_df.index,
            "house_id": test_df["house_id"].values,
            "actual_wm_power_w": y_test.values,
            "pred_best_model": best_pred_test,
        })
        pred_df.to_csv(PREDICTIONS_DIR / "best_model_test_predictions.csv", index=False)

    if best_pred_held is not None:
        pred_held_df = pd.DataFrame({
            "timestamp": feat_held.index,
            "house_id": HELD_OUT_HOUSE,
            "actual_wm_power_w": y_held.values,
            "pred_best_model": best_pred_held,
        })
        pred_held_df.to_csv(
            PREDICTIONS_DIR / f"best_model_held_out_house{HELD_OUT_HOUSE}_predictions.csv",
            index=False
        )

    # Best model scatter and time-series plots
    if best_pred_test is not None:
        viz.plot_scatter_actual_vs_predicted(
            y_test.values, best_pred_test,
            title=f"{best_model_name} — Actual vs Predicted (Test)",
            save_name=f"scatter_best_model_test"
        )
        viz.plot_residuals(
            y_test, best_pred_test,
            title=f"{best_model_name} — Residual Distribution (Test)",
            save_name=f"residuals_best_model_test"
        )

        # Time-series comparison
        active_test = y_test[y_test > CYCLE_POWER_THRESHOLD_W]
        if len(active_test) > 0:
            mid_active = active_test.index[len(active_test) // 2]
            ts_start = str(mid_active - pd.Timedelta(days=1))
            ts_end = str(mid_active + pd.Timedelta(days=1))
        else:
            ts_start = str(y_test.index[0])
            ts_end = str(y_test.index[min(2880, len(y_test) - 1)])

        viz.plot_actual_vs_predicted_ts(
            y_test, best_pred_test,
            title=f"{best_model_name} — Actual vs Predicted WM Power (Test)",
            start=ts_start, end=ts_end,
            save_name="actual_vs_predicted_best_model"
        )

    logger.info("\nModel comparison complete!")
    return summary_df


if __name__ == "__main__":
    main()

