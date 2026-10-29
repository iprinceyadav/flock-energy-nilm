"""
visualization.py — Publication-quality plotting functions for the REFIT project.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Global style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.figsize": (14, 5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def save_fig(fig, name: str, subdir: str = ""):
    """Save figure to outputs/figures/."""
    out = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{name}.png"
    fig.savefig(fp, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure: %s", fp)
    return fp


# ── Section 1: Raw vs Cleaned ────────────────────────────────────────

def plot_power_timeseries(
    series: pd.Series,
    title: str,
    ylabel: str = "Power (W)",
    color: str = "#2196F3",
    figsize: tuple = (14, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, linewidth=0.5, color=color, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    return fig


def plot_raw_vs_cleaned(
    raw: pd.Series,
    cleaned: pd.Series,
    col_name: str,
    start: str,
    end: str,
    save_name: str = "",
) -> plt.Figure:
    """Side-by-side or overlaid plot of raw vs cleaned power."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    raw_slice = raw.loc[start:end]
    cln_slice = cleaned.loc[start:end]

    axes[0].plot(raw_slice.index, raw_slice.values, linewidth=0.5, color="#E53935", alpha=0.8)
    axes[0].set_title(f"{col_name} — Raw")
    axes[0].set_ylabel("Power (W)")

    axes[1].plot(cln_slice.index, cln_slice.values, linewidth=0.5, color="#2196F3", alpha=0.8)
    axes[1].set_title(f"{col_name} — Cleaned & Resampled (1 min)")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_xlabel("Time")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_name:
        save_fig(fig, save_name, subdir="section1_cleaning")
    return fig


def plot_sampling_interval_distribution(
    intervals_sec: pd.Series,
    title: str = "Sampling Interval Distribution",
    save_name: str = "",
) -> plt.Figure:
    """Histogram of timestamp gaps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    clipped = intervals_sec[intervals_sec <= 120]  # Focus on short intervals
    ax.hist(clipped, bins=100, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Interval (seconds)")
    ax.set_ylabel("Count")
    ax.axvline(8, color="red", linestyle="--", linewidth=1, label="Nominal 8 s")
    ax.legend()
    if save_name:
        save_fig(fig, save_name, subdir="section1_cleaning")
    return fig


def plot_missing_data(
    df: pd.DataFrame,
    cols: List[str],
    title: str = "Missing Data Heatmap",
    save_name: str = "",
) -> plt.Figure:
    """Visualize missing data pattern."""
    # Resample to hourly for readability
    hourly_missing = df[cols].resample("1h").apply(lambda x: x.isna().mean())
    fig, ax = plt.subplots(figsize=(14, max(4, len(cols) * 0.6)))
    sns.heatmap(
        hourly_missing.T,
        cmap="YlOrRd",
        cbar_kws={"label": "Fraction Missing"},
        ax=ax,
        xticklabels=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (hourly)")
    if save_name:
        save_fig(fig, save_name, subdir="section1_cleaning")
    return fig


# ── Section 2: Washing Machine EDA ──────────────────────────────────

def plot_24h_trace(
    agg: pd.Series,
    wm: pd.Series,
    date_str: str,
    house_id: int,
    save_name: str = "",
) -> plt.Figure:
    """24-hour aggregate + WM power overlay."""
    day_start = pd.Timestamp(date_str)
    day_end = day_start + pd.Timedelta(days=1)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(
        agg.loc[day_start:day_end].index,
        agg.loc[day_start:day_end].values,
        alpha=0.3, color="#90CAF9", label="Aggregate"
    )
    ax1.plot(
        agg.loc[day_start:day_end].index,
        agg.loc[day_start:day_end].values,
        linewidth=0.8, color="#1565C0", alpha=0.7
    )
    ax1.plot(
        wm.loc[day_start:day_end].index,
        wm.loc[day_start:day_end].values,
        linewidth=1.2, color="#E53935", label="Washing Machine"
    )
    ax1.set_title(f"House {house_id} — 24-Hour Power Trace ({date_str})")
    ax1.set_ylabel("Power (W)")
    ax1.set_xlabel("Time")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    if save_name:
        save_fig(fig, save_name, subdir="section2_eda")
    return fig


def plot_detected_cycles(
    wm: pd.Series,
    cycles_df: pd.DataFrame,
    house_id: int,
    n_cycles: int = 3,
    save_name: str = "",
) -> plt.Figure:
    """Plot individual detected cycles with markers."""
    if cycles_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cycles detected", ha="center", va="center")
        return fig

    show = cycles_df.head(n_cycles)
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 4 * len(show)))
    if len(show) == 1:
        axes = [axes]

    for i, (_, row) in enumerate(show.iterrows()):
        pad = pd.Timedelta(minutes=15)
        s = row["start"] - pad
        e = row["end"] + pad
        segment = wm.loc[s:e]

        axes[i].plot(segment.index, segment.values, linewidth=1, color="#1565C0")
        axes[i].axvspan(row["start"], row["end"], alpha=0.2, color="#E53935",
                        label=f"Cycle {i+1}: {row['duration_min']} min, "
                              f"{row['peak_power_w']:.0f} W peak, "
                              f"{row['energy_kwh']:.3f} kWh")
        axes[i].set_ylabel("Power (W)")
        axes[i].legend(fontsize=9)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"House {house_id} — Detected Washing Machine Cycles", y=1.02)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name, subdir="section2_eda")
    return fig


def plot_hourly_usage(
    cycles_df: pd.DataFrame,
    house_id: int,
    save_name: str = "",
) -> plt.Figure:
    """Hour-of-day usage distribution."""
    if cycles_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cycles", ha="center")
        return fig

    hours = cycles_df["start"].dt.hour
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(24), [int((hours == h).sum()) for h in range(24)],
           color="#42A5F5", edgecolor="white")
    ax.set_title(f"House {house_id} — WM Usage by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of Cycles")
    ax.set_xticks(range(24))
    if save_name:
        save_fig(fig, save_name, subdir="section2_eda")
    return fig


def plot_dow_usage(
    cycles_df: pd.DataFrame,
    house_id: int,
    save_name: str = "",
) -> plt.Figure:
    """Day-of-week usage distribution."""
    if cycles_df.empty:
        fig, ax = plt.subplots()
        return fig

    days = cycles_df["start"].dt.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = [int((days == d).sum()) for d in range(7)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(day_names, counts, color="#66BB6A", edgecolor="white")
    ax.set_title(f"House {house_id} — WM Usage by Day of Week")
    ax.set_ylabel("Number of Cycles")
    if save_name:
        save_fig(fig, save_name, subdir="section2_eda")
    return fig


def plot_cross_household_comparison(
    all_stats: Dict,
    metric: str = "energy_kwh",
    title: str = "Cross-Household Comparison",
    save_name: str = "",
) -> plt.Figure:
    """Box plot comparing a cycle metric across households."""
    fig, ax = plt.subplots(figsize=(12, 5))
    data = []
    labels = []
    for house_id, cycles_df in all_stats.items():
        if not cycles_df.empty and metric in cycles_df.columns:
            data.append(cycles_df[metric].values)
            labels.append(f"House {house_id}")

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    if save_name:
        save_fig(fig, save_name, subdir="section2_eda")
    return fig


def plot_data_availability(
    availability: Dict[int, Dict],
    save_name: str = "",
) -> plt.Figure:
    """Bar chart of data availability across households."""
    fig, ax = plt.subplots(figsize=(12, 5))
    houses = sorted(availability.keys())
    valid_pcts = [availability[h].get("valid_pct", 0) for h in houses]
    missing_pcts = [availability[h].get("missing_pct", 0) for h in houses]

    x = range(len(houses))
    ax.bar(x, valid_pcts, label="Valid", color="#66BB6A", edgecolor="white")
    ax.bar(x, missing_pcts, bottom=valid_pcts, label="Missing", color="#EF5350", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in houses])
    ax.set_ylabel("Percentage")
    ax.set_title("Data Availability by Household")
    ax.legend()
    if save_name:
        save_fig(fig, save_name, subdir="section2_eda")
    return fig


# ── Section 3: Model Evaluation ──────────────────────────────────────

def plot_actual_vs_predicted_ts(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted WM Power",
    start: str = None,
    end: str = None,
    save_name: str = "",
) -> plt.Figure:
    """Time-series overlay of actual and predicted WM power."""
    fig, ax = plt.subplots(figsize=(14, 5))

    if start and end:
        mask = (y_true.index >= start) & (y_true.index <= end)
        y_t = y_true[mask]
        y_p = y_pred[mask] if isinstance(y_pred, pd.Series) else pd.Series(y_pred, index=y_true.index)[mask]
    else:
        y_t = y_true
        y_p = y_pred if isinstance(y_pred, pd.Series) else pd.Series(y_pred, index=y_true.index)

    ax.plot(y_t.index, y_t.values, linewidth=0.7, color="#1565C0", label="Actual", alpha=0.8)
    ax.plot(y_t.index, y_p.values, linewidth=0.7, color="#E53935", label="Predicted", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Washing Machine Power (W)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    if save_name:
        save_fig(fig, save_name, subdir="section3_model")
    return fig


def plot_metrics_comparison(
    metrics_list: List[Dict],
    save_name: str = "",
) -> plt.Figure:
    """Bar chart comparing MAE, RMSE across models."""
    df = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = df["label"]
    colors = ["#90CAF9", "#66BB6A", "#FF8A65", "#CE93D8"][:len(df)]

    for i, (metric, title) in enumerate([
        ("MAE_W", "Mean Absolute Error (W)"),
        ("RMSE_W", "Root Mean Squared Error (W)"),
        ("R2", "R² Score"),
    ]):
        axes[i].bar(labels, df[metric], color=colors, edgecolor="white")
        axes[i].set_title(title)
        axes[i].tick_params(axis="x", rotation=20)
        for j, v in enumerate(df[metric]):
            axes[i].text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name, subdir="section3_model")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Distribution",
    save_name: str = "",
) -> plt.Figure:
    """Histogram of residuals (actual - predicted)."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=100, color="#7E57C2", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"{title}")
    axes[0].set_xlabel("Residual (W)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)

    # Clip for readability
    clipped = residuals[(residuals > -500) & (residuals < 500)]
    axes[1].hist(clipped, bins=100, color="#26A69A", edgecolor="white", alpha=0.85)
    axes[1].set_title(f"{title} (clipped ±500 W)")
    axes[1].set_xlabel("Residual (W)")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name, subdir="section3_model")
    return fig


def plot_scatter_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Scatter",
    save_name: str = "",
) -> plt.Figure:
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=1, alpha=0.1, color="#1565C0")
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual WM Power (W)")
    ax.set_ylabel("Predicted WM Power (W)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, min(max_val, 3000))
    ax.set_ylim(0, min(max_val, 3000))
    if save_name:
        save_fig(fig, save_name, subdir="section3_model")
    return fig
