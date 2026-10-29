"""
generate_notebooks.py — Script to generate the three clean, reproducible Jupyter notebooks
in notebooks/ adhering to the assignment specification.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src
    }

def code_cell(source):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }

# ==============================================================================
# Notebook 1: Raw Data Cleaning
# ==============================================================================
nb1_cells = [
    md_cell([
        "# Section 1: Raw Data Inspection and Cleaning (House 1)",
        "This notebook walks through the inspection, data-quality profiling, cleaning,",
        "and 1-minute resampling of raw REFIT data for House 1."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from IPython.display import display, JSON",
        "",
        "# Add project root to sys.path",
        "PROJECT_ROOT = Path('.').resolve().parent",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "",
        "from src.data_loader import load_raw_house1",
        "from src.cleaning import inspect_raw_data, clean_raw_data",
        "from src.resampling import resample_to_1min",
        "from src.visualization import (",
        "    plot_raw_vs_cleaned,",
        "    plot_missing_data_heatmap,",
        "    plot_sampling_intervals",
        ")"
    ]),
    md_cell([
        "## 1. Load Raw House 1 Data",
        "House 1 data is supplied in two parts: Part1 and Part2. We load and concatenate them chronologically."
    ]),
    code_cell([
        "print('Loading raw House 1 data...')",
        "df_raw = load_raw_house1()",
        "print(f'Raw data shape: {df_raw.shape}')",
        "df_raw.head()"
    ]),
    md_cell([
        "## 2. Inspect Raw Data Quality",
        "We identify missing values, duplicate timestamps, non-monotonic records, sampling interval distribution, and extreme spikes."
    ]),
    code_cell([
        "quality_report = inspect_raw_data(df_raw, house_id=1)",
        "print('=== Key Quality Findings ===')",
        "print(f'Total Rows: {quality_report[\"total_rows\"]:,}')",
        "print(f'Duplicate Timestamps: {quality_report[\"duplicate_timestamps\"]:,}')",
        "print(f'Sampling Cadence: mean={quality_report[\"sampling_interval_seconds\"][\"mean\"]:.2f}s, '",
        "      f'median={quality_report[\"sampling_interval_seconds\"][\"median\"]:.1f}s')",
        "print(f'Aggregate Extreme Spikes (>4000 W): {quality_report[\"extreme_readings_gt_4000w\"][\"Aggregate\"]:,}')",
        "print(f'Major Outages (>60s): {quality_report[\"significant_gaps_gt_60s_count\"]:,}')"
    ]),
    md_cell([
        "## 3. Clean Raw Data",
        "We apply reproducible cleaning rules:",
        "1. Deduplicate timestamps by retaining the latest reading.",
        "2. Clip negative power readings to 0 W.",
        "3. Replace impossible power spikes (>4000 W) with NaN to preserve time index.",
        "4. Sort chronologically."
    ]),
    code_cell([
        "df_clean, cleaning_stats = clean_raw_data(df_raw)",
        "print('Cleaning Stats:', cleaning_stats)",
        "print(f'Rows before: {cleaning_stats[\"rows_before\"]:,} -> after: {cleaning_stats[\"rows_after\"]:,}')"
    ]),
    md_cell([
        "## 4. Resample to 1-Minute Resolution",
        "1. Instantaneous power is averaged over 1-minute intervals (unit remains Watts).",
        "2. Short sensor dropouts (<= 3 minutes) are conservatively interpolated.",
        "3. Prolonged outages (> 15 minutes) are flagged and preserved as explicit missingness (never fabricated)."
    ]),
    code_cell([
        "df_1min, resamp_stats = resample_to_1min(df_clean, max_interp_gap_min=3, outage_gap_min=15)",
        "print('Resampling Stats:', resamp_stats)",
        "df_1min.head()"
    ]),
    md_cell([
        "## 5. Visualizations",
        "Reviewing before/after plots, sampling interval distribution, and outage heatmaps."
    ]),
    code_cell([
        "# Plot sampling intervals",
        "from PIL import Image",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section1_cleaning/house1_sampling_intervals.png')"
    ]),
    code_cell([
        "# Plot Aggregate Raw vs Cleaned",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section1_cleaning/house1_aggregate_raw_vs_cleaned.png')"
    ]),
    code_cell([
        "# Plot Missing Data Heatmap",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section1_cleaning/house1_missing_data_heatmap.png')"
    ])
]

# ==============================================================================
# Notebook 2: Washing Machine EDA
# ==============================================================================
nb2_cells = [
    md_cell([
        "# Section 2: Washing Machine Exploratory Data Analysis (EDA)",
        "This notebook explores washing machine operational signatures, detects wash cycles",
        "using a transparent rule-based algorithm, and evaluates cross-household usage characteristics."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import json",
        "from PIL import Image",
        "",
        "PROJECT_ROOT = Path('.').resolve().parent",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "",
        "from src.data_loader import load_cleaned_house, get_appliance_channel",
        "from src.resampling import resample_to_1min",
        "from src.cycle_detection import detect_cycles, cycle_summary_stats"
    ]),
    md_cell([
        "## 1. Household Selection & Summary Statistics",
        "We selected 6 households with verified washing machine sub-meters: Houses 2, 5, 7, 9, 17, and 20.",
        "Let's load the generated summary stats."
    ]),
    code_cell([
        "stats_path = PROJECT_ROOT / 'outputs/metrics/eda_household_stats.json'",
        "with open(stats_path) as f:",
        "    eda_stats = json.load(f)",
        "",
        "summary_df = pd.DataFrame.from_dict(eda_stats, orient='index')",
        "cols = ['wm_channel', 'n_cycles', 'duration_mean_min', 'peak_power_mean_w', 'energy_mean_kwh', 'energy_total_kwh', 'valid_pct', 'zero_off_pct']",
        "summary_df[cols]"
    ]),
    md_cell([
        "## 2. Washing Machine Cycle Detection Rule",
        "A cycle is detected when:",
        "- Active power $P \\ge 20$ W (filters standby sensor noise).",
        "- Short gaps $< 3$ minutes between wash/rinse/spin stages are bridged.",
        "- Minimum cycle duration is $\\ge 10$ minutes.",
        "- Energy is computed as: $E = \\sum \\frac{P_t \\times 1}{60,000}$ kWh."
    ]),
    code_cell([
        "# Representative 24-hour trace for House 2",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/house2_24h_trace.png')"
    ]),
    code_cell([
        "# Detected cycles overlay for House 2",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/house2_detected_cycles.png')"
    ]),
    md_cell([
        "## 3. Usage Patterns (Hour of Day and Day of Week)"
    ]),
    code_cell([
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/house2_hourly_usage.png')"
    ]),
    code_cell([
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/house2_dow_usage.png')"
    ]),
    md_cell([
        "## 4. Cross-Household Comparisons"
    ]),
    code_cell([
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/cross_household_duration.png')"
    ]),
    code_cell([
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/cross_household_energy.png')"
    ]),
    code_cell([
        "Image.open(PROJECT_ROOT / 'outputs/figures/section2_eda/cross_household_peak_power.png')"
    ])
]

# ==============================================================================
# Notebook 3: NILM Disaggregation Modeling
# ==============================================================================
nb3_cells = [
    md_cell([
        "# Section 3: Appliance Disaggregation (NILM) Modeling",
        "This notebook presents the continuous power disaggregation model predicting washing machine power",
        "from whole-house aggregate power at 1-minute resolution."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import json",
        "from PIL import Image",
        "",
        "PROJECT_ROOT = Path('.').resolve().parent",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "",
        "from src.models import AlwaysOffBaseline, UsageProfileBaseline, HistGBRDisaggregator",
        "from src.evaluation import compute_regression_metrics, compute_active_period_metrics"
    ]),
    md_cell([
        "## 1. Model Evaluation Metrics",
        "We compare HistGradientBoostingRegressor against Always-Off and Usage-Profile baselines",
        "across both chronological test data and unseen held-out House 17."
    ]),
    code_cell([
        "metrics_df = pd.read_csv(PROJECT_ROOT / 'outputs/metrics/model_comparison_metrics.csv')",
        "metrics_df"
    ]),
    md_cell([
        "## 2. Active-Period Conditional Performance",
        "Because washing machines are off >93% of the time, overall MAE can be misleading.",
        "Evaluating conditionally on active minutes ($P > 20$ W) reveals true disaggregation capability."
    ]),
    code_cell([
        "with open(PROJECT_ROOT / 'outputs/metrics/active_period_metrics.json') as f:",
        "    active_metrics = json.load(f)",
        "pd.DataFrame(active_metrics)"
    ]),
    md_cell([
        "## 3. Disaggregation Plots"
    ]),
    code_cell([
        "# Actual vs Predicted Time-Series Trace",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section3_model/actual_vs_predicted_timeseries.png')"
    ]),
    code_cell([
        "# Model Comparison Chart (Test Split)",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section3_model/model_comparison_test.png')"
    ]),
    code_cell([
        "# Model Comparison Chart (Held-Out House 17)",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section3_model/model_comparison_house17.png')"
    ]),
    code_cell([
        "# Residual Distribution",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section3_model/residuals_test.png')"
    ]),
    code_cell([
        "# Actual vs Predicted Scatter",
        "Image.open(PROJECT_ROOT / 'outputs/figures/section3_model/scatter_actual_vs_predicted.png')"
    ]),
    md_cell([
        "## 4. Failure Mode Analysis"
    ]),
    code_cell([
        "with open(PROJECT_ROOT / 'outputs/metrics/failure_analysis.json') as f:",
        "    failures = json.load(f)",
        "print(json.dumps(failures, indent=2))"
    ])
]

# Write notebooks
nb1_path = NOTEBOOKS_DIR / "01_raw_data_cleaning.ipynb"
nb2_path = NOTEBOOKS_DIR / "02_washing_machine_eda.ipynb"
nb3_path = NOTEBOOKS_DIR / "03_nilm_modeling.ipynb"

with open(nb1_path, "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

with open(nb2_path, "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

with open(nb3_path, "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

print(f"Generated notebooks in {NOTEBOOKS_DIR}:")
print(f" - {nb1_path.name}")
print(f" - {nb2_path.name}")
print(f" - {nb3_path.name}")
