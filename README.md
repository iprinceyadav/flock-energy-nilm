# Flock Energy REFIT — Non-Intrusive Load Monitoring (NILM) Pipeline

## Project Overview

This project implements a complete data science pipeline for the **REFIT Electrical Load Measurements** dataset, focusing on:

1. **Raw Data Inspection & Cleaning** — Inspect and clean House 1 raw data, resample to 1-minute intervals
2. **Washing-Machine Exploratory Data Analysis** — Detect and analyze washing-machine usage patterns across 6 households
3. **Appliance Disaggregation (NILM)** — Train a machine learning model to predict washing-machine power from whole-house aggregate readings
4. **Practical Recommendations** — Energy-saving opportunities and deployment considerations

## Problem Statement

Non-Intrusive Load Monitoring (NILM) aims to decompose a household's total power consumption into individual appliance signals using only the aggregate meter reading. This project focuses on **washing-machine disaggregation** — estimating washing-machine power (watts) at 1-minute resolution from whole-house data.

## Dataset

The REFIT dataset contains electrical load measurements from 20 UK households at ~8-second intervals, collected 2013–2015. Each house has 1 aggregate meter and 9 individual appliance monitors (IAMs).

See [DATA.md](DATA.md) for full dataset documentation, channel mappings, and data quality decisions.

## Project Structure

```
flock-energy-refit/
├── data/
│   ├── raw/                    # REFIT raw CSVs (not in git)
│   ├── cleaned/                # REFIT cleaned CSVs (not in git)
│   └── processed/              # Pipeline outputs (parquet, models)
├── notebooks/
│   ├── 01_raw_data_cleaning.ipynb   # Raw inspection & cleaning
│   ├── 02_washing_machine_eda.ipynb # WM cycle detection & usage EDA
│   └── 03_nilm_modeling.ipynb       # NILM disaggregation modeling
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration
│   ├── data_loader.py          # CSV loading utilities
│   ├── cleaning.py             # Raw data cleaning pipeline
│   ├── resampling.py           # 1-minute resampling
│   ├── cycle_detection.py      # Rule-based WM cycle detection
│   ├── features.py             # Feature engineering for NILM
│   ├── models.py               # Baseline and primary models
│   ├── evaluation.py           # Evaluation metrics
│   └── visualization.py        # Plotting functions
├── scripts/
│   ├── run_pipeline.py         # Main entry point
│   └── verify_data.py          # Data verification
├── tests/
│   ├── test_cleaning.py
│   ├── test_resampling.py
│   ├── test_cycle_detection.py
│   └── test_evaluation.py
├── outputs/
│   ├── figures/                # Generated plots
│   ├── metrics/                # JSON/CSV metrics
│   ├── predictions/            # Model predictions
│   └── reports/                # Text reports
├── README.md                   # This file
├── DATA.md                     # Dataset documentation
├── RESULTS.md                  # Analysis results
├── Recommendation.md           # Utility-facing recommendations
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Environment Setup

### Prerequisites
- Python 3.10+
- ~8 GB RAM (for loading large CSVs)
- REFIT dataset files in `data/raw/` and `data/cleaned/`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd flock-energy-refit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the REFIT dataset:
   - Raw: [REFIT_RAW_081116.7z](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
   - Cleaned: [CLEAN_REFIT_081116.7z](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)

2. Extract files:
   ```
   data/raw/      ← RAW_House*_Part*.csv files
   data/cleaned/  ← CLEAN_House*.csv files
   ```

3. Verify data:
   ```bash
   python scripts/verify_data.py
   ```

### Configuring Dataset Paths

Edit `src/config.py` to change:
- `RAW_DIR` / `CLEANED_DIR`: Point to your local data locations
- `EDA_HOUSES`: Households for washing-machine EDA
- `MODEL_HOUSES` / `HELD_OUT_HOUSE`: Households for NILM model

Or pass paths via command-line:
```bash
python scripts/run_pipeline.py --raw-dir /path/to/raw --cleaned-dir /path/to/cleaned
```

## Running the Pipeline

### Full pipeline
```bash
python scripts/run_pipeline.py
```

### Skip specific sections
```bash
python scripts/run_pipeline.py --skip-section1        # Skip raw cleaning
python scripts/run_pipeline.py --skip-section2        # Skip WM EDA
python scripts/run_pipeline.py --skip-section3        # Skip NILM model
```

### Run tests
```bash
python -m pytest tests/ -v
```

### Expected Output

After a successful run, find results in:
- `outputs/figures/` — All generated plots (PNG)
- `outputs/metrics/` — Evaluation metrics (JSON/CSV)
- `outputs/predictions/` — Model predictions with timestamps
- `outputs/reports/` — Data quality reports

## Methodology

### Section 1: Raw Data Cleaning
- Parse and sort timestamps
- Remove duplicates (keep last reading)
- Clip negative readings to 0
- Replace extreme readings (> 4 kW) with NaN
- Resample to 1-minute mean aggregation
- Interpolate short gaps (≤ 3 min), flag long outages (> 15 min)

### Section 2: Washing-Machine EDA
- Use officially cleaned REFIT data
- Resample to 1-minute resolution
- Detect WM cycles using configurable thresholds (power ≥ 20 W, duration ≥ 10 min)
- Bridge short interruptions (≤ 3 min) within cycles
- Analyze usage by hour-of-day and day-of-week
- Compare cycle characteristics across households

### Section 3: NILM Disaggregation Model
- **Task**: Predict WM power (continuous, watts) from aggregate power
- **Features**: Lagged aggregate values, rolling statistics, time-of-day (all causal)
- **Baselines**: Always-Off (zero prediction), Usage-Profile (hourly average)
- **Primary model**: HistGradientBoostingRegressor
- **Validation**: Chronological train/val/test split (70/15/15%)
- **Generalization**: Held-out household evaluation (House 17)

### Key Design Decisions
1. **Mean resampling**: Instantaneous power averaged over 1-minute windows (unit remains watts)
2. **Causal features only**: Lagged and rolling features use only past/current values
3. **No target leakage**: WM power is never used as an input feature
4. **Chronological splits**: No random mixing of overlapping time windows

## Limitations

- REFIT contains UK household data only — performance on Indian homes is unknown
- Overlapping appliance signatures (e.g., dishwasher, kettle) confuse the model
- Threshold-based cycle detection is approximate, not ground truth
- The model generalizes poorly to unseen households with different WM models
- Data outages (especially Feb 2014) reduce effective training coverage
- 1-minute resolution loses short transient events visible at 8-second sampling

## References

1. Murray, D., Liao, J., Stankovic, L., Stankovic, V., et al. (2015). "A data management platform for personalised real-time energy feedback." *Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting*.
2. Hart, G.W. (1992). "Nonintrusive appliance load monitoring." *Proceedings of the IEEE*, 80(12), 1870–1891.
3. Kelly, J. & Knottenbelt, W. (2015). "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation." *BuildSys '15*.
