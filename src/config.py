"""
Central configuration for the Flock Energy REFIT project.
All paths, thresholds, and parameters are defined here so that
notebooks and scripts share a single source of truth.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
CLEANED_DIR  = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUT_DIR / "figures"
METRICS_DIR  = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR  = OUTPUT_DIR / "reports"

# Create output directories on import
for d in [PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset parameters ─────────────────────────────────────────────────
ORIGINAL_SAMPLE_INTERVAL_SEC = 8        # Nominal 6‒8 s, documented as ~8 s
RESAMPLE_FREQ = "1min"                  # Target resolution
POWER_UNIT = "watts"
MAX_VALID_POWER_W = 4000                # Spikes >4 kW removed in cleaned set
SHORT_GAP_LIMIT_MIN = 3                 # Interpolate gaps ≤ 3 min
LONG_OUTAGE_THRESHOLD_MIN = 15          # Flag gaps > 15 min as outages

# ── Washing‑machine appliance mapping (channel numbers are 1-indexed) ─
# Source: official REFIT CLEAN_READ_ME_081116.txt
WM_CHANNEL_MAP = {
    1:  "Appliance5",    # Washing Machine, Beko WMC6140
    2:  "Appliance2",    # Washing Machine, LG F1289TD
    3:  "Appliance6",    # Washing Machine
    4:  "Appliance4",    # Washing Machine (1), Servis 6065
    5:  "Appliance3",    # Washing Machine, AEG L99695HWD
    6:  "Appliance2",    # Washing Machine, Bosch Classixx 1200 Express
    7:  "Appliance5",    # Washing Machine, Bosch
    8:  "Appliance4",    # Washing Machine
    9:  "Appliance3",    # Washing Machine, Bosch Classixx 6 1200 Express
    10: "Appliance5",    # Washing Machine, Beko WI1382
    11: "Appliance3",    # Washing Machine
    13: "Appliance3",    # Washing Machine
    15: "Appliance3",    # Washing Machine, Beko WMB91242LB
    16: "Appliance5",    # Washing Machine, Bosch WAB24262GB
    17: "Appliance4",    # Washing Machine, Bosch Exxcel 8
    18: "Appliance5",    # Washing Machine
    19: "Appliance2",    # Washing Machine, Bosch WAE24060GB
    20: "Appliance4",    # Washing Machine
    21: "Appliance3",    # Washing Machine, Beko WMB81241LW
}

# Houses selected for EDA (at least 4, extended if data permits)
EDA_HOUSES = [2, 5, 7, 9, 17, 20]

# ── Cycle detection defaults ───────────────────────────────────────────
CYCLE_POWER_THRESHOLD_W = 20        # Active-power threshold (watts)
CYCLE_MIN_DURATION_MIN = 10         # Minimum cycle duration (minutes)
CYCLE_MERGE_GAP_MIN = 3             # Bridge very short interruptions

# ── Modelling ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
MODEL_HOUSES = [2, 5, 7, 9, 20]    # Houses for NILM model training / eval
TRAIN_FRAC = 0.7                    # Chronological train fraction
VAL_FRAC = 0.15                     # Chronological validation fraction
TEST_FRAC = 0.15                    # Chronological test fraction
HELD_OUT_HOUSE = 17                 # Generalization test household
