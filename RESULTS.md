# RESULTS.md — Flock Energy REFIT Data Science Results

This report documents the empirical results, evaluation metrics, and analytical findings produced by executing the complete data science pipeline on the **REFIT Electrical Load Measurements** dataset. All numerical values and figures cited in this document were computed directly from the dataset.

---

## 1. Washing Machine Exploratory Data Analysis (EDA)

### 1.1 Household Selection & Data Availability
Six households were analyzed using the **OFFICIALLY CLEANED REFIT** dataset at 1-minute resampled resolution. All households contain verified washing machine (WM) channels as per the official REFIT metadata:

| House ID | WM Channel | Date Range | Total Minutes | Valid Minutes | Missing % | Zero / Off % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **House 2** | Appliance 2 | 2013-09-17 → 2015-05-28 | 889,078 | 672,075 | 24.41% | 94.38% |
| **House 5** | Appliance 3 | 2013-09-26 → 2015-07-06 | 933,593 | 850,387 | 8.91% | 52.10% |
| **House 7** | Appliance 5 | 2013-11-01 → 2015-07-08 | 882,970 | 773,064 | 12.45% | 74.85% |
| **House 9** | Appliance 3 | 2013-12-17 → 2015-07-08 | 818,007 | 693,170 | 15.26% | 90.62% |
| **House 17** | Appliance 4 | 2014-03-06 → 2015-06-19 | 676,710 | 636,004 | 6.02% | 96.60% |
| **House 20** | Appliance 4 | 2014-03-20 → 2015-06-23 | 662,864 | 623,738 | 5.90% | 97.19% |

*Key Observation:* In most homes (House 2, 9, 17, 20), washing machines are idle >90% of the time, illustrating severe class imbalance. House 5 exhibited low zero percentage (52.1%) due to a continuous low-level standby draw or sub-metering sharing, which our cycle detector successfully handled via the 20 W active threshold.

**Plot Reference:**
- Data Availability Comparison: [data_availability_comparison.png](outputs/figures/section2_eda/data_availability_comparison.png)

---

### 1.2 Washing Machine Cycle Detection

#### Detection Rule and Parameters:
1. **Active Threshold**: $P \ge 20\text{ W}$ (filters out standby power and sensor drift).
2. **Short-Gap Bridging**: Intermittent dips $< 3\text{ minutes}$ below threshold during a wash (e.g., between wash, rinse, and spin cycles) are bridged.
3. **Minimum Duration**: An operating cycle must span at least **10 minutes**.
4. **Outage Boundary**: Missing data $> 3\text{ minutes}$ terminates a cycle candidate, preventing artificial merging across power outages.
5. **Energy Formula**: $E_{\text{cycle}}\text{ (kWh)} = \sum_{t=1}^{T} \frac{P_t \times 1\text{ min}}{60,000}$.

#### Detected Cycle Summary Statistics:
| House ID | Detected Cycles | Mean Duration (min) | Median Duration (min) | Std Duration (min) | Mean Peak (W) | Median Peak (W) | Mean Energy (kWh) | Total Energy (kWh) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **House 2** | 356 | 93.7 | 95.0 | 30.9 | 1,963.5 | 2,088.0 | 0.568 | 202.03 |
| **House 5** | 938 | 71.4 | 39.0 | 157.6 | 1,494.3 | 2,045.5 | 0.444 | 416.29 |
| **House 7** | 909 | 71.4 | 60.0 | 24.0 | 2,066.1 | 2,081.4 | 0.537 | 488.19 |
| **House 9** | 257 | 72.5 | 73.0 | 12.6 | 2,154.3 | 2,209.1 | 0.673 | 172.87 |
| **House 17** | 349 | 46.2 | 45.0 | 18.7 | 1,642.8 | 2,030.5 | 0.310 | 108.02 |
| **House 20** | 208 | 74.8 | 73.0 | 14.6 | 2,198.0 | 2,269.0 | 0.398 | 82.72 |

**Plot References:**
- Representative 24h traces: [house2_24h_trace.png](outputs/figures/section2_eda/house2_24h_trace.png), [house20_24h_trace.png](outputs/figures/section2_eda/house20_24h_trace.png)
- Detected Cycles overlay: [house2_detected_cycles.png](outputs/figures/section2_eda/house2_detected_cycles.png), [house17_detected_cycles.png](outputs/figures/section2_eda/house17_detected_cycles.png)
- Cross-Household Comparisons:
  - Duration Distribution: [cross_household_duration.png](outputs/figures/section2_eda/cross_household_duration.png)
  - Energy Distribution: [cross_household_energy.png](outputs/figures/section2_eda/cross_household_energy.png)
  - Peak Power Distribution: [cross_household_peak_power.png](outputs/figures/section2_eda/cross_household_peak_power.png)

---

### 1.3 Usage Patterns by Hour and Weekday
- **Hour of Day**: Peak washing machine operations across almost all households occur between **08:00 and 12:00 in the morning**, with a secondary minor peak around **18:00–20:00**.
- **Day of Week**: In family households (House 5, 7, 2), operations peak distinctly on **Saturday and Sunday** (laundry batching), whereas single/retired occupants (e.g., House 9, 20) demonstrate more evenly distributed mid-week washes.

**Plot References:**
- Hourly Usage: [house2_hourly_usage.png](outputs/figures/section2_eda/house2_hourly_usage.png), [house7_hourly_usage.png](outputs/figures/section2_eda/house7_hourly_usage.png)
- Day of Week Usage: [house2_dow_usage.png](outputs/figures/section2_eda/house2_dow_usage.png), [house7_dow_usage.png](outputs/figures/section2_eda/house7_dow_usage.png)

---

## 2. NILM Disaggregation Modeling

### 2.1 Model Architecture & Feature Engineering
- **Model**: `HistGradientBoostingRegressor` (500 maximum iterations, `learning_rate=0.05`, `max_leaf_nodes=63`, `min_samples_leaf=50`).
- **Input Features (31 total, strictly causal)**:
  - Instantaneous Aggregate Power ($P_t$)
  - Lagged Aggregate Power: $P_{t-1}, P_{t-2}, P_{t-3}, P_{t-5}, P_{t-10}, P_{t-15}, P_{t-30}, P_{t-45}, P_{t-60}$
  - Differences / Steps: $\Delta P_{t, t-1}, \Delta P_{t, t-2}, \Delta P_{t, t-5}$
  - Rolling Statistics (causal backward windows of 5, 15, 30, 60 minutes): Mean, Std Dev, Max, Min
  - Cyclical Calendar Features: $\sin/\cos(\text{hour})$, $\sin/\cos(\text{day of week})$, $\sin/\cos(\text{month})$
  - Binary Flags: `is_weekend`, `is_night` (23:00–06:00)
- **Target**: Washing machine power ($P_{\text{WM}, t}$ in Watts).

### 2.2 Validation Strategy & Leakage Prevention
1. **Chronological In-Household Split**: Houses 2, 5, 7, 9, 20 combined chronologically:
   - **Train**: 2,528,700 samples (70%)
   - **Validation**: 541,863 samples (15%)
   - **Test**: 541,871 samples (15%)
2. **Unseen Held-Out Generalization Evaluation**:
   - **House 17**: 636,004 valid samples completely withheld from feature scaling, hyperparameter tuning, and model training. Evaluates transferability to an unseen house and appliance.
3. **Strict Causal Discipline**: No future windows used. All statistics computed strictly over past observations.

---

### 2.3 Evaluation Metrics and Baselines

Two baselines were implemented:
1. **Baseline A — Always Off**: Predicts $0.0\text{ W}$ for all minutes.
2. **Baseline B — Usage Profile**: Predicts the training-set conditional mean power by `(hour_of_day, is_weekend)`.

#### Comprehensive Results Table:
| Split / Evaluation | Model / Baseline | N Samples | MAE (W) | RMSE (W) | $R^2$ | Actual Energy (kWh) | Pred Energy (kWh) | Energy Error (kWh) | Error % |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chronological Test** | Always Off | 541,871 | 27.49 | 204.65 | -0.0184 | 248.24 | 0.00 | -248.24 | -100.0% |
| **Chronological Test** | Usage Profile | 541,871 | 45.19 | 202.27 | 0.0052 | 248.24 | 196.29 | -51.95 | -20.9% |
| **Chronological Test** | **HistGBR (Ours)** | 541,871 | **30.75** | **155.53** | **0.4118** | **248.24** | **205.27** | **-42.97** | **-17.3%** |
| **Held-out House 17** | Always Off | 636,004 | 10.46 | 126.93 | -0.0068 | 110.91 | 0.00 | -110.91 | -100.0% |
| **Held-out House 17** | Usage Profile | 636,004 | 30.56 | 127.23 | -0.0115 | 110.91 | 230.56 | +119.65 | +107.9% |
| **Held-out House 17** | **HistGBR (Ours)** | 636,004 | **16.67** | **109.37** | **0.2525** | **110.91** | **124.29** | **+13.38** | **+12.1%** |

#### Conditional Evaluation on Active Periods Only ($P_{\text{WM}} > 20\text{ W}$):
Active minutes represent 6.6% (35,775 minutes) of the chronological test set:
- **Always Off (Active Only)**: MAE = 414.53 W, RMSE = 796.47 W, $R^2 = -0.3715$
- **HistGBR (Active Only)**: **MAE = 287.24 W, RMSE = 561.47 W, $R^2 = 0.3184$**

*Critical Metric Insight:*
Notice how the "Always Off" baseline achieves a deceptively low overall MAE of 27.49 W simply because the appliance is idle 93.4% of the time. However, its $R^2$ is negative (-0.0184) and it misses 100% of the actual energy (248 kWh error). HistGBR achieves an $R^2 = 0.4118$ on the chronological test set and an $R^2 = 0.2525$ on the completely unseen held-out House 17, estimating total energy within **-17.3%** and **+12.1%** error respectively.

**Plot References:**
- Actual vs. Predicted Time-Series Trace: [actual_vs_predicted_timeseries.png](outputs/figures/section3_model/actual_vs_predicted_timeseries.png)
- Model Comparison on Test Split: [model_comparison_test.png](outputs/figures/section3_model/model_comparison_test.png)
- Model Comparison on Held-Out House 17: [model_comparison_house17.png](outputs/figures/section3_model/model_comparison_house17.png)
- Residual Distribution: [residuals_test.png](outputs/figures/section3_model/residuals_test.png)
- Actual vs. Predicted Scatter: [scatter_actual_vs_predicted.png](outputs/figures/section3_model/scatter_actual_vs_predicted.png)

---

### 2.4 Multi-Model Exploration & Error Reduction

To improve predictive accuracy and reduce disaggregation error, a comparative benchmark was conducted evaluating advanced tree algorithms (XGBoost, LightGBM) and an ensemble blend against the baseline model, utilizing an expanded set of 51 causal features (power-surge step detection, second-order power acceleration, multi-scale rolling spreads, and load regime indicators):

| Model | Test MAE (W) | Test RMSE (W) | Test $R^2$ | Test Energy Err % | House 17 RMSE (W) | House 17 $R^2$ | Active Test MAE (W) | Active Test $R^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGBR (Baseline)** | 30.75 | 155.53 | 0.4118 | -17.30% | 109.37 | 0.2525 | 287.24 | 0.3184 |
| **LightGBM** | 32.01 | 154.83 | 0.4171 | -12.72% | 109.79 | 0.2467 | 286.50 | 0.3246 |
| **HistGBR-Enhanced** | 32.25 | 154.07 | 0.4228 | -7.06% | **108.98** | **0.2578** | 280.04 | 0.3526 |
| **HistGBR-Tuned** | **31.73** | 153.79 | 0.4250 | **-6.19%** | 112.27 | 0.2123 | **275.85** | **0.3637** |
| **XGBoost** | 32.16 | 153.71 | 0.4255 | -9.31% | 109.36 | 0.2526 | 281.61 | 0.3424 |
| **WeightedBlend (Best)** | 32.01 | **153.28** | **0.4287** | -9.68% | 109.25 | 0.2541 | 281.15 | 0.3436 |

#### Key Takeaways:
1. **Lowest Test RMSE & Highest $R^2$**: The **WeightedBlend** (0.10 HistGBR + 0.70 XGBoost + 0.20 LightGBM) achieved the highest overall performance on the test split with **$R^2 = 0.4287$** and **$\text{RMSE} = 153.28\text{ W}$** (improving over the original 0.4118 / 155.53 W).
2. **Best Generalization on Unseen House 17**: **HistGBR-Enhanced** delivered the best cross-household transfer with **$\text{RMSE} = 108.98\text{ W}$** and **$R^2 = 0.2578$** (improved from 109.37 W and 0.2525).
3. **Substantial Energy Estimation Error Reduction**: Test split total energy error was reduced from **-17.3%** down to **-6.19%** (HistGBR-Tuned) and **-7.06%** (HistGBR-Enhanced).
4. **Active Period Improvement**: On active wash minutes ($P_{\text{WM}} > 20\text{ W}$), MAE was reduced from 287.24 W down to **275.85 W**, while active $R^2$ improved from 0.3184 to **0.3637**.

**Enhanced Model Visualizations:**
- Multi-Model Test Comparison: [model_comparison_enhanced_test.png](outputs/figures/section3_model/model_comparison_enhanced_test.png)
- Multi-Model Held-Out Comparison: [model_comparison_enhanced_house17.png](outputs/figures/section3_model/model_comparison_enhanced_house17.png)
- Best Model Time-Series Overlay: [actual_vs_predicted_best_model.png](outputs/figures/section3_model/actual_vs_predicted_best_model.png)
- Best Model Residual Distribution: [residuals_best_model_test.png](outputs/figures/section3_model/residuals_best_model_test.png)
- Best Model Scatter Plot: [scatter_best_model_test.png](outputs/figures/section3_model/scatter_best_model_test.png)

---

## 3. Failure Analysis

Quantitative analysis of model predictions (`outputs/metrics/failure_analysis.json`) revealed three distinct modes of failure:

1. **Water-Heating Pulse Under-estimation (High-Error Outliers)**:
   - There are 5,419 test minutes (1.0% of test data) with absolute errors $> 663.5\text{ W}$.
   - During these minutes, the actual washing machine average power was **1,593.4 W** (the resistive heating phase), whereas the model predicted an average of **625.0 W**.
   - *Cause*: In a 1-minute aggregate signal, a 2,000 W heater pulse is indistinguishable from a kettle or electric shower, causing the gradient boosting model to hedge predictions toward conditional probabilities.
2. **False Positives When Appliance is Off (Signatures Overlap)**:
   - Across 502,663 minutes when the washing machine was genuinely off, the model predicted $>50\text{ W}$ in **20,977 minutes** (4.17% false-positive rate).
   - *Cause*: Dishwashers and immersion heaters create multi-minute plateau loads of 1,500–2,000 W that mirror washing machine wash cycles.
3. **Missed Low-Power Events (Spin / Rinse Cycles)**:
   - Across 35,758 active minutes, the model predicted $< 20\text{ W}$ in **6,477 minutes** (18.11% miss rate).
   - *Cause*: Low-power tumbling and drain pump operations (50–150 W) get buried under household baseline noise (refrigeration, electronics, lighting).

---

## 4. Key Limitations & Takeaways

1. **Generalization across Homes**: Model $R^2$ dropped from 0.41 on known homes to 0.25 on held-out House 17. Different appliance models have different heating profiles and cycle lengths.
2. **Resolution Constraints**: 1-minute aggregation smooths out the high-frequency switching transients (motor PWM, valve clicks) that would otherwise allow disambiguation from dishwashers.
3. **Suitability**: The model is well-suited for **aggregate weekly/monthly energy budgeting and time-of-use scheduling feedback**, but **unsuitable for minute-level sub-metering or direct billing**.
