# Recommendation.md — Utility-Facing Summary

## Executive Summary

This analysis demonstrates that **washing-machine usage patterns can be partially recovered from whole-house smart meter data**, but with important caveats about accuracy and deployment suitability. The model provides useful aggregate insights for customer engagement programs while remaining insufficient for precise per-appliance billing or automated control.

---

## Key Findings

1. **Washing machines account for a small fraction of total household energy**, but their usage patterns are distinctive and detectable from aggregate power signals.

2. **Cross-household variability is significant** — cycle durations, peak power, and usage times vary substantially between homes, reflecting different appliance models, household sizes, and lifestyles.

3. **The NILM model outperforms naive baselines** on metrics like MAE and RMSE, particularly during active washing-machine periods. However, the "always-off" baseline achieves deceptively good aggregate metrics because the washing machine is idle for ~95%+ of all minutes.

4. **Generalization to unseen households is limited** — models trained on a subset of homes show degraded performance on held-out households with different appliance makes and usage patterns.

---

## Practical Opportunity: Washing-Machine Usage Insight Program

### Who Would Use It
- **Energy utility customer engagement teams** designing energy-awareness campaigns
- **Customers** interested in understanding their appliance-level consumption
- **Demand-response program managers** targeting flexible loads for shifting

### Proposed Intervention
Provide participating households with a monthly **"Washing-Machine Usage Report"** derived from smart meter data, showing:
- Estimated number of wash cycles per week
- Typical washing hours (highlighting off-peak opportunities)
- Comparison with similar households
- Estimated washing-machine energy cost

### What Action Could Be Taken
- Households washing during **peak hours** could be nudged toward **off-peak washing** (e.g., after 10 PM or before 7 AM)
- Households with **unusually frequent cycles** (e.g., >12/week) could receive tips on load consolidation
- Time-of-use tariff customers could be shown potential savings from shifting wash times

### Evaluation Plan
1. **Pilot design**: Recruit 200 households (100 treatment + 100 control) matched by household size, tariff type, and baseline consumption
2. **Intervention**: Treatment group receives monthly washing-machine reports for 6 months
3. **Primary metric**: Change in estimated peak-hour washing-machine energy (kWh/month)
4. **Secondary metrics**: Customer satisfaction scores, report engagement rates, total household consumption change
5. **Seasonality control**: Compare same months year-over-year, or use a matched difference-in-differences design
6. **Minimum detectable effect**: With 100 per group, a 15% shift in peak-hour washing energy is detectable at α=0.05, β=0.80

### Why Disaggregation Estimates Should Not Be Treated as Exact
- The model produces **probabilistic estimates**, not exact measurements
- False positives (attributing other appliance usage to the washing machine) and missed events both occur
- Reports should use language like "estimated" and "approximate" and should be presented as trends, not precise figures
- Individual washing-machine energy estimates may deviate significantly from reality; the value lies in **aggregate patterns and relative comparisons**

---

## Key Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Overlapping appliance signatures** | Dishwashers and kettle spikes can be misidentified as WM activity | Use cycle-level filtering (duration, power profile) |
| **Missing data and outages** | Gaps in Feb 2014 and sporadic outages reduce coverage | Report coverage alongside estimates; flag low-confidence periods |
| **Household-specific behavior** | Model trained on UK homes; appliance models and habits vary | Retrain on local data before deployment; treat initial model as a prototype |
| **REFIT ≠ Indian households** | Voltage (230V UK vs 230V India nominal), appliance types, usage patterns, and household structures differ significantly | Collect a calibration dataset from target market before claiming transferability |
| **Threshold-based cycle detection** | Not true ground truth; may split or merge cycles incorrectly | Present cycle counts as estimates; validate against a sample of households with sub-metering |
| **Model generalization** | Performance degrades on unseen households | Include household-specific recalibration; ensemble approaches may help |

---

## Implications of Lower-Frequency Smart Meter Data

Most deployed smart meters record at **15-minute or 30-minute intervals**, far coarser than the 1-minute resolution used in this analysis.

### What Becomes Harder at 15–30 Minute Resolution
- **Temporal information loss**: A typical washing-machine cycle (30–90 minutes) spans only 2–6 readings at 15-minute resolution, making it difficult to distinguish cycle phases
- **Cycle boundary detection**: Start and end times become ambiguous; a 10-minute warm-up phase is invisible in a 15-minute reading
- **Peak power estimation**: Peak power during the spin cycle (often 500–2000 W for ~5 minutes) is averaged away
- **Short events disappear**: Quick-wash cycles (< 15 min) may be entirely contained within a single interval
- **Multiple appliances merge**: A kettle boil (3 min) overlapping with a washing cycle is indistinguishable from a higher washing-machine power reading
- **Feature degradation**: Lagged features and rolling statistics become less informative with fewer data points per cycle

### Key Implication
A model trained at 1-minute resolution **cannot simply be applied to 15-minute or 30-minute data** by averaging or downsampling. The entire feature pipeline, model architecture, and evaluation must be redesigned for the lower resolution. Performance should be expected to degrade significantly, and separate validation on appropriately-sampled data is essential.

---

## Suitable and Unsuitable Uses

### ✅ Suitable Uses
- **Population-level insights**: Average washing times, peak-hour usage fractions across thousands of households
- **Customer engagement**: Approximate usage reports that encourage awareness without claiming measurement precision
- **Demand-response targeting**: Identifying households with significant flexible washing loads
- **Research and pilot programs**: Testing hypotheses about behavior change

### ❌ Unsuitable Uses
- **Per-appliance billing**: Accuracy is insufficient for individual metering replacement
- **Automated appliance control**: False positives/negatives make automated scheduling unreliable
- **Regulatory compliance**: Cannot substitute for certified sub-metering
- **Individual household energy audits**: Estimates may deviate substantially from reality for any single home
