# Power Outages: An Analysis and Prediction Model (v2)

UCSD DSC 80 Data Science Project

### By: Sonali Singh

> This repository is a personal rework of a project originally completed jointly with **Susana Haing** for DSC 80 — the original lives at [shaing04/power-outages-analysis](https://github.com/shaing04/power-outages-analysis). Redone here with Susana's permission, in my own repository. See **Attribution** and **Changelog** below for exactly what's carried over vs. rebuilt.

## Attribution

- **Steps 1-4 (Introduction, Data Cleaning & EDA, Assessment of Missingness, Hypothesis Testing): joint work**, carried over from the original DSC 80 project with Susana Haing. Credit for the data pipeline and hypothesis test is shared equally.
- **Steps 5-8 (Framing the Prediction Problem, Baseline Model, Final Model, Fairness Analysis): independent rework**, done solely by me for this repository. The modeling section was rebuilt from scratch — different features, different target transformation, a different model family, and an additional classification framing — not just re-polished.

## Changelog (v1 → v2)

The original v1 modeling section (linear regression on raw-minute outage duration) topped out at **R² = 0.045**. Its writeup also contained a factual error: it claimed RMSE *decreased* from baseline (7264 min) to final model (7422 min) — it actually increased. This rework:

1. **Fixes the RMSE sentence.** The comparison table in this README and the notebook state the real deltas, including where a metric got worse.
2. **Drops `U.S._STATE` as a feature.** With ~1,400 usable rows across 49 states, one-hot-encoding state adds a large, sparse, overfitting-prone block without much signal beyond what `CLIMATE.REGION` / `NERC.REGION` already capture.
3. **Adds `SEASON`** (derived from `MONTH`) as a lower-cardinality seasonal feature.
4. **Predicts `log1p(duration)` instead of raw minutes**, since duration is massively right-skewed (a long tail of multi-week outages dominates squared-error loss on the raw scale).
5. **Replaces linear regression with gradient boosting** (`HistGradientBoostingRegressor`) to capture non-linear interactions between cause, region, and season.
6. **Reports MAE in hours** as the headline metric, with R² shown on both the log scale (what the model actually optimizes) and the raw-minutes scale (flagged as unstable, not hidden).
7. **Adds a classification reframing**: duration buckets (`<1 day` / `1-3 days` / `3+ days`) via `HistGradientBoostingClassifier` — a more forgiving, more decision-useful framing ("should we tell customers to expect same-day power, or a multi-day outage?").
8. **Rebuilds the fairness analysis** on the new model, and is explicit about a real limitation the original didn't flag: the Equipment Failure test-set group has only 12 examples, so the fairness test has limited statistical power regardless of outcome.

All of this lives in [`power_outage_analysis.ipynb`](power_outage_analysis.ipynb), executed end-to-end against [`outage.csv`](outage.csv) (same dataset as the original project).

## Introduction

In this project, we examine [data](https://engineering.purdue.edu/LASCI/research-data/outages) from Purdue University's Laboratory for Advancing Sustainable Critical Infrastructure on major power outages in the continental United States from January 2000 to July 2016.

The data includes geographical location, climate/weather, outage cause, and the number of people affected for each major outage.

Our research question: **how do different causes affect power outage duration, and can we predict how long an outage will last?** We test whether severe weather has a measurable effect on duration, then build a model to predict duration from information available around the time an outage begins — useful for utilities triaging restoration communication with customers.

| Variable | Description |
|---|---|
| YEAR | The year an outage occurred |
| MONTH | The month an outage occurred |
| U.S._STATE | The specific US State the power outage occurred in |
| NERC.REGION | North American Electric Reliability Corporation region of an outage |
| CLIMATE.REGION | The 9 NOAA-defined climate regions in the United States |
| OUTAGE.START.DATE / TIME | Date and time an outage started |
| OUTAGE.RESTORATION.DATE / TIME | Date and time an outage was resolved |
| CAUSE.CATEGORY | Cause of the power outage |
| CLIMATE.CATEGORY | Whether it was a cold, warm, or normal climate period |
| OUTAGE.DURATION | Duration of the outage, in minutes |
| CUSTOMERS.AFFECTED | Number of customers affected by an outage |
| POPDEN_URBAN | Population density of urban areas (persons / sq. mile) |

## Data Cleaning and Exploratory Data Analysis *(joint work)*

The original dataset has 57 variables; we keep only the columns above. `OUTAGE.START.DATE`/`OUTAGE.START.TIME` are condensed into a single `OUTAGE.START` datetime column, and likewise for `OUTAGE.RESTORATION`. Rows with `OUTAGE.DURATION == 0` are set to `NaN` — a *major* outage lasting 0 minutes isn't physically meaningful, so a recorded 0 more likely reflects a data-entry artifact.

**Power outages by state:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/outages_map.html" width="100%" height="600" frameborder="0"></iframe>
</div>

**Power outages over time:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/outages_overtime.html" width="100%" height="600" frameborder="0"></iframe>
</div>

**Power outages by climate region:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/outages_climate_region.html" width="100%" height="600" frameborder="0"></iframe>
</div>

**Outage duration by climate region:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/outage_duration_climateregion.html" width="100%" height="600" frameborder="0"></iframe>
</div>

**Outage duration by cause category:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/outage_duration_cause_cat.html" width="100%" height="600" frameborder="0"></iframe>
</div>

**Aggregation — duration stats by cause category:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/agg_summary_table.html" width="100%" height="260" frameborder="0"></iframe>
</div>

**Pivot — outage count by climate region × cause category:**
<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/pivot_table_year.html" width="100%" height="300" frameborder="0"></iframe>
</div>

## Assessment of Missingness *(joint work)*

This section compares outage durations for *Severe Weather* vs. all *Other Causes*, as a visual precursor to the formal hypothesis test below.

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/missingness_proportion.html" width="100%" height="400" frameborder="0"></iframe>
</div>

**Boxplot insight:** Severe Weather outages have a higher median and wider IQR — both longer and more variable durations, with more extreme outliers.

**Histogram insight:** Severe Weather durations are right-skewed with a long tail past 500 hours; Other Causes cluster tightly around 20-40 hours. This skew is exactly what motivates the log-transform used in the v2 modeling section below.

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/missingness_box.html" width="100%" height="400" frameborder="0"></iframe>
</div>

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/missingness_hist.html" width="100%" height="400" frameborder="0"></iframe>
</div>

## Hypothesis Testing *(joint work)*

**Null Hypothesis (H₀):** Severe weather has no effect on the length of a power outage.

**Alternative Hypothesis (H₁):** Severe weather has an effect on the length of a power outage.

**Test statistic:** Difference in means (Severe Weather − Other Causes).

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/permutation_test_5.html" width="100%" height="400" frameborder="0"></iframe>
</div>

**Results:** Observed difference: 2399.86 minutes. Permutation p-value: 0.0000 (10,000 permutations).

Since p < 0.05, we **reject the null hypothesis**: severe weather significantly increases outage duration.

## Framing the Prediction Problem *(v2 — independent rework)*

Everything from here on is an independent rebuild of the modeling section. See the Changelog above for what changed and why.

**Features used** (all knowable at or shortly after outage onset — no information about how the outage was ultimately resolved): `CAUSE.CATEGORY`, `CLIMATE.REGION`, `NERC.REGION`, `SEASON` (derived from `MONTH`), `MONTH`, `YEAR`, `START_HOUR` (derived from `OUTAGE.START`), `POPDEN_URBAN`.

**Evaluation metric:** MAE in hours as the headline number, with R² reported on both the log scale (what gradient boosting is optimized against) and the raw-minutes scale (shown, but flagged as unstable given the long right tail).

## Baseline Model *(v2)*

Two baselines, both on the same 1,393-row cleaned sample and 80/20 train/test split (`random_state=42`):

- **Dummy (predict the mean):** MAE 53.41 hrs, RMSE 115.10 hrs, R² −0.0009.
- **Linear Regression** (v2 feature set, no log-transform): MAE 43.36 hrs, RMSE 103.86 hrs, R² 0.1851.

For reference, the **original v1** final linear regression model (different feature set, including `U.S._STATE`) reported MAE ≈ 2934.65 min (≈ 48.9 hrs), RMSE ≈ 7422.66 min (≈ 123.7 hrs), R² ≈ 0.0449 — and its writeup incorrectly claimed RMSE had decreased from the v1 baseline (7264 min); it had actually increased. That's the sentence this rework corrects.

Just swapping `U.S._STATE` for `CLIMATE.REGION` + `SEASON` — before gradient boosting or the log-transform even enter the picture — roughly quadruples R² (0.045 → 0.185). Most of the v1 model's weakness was feature framing, not an unbeatable ceiling.

## Final Model *(v2)*

**Gradient boosting on log-duration:** `HistGradientBoostingRegressor` predicting `log1p(OUTAGE.DURATION)`.

- **MAE: 38.59 hours**
- **RMSE: 110.37 hours** (raw-minutes scale)
- **R² (log scale, matches optimization target): 0.5321**
- **R² (raw-minutes scale): 0.0798** — shown for transparency, but this number is dominated by a handful of multi-week outages and is not a reliable summary of fit for this target.

**Reading these honestly:** MAE improves from 43.4 → 38.6 hours going from linear/raw-minutes to gradient-boosting/log-duration — a real but modest gain, not a breakthrough. The gap between log-scale R² (0.53) and raw-minutes R² (0.08) is the point: raw-minutes R² is the wrong lens for a target this skewed. We do not claim this solves duration prediction — real drivers like crew dispatch time and physical damage extent aren't in this dataset, so a real ceiling on predictability here is a legitimate, well-diagnosed finding rather than a modeling shortfall.

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/final_model_v2_residuals.html" width="100%" height="400" frameborder="0"></iframe>
</div>

### Reframing as Classification — Duration Buckets

Predicting exact duration is hard from these features; predicting a **triage bucket** (`<1 day` / `1-3 days` / `3+ days`) is more forgiving and arguably more useful — it mirrors what a utility would actually tell customers.

- **Accuracy: 0.681** vs. **0.599** majority-class baseline.
- **F1 (macro): 0.601.**
- Best-distinguished class: `<1 day` (precision 0.82, recall 0.81) — the practically important distinction for triage. `1-3 days` vs. `3+ days` is a harder boundary given these features.

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/bucket_confusion_matrix.html" width="100%" height="500" frameborder="0"></iframe>
</div>

### Summary

| Model | Target | MAE | R² | Notes |
|---|---|---|---|---|
| v1 (original, joint project) — Linear Regression | raw minutes | ≈ 48.9 hrs | 0.045 | original `U.S._STATE`-based feature set |
| v2 Baseline — Linear Regression | raw minutes | 43.4 hrs | 0.185 | same v2 feature set as Final Model, no log-transform/GB |
| v2 Final — Gradient Boosting | log1p(minutes) | 38.6 hrs | 0.53 (log) / 0.08 (raw) | headline model |
| v2 Classification — Gradient Boosting | duration bucket | — | acc. 0.68 vs. 0.60 baseline; F1 (macro) 0.60 | more forgiving, actionable framing |

## Fairness Analysis *(v2)*

We check whether the final regression model is equally accurate across outage causes — **Severe Weather** vs. **Equipment Failure**.

**Null Hypothesis (H₀):** The model is fair — RMSE is the same for both groups, and any observed difference is due to chance.

**Alternative Hypothesis (H₁):** The model is unfair — RMSE differs meaningfully between the two groups.

**Test statistic:** Difference in RMSE (Severe Weather − Equipment Failure), via permutation test.

**Caveat:** the test set has only **12** Equipment Failure examples (vs. 151 Severe Weather). RMSE for that group is estimated from very few points and will be noisy regardless of what the model does.

- RMSE, Severe Weather: 62.20 hours
- RMSE, Equipment Failure: 371.88 hours
- Observed difference: −309.68 hours
- **Permutation p-value (two-sided, 2000 reps): 0.0795**

<div style="width: 100%; margin: 0 auto;">
  <iframe src="assets/fairness_analysis_v2.html" width="100%" height="400" frameborder="0"></iframe>
</div>

**Conclusion:** At the 0.05 level we **fail to reject** the null — no significant evidence of unfairness between these two groups. Given the very small Equipment Failure sample, this test has limited power in either direction; a larger, more balanced sample would be needed for a confident fairness conclusion.

## Reproducing this analysis

```
pip install pandas numpy scikit-learn plotly seaborn matplotlib scipy us jupyter
jupyter nbconvert --to notebook --execute power_outage_analysis.ipynb
```

Requires `outage.csv` (included in this repo) in the working directory.
