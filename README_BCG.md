# BCG Data Science Job Simulation — PowerCo Churn Analysis

> **Programme:** Boston Consulting Group (BCG) Data Science Virtual Experience  
> **Client:** PowerCo — SME Electricity Division  
> **Objective:** Identify and predict customer churn using data science techniques, and communicate findings to senior stakeholders

---

## Project Overview

PowerCo, a major gas and electricity utility, is concerned about its customers leaving for competitors in the increasingly deregulated European energy market. The Head of the SME Division hypothesised that **price sensitivity** is the primary driver of churn among small and medium enterprise (SME) customers.

BCG was engaged to investigate this hypothesis, build a predictive churn model, and deliver actionable recommendations.

---

## Repository Structure

```
├── client_data__1_.csv               # Raw historical customer data
├── price_data__1_.csv                # Raw historical pricing data
├── data_for_predictions.csv          # Final engineered dataset (provided by Estelle)
│
├── Task_3_EDA.ipynb                  # Task 3 — Exploratory Data Analysis
├── Task_4_Feature_Engineering.ipynb  # Task 4 — Feature Engineering
├── Task_5_Modelling.ipynb            # Task 5 — Random Forest Modelling & Evaluation
│
├── Task6_Executive_Abstract.pptx     # Task 6 — Steering Committee Presentation
│
└── README.md                         # This file
```

---

## Task Breakdown

### Task 2 — Price Sensitivity Framing *(conceptual)*

Before any data work, the team defined what **price sensitivity** means in the context of energy churn:

- A customer is price-sensitive if their probability of churning increases meaningfully when exposed to a cheaper competitor offer
- This framing shaped the hypothesis to be tested and guided the feature engineering choices in later tasks
- Key question: *Is price the primary churn driver, or are other factors (margin, tenure, consumption) more predictive?*

---

### Task 3 — Exploratory Data Analysis

**File:** `Task_3_EDA.ipynb`

#### Datasets Analysed
| Dataset | Rows | Columns | Description |
|---|---|---|---|
| `client_data__1_.csv` | 14,606 | 26 | Customer usage, contract dates, margins, churn label |
| `price_data__1_.csv` | ~175,000 | 8 | Monthly variable and fixed prices per customer |

#### Framework Applied
1. **Data type inspection** — identified date columns stored as strings, binary categoricals, and pricing floats
2. **Missing value analysis** — both datasets are complete with zero missing values
3. **Descriptive statistics** — revealed highly right-skewed consumption, low gas usage across the base, and mixed-sign margins
4. **Distribution visualisations** — histograms for consumption, margin, forecast, and price columns split by churn/retention
5. **Categorical analysis** — stacked bar charts for sales channel, origin campaign, and gas subscription vs churn rate
6. **Correlation analysis** — heatmap and feature-vs-churn bar chart for all numeric columns

#### Key Findings
- Overall churn rate: **9.7%** (1,419 / 14,606 customers) — imbalanced dataset
- `cons_12m` (12-month consumption) is heavily right-skewed; a small number of customers consume vastly more than the median
- Most customers do not use gas (`cons_gas_12m` = 0 for 75%+ of the base)
- Certain sales channels show notably higher churn rates than others
- `price_peak_var` and `price_mid_peak_var` are 0 for the majority of customers — off-peak pricing dominates

---

### Task 4 — Feature Engineering

**File:** `Task_4_Feature_Engineering.ipynb`

#### Framework Applied

**Step 1 — Remove redundant columns**
- Dropped `forecast_cons_year`: near-identical to `forecast_cons_12m` (Pearson r ≈ 1.0)

**Step 2 — Extract date features**
| New Feature | Source | Rationale |
|---|---|---|
| `contract_duration_days` | `date_end` − `date_activ` | Captures contract length |
| `days_to_renewal` | `date_renewal` − reference date | Proximity to renewal = churn risk window |
| `days_since_modif` | reference date − `date_modif_prod` | Recency of last product change |
| `month_activ` | `date_activ` month | Seasonality of sign-up |
| `month_end` | `date_end` month | Seasonality of contract end |
| `year_activ` | `date_activ` year | Cohort effect |

**Step 3 — Create derived features**
| New Feature | Formula | Rationale |
|---|---|---|
| `cons_ratio_last_to_12m` | `cons_last_month / (cons_12m / 12)` | Detects unusual recent consumption |
| `gas_to_elec_ratio` | `cons_gas_12m / cons_12m` | Energy mix profile |
| `forecast_vs_actual` | `forecast_cons_12m − cons_12m` | Forecast accuracy / demand change |
| `log_cons_12m` | `log1p(cons_12m)` | Reduces right skew |
| `net_margin_per_unit` | `net_margin / cons_12m` | Profitability relative to usage |
| `margin_diff_gross_net` | `margin_gross_pow_ele − margin_net_pow_ele` | Overhead indicator |

**Step 4 — Encode categoricals**
- `has_gas`: Y/N → 1/0
- `channel_sales`: one-hot encoded (drop first)
- `origin_up`: one-hot encoded (drop first)

**Step 5 — Aggregate price data per customer**
- Computed mean, max, min, and standard deviation of all 6 price columns across the 12 monthly records per customer
- Created **price sensitivity features**:
  - `price_sensitivity_var`: off-peak vs peak variable price gap (most recent month)
  - `price_sensitivity_fix`: off-peak vs peak fixed price gap (most recent month)
  - `price_off_peak_var_change`: change in off-peak variable price over the observed period

**Step 6 — Merge datasets**
- Joined client features with aggregated price features on `id`
- Final dataset: **14,606 rows × 63 features** (plus target `churn`)
- Saved as `engineered_features.csv`

---

### Task 5 — Random Forest Modelling & Evaluation

**File:** `Task_5_Modelling.ipynb`  
**Input data:** `data_for_predictions.csv` (final dataset provided by Estelle)

#### Model Configuration
```python
RandomForestClassifier(
    n_estimators   = 1000,       # 1,000 decision trees for stable, converged predictions
    max_features   = 'sqrt',     # sqrt(n_features) considered at each split — reduces tree correlation
    min_samples_leaf = 5,        # Minimum 5 samples per leaf — controls overfitting
    random_state   = 42,         # Reproducibility
    n_jobs         = -1          # Use all CPU cores
)
```

**Train/Test split:** 75% training / 25% test (`random_state=42`)

#### Evaluation Metrics — Results

| Metric | Value | Class |
|---|---|---|
| Accuracy | **90.25%** | All |
| Precision | **100%** | Churn (1) |
| Recall | **2.73%** | Churn (1) |
| F1 Score | **5.32%** | Churn (1) |
| ROC-AUC | **0.6632** | — |

**Confusion Matrix:**
```
                    Predicted: Retained   Predicted: Churned
Actual: Retained          3,286                  0
Actual: Churned             356                 10
```

#### Why These Metrics?

| Metric | Justification |
|---|---|
| **Accuracy** | Baseline indicator — but misleading on imbalanced data (a model always predicting "retained" also gets ~90%) |
| **Precision** | Measures false alarm rate — important commercially as retention interventions cost money |
| **Recall** | Most business-critical — catching actual churners before they leave; missing them means lost revenue |
| **F1 Score** | Harmonic mean of precision and recall — balanced single metric for imbalanced classes |
| **ROC-AUC** | Threshold-agnostic discrimination ability — ideal for comparing models |

#### Is Model Performance Satisfactory?

**No.** Despite 90.25% accuracy, the model only catches **10 out of 366 actual churners** (recall = 2.73%). If deployed, PowerCo would miss 97% of customers about to leave.

The root cause is **class imbalance** (~90% retained / ~10% churned): the model learns that always predicting "retained" is almost always safe. The ROC-AUC of 0.663 shows the underlying signal does exist — the problem is the decision threshold and weighting, not the absence of predictive features.

#### Top Predictive Features
1. `margin_net_pow_ele` — Net power electricity margin
2. `margin_gross_pow_ele` — Gross power electricity margin
3. `cons_12m` — 12-month electricity consumption
4. `forecast_meter_rent_12m` — Forecasted meter rent
5. `net_margin` — Overall net margin
6. `num_years_antig` / `months_activ` — Customer tenure
7. `off_peak_peak_var_mean_diff` — Engineered price sensitivity feature *(supports BCG hypothesis)*

#### Recommended Model Improvements
- Apply `class_weight='balanced'` to penalise minority class misclassifications
- Use SMOTE (Synthetic Minority Oversampling Technique) on the training set
- Lower the classification threshold (e.g. flag churn if probability > 0.2–0.3)
- Perform hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`
- Evaluate precision-recall curves (more informative than ROC for imbalanced datasets)

---

### Task 6 — Executive Abstract (Steering Committee Presentation)

**Files:** `Task6_Executive_Abstract.pptx`

A BCG-style 2-slide executive summary prepared for the Head of the SME Division and key client stakeholders.

#### Slide 1 — Cover
Title slide with BCG branding, client identification, and project track overview.

#### Slide 2 — Abstract
A structured single-page summary containing:

| Section | Content |
|---|---|
| **Situation** | PowerCo's SME churn problem and the price sensitivity hypothesis |
| **Key Data Finding** | 9.7% churn rate; margin, consumption, and tenure are the dominant drivers — not price alone |
| **Price Sensitivity Insight** | Price gap features do contribute but rank below financial margin variables |
| **Business Impact** | Estimated ~213 additional retained customers per period if model is improved and discount programme deployed |
| **Model Performance** | 4-metric grid: Accuracy 90.3%, Precision 100%, Recall 2.7% (flagged red), ROC-AUC 0.663 |
| **Recommended Next Steps** | 4 numbered actions: fix class imbalance → target high-risk customers → deploy score-based discounts → run A/B test |

---

## Key Takeaways

| # | Finding |
|---|---|
| 1 | **Price sensitivity is a contributing factor to churn — but not the primary driver.** Power margin, consumption, and tenure are stronger predictors. |
| 2 | **The churn dataset is imbalanced (~10% churn rate)**, which makes accuracy a misleading evaluation metric. Recall is the most business-critical measure. |
| 3 | **The baseline Random Forest model is insufficient for deployment** — it catches only 2.7% of churners at the default threshold. |
| 4 | **The model's ROC-AUC of 0.663 confirms that predictive signal exists in the features** — the issue is addressable with class balancing and threshold tuning. |
| 5 | **Targeted retention interventions (discounts)** should be focused on customers with low power margins, high consumption variability, and short tenure — not broad discounts to all customers. |

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualisations |
| scikit-learn | Random Forest model, metrics, train/test split |
| PptxGenJS | Programmatic PowerPoint generation |
| Jupyter Notebook | Analysis and documentation environment |

---

## How to Run

1. Place `client_data__1_.csv`, `price_data__1_.csv`, and `data_for_predictions.csv` in the same directory as the notebooks
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Run notebooks in order:
   ```
   Task_3_EDA.ipynb
   Task_4_Feature_Engineering.ipynb
   Task_5_Modelling.ipynb
   ```

---

*BCG Data Science Virtual Experience Programme — PowerCo SME Churn Project*
