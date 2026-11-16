# Customer Churn Classification Model

An end-to-end customer churn classifier built on **Databricks** with **Unity Catalog**, **Delta Lake** and **MLflow**. The notebook builds point-in-time monthly feature snapshots, evaluates models with an expanding-window time-series backtest (no random splits), compares a logistic-regression baseline against a tuned gradient-boosting model, evaluates the winner exactly once on a held-out month, registers it in Unity Catalog, and publishes a scored risk table for marketing.

**Notebook:** [`Customer_Churn_Classification_Model.ipynb`](./Customer_Churn_Classification_Model.ipynb)

---

## Highlights

| | |
|---|---|
| **Champion model** | Tuned `HistGradientBoostingClassifier` (scikit-learn) |
| **Holdout ROC-AUC** | **0.872** (2025-09-30 snapshot, evaluated once) |
| **Holdout PR-AUC** | **0.708** (base churn rate 18.3%) |
| **Top-decile lift** | **4.56x** |
| **Backtest ROC-AUC** | 0.852 ± 0.012 across 6 time folds (baseline LR: 0.846 ± 0.011) |
| **Customers scored** | 1,558 (338 High risk, 840 flagged at the profit-optimal threshold) |
| **Reproducibility** | Fixed random seed `42` everywhere |

---

## Problem definition

At each month-end **cutoff date `c`**:

- **Eligible customer:** signed up before `c` **and** has at least one *Sold* transaction in the 180 days ending at `c`.
- **Label:** `churned = 1` if the customer has **zero** *Sold* transactions in the **90 days after** `c`.
- **Features:** computed **only from data on or before `c`**, so nothing from the future leaks in.

Sixteen monthly snapshots (2024-06-30 → 2025-09-30) produce **25,712 customer-month rows** with an overall churn rate of **15.1%** (12.2%–18.6% per month).

---

## Data sources

All inputs are cleansed **gold** tables from an upstream medallion pipeline in Unity Catalog (`customerchurn.customerdb`):

| Table | Rows | Description |
|---|---:|---|
| `customers_gold` | 1,969 | Customer profile, membership tier, signup date, home store, email opt-in |
| `transactions_gold` | 65,348 | Transactions (Sold / Returned); walk-ins with null `Customer_id` are excluded |
| `engagement_gold` | — | Monthly emails sent/opened and app logins per customer |
| `stores_gold` | 12 | Store location and region |

---

## Features (18)

**Numeric (15)**

| Group | Features |
|---|---|
| Recency | `recency_days` |
| Frequency | `frequency_30d`, `frequency_90d`, `frequency_180d` |
| Monetary | `spend_90d`, `spend_180d`, `avg_basket` |
| Behaviour | `return_rate_180d`, `trend_ratio` (last 90d vs. prior 90d), `distinct_stores_180d` |
| Tenure | `tenure_days` |
| Engagement | `emails_open_rate_3m`, `app_logins_3m`, `emails_opened_delta_3m`, `app_logins_delta_3m` |

**Categorical (3):** `Membership_tier`, `Store_Region`, `Email_opt_in`

Preprocessing is a scikit-learn `ColumnTransformer`: median imputation for numerics (plus scaling for the linear model) and most-frequent imputation + one-hot encoding for categoricals.

---

## Notebook walkthrough

| # | Section | What it does |
|---|---|---|
| 0 | **Config** | Table names, MLflow experiment, UC model name, seed |
| — | **Data** | Loads the gold tables, prints schemas, row counts and profiles |
| 1 | **Labels + point-in-time features** | Builds 16 monthly snapshots → `churn_features` Delta table |
| 2 | **Backtest design** | Six expanding-window folds plus one reserved holdout |
| 3 | **Baseline** | `LogisticRegression(class_weight="balanced")` with MLflow parent/child runs |
| 4 | **Main model** | Gradient boosting tuned on mean backtest AUC (15 trials) |
| 5 | **Champion + holdout + registration** | Picks the winner, retrains, scores the holdout once, registers in UC with the `@champion` alias |
| 6 | **Model analysis** | Calibration, threshold/profit sweep, segments, explainability, leakage checks |
| 7 | **Scoring & publishing** | Loads `@champion` from the registry and writes `churn_analysis` |
| 8 | **ML traces** | MLflow tracing of the scoring job (span tree visible in the experiment's Traces tab) |

### Validation strategy (no random splits)

| Split | Cutoffs | Purpose |
|---|---|---|
| Backtest folds (6) | Test = 2025-01-31 … 2025-06-30; train = all cutoffs ≤ test − 3 months | Model comparison and hyperparameter tuning |
| Final holdout | 2025-09-30 | Scored **exactly once**, after the champion is chosen |

The **3-month gap** between the newest training cutoff and the test cutoff keeps the 90-day label windows from overlapping, which prevents leakage between adjacent folds.

| Test cutoff | # train cutoffs | Train rows | Test rows | Test churn |
|---|---:|---:|---:|---:|
| 2025-01-31 | 5 | 7,885 | 1,623 | 14.5% |
| 2025-02-28 | 6 | 9,495 | 1,620 | 14.3% |
| 2025-03-31 | 7 | 11,115 | 1,637 | 14.5% |
| 2025-04-30 | 8 | 12,738 | 1,644 | 15.7% |
| 2025-05-31 | 9 | 14,358 | 1,650 | 15.5% |
| 2025-06-30 | 10 | 15,995 | 1,646 | 16.4% |

### Modeling

- **Library fallback chain:** LightGBM → XGBoost → scikit-learn `HistGradientBoostingClassifier` (the first one installed is used; this run used `hist_gb`).
- **Tuner fallback chain:** Optuna (TPE) → Hyperopt (TPE) → seeded random search (this run used random search, 15 trials).
- **Objective:** mean ROC-AUC across the same time-based folds.
- **Class imbalance:** balanced class weights / sample weights.
- **Metrics:** ROC-AUC, PR-AUC, accuracy, precision, recall, F1 and Brier score, plus decile lift/gains.

---

## Results

### Model comparison (backtest)

| Model | Mean ROC-AUC | Pooled OOF ROC-AUC | Pooled OOF PR-AUC | Top-decile lift |
|---|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.846 ± 0.011 | 0.846 | 0.606 | 4.42x |
| **Tuned HistGradientBoosting** | **0.852 ± 0.012** | **0.852** | **0.616** | **4.51x** |

### Holdout (2025-09-30, single evaluation)

| ROC-AUC | PR-AUC | Brier | Top-decile lift |
|---:|---:|---:|---:|
| 0.872 | 0.708 | 0.144 | 4.56x |

### Top churn drivers (permutation importance)

1. `recency_days`: by far the strongest signal
2. `emails_open_rate_3m`
3. `app_logins_3m`
4. `spend_180d`
5. `frequency_180d`

### Business threshold

A threshold sweep with assumed economics (**$10** contact cost per customer, **$500** value per retained churner, **30%** retention success rate) gives a **profit-optimal threshold of 0.30**. At that threshold the model targets 840 customers and reaches 93% of actual churners.

### Segments (holdout)

| Membership tier | Customers | Actual churn |
|---|---:|---:|
| Bronze | 729 | 25.5% |
| Silver | 537 | 14.2% |
| Gold | 292 | 7.9% |

The Southwest region has the highest actual churn (26.0%); other regions fall between 14.9% and 17.2%.

### Calibration note

Because training uses balanced class weights, predicted probabilities run **higher than observed churn rates** in the low and middle bins. The model ranks customers well (high AUC and lift), but the raw scores should be read as relative risk, not literal probabilities. Adding a calibration step (e.g. `CalibratedClassifierCV` / isotonic regression) is a natural next improvement.

### Leakage checks

- The holdout **label** recomputed independently from the raw gold tables with Spark: **0 mismatches**
- Holdout **recency** recomputed the same way: **0 mismatches**
- Feature–label correlations sanity-checked

---

## Outputs

| Artifact | Description |
|---|---|
| `customerchurn.customerdb.churn_features` | 16 point-in-time monthly snapshots with leakage-free features + labels |
| MLflow experiment `churn_model_experiment` | Baseline and tuned-model parent runs, nested per-fold runs, tuning trials, ROC/PR/calibration plots, lift tables, threshold sweep, segment tables, feature importance |
| `customerchurn.customerdb.churn_model` | UC-registered champion pipeline (alias **`@champion`**) with signature and input example |
| `customerchurn.customerdb.churn_analysis` | Marketing-facing scored table (see schema below) |

### `churn_analysis` schema

| Column | Description |
|---|---|
| `Customer_Id` | Customer identifier |
| `snapshot_date` | Feature cutoff the customer was scored at |
| `churn_probability` | Model score for "zero Sold transactions in the next 90 days" |
| `risk_tier` | `High` ≥ 0.60, `Medium` ≥ 0.30, `Low` otherwise |
| `predicted_churn` | 1 when the score ≥ profit-optimal threshold (0.30) |
| `top_churn_drivers` | Top-3 per-customer churn drivers |
| `Membership_tier`, `Store_Region` | Profile fields |
| `model_name`, `model_version`, `model_run_id`, `scored_at` | Full model lineage |

The table and all its columns carry Unity Catalog comments.

---

## Tech stack

- **Platform:** Databricks (PySpark, Unity Catalog, Delta Lake)
- **ML:** scikit-learn (pipelines, `LogisticRegression`, `HistGradientBoostingClassifier`, permutation importance); optional LightGBM / XGBoost / Optuna / Hyperopt / SHAP
- **MLOps:** MLflow tracking, nested runs, artifacts, UC Model Registry with aliases, `mlflow.models.predict` validation in an isolated `uv` environment, MLflow Tracing
- **Analysis:** pandas, NumPy, matplotlib

---

## How to run

1. Import the notebook into a Databricks workspace with Unity Catalog enabled.
2. Make sure the four gold tables exist under `customerchurn.customerdb` (or edit the table names in the **Config** cell).
3. Set `EXPERIMENT` in the Config cell to your own MLflow experiment path (e.g. `/Users/<your-user>/churn_model_experiment`).
4. Attach to a cluster running Databricks ML Runtime (or install `mlflow`, `scikit-learn`, and optionally `lightgbm`/`xgboost`/`optuna`/`shap`).
5. Run all cells top to bottom. Each section ends with a printed `SUMMARY` line.

---

## Next steps

- Schedule Section 7 as a monthly Databricks Job to re-score the newest snapshot.
- Monitor drift in champion probabilities and feature distributions.
- Re-run the full notebook quarterly to refresh labels, re-tune and re-validate on a fresh holdout.
- Add probability calibration and SHAP-based per-customer explanations.
