# MMM Random Forest Model + Plots

**Repository**: MMM Random Forest Model + Plots

A reproducible notebook and supporting code that fits a two-stage Marketing Mix Modeling (MMM) pipeline using Random Forests. The pipeline: (1) predict `google_spend` from social channels (mediator stage), and (2) model log(revenue) using the mediator plus marketing levers and controls. The repo includes plotting and diagnostic code (feature importances, predicted vs actual, partial dependence plots).

---

## Contents

* `notebook.ipynb` — Reproducible Jupyter notebook that runs the full pipeline (data preparation, stage 1 and stage 2 modeling, plots, diagnostics).
* `README.md` — This file.
* `requirements.txt` — Python package requirements for reproducibility.
* `data/Assessment 2 - MMM Weekly.csv` — (Not included) Put your weekly CSV here.
* `scripts/` — Helper scripts (optional): data cleaning, model training wrappers, plotting utilities.

---

## Quick start

1. Clone the repo

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Create a Python environment and install requirements

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Place the dataset `Assessment 2 - MMM Weekly.csv` into the `data/` folder.

4. Run the notebook

```bash
jupyter lab   # or jupyter notebook, then open `notebook.ipynb`
```

Or run the core script(s) from `scripts/` (if provided):

```bash
python scripts/run_pipeline.py --data data/Assessment\ 2\ -\ MMM\ Weekly.csv
```

---

## Requirements

Minimum tested with Python 3.9+. Key libraries:

* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn

A `requirements.txt` with pinned versions is included to lock the environment.

---

## Notebook / Code overview

The notebook follows these sections (mirror of the provided script):

1. **Imports & global plotting settings** — numpy, pandas, sklearn, matplotlib, seaborn, PartialDependenceDisplay.

2. **Load dataset** — read CSV and coerce `week` to `datetime`.

3. **Time features** — derive `week_number`, `year`, `month`, and `quarter`. These help capture seasonal/trend effects if included in models or diagnostics.

4. **Lag features** — `revenue_lag1`, `revenue_lag2` to provide short-term autoregressive signal. Rows with NA due to lagging are dropped.

5. **Log-transform spend & revenue** — all spend channels and revenue are `log1p` transformed to stabilize variance and handle zeros.

6. **Categorical feature handling** — `promotions` encoded with one-hot dummies (drop-first to avoid multicollinearity).

7. **Other features** — scale `emails_send` and `sms_send` (into thousands), create `followers_growth` and scale it.

8. **Stage 1 (mediator)** — RandomForestRegressor trained to predict `google_spend_log` from social channels. Save prediction as `google_spend_pred`.

9. **Stage 2 (revenue)** — RandomForestRegressor trained to predict `revenue_log` using `google_spend_pred`, levers (emails, sms, price), controls (promotions dummies), and scaled features.

10. **Diagnostics & plots** — feature importances, predicted vs actual scatter, partial dependence plots for top features, and printed RMSE/R².

---

## Reproducibility & recommended improvements

### Data preparation (addressing your write-up point 1)

* **Weekly seasonality & trend**: derive `week_number` and optionally include `sin/cos` transforms of weekly cycle or include `month`/`quarter` dummies to model seasonal patterns. To isolate trend, add a rolling mean or explicit `trend = week_number` variable.
* **Zero-spend periods**: use `log1p()` to keep zero spends in the model. Consider a two-part model (classification for spend > 0 and regression conditional on spend) if zeros are frequent and structurally different.
* **Feature scaling & transforms**: continuous skewed variables use `log1p`; growth rates are clipped and standardized. Keep track of scalers so production transforms are identical.

### Modeling approach (write-up point 2)

* **Why Random Forest?** Nonlinear, robust to monotonic transformations, handles interactions, and provides interpretable feature importances and partial dependence plots. Good baseline for heterogeneous weekly data.
* **Hyperparameters**: the notebook uses conservative trees (`max_depth`, `min_samples_leaf`) to avoid overfitting. Example choices:

  * Stage 1: `n_estimators=200`, `max_depth=5`, `random_state=42` — simpler mediator model.
  * Stage 2: `n_estimators=300`, `max_depth=6`, `max_features='sqrt'`, `min_samples_split=5`, `min_samples_leaf=3` — stronger regularization.
* **Regularization / feature selection**: avoid very deep trees and use `min_samples_leaf`. Evaluate dropping low-importance features or use permutation importance to validate.
* **Validation plan**: time-aware split used in notebook (first 80% train, last 20% test). Prefer blocked/rolling cross-validation for time series (see Diagnostics).

### Causal framing & mediator assumption (point 3)

* The two-stage approach treats `google_spend` as **partially mediated** by social activity. Stage 1 predicts `google_spend` from social channels and stage 2 uses the predicted `google_spend_pred` as a mediator.
* **Assumptions**: no unobserved confounders that simultaneously affect social spend and revenue (back-door paths). If users/promotions or seasonality drive both social and revenue, include those controls (promotions dummies, seasonality features) to block confounding paths.
* **Leakage**: avoid including future-looking variables (e.g., revenue lags beyond what would be available at prediction time). Ensure predictions are created using only information available up to the prediction week.

### Diagnostics (point 4)

* **Out-of-sample performance**: report RMSE and R² on holdout. For time series, use blocked CV or expanding-window CV to avoid leakage.
* **Stability checks**: run rolling-window retraining and check changes in feature importances across windows. Also run model on subsets (e.g., high vs low promotions) to test stability.
* **Residual analysis**: plot residuals vs. fitted, residuals over time, ACF of residuals to detect autocorrelation. If residuals are autocorrelated, consider adding lag features or using time-series specific models.
* **Sensitivity checks**: vary average price and promotions and evaluate predicted revenue changes. Use Partial Dependence and SHAP (recommended) for local/feature-level explanations.

### Insights & recommendations (point 5)

* Use feature importances + partial dependence plots to identify which levers move `revenue_log` the most. Interpret with caution because Random Forests show associations — the two-stage causal framing helps isolate mediation by google spend.
* Watch out for **collinearity** (e.g., social spends highly correlated with each other) — the mediator stage helps but also consider decorrelation techniques (PCA) or hierarchical models.
* Operational recommendations:

  * Test media reallocation experiments (A/B or geo experiments) where possible to validate causal claims.
  * If price has a strong negative partial dependence, incorporate price elasticity analysis and simulate revenue under alternative pricing.

---

## How to extend

* Replace Random Forests with gradient boosting (XGBoost, LightGBM) for potential predictive gains.
* Add SHAP analysis for stronger local explanations.
* Build a proper cross-validation loop (TimeSeriesSplit or blocked CV) and hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV` with `cv=TimeSeriesSplit(...)`.
* Add a small experiment design plan (holdouts or geo experiments) to validate causal statements.

---

## Contact

If you want me to:

* convert this into a runnable Python script with CLI flags,
* add SHAP plots and save all figures to `outputs/`, or
* prepare a 1-page slide summarizing results — tell me which one and I will add it.

---

*Generated for the provided MMM Random Forest pipeline. Replace placeholders (paths, dataset) with your project-specific files.*
