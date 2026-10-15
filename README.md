# Airline Passenger Satisfaction — End-to-End ML Project

Predicting whether an airline passenger is satisfied with their flight, from raw data through a
tuned model to a live, monitored production API — not just a notebook.

**Live demo:** https://airline-satisfaction-monitor.onrender.com — try the model yourself, or jump
straight to the [monitoring dashboard](https://airline-satisfaction-monitor.onrender.com/dashboard)
*(free-tier hosting — the first request after 15 min of inactivity takes ~50s to wake up)*

**Dataset:** [Kaggle: Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
— ~700k rows, 21 raw features (service ratings, flight details, demographics), binary target
(`satisfaction`: satisfied / not satisfied).

---

## 1. Exploratory Data Analysis

Full analysis covered missing values, feature distributions, correlations, and each feature's
relationship to the target. Key findings that shaped everything downstream:

- **Missing data is negligible**: only `Arrival Delay in Minutes` has nulls (0.04% of rows).
- **A hidden data-quality issue**: the 1–5 rating columns (`Inflight wifi service`,
  `Online boarding`, etc.) contain a cluster of literal `0` values — but `Baggage handling` *never*
  has a `0`. That asymmetry means `0` isn't a genuine "worst" rating, it's a sentinel for
  "not applicable / not rated." Treating it as a real score would bias the model. → led to the
  `was_rated_*` feature engineering below.
- **Target is mildly imbalanced**: 55.6% not satisfied vs. 44.4% satisfied.
- **Strongest predictors** (by correlation with target): `Online boarding` (0.56),
  `Inflight entertainment` (0.43), `Seat comfort` (0.40), `Flight Distance` (0.37).
- **Weak/noisy predictors**: `Gate location` (0.02), flight delays (-0.04 to -0.05) — delays barely
  matter on their own.
- **Redundant feature clusters** worth knowing about (not removed — tree models handle correlated
  features fine, but relevant if you ever simplify the feature set): `Seat comfort` /
  `Cleanliness` / `Inflight entertainment` / `Food and drink` correlate 0.65–0.76 with each other;
  `Inflight wifi service` / `Ease of Online booking` correlate 0.77.
- **Categorical features matter a lot**: `Type of Travel` (Business 58% satisfied vs. Personal 10%)
  and `Class` (Business 73% vs. Eco 17%) are both strong standalone signals.

## 2. Feature Engineering

Applied identically at training time and at inference time (same `engineer_features()` logic in
both the training scripts and the serving API, so there's no train/serve skew):

- **`was_rated_<col>` flags**: for every rating column that contains the `0`-as-"not rated"
  sentinel, add a binary flag for whether it was rated at all, then replace `0` → `NaN` so the
  model doesn't treat "not rated" as "rated worst."
- **`Total Delay in Minutes`**: sum of departure + arrival delay.

## 3. Modeling — Random Forest vs. XGBoost

Both tuned via `RandomizedSearchCV` (25 candidates, 3-fold CV, scored on ROC-AUC), evaluated on an
identical 20% held-out validation split:

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 92.33% | **92.46%** |
| Precision | 92.60% | **92.64%** |
| Recall | 89.89% | **90.15%** |
| F1 | 91.22% | **91.38%** |
| ROC-AUC | 95.61% | **95.75%** |
| Model file size | 118 MB | **1.5 MB** |
| Full tuning + training time | ~3 min | **~15s** |

**XGBoost won on every metric *and* is ~80x smaller and ~12x faster to train** — the clear choice
for production. Both models agree on the top drivers: `Online boarding`, `Class`,
`Type of Travel`, `Inflight wifi service`.

See [`1. RF/`](./1.%20RF) and [`2. XGBoost/`](./2.%20XGBoost) for the full training scripts,
hyperparameter search results, confusion matrices, ROC curves, and feature importance plots.

### A methodology detail that matters for what comes next

The model trained on **100% of the data** (for the best possible Kaggle submission) is *not* the
same model that's deployed for serving. The serving model is trained on only the **80% training
split**, so the remaining **20% (139,927 rows) is a genuine holdout it has never seen** — this is
what makes the production-monitoring setup below honest rather than circular (replaying training
data back through a model that already memorized it would show fake, inflated accuracy).

## 4. Deployment — FastAPI + Monitoring Dashboard

The serving model is wrapped in a FastAPI app ([`3. Deployment/app.py`](./3.%20Deployment/app.py))
with:

- `POST /predict` — score a passenger, returns a probability + prediction + a
  [SHAP](https://github.com/shap/shap) explanation (which features pushed this specific
  prediction toward/away from "satisfied", via `TreeExplainer` — exact and ~2ms per request for
  tree models, no approximation needed)
- `POST /feedback` — attach a (delayed) true label to a past prediction, simulating how ground
  truth actually arrives in production (e.g. a satisfaction survey days later)
- `/demo` — a Gradio UI with model info (validation metrics, SHAP-based global feature
  importances, feature lists) and an interactive "enter your own values" predictor that shows
  both the per-prediction SHAP explanation and a plain-English summary generated by a free LLM
  ([Groq](https://groq.com), `llama-3.1-8b-instant`) — falls back to a templated sentence if no
  API key is configured, so the demo never breaks
- `/dashboard` — a Gradio UI (mounted on the same FastAPI app) tracking:
  - **Operational metrics**: requests/minute, p50/p95 latency, error rate
  - **Performance metrics**: rolling accuracy and ROC-AUC, computed as feedback arrives
  - **Drift metrics**: how far live traffic's feature distributions have moved from training

  Manual tries from `/demo` are logged separately and excluded from every metric above — there's
  no real outcome to compare a hypothetical input against, so counting them as "traffic" would
  distort what the dashboard is meant to show.

[`replay_client.py`](./3.%20Deployment/replay_client.py) simulates production traffic by replaying
the held-out 20% against the live API with realistic delays, and can deliberately inject drift
into a couple of features (`--inject-drift-after-batch`) to demonstrate the drift panel catching it.

### Two drift metrics, not one

Population Stability Index (PSI) is the standard drift metric, but it only works on genuinely
continuous features — it breaks down (produces misleadingly huge values) on low-cardinality data
like 1–5 rating scales or binary flags, because its quantile bins degenerate. Found this the hard
way while building the monitor: `was_rated_*` flags were showing PSI values of 13+ despite being
completely untouched. Fixed by classifying each feature at training time as continuous (gets real
PSI) or discrete (rating scales, binary flags, zero-inflated delay columns — gets a direct
frequency/L1-distance comparison instead, same treatment as categoricals).

### Where it's hosted, and why

Originally built to deploy on Hugging Face Spaces, but HF now requires a PRO subscription to host
any Docker or Gradio Space, even on free compute. Re-researched current free-tier alternatives and
deployed on
**[Render](https://render.com)** instead — genuinely free, no credit card required, deploys
straight from this repo's `Dockerfile`.

*(Free tier caveat: the instance spins down after 15 minutes idle and its local request log
resets on restart — fine for a demo, would move to persistent/external logging for anything
longer-lived.)*

## Repo structure

```
1. RF/            Random Forest training script, EDA outputs, model evaluation plots
2. XGBoost/        XGBoost training script, model evaluation plots, tuned pipeline
3. Deployment/     FastAPI serving API + Gradio monitoring dashboard, Dockerfile, traffic replay client
```

`Data/` (raw Kaggle CSVs) and the full-size Random Forest model (118MB, exceeds GitHub's 100MB
file limit) are excluded — download the dataset from Kaggle and re-run `1. RF/rf_model.py` /
`2. XGBoost/xgb_model.py` to regenerate everything.

## Stack

Python, pandas, scikit-learn, XGBoost, FastAPI, Gradio, Plotly, Docker, Render.
