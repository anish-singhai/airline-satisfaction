"""
XGBoost model for Airline Passenger Satisfaction, with hyperparameter tuning.
Mirrors the structure of ../1. RF/rf_model.py for an apples-to-apples comparison.

Pipeline:
1. Load train/test, impute missing values, one-hot encode categoricals
2. Stratified train/validation split
3. RandomizedSearchCV over XGBClassifier hyperparameters
   (searched on a stratified subsample for speed, scored on ROC-AUC via CV)
4. Refit best params on the full training split, evaluate on the held-out validation set
5. Feature importances
6. Refit on ALL training data, predict on test.csv, write Kaggle submission file

Run from inside this folder: `python3 xgb_model.py`
"""

import time
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, RocCurveDisplay,
)
from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42
TRAIN_PATH = "../Data/train.csv"
TEST_PATH = "../Data/test.csv"
TARGET = "satisfaction"
OUT_DIR = "model_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Rating columns where 0 is observed in the data but the scale is 1-5 (EDA showed
# these have a sentinel-like cluster of 0s, e.g. "Departure/Arrival time convenient"
# at 3.68%, while "Baggage handling" never has a 0) -- treat 0 as "not rated"
# rather than a genuine worst-score rating.
ZERO_AS_NA_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Checkin service", "Cleanliness",
]


def engineer_features(df):
    df = df.copy()
    for col in ZERO_AS_NA_COLS:
        df[f"was_rated_{col}"] = (df[col] != 0).astype(int)
        df[col] = df[col].replace(0, np.nan)
    df["Total Delay in Minutes"] = df["Departure Delay in Minutes"].add(
        df["Arrival Delay in Minutes"], fill_value=0
    )
    return df


# ----------------------------------------------------------------------
# 1. Load + prep
# ----------------------------------------------------------------------
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

train_raw[TARGET] = (train_raw[TARGET].astype(str).str.strip().str.lower() == "true").astype(int)

y = train_raw[TARGET]
X = engineer_features(train_raw.drop(columns=[TARGET, "id"]))
test_ids = test_raw["id"]
X_test_final = engineer_features(test_raw.drop(columns=["id"]))

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ]
)

# ----------------------------------------------------------------------
# 2. Train / validation split
# ----------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# ----------------------------------------------------------------------
# 3. Hyperparameter tuning (RandomizedSearchCV)
#    Tuned on a stratified subsample of the training split for speed;
#    final model is refit on the FULL training split afterwards.
# ----------------------------------------------------------------------
SEARCH_SAMPLE_SIZE = 100_000
if len(X_train) > SEARCH_SAMPLE_SIZE:
    X_search, _, y_search, _ = train_test_split(
        X_train, y_train,
        train_size=SEARCH_SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
else:
    X_search, y_search = X_train, y_train

param_distributions = {
    "clf__n_estimators": [200, 300, 400, 500, 600],
    "clf__max_depth": [3, 4, 5, 6, 8, 10],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [1, 3, 5, 7],
    "clf__gamma": [0, 0.1, 0.3, 0.5],
    "clf__reg_alpha": [0, 0.01, 0.1, 1],
    "clf__reg_lambda": [0.5, 1, 1.5, 2],
}

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )),
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="roc_auc",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)

print("\n" + "=" * 70)
print(f"RANDOMIZED SEARCH (tuning on {len(X_search)}-row subsample, 3-fold CV, 25 iters)")
print("=" * 70)
t0 = time.time()
search.fit(X_search, y_search)
print(f"Search done in {time.time() - t0:.1f}s")

print("\nBest params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV ROC-AUC (on search subsample): {search.best_score_:.4f}")

cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
cv_results.to_csv(f"{OUT_DIR}/cv_results.csv", index=False)

# ----------------------------------------------------------------------
# 4. Refit best params on FULL training split, evaluate on validation set
# ----------------------------------------------------------------------
best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
final_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        **best_params,
    )),
])

print("\nRefitting best params on full training split...")
t0 = time.time()
final_pipe.fit(X_train, y_train)
print(f"Refit done in {time.time() - t0:.1f}s")

val_proba = final_pipe.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)

print("\n" + "=" * 70)
print("VALIDATION METRICS")
print("=" * 70)
print(f"Accuracy : {accuracy_score(y_val, val_pred):.4f}")
print(f"Precision: {precision_score(y_val, val_pred):.4f}")
print(f"Recall   : {recall_score(y_val, val_pred):.4f}")
print(f"F1       : {f1_score(y_val, val_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, val_proba):.4f}")
print("\nClassification report:")
print(classification_report(y_val, val_pred, target_names=["Not satisfied", "Satisfied"]))

cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not satisfied", "Satisfied"],
            yticklabels=["Not satisfied", "Satisfied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (validation)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

RocCurveDisplay.from_predictions(y_val, val_proba)
plt.title("ROC Curve (validation)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5. Feature importances (SHAP) -- mean |SHAP value| over a sample of the
#    training data. More rigorous than XGBoost's built-in gain-based
#    importances, and the same TreeExplainer is reused at inference time
#    for per-prediction explanations (see 3. Deployment/app.py).
# ----------------------------------------------------------------------
feature_names = final_pipe.named_steps["prep"].get_feature_names_out()

SHAP_SAMPLE_SIZE = 2000
X_shap_sample = X_train.sample(n=min(SHAP_SAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE)
X_shap_sample_transformed = final_pipe.named_steps["prep"].transform(X_shap_sample)
explainer = shap.TreeExplainer(final_pipe.named_steps["clf"])
shap_values_sample = explainer.shap_values(X_shap_sample_transformed)
mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

print("\nTop 20 feature importances (mean |SHAP value|):")
print(shap_importance.head(20))

plt.figure(figsize=(8, 8))
sns.barplot(x=shap_importance.head(20).values, y=shap_importance.head(20).index, color="#DD8452")
plt.title("Top 20 Feature Importances (mean |SHAP value|)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importances.png", dpi=150)
plt.close()

# Export metrics + feature lists for the deployed app's "Model Info" tab
eval_metrics = {
    "validation_metrics": {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "precision": float(precision_score(y_val, val_pred)),
        "recall": float(recall_score(y_val, val_pred)),
        "f1": float(f1_score(y_val, val_pred)),
        "roc_auc": float(roc_auc_score(y_val, val_proba)),
    },
    "best_hyperparameters": best_params,
    "top_shap_importances": [
        {"feature": name, "mean_abs_shap": float(score)}
        for name, score in shap_importance.head(20).items()
    ],
    "raw_input_features": sorted(
        set(train_raw.columns) - {TARGET, "id"}
    ),
    "engineered_features": numeric_cols + categorical_cols,
}
with open(f"{OUT_DIR}/eval_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)
print(f"Eval metrics + feature lists saved to {OUT_DIR}/eval_metrics.json")

# ----------------------------------------------------------------------
# 5.5 Export serving artifacts for deployment/monitoring
#     `final_pipe` at this point is fit ONLY on X_train (80%) -- X_val (20%)
#     has never been trained or tuned on, so it's a genuine holdout we can
#     replay as "incoming production traffic" to honestly track live
#     performance/drift. This model is intentionally distinct from the
#     all-data refit below, which is only for the Kaggle submission.
# ----------------------------------------------------------------------
dump(final_pipe, f"{OUT_DIR}/xgboost_pipeline_serving.joblib")

# Save the RAW (pre-engineering) columns for the holdout rows -- this is what a
# real incoming request looks like; the serving API applies engineer_features()
# itself, so the replay client should send raw data, not already-engineered data.
raw_val = train_raw.drop(columns=[TARGET, "id"]).loc[X_val.index].copy()
raw_val[TARGET] = y_val.values
raw_val.to_csv(f"{OUT_DIR}/simulation_holdout.csv", index=False)
print(f"\nServing model saved to {OUT_DIR}/xgboost_pipeline_serving.joblib")
print(f"Simulation holdout ({len(raw_val)} rows) saved to {OUT_DIR}/simulation_holdout.csv")

# Numeric columns with few distinct values (rating scales 0-5, binary was_rated_* flags)
# are drift-checked like categoricals (frequency comparison) -- quantile-based PSI bins
# degenerate on low-cardinality data and produce misleading spikes. Columns that ARE
# high-cardinality but still zero-inflated (delay minutes: ~87% exactly 0) also produce
# degenerate/duplicate quantile edges, so those fall back to a coarse zero-vs-nonzero
# frequency split instead. Only genuinely continuous, non-degenerate columns (Age,
# Flight Distance) get real PSI quantile-bin treatment.
DISCRETE_NUMERIC_THRESHOLD = 10


def canon_key(x):
    """Canonical string form for a numeric value, independent of int/float dtype
    (pandas upcasts a whole column to float64 only when it actually contains a NaN,
    which differs between the full training set and a single live request -- without
    this, '4' vs '4.0' would silently fail to match and look like fake drift)."""
    f = float(x)
    return str(int(f)) if f.is_integer() else str(f)


def value_freq_dict(s):
    freq = {}
    for k, v in s.value_counts(normalize=True).to_dict().items():
        key = canon_key(k)
        freq[key] = freq.get(key, 0) + v
    return freq


def classify_numeric(s):
    if s.nunique() <= DISCRETE_NUMERIC_THRESHOLD:
        return {"type": "discrete", "binning": "exact", "value_freq": value_freq_dict(s)}

    p10, p50, p90 = s.quantile(0.10), s.quantile(0.50), s.quantile(0.90)
    if len({p10, p50, p90}) < 3:
        prop_zero = float((s == 0).mean())
        value_freq = {"0": prop_zero, "nonzero": 1 - prop_zero}
        return {"type": "discrete", "binning": "zero_vs_nonzero", "value_freq": value_freq}

    return {
        "type": "continuous",
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
    }


reference_stats = {"numeric": {}, "categorical": {}}
for col in numeric_cols:
    s = X_train[col].dropna()
    reference_stats["numeric"][col] = classify_numeric(s)
for col in categorical_cols:
    freq = X_train[col].value_counts(normalize=True)
    reference_stats["categorical"][col] = freq.to_dict()

with open(f"{OUT_DIR}/reference_stats.json", "w") as f:
    json.dump(reference_stats, f, indent=2)
print(f"Reference distribution stats saved to {OUT_DIR}/reference_stats.json")

# ----------------------------------------------------------------------
# 6. Refit on ALL training data, predict on test set, write submission
# ----------------------------------------------------------------------
print("\nRefitting best model on ALL training data for final predictions...")
t0 = time.time()
final_pipe.fit(X, y)
print(f"Full-data refit done in {time.time() - t0:.1f}s")

dump(final_pipe, f"{OUT_DIR}/xgboost_pipeline.joblib")

test_proba = final_pipe.predict_proba(X_test_final)[:, 1]
submission = pd.DataFrame({"id": test_ids, "satisfaction": test_proba})
submission.to_csv(f"{OUT_DIR}/submission.csv", index=False)
print(f"\nSubmission saved to {OUT_DIR}/submission.csv")
print(submission.head())

print(f"\nAll outputs saved to ./{OUT_DIR}/")
