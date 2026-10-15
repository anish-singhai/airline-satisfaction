"""
Random Forest model for Airline Passenger Satisfaction, with hyperparameter tuning.

Pipeline:
1. Load train/test, impute missing values, one-hot encode categoricals
2. Stratified train/validation split
3. RandomizedSearchCV over RandomForestClassifier hyperparameters
   (searched on a stratified subsample for speed, scored on ROC-AUC via CV)
4. Refit best params on the full training split, evaluate on the held-out validation set
5. Feature importances
6. Refit on ALL training data, predict on test.csv, write Kaggle submission file
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, RocCurveDisplay,
)

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
    "clf__max_depth": [None, 10, 15, 20, 25, 30],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 4, 8],
    "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
    "clf__bootstrap": [True, False],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
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
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)),
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
# 5. Feature importances
# ----------------------------------------------------------------------
feature_names = final_pipe.named_steps["prep"].get_feature_names_out()
importances = final_pipe.named_steps["clf"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\nTop 20 feature importances:")
print(fi.head(20))

plt.figure(figsize=(8, 8))
sns.barplot(x=fi.head(20).values, y=fi.head(20).index, color="#4C72B0")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importances.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. Refit on ALL training data, predict on test set, write submission
# ----------------------------------------------------------------------
print("\nRefitting best model on ALL training data for final predictions...")
t0 = time.time()
final_pipe.fit(X, y)
print(f"Full-data refit done in {time.time() - t0:.1f}s")

dump(final_pipe, f"{OUT_DIR}/random_forest_pipeline.joblib")

test_proba = final_pipe.predict_proba(X_test_final)[:, 1]
submission = pd.DataFrame({"id": test_ids, "satisfaction": test_proba})
submission.to_csv(f"{OUT_DIR}/submission.csv", index=False)
print(f"\nSubmission saved to {OUT_DIR}/submission.csv")
print(submission.head())

print(f"\nAll outputs saved to ./{OUT_DIR}/")
