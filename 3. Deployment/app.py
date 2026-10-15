"""
Airline Passenger Satisfaction -- serving API + monitoring dashboard.

- POST /predict  : score a passenger, logs the request (features, prediction, latency)
- POST /feedback : attach a delayed ground-truth label to a previous prediction
- GET  /health   : liveness check
- /dashboard     : Gradio UI (operational metrics, rolling accuracy/ROC-AUC, feature drift)

The served model (`xgboost_pipeline_serving.joblib`) was trained ONLY on the 80% training
split -- it has never seen the 20% holdout used by replay_client.py to simulate traffic,
so performance/drift measured here are honest, not memorized.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "xgboost_pipeline_serving.joblib"
REFERENCE_STATS_PATH = BASE_DIR / "reference_stats.json"
DB_PATH = BASE_DIR / "logs.db"

# ----------------------------------------------------------------------
# Feature engineering (must exactly match 2. XGBoost/xgb_model.py)
# ----------------------------------------------------------------------
ZERO_AS_NA_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Checkin service", "Cleanliness",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_AS_NA_COLS:
        df[f"was_rated_{col}"] = (df[col] != 0).astype(int)
        df[col] = df[col].replace(0, np.nan)
    df["Departure Delay in Minutes"] = pd.to_numeric(df["Departure Delay in Minutes"], errors="coerce")
    df["Arrival Delay in Minutes"] = pd.to_numeric(df["Arrival Delay in Minutes"], errors="coerce")
    df["Total Delay in Minutes"] = df["Departure Delay in Minutes"].add(
        df["Arrival Delay in Minutes"], fill_value=0
    )
    return df


# ----------------------------------------------------------------------
# Request / response schemas
# ----------------------------------------------------------------------
class PassengerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(alias="Age")
    flight_distance: int = Field(alias="Flight Distance")
    inflight_wifi_service: int = Field(alias="Inflight wifi service")
    departure_arrival_time_convenient: int = Field(alias="Departure/Arrival time convenient")
    ease_of_online_booking: int = Field(alias="Ease of Online booking")
    gate_location: int = Field(alias="Gate location")
    food_and_drink: int = Field(alias="Food and drink")
    online_boarding: int = Field(alias="Online boarding")
    seat_comfort: int = Field(alias="Seat comfort")
    inflight_entertainment: int = Field(alias="Inflight entertainment")
    on_board_service: int = Field(alias="On-board service")
    leg_room_service: int = Field(alias="Leg room service")
    baggage_handling: int = Field(alias="Baggage handling")
    checkin_service: int = Field(alias="Checkin service")
    cleanliness: int = Field(alias="Cleanliness")
    departure_delay_in_minutes: float = Field(alias="Departure Delay in Minutes")
    arrival_delay_in_minutes: Optional[float] = Field(default=None, alias="Arrival Delay in Minutes")
    gender: str = Field(alias="Gender")
    customer_type: str = Field(alias="Customer Type")
    type_of_travel: str = Field(alias="Type of Travel")
    class_: str = Field(alias="Class")


class PredictResponse(BaseModel):
    request_id: str
    satisfaction_probability: float
    prediction: int


class FeedbackRequest(BaseModel):
    request_id: str
    true_label: int


# ----------------------------------------------------------------------
# SQLite logging
# ----------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            request_id TEXT PRIMARY KEY,
            created_at REAL,
            latency_ms REAL,
            features_json TEXT,
            probability REAL,
            prediction INTEGER,
            true_label INTEGER,
            feedback_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL,
            endpoint TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(request_id, features, probability, prediction, latency_ms):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (request_id, created_at, latency_ms, features_json, probability, prediction) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (request_id, time.time(), latency_ms, json.dumps(features), probability, prediction),
    )
    conn.commit()
    conn.close()


def log_feedback(request_id, true_label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE predictions SET true_label = ?, feedback_at = ? WHERE request_id = ?",
        (true_label, time.time(), request_id),
    )
    conn.commit()
    found = cur.rowcount > 0
    conn.close()
    if not found:
        raise ValueError(f"Unknown request_id: {request_id}")


def log_error(endpoint, message):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO errors (created_at, endpoint, message) VALUES (?, ?, ?)",
        (time.time(), endpoint, message),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Load model + reference stats, init DB
# ----------------------------------------------------------------------
pipeline = joblib.load(MODEL_PATH)
with open(REFERENCE_STATS_PATH) as f:
    REFERENCE_STATS = json.load(f)
init_db()

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Airline Satisfaction Serving API")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PassengerFeatures):
    t0 = time.time()
    raw = payload.model_dump(by_alias=True)
    try:
        df = pd.DataFrame([raw])
        df_eng = engineer_features(df)
        proba = float(pipeline.predict_proba(df_eng)[:, 1][0])
        pred = int(proba >= 0.5)
        latency_ms = (time.time() - t0) * 1000
        request_id = str(uuid.uuid4())
        # Log the ENGINEERED features (not raw) -- reference_stats.json was computed on
        # engineered X_train, so drift comparisons need to be apples-to-apples.
        engineered_record = df_eng.iloc[0].where(pd.notnull(df_eng.iloc[0]), None).to_dict()
        log_prediction(request_id, engineered_record, proba, pred, latency_ms)
        return PredictResponse(request_id=request_id, satisfaction_probability=proba, prediction=pred)
    except Exception as e:
        log_error("/predict", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    try:
        log_feedback(payload.request_id, payload.true_label)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_error("/feedback", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Monitoring dashboard (Gradio, mounted at /dashboard)
# ----------------------------------------------------------------------
def get_operational_metrics():
    conn = sqlite3.connect(DB_PATH)
    preds = pd.read_sql("SELECT * FROM predictions", conn)
    errors = pd.read_sql("SELECT * FROM errors", conn)
    conn.close()

    total = len(preds)
    total_errors = len(errors)
    error_rate = round(100 * total_errors / max(total + total_errors, 1), 2)
    p50 = round(preds["latency_ms"].quantile(0.5), 1) if total else 0.0
    p95 = round(preds["latency_ms"].quantile(0.95), 1) if total else 0.0

    if total:
        preds["minute"] = pd.to_datetime(preds["created_at"], unit="s").dt.floor("min")
        counts = preds.groupby("minute").size().reset_index(name="requests")
        fig = px.bar(counts, x="minute", y="requests", title="Requests per minute")
    else:
        fig = px.bar(title="No requests yet")

    return {"total": total, "error_rate": error_rate, "p50": p50, "p95": p95, "requests_fig": fig}


def get_performance_metrics(batch_size=20):
    conn = sqlite3.connect(DB_PATH)
    preds = pd.read_sql(
        "SELECT * FROM predictions WHERE true_label IS NOT NULL ORDER BY feedback_at", conn
    )
    conn.close()

    if len(preds) < 2:
        empty = px.line(title="Not enough feedback yet")
        return empty, empty

    preds = preds.reset_index(drop=True)
    preds["batch"] = preds.index // batch_size
    rows = []
    for batch, g in preds.groupby("batch"):
        acc = accuracy_score(g["true_label"], (g["probability"] >= 0.5).astype(int))
        auc = roc_auc_score(g["true_label"], g["probability"]) if g["true_label"].nunique() > 1 else None
        rows.append({"batch": batch, "accuracy": acc, "roc_auc": auc})
    metrics_df = pd.DataFrame(rows)

    acc_fig = px.line(metrics_df, x="batch", y="accuracy", markers=True, title="Rolling accuracy")
    acc_fig.update_yaxes(range=[0, 1])
    auc_fig = px.line(metrics_df, x="batch", y="roc_auc", markers=True, title="Rolling ROC-AUC")
    auc_fig.update_yaxes(range=[0, 1])
    return acc_fig, auc_fig


def _compute_psi(ref_props, current_values, edges):
    bin_edges = [-np.inf] + edges + [np.inf]
    current_counts, _ = np.histogram(current_values, bins=bin_edges)
    current_props = current_counts / max(current_counts.sum(), 1)
    psi = 0.0
    for ref_p, cur_p in zip(ref_props, current_props):
        ref_p = max(ref_p, 1e-6)
        cur_p = max(cur_p, 1e-6)
        psi += (cur_p - ref_p) * np.log(cur_p / ref_p)
    return psi


def get_drift_metrics(window=200):
    conn = sqlite3.connect(DB_PATH)
    preds = pd.read_sql(
        "SELECT features_json, created_at FROM predictions ORDER BY created_at DESC LIMIT ?",
        conn, params=(window,),
    )
    conn.close()

    if preds.empty:
        empty = px.bar(title="No data yet")
        return empty, empty

    feats = pd.DataFrame([json.loads(x) for x in preds["features_json"]])

    # PSI only for genuinely continuous numeric features (quantile-bin PSI degenerates
    # on low-cardinality data, see DISCRETE_NUMERIC_THRESHOLD in xgb_model.py).
    psi_rows = []
    for col, stats in REFERENCE_STATS["numeric"].items():
        if stats.get("type") != "continuous" or col not in feats.columns:
            continue
        vals = pd.to_numeric(feats[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        edges = [stats["p10"], stats["p50"], stats["p90"]]
        psi = _compute_psi([0.10, 0.40, 0.40, 0.10], vals.values, edges)
        psi_rows.append({"feature": col, "psi": psi})
    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False) if psi_rows else pd.DataFrame(
        columns=["feature", "psi"])
    psi_fig = px.bar(psi_df, x="feature", y="psi", title="Continuous feature drift (PSI)")
    psi_fig.add_hline(y=0.2, line_dash="dash", annotation_text="moderate (0.2)")
    psi_fig.add_hline(y=0.25, line_dash="dot", annotation_text="significant (0.25)")
    psi_fig.update_xaxes(tickangle=45)

    # L1 (total variation) distance for categoricals AND low-cardinality numeric
    # features (rating scales, was_rated_* binary flags) -- both are small, fixed
    # value sets, so a direct frequency comparison is more honest than PSI bins.
    def l1_distance(ref_freq, cur_freq):
        all_keys = set(ref_freq) | set(cur_freq)
        return sum(abs(ref_freq.get(k, 0) - cur_freq.get(k, 0)) for k in all_keys) / 2

    def canon_key(x):
        # Must match canon_key() in xgb_model.py exactly -- a column stays int64 in a
        # single live request unless that request happens to contain a 0, but the bulk
        # training column upcasts to float64 as soon as ANY row has a 0. Without this,
        # "4" vs "4.0" silently fail to match and look like fake drift.
        f = float(x)
        return str(int(f)) if f.is_integer() else str(f)

    def value_freq_dict(series):
        freq = {}
        for k, v in series.value_counts(normalize=True).to_dict().items():
            key = canon_key(k)
            freq[key] = freq.get(key, 0) + v
        return freq

    cat_rows = []
    for col, ref_freq in REFERENCE_STATS["categorical"].items():
        if col not in feats.columns:
            continue
        cur_freq = feats[col].value_counts(normalize=True).to_dict()
        cat_rows.append({"feature": col, "l1_distance": l1_distance(ref_freq, cur_freq)})
    for col, stats in REFERENCE_STATS["numeric"].items():
        if stats.get("type") != "discrete" or col not in feats.columns:
            continue
        if stats.get("binning") == "zero_vs_nonzero":
            vals = pd.to_numeric(feats[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            prop_zero = float((vals == 0).mean())
            cur_freq = {"0": prop_zero, "nonzero": 1 - prop_zero}
        else:
            cur_freq = value_freq_dict(feats[col])
        cat_rows.append({"feature": col, "l1_distance": l1_distance(stats["value_freq"], cur_freq)})
    cat_df = pd.DataFrame(cat_rows).sort_values("l1_distance", ascending=False) if cat_rows else pd.DataFrame(
        columns=["feature", "l1_distance"])
    cat_fig = px.bar(cat_df, x="feature", y="l1_distance", title="Discrete feature drift (L1 distance)")
    cat_fig.update_xaxes(tickangle=45)

    return psi_fig, cat_fig


def refresh_all():
    op = get_operational_metrics()
    acc_fig, auc_fig = get_performance_metrics()
    psi_fig, cat_fig = get_drift_metrics()
    return (
        op["total"], op["error_rate"], op["p50"], op["p95"],
        op["requests_fig"], acc_fig, auc_fig, psi_fig, cat_fig,
    )


def build_dashboard():
    with gr.Blocks(title="Airline Satisfaction - Model Monitoring") as demo:
        gr.Markdown("# Airline Satisfaction -- Model Monitoring Dashboard")
        gr.Markdown(
            "Served model was trained only on the 80% training split; everything here reflects "
            "the never-trained-on 20% holdout replayed as simulated production traffic."
        )
        refresh_btn = gr.Button("Refresh")
        with gr.Row():
            total_box = gr.Number(label="Total requests", interactive=False)
            error_box = gr.Number(label="Error rate (%)", interactive=False)
            p50_box = gr.Number(label="p50 latency (ms)", interactive=False)
            p95_box = gr.Number(label="p95 latency (ms)", interactive=False)
        requests_plot = gr.Plot(label="Requests per minute")
        with gr.Row():
            acc_plot = gr.Plot(label="Rolling accuracy")
            auc_plot = gr.Plot(label="Rolling ROC-AUC")
        psi_plot = gr.Plot(label="Numeric feature drift (PSI)")
        cat_plot = gr.Plot(label="Categorical feature drift")

        outputs = [total_box, error_box, p50_box, p95_box,
                   requests_plot, acc_plot, auc_plot, psi_plot, cat_plot]
        refresh_btn.click(refresh_all, outputs=outputs)
        demo.load(refresh_all, outputs=outputs)
    return demo


app = gr.mount_gradio_app(app, build_dashboard(), path="/dashboard")
