"""
Airline Passenger Satisfaction -- serving API + monitoring dashboard + showcase demo.

- POST /predict  : score a passenger, logs the request (features, prediction, latency)
- POST /feedback : attach a delayed ground-truth label to a previous prediction
- GET  /health   : liveness check
- GET  /         : redirects to /demo
- /demo          : Gradio UI (model info + interactive "try it yourself" predictor)
- /dashboard     : Gradio UI (operational metrics, rolling accuracy/ROC-AUC, feature drift)

The served model (`xgboost_pipeline_serving.joblib`) was trained ONLY on the 80% training
split -- it has never seen the 20% holdout used by replay_client.py to simulate traffic,
so performance/drift measured here are honest, not memorized. Manual predictions made via
/demo are logged with source='manual' and excluded from every /dashboard metric, since
there's no real outcome to compare them against.
"""

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "xgboost_pipeline_serving.joblib"
REFERENCE_STATS_PATH = BASE_DIR / "reference_stats.json"
EVAL_METRICS_PATH = BASE_DIR / "eval_metrics.json"
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


class ShapContribution(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    request_id: str
    satisfaction_probability: float
    prediction: int
    shap_contributions: list[ShapContribution]


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
            feedback_at REAL,
            source TEXT NOT NULL DEFAULT 'api'
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


def log_prediction(request_id, features, probability, prediction, latency_ms, source="api"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions "
        "(request_id, created_at, latency_ms, features_json, probability, prediction, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (request_id, time.time(), latency_ms, json.dumps(features), probability, prediction, source),
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
with open(EVAL_METRICS_PATH) as f:
    EVAL_METRICS = json.load(f)
init_db()

# SHAP explainer (TreeExplainer works directly against the XGBoost booster inside
# the pipeline; no background dataset needed -- it's exact for tree models, and
# ~2ms per prediction, so it's cheap enough to run on every /predict call too).
SHAP_EXPLAINER = shap.TreeExplainer(pipeline.named_steps["clf"])
TRANSFORMED_FEATURE_NAMES = pipeline.named_steps["prep"].get_feature_names_out()

# Groq (free tier, no credit card) powers the plain-English summary on the manual
# "Try it Yourself" tab. If GROQ_API_KEY isn't set (e.g. local dev without it),
# generate_llm_summary() falls back to a templated sentence -- the demo never
# breaks just because the key is missing or the API call fails.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Airline Satisfaction Serving API")


@app.get("/")
def root():
    # Mounting a Gradio app directly at "/" breaks Starlette's auto-redirect for
    # other mounted sub-paths (e.g. "/dashboard" starts 404ing) -- a known quirk of
    # mounting at the literal root. So the showcase app lives at /demo instead, and
    # this just forwards the bare URL there for convenience.
    return RedirectResponse(url="/demo")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


def score_passenger(raw: dict):
    """Run one raw passenger record through feature engineering + the model.
    Shared by /predict and the in-process 'Try it Yourself' demo tab, so there's
    exactly one place that knows how to turn raw input into a prediction."""
    df = pd.DataFrame([raw])
    df_eng = engineer_features(df)
    proba = float(pipeline.predict_proba(df_eng)[:, 1][0])
    pred = int(proba >= 0.5)
    # Engineered (not raw) features are what reference_stats.json was computed on,
    # so drift comparisons need to be apples-to-apples.
    engineered_record = df_eng.iloc[0].where(pd.notnull(df_eng.iloc[0]), None).to_dict()
    return proba, pred, engineered_record, df_eng


def compute_shap_contributions(df_eng, top_n=15):
    """Per-prediction SHAP values for one engineered row, sorted by |impact|.
    Positive = pushes toward 'satisfied', negative = pushes toward 'not satisfied'."""
    X_transformed = pipeline.named_steps["prep"].transform(df_eng)
    raw_shap = SHAP_EXPLAINER.shap_values(X_transformed)[0]
    sv = pd.Series(raw_shap, index=TRANSFORMED_FEATURE_NAMES).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    return [{"feature": f, "shap_value": float(v)} for f, v in sv.head(top_n).items()]


def build_shap_figure(shap_contributions):
    df = pd.DataFrame(shap_contributions)
    df["feature"] = df["feature"].str.split("__", n=1).str[-1]  # strip num__/cat__ prefix
    df = df.sort_values("shap_value")
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in df["shap_value"]]
    fig = go.Figure(go.Bar(x=df["shap_value"], y=df["feature"], orientation="h", marker_color=colors))
    fig.update_layout(
        title="What influenced this prediction (SHAP values)",
        xaxis_title="SHAP value (positive pushes toward 'satisfied')",
    )
    return fig


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PassengerFeatures):
    t0 = time.time()
    raw = payload.model_dump(by_alias=True)
    try:
        proba, pred, engineered_record, df_eng = score_passenger(raw)
        shap_contributions = compute_shap_contributions(df_eng)
        latency_ms = (time.time() - t0) * 1000
        request_id = str(uuid.uuid4())
        log_prediction(request_id, engineered_record, proba, pred, latency_ms, source="api")
        return PredictResponse(
            request_id=request_id, satisfaction_probability=proba, prediction=pred,
            shap_contributions=shap_contributions,
        )
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


def generate_llm_summary(raw, proba, pred, shap_contributions):
    """Plain-English summary of a manual prediction via Groq (free tier). Falls back
    to a templated sentence if GROQ_API_KEY isn't set or the API call fails -- the
    demo tab should never break just because an external LLM call didn't go through."""
    label = "satisfied" if pred == 1 else "not satisfied"
    top_factors = ", ".join(
        f"{c['feature'].split('__', 1)[-1]} ({c['shap_value']:+.2f})"
        for c in shap_contributions[:3]
    )
    fallback = (
        f"Predicted **{label}** ({proba:.0%} confidence). "
        f"Most influential factors: {top_factors}."
    )
    if groq_client is None:
        return fallback

    prompt = (
        f"An airline satisfaction model predicted a passenger would be '{label}' "
        f"with {proba:.0%} confidence. Passenger: {raw['Age']}yo {raw['Gender']}, "
        f"{raw['Customer Type']}, {raw['Type of Travel']}, {raw['Class']} class, "
        f"{raw['Flight Distance']} mile flight. Top contributing factors (SHAP values, "
        f"positive pushes toward satisfied, negative toward not satisfied): {top_factors}. "
        "Write a 2-3 sentence plain-English summary of this prediction for a "
        "non-technical reader."
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            timeout=5,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log_error("llm_summary", str(e))
        return fallback + " (LLM summary unavailable.)"


# ----------------------------------------------------------------------
# Monitoring dashboard (Gradio, mounted at /dashboard)
# ----------------------------------------------------------------------
def get_manual_request_count():
    conn = sqlite3.connect(DB_PATH)
    count = pd.read_sql("SELECT COUNT(*) AS n FROM predictions WHERE source = 'manual'", conn)["n"][0]
    conn.close()
    return int(count)


def get_operational_metrics():
    # Only 'api' traffic (replay_client.py simulating production) counts here -- manual
    # tries from the "Try it Yourself" tab are tracked separately (get_manual_request_count)
    # so they don't distort what real traffic patterns look like.
    conn = sqlite3.connect(DB_PATH)
    preds = pd.read_sql("SELECT * FROM predictions WHERE source = 'api'", conn)
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
        "SELECT * FROM predictions WHERE source = 'api' AND true_label IS NOT NULL "
        "ORDER BY feedback_at", conn
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
        "SELECT features_json, created_at FROM predictions WHERE source = 'api' "
        "ORDER BY created_at DESC LIMIT ?",
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
    manual_count = get_manual_request_count()
    return (
        op["total"], op["error_rate"], op["p50"], op["p95"], manual_count,
        op["requests_fig"], acc_fig, auc_fig, psi_fig, cat_fig,
    )


def build_dashboard():
    with gr.Blocks(title="Airline Satisfaction - Model Monitoring") as demo:
        gr.Markdown("# Airline Satisfaction -- Model Monitoring Dashboard")
        gr.Markdown(
            "Served model was trained only on the 80% training split; everything here reflects "
            "the never-trained-on 20% holdout replayed as simulated production traffic. Manual "
            "tries from the [demo predictor](/demo) are tracked separately and excluded from every "
            "metric below so they don't distort what real traffic looks like."
        )
        refresh_btn = gr.Button("Refresh")
        with gr.Row():
            total_box = gr.Number(label="Total requests (simulated traffic)", interactive=False)
            error_box = gr.Number(label="Error rate (%)", interactive=False)
            p50_box = gr.Number(label="p50 latency (ms)", interactive=False)
            p95_box = gr.Number(label="p95 latency (ms)", interactive=False)
            manual_box = gr.Number(label="Manual requests (excluded above)", interactive=False)
        requests_plot = gr.Plot(label="Requests per minute")
        with gr.Row():
            acc_plot = gr.Plot(label="Rolling accuracy")
            auc_plot = gr.Plot(label="Rolling ROC-AUC")
        psi_plot = gr.Plot(label="Numeric feature drift (PSI)")
        cat_plot = gr.Plot(label="Categorical feature drift")

        outputs = [total_box, error_box, p50_box, p95_box, manual_box,
                   requests_plot, acc_plot, auc_plot, psi_plot, cat_plot]
        refresh_btn.click(refresh_all, outputs=outputs)
        demo.load(refresh_all, outputs=outputs)
    return demo


# ----------------------------------------------------------------------
# Showcase app (Gradio, mounted at /demo): model info + interactive predictor.
# Separate from the monitoring dashboard on purpose -- a visitor following the
# demo link should land on "try the model," not latency percentiles.
# ----------------------------------------------------------------------
def build_showcase():
    with gr.Blocks(title="Airline Satisfaction - Demo") as demo:
        gr.Markdown("# Airline Passenger Satisfaction Predictor")
        with gr.Tabs():
            with gr.Tab("Try it Yourself"):
                gr.Markdown(
                    "Enter a passenger's details and get a live prediction from the deployed "
                    "XGBoost model. These manual tries are tracked separately from the "
                    "[monitoring dashboard](/dashboard) -- there's no real outcome to compare "
                    "them against, so they're excluded from its accuracy/drift metrics."
                )
                with gr.Row():
                    age = gr.Number(label="Age", value=35, precision=0)
                    flight_distance = gr.Number(label="Flight Distance", value=1000, precision=0)
                with gr.Row():
                    gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                    customer_type = gr.Dropdown(
                        ["Loyal Customer", "disloyal Customer"], label="Customer Type",
                        value="Loyal Customer")
                    type_of_travel = gr.Dropdown(
                        ["Business travel", "Personal Travel"], label="Type of Travel",
                        value="Business travel")
                    travel_class = gr.Dropdown(
                        ["Eco", "Eco Plus", "Business"], label="Class", value="Business")

                gr.Markdown("### Service ratings (0 = not rated, 1-5 = rated)")
                rating_sliders = {}
                with gr.Row():
                    for col in ZERO_AS_NA_COLS[:6]:
                        rating_sliders[col] = gr.Slider(0, 5, step=1, value=3, label=col)
                with gr.Row():
                    for col in ZERO_AS_NA_COLS[6:]:
                        rating_sliders[col] = gr.Slider(0, 5, step=1, value=3, label=col)
                baggage_handling = gr.Slider(1, 5, step=1, value=3, label="Baggage handling")

                with gr.Row():
                    dep_delay = gr.Number(label="Departure Delay in Minutes", value=0, precision=0)
                    arr_delay = gr.Number(label="Arrival Delay in Minutes", value=0, precision=0)

                predict_btn = gr.Button("Predict", variant="primary")
                result = gr.Markdown()
                shap_plot = gr.Plot(label="What influenced this prediction (SHAP values)")
                summary_md = gr.Markdown(label="Summary")

                def run_demo_prediction(age, flight_distance, gender, customer_type,
                                         type_of_travel, travel_class, baggage_handling,
                                         dep_delay, arr_delay, *rating_values):
                    raw = {
                        "Age": int(age),
                        "Flight Distance": int(flight_distance),
                        "Gender": gender,
                        "Customer Type": customer_type,
                        "Type of Travel": type_of_travel,
                        "Class": travel_class,
                        "Baggage handling": int(baggage_handling),
                        "Departure Delay in Minutes": float(dep_delay),
                        "Arrival Delay in Minutes": float(arr_delay),
                    }
                    for col, val in zip(ZERO_AS_NA_COLS, rating_values):
                        raw[col] = int(val)

                    t0 = time.time()
                    proba, pred, engineered_record, df_eng = score_passenger(raw)
                    shap_contributions = compute_shap_contributions(df_eng)
                    latency_ms = (time.time() - t0) * 1000
                    request_id = str(uuid.uuid4())
                    log_prediction(request_id, engineered_record, proba, pred, latency_ms, source="manual")

                    label = "Satisfied" if pred == 1 else "Not Satisfied"
                    result_md = f"## {label}\n\n**Confidence**: {proba:.1%} probability of satisfaction"
                    fig = build_shap_figure(shap_contributions)
                    summary = generate_llm_summary(raw, proba, pred, shap_contributions)
                    return result_md, fig, summary

                predict_btn.click(
                    run_demo_prediction,
                    inputs=[age, flight_distance, gender, customer_type, type_of_travel,
                            travel_class, baggage_handling, dep_delay, arr_delay]
                           + [rating_sliders[c] for c in ZERO_AS_NA_COLS],
                    outputs=[result, shap_plot, summary_md],
                )

            with gr.Tab("Model Info"):
                vm = EVAL_METRICS["validation_metrics"]
                metrics_md = (
                    "### Validation Metrics (held-out 20%, never trained on)\n\n"
                    "| Metric | Score |\n|---|---|\n"
                    f"| Accuracy | {vm['accuracy']:.2%} |\n"
                    f"| Precision | {vm['precision']:.2%} |\n"
                    f"| Recall | {vm['recall']:.2%} |\n"
                    f"| F1 | {vm['f1']:.2%} |\n"
                    f"| ROC-AUC | {vm['roc_auc']:.2%} |\n"
                )
                gr.Markdown(metrics_md)

                fi_df = pd.DataFrame(EVAL_METRICS["top_shap_importances"])
                fi_fig = px.bar(fi_df, x="mean_abs_shap", y="feature", orientation="h",
                                 title="Top 20 Feature Importances (mean |SHAP value|)")
                fi_fig.update_yaxes(autorange="reversed")
                gr.Plot(value=fi_fig)

                raw_feats = ", ".join(f"`{c}`" for c in EVAL_METRICS["raw_input_features"])
                eng_feats = ", ".join(f"`{c}`" for c in EVAL_METRICS["engineered_features"])
                gr.Markdown(
                    f"### Raw input features ({len(EVAL_METRICS['raw_input_features'])})\n\n"
                    f"{raw_feats}\n\n"
                    f"### Engineered features fed to the model "
                    f"({len(EVAL_METRICS['engineered_features'])})\n\n{eng_feats}"
                )

                hp_md = "### Best Hyperparameters (RandomizedSearchCV)\n\n" + "\n".join(
                    f"- `{k}`: {v}" for k, v in EVAL_METRICS["best_hyperparameters"].items()
                )
                gr.Markdown(hp_md)
    return demo


app = gr.mount_gradio_app(app, build_dashboard(), path="/dashboard")
app = gr.mount_gradio_app(app, build_showcase(), path="/demo")
