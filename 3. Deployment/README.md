# Airline Satisfaction — Serving API + Monitoring

FastAPI service that scores airline passenger satisfaction with a tuned XGBoost pipeline, plus a
Gradio dashboard (`/dashboard`) tracking operational and model-quality metrics over time. Deployed
on [Render](https://render.com) — see the [top-level README](../README.md) for the full project
story (EDA, model comparison, why Render instead of Hugging Face Spaces).

The served model was trained **only** on an 80% split of the training data; the other 20% was
never trained or tuned on and is replayed by `replay_client.py` (run locally, not part of this
deployment) to simulate live production traffic with delayed ground-truth feedback.

## Endpoints

- `GET /health` — liveness check
- `POST /predict` — score a passenger (raw Kaggle feature columns as JSON body), returns
  `{request_id, satisfaction_probability, prediction}`
- `POST /feedback` — attach a true label to a previous prediction:
  `{"request_id": "...", "true_label": 0 or 1}`
- `/dashboard` — Gradio UI: requests/min, latency percentiles, error rate, rolling
  accuracy/ROC-AUC (as feedback arrives), and feature drift (PSI for continuous numeric features,
  L1/total-variation distance for categorical and discrete numeric features) vs. the training
  distribution.

## Running locally

```
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 7860
```

Or via Docker:
```
docker build -t airline-satisfaction .
docker run -p 7860:7860 airline-satisfaction
```

## Known limitation

Render's free tier spins down after 15 minutes idle, which wipes the local SQLite log (`logs.db`)
since it's on ephemeral container storage. Fine for a demo/learning project; persisting logs to an
external database would be the natural next step for anything longer-lived.
