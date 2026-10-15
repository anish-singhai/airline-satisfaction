---
title: Airline Satisfaction Monitor
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Airline Satisfaction — Serving API + Monitoring

FastAPI service that scores airline passenger satisfaction with a tuned XGBoost pipeline, plus a
Gradio dashboard (`/dashboard`) tracking operational and model-quality metrics over time.

The served model was trained **only** on an 80% split of the training data; the other 20% was
never trained or tuned on and is replayed by `replay_client.py` (run locally, not part of this
Space) to simulate live production traffic with delayed ground-truth feedback.

## Endpoints

- `GET /health` — liveness check
- `POST /predict` — score a passenger (raw Kaggle feature columns as JSON body), returns
  `{request_id, satisfaction_probability, prediction}`
- `POST /feedback` — attach a true label to a previous prediction:
  `{"request_id": "...", "true_label": 0 or 1}`
- `/dashboard` — Gradio UI: requests/min, latency percentiles, error rate, rolling
  accuracy/ROC-AUC (as feedback arrives), and feature drift (PSI for numeric, L1 distance for
  categorical) vs. the training distribution.

## Known limitation

Free-tier Spaces can go idle and restart, which wipes the local SQLite log (`logs.db`) since it's
on ephemeral container storage. Fine for a demo/learning project; persisting logs to an HF
Dataset repo would be the natural next step for anything longer-lived.
