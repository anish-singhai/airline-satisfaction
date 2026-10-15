FROM python:3.11-slim

WORKDIR /app

# libgomp1 is the OpenMP runtime XGBoost needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY xgboost_pipeline_serving.joblib .
COPY reference_stats.json .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
