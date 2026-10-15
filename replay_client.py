"""
Replay the held-out 20% (model never trained on it) against the deployed API as if it were
live production traffic, with ground-truth feedback arriving after a delay. Run this locally --
it is the "client" generating the traffic the dashboard tracks, not part of the Space itself.

Example:
    python3 replay_client.py --base-url https://anishsinghai-airline-satisfaction-monitor.hf.space \
        --num-batches 15 --batch-size 20 --row-sleep 0.1 --feedback-delay 8 \
        --inject-drift-after-batch 10
"""

import argparse
import time
from collections import deque

import pandas as pd
import requests

INT_COLS = [
    "Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
FLOAT_COLS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
STR_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def row_to_payload(row, inject_drift=False):
    payload = {}
    for col in INT_COLS:
        payload[col] = int(row[col])
    for col in FLOAT_COLS:
        v = row[col]
        payload[col] = None if pd.isna(v) else float(v)
    for col in STR_COLS:
        payload[col] = str(row[col])
    if inject_drift:
        # Push Flight Distance and Age well outside the training distribution so the
        # drift panel has something real to flag.
        payload["Flight Distance"] = int(payload["Flight Distance"] * 3)
        payload["Age"] = min(int(payload["Age"] + 40), 95)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                         help="Deployed Space URL, e.g. https://anishsinghai-airline-satisfaction-monitor.hf.space")
    parser.add_argument("--holdout-csv", default="../2. XGBoost/model_outputs/simulation_holdout.csv")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--row-sleep", type=float, default=0.1,
                         help="Seconds between individual predict requests")
    parser.add_argument("--feedback-delay", type=float, default=10.0,
                         help="Seconds to wait before sending a batch's ground truth")
    parser.add_argument("--inject-drift-after-batch", type=int, default=None,
                         help="Start perturbing Flight Distance/Age from this batch index onward")
    args = parser.parse_args()

    df = pd.read_csv(args.holdout_csv)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    predict_url = f"{args.base_url.rstrip('/')}/predict"
    feedback_url = f"{args.base_url.rstrip('/')}/feedback"

    pending = deque()  # (due_time, [(request_id, true_label), ...])

    for batch_idx in range(args.num_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch_df = df.iloc[start:end]
        if batch_df.empty:
            break

        inject = (
            args.inject_drift_after_batch is not None
            and batch_idx >= args.inject_drift_after_batch
        )
        batch_results = []
        for _, row in batch_df.iterrows():
            payload = row_to_payload(row, inject_drift=inject)
            try:
                resp = requests.post(predict_url, json=payload, timeout=10)
                resp.raise_for_status()
                request_id = resp.json()["request_id"]
                batch_results.append((request_id, int(row["satisfaction"])))
            except Exception as e:
                print(f"[batch {batch_idx}] predict failed: {e}")
            time.sleep(args.row_sleep)

        print(f"Batch {batch_idx}: sent {len(batch_results)} predictions"
              f"{' (drift injected)' if inject else ''}")
        pending.append((time.time() + args.feedback_delay, batch_results))

        while pending and pending[0][0] <= time.time():
            _, results = pending.popleft()
            for request_id, true_label in results:
                try:
                    requests.post(feedback_url,
                                  json={"request_id": request_id, "true_label": true_label}, timeout=10)
                except Exception as e:
                    print(f"feedback failed for {request_id}: {e}")
            print(f"  -> sent feedback for {len(results)} earlier predictions")

    print("Draining remaining feedback...")
    while pending:
        due, results = pending.popleft()
        wait = max(0, due - time.time())
        if wait:
            time.sleep(wait)
        for request_id, true_label in results:
            try:
                requests.post(feedback_url,
                              json={"request_id": request_id, "true_label": true_label}, timeout=10)
            except Exception as e:
                print(f"feedback failed for {request_id}: {e}")
        print(f"  -> sent feedback for {len(results)} predictions")

    print(f"Done. Open {args.base_url.rstrip('/')}/dashboard to view metrics.")


if __name__ == "__main__":
    main()
