"""
Create (or update) the private Hugging Face Space serving this app.
Requires: `hf auth login` already run.
"""

from huggingface_hub import HfApi, create_repo

REPO_ID = "anishsinghai/airline-satisfaction-monitor"

api = HfApi()

create_repo(REPO_ID, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

api.upload_folder(
    repo_id=REPO_ID,
    repo_type="space",
    folder_path=".",
    allow_patterns=[
        "app.py", "Dockerfile", "requirements.txt", "README.md",
        "xgboost_pipeline_serving.joblib", "reference_stats.json",
    ],
    commit_message="Deploy serving API + monitoring dashboard",
)

print(f"Pushed to https://huggingface.co/spaces/{REPO_ID}")
