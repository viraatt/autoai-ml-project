"""
run_autoai_wml.py

Comprehensive, ready-to-run script to: 
- clone your GitHub repo (if needed),
- find a CSV dataset in the repo,
- authenticate to IBM Watson Machine Learning (AutoAI) using an API key,
- upload the dataset to the project/space as a data asset,
- start an AutoAI experiment programmatically,
- poll for completion, fetch the leaderboard,
- save the best pipeline as a model and create an online deployment,
- score a sample file and save results locally.

USAGE:
1. Put this file inside your local copy of the repo or run it from a place that can access the dataset in the repo.
2. Set environment variables before running (recommended):
   export WML_APIKEY="<your api key>"
   export WML_URL="https://us-south.ml.cloud.ibm.com"            # or your region's URL
   export WML_PROJECT_ID="<optional_project_id>"               # optional: project-based flow
   export WML_SPACE_ID="<optional_space_id>"                   # optional: space-based flow
   export DATASET_PATH="path/to/your/dataset.csv"              # relative path inside repo
3. Run: python run_autoai_wml.py

IMPORTANT SECURITY NOTE:
- Do NOT hardcode API keys inside files you commit to GitHub. Use environment variables or a secure secret store.

This script makes best-effort assumptions when required metadata isn't provided. If you do not have a Project or Space ID, set WML_SPACE_ID or WML_PROJECT_ID in your environment. See comments below for where to find them in Watson Studio.

"""

import os
import time
import json
import sys
import subprocess
from pathlib import Path
import pandas as pd

try:
    from ibm_watson_machine_learning import APIClient
    from ibm_watson_machine_learning.experiment import AutoAI
except Exception as e:
    print("Missing ibm-watson-machine-learning SDK. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ibm-watson-machine-learning==1.1.0"])
    from ibm_watson_machine_learning import APIClient
    from ibm_watson_machine_learning.experiment import AutoAI

# ---------------------------
# Configuration (from env)
# ---------------------------
WML_APIKEY = os.environ.get("WML_APIKEY") or os.environ.get("CPD_APIKEY")
WML_URL = os.environ.get("WML_URL", "https://us-south.ml.cloud.ibm.com")
WML_PROJECT_ID = os.environ.get("WML_PROJECT_ID")  # optional
WML_SPACE_ID = os.environ.get("WML_SPACE_ID")      # optional
DATASET_PATH = os.environ.get("DATASET_PATH", "data/your_dataset.csv")
REPO_URL = os.environ.get("REPO_URL", "https://github.com/viraatt/autoai-ml-project.git")
LOCAL_REPO_DIR = Path("./autoai-ml-project")

if not WML_APIKEY:
    print("ERROR: WML_APIKEY is not set. Please set it as an environment variable and re-run.")
    sys.exit(1)

# Optional: clone the repo if not already present
if not LOCAL_REPO_DIR.exists():
    print(f"Cloning repo {REPO_URL} into {LOCAL_REPO_DIR}...")
    subprocess.check_call(["git", "clone", REPO_URL, str(LOCAL_REPO_DIR)])
else:
    print(f"Repo already present at {LOCAL_REPO_DIR}. Pulling latest changes...")
    try:
        subprocess.check_call(["git", "-C", str(LOCAL_REPO_DIR), "pull"])
    except Exception:
        print("Could not pull; continuing with existing files.")

# If dataset path is relative, try inside the repo
candidate_paths = [Path(DATASET_PATH), LOCAL_REPO_DIR / DATASET_PATH]
dataset_file = None
for p in candidate_paths:
    if p.exists():
        dataset_file = p
        break

if dataset_file is None:
    # Try heuristics: look for csv files in repo root or data/
    csvs = list(LOCAL_REPO_DIR.rglob('*.csv'))[:20]
    if len(csvs) == 0:
        print("No CSV dataset found in the repository. Please set DATASET_PATH to point at your CSV.")
        sys.exit(1)
    print("Multiple/No dataset path provided. Using the first CSV found in repo:", csvs[0])
    dataset_file = csvs[0]

print("Using dataset:", dataset_file)

# Basic local preview
try:
    df_preview = pd.read_csv(dataset_file, nrows=5)
    print("Dataset preview (first 5 rows):")
    print(df_preview)
except Exception as e:
    print("Warning: couldn't read dataset preview:", e)

# ---------------------------
# WML Authentication
# ---------------------------
wml_credentials = {
    "apikey": WML_APIKEY,
    "url": WML_URL
}
print("Authenticating to Watson Machine Learning at", WML_URL)
client = APIClient(wml_credentials)

# Helper: set either project or space if provided
if WML_PROJECT_ID:
    try:
        client.set.default_project(WML_PROJECT_ID)
        print("Set default project to:", WML_PROJECT_ID)
    except Exception as e:
        print("Could not set project (you may not have permissions or the project id is incorrect):", e)
elif WML_SPACE_ID:
    try:
        client.set.default_space(WML_SPACE_ID)
        print("Set default space to:", WML_SPACE_ID)
    except Exception as e:
        print("Could not set default space:", e)
else:
    print("No project or space ID set. The script will attempt to operate without explicitly setting them, but you may need to set WML_PROJECT_ID or WML_SPACE_ID.")

# ---------------------------
# Upload data asset
# ---------------------------
print("Uploading dataset as a data asset to the project/space...")
asset_name = f"autoai_dataset_{int(time.time())}.csv"
try:
    meta_props = {
        client.data_assets.ConfigurationMetaNames.NAME: asset_name,
        client.data_assets.ConfigurationMetaNames.DESCRIPTION: "Dataset uploaded by run_autoai_wml.py"
    }
    data_asset_details = client.data_assets.create(name=str(dataset_file.name), meta_props=meta_props, file_path=str(dataset_file))
    data_asset_id = data_asset_details['metadata']['asset_id'] if 'metadata' in data_asset_details and 'asset_id' in data_asset_details['metadata'] else data_asset_details['metadata'].get('id')
    print("Uploaded data asset id:", data_asset_id)
except Exception as e:
    print("Failed to upload data asset via client.data_assets.create(). Trying fallback 'store' method...", e)
    try:
        data_asset_details = client.data_assets.store(meta_props={
            client.data_assets.ConfigurationMetaNames.NAME: asset_name
        }, file_path=str(dataset_file))
        data_asset_id = data_asset_details['metadata']['asset_id']
        print("Uploaded data asset id (fallback):", data_asset_id)
    except Exception as e2:
        print("ERROR: Could not upload dataset to Watson Machine Learning.\nDetails:", e2)
        sys.exit(1)

# ---------------------------
# Start AutoAI experiment
# ---------------------------
print("Starting AutoAI experiment. Please set these variables below as needed.")
# Auto-detect the prediction column if possible (naive heuristic)
try:
    df = pd.read_csv(dataset_file, nrows=200)
    # choose the last column as target if it looks categorical
    target_col = os.environ.get('TARGET_COLUMN')
    if not target_col:
        target_col = df.columns[-1]
        print(f"No TARGET_COLUMN provided. Heuristic chose '{target_col}' as target (last column). Set TARGET_COLUMN env var to override.")
except Exception:
    target_col = os.environ.get('TARGET_COLUMN', 'target')
    print("Falling back to TARGET_COLUMN=", target_col)

prediction_type = os.environ.get('PREDICTION_TYPE', 'classification')  # or 'regression'
max_wait_seconds = int(os.environ.get('MAX_WAIT_SECONDS', 60 * 30))
max_estimators = int(os.environ.get('MAX_ESTIMATORS', 50))

print(f"AutoAI params -> prediction_type: {prediction_type}, target: {target_col}, max_wait_seconds: {max_wait_seconds}, max_estimators: {max_estimators}")

autoai = AutoAI(wml_client=client)

experiment_meta = {
    "name": f"autoai_experiment_{int(time.time())}",
    "prediction_type": prediction_type,
    "prediction_column": target_col,
    # You can add metric, holdout, etc.
    "max_number_of_estimators": max_estimators,
}

print("Submitting AutoAI run... this can take a long time depending on the max_estimators and search depth.")
try:
    experiment_run = autoai.run(training_data=data_asset_id, meta_props=experiment_meta, max_wait=max_wait_seconds)
    run_id = experiment_run['metadata']['guid']
    print("Started AutoAI run id:", run_id)
except Exception as e:
    print("Failed to start AutoAI run:", e)
    # Attempt a more explicit call for older SDK versions
    try:
        experiment_run = autoai.run(training_data=str(dataset_file), meta_props=experiment_meta, max_wait=max_wait_seconds)
        run_id = experiment_run['metadata']['guid']
        print("Started AutoAI run id (fallback):", run_id)
    except Exception as e2:
        print("ERROR: Could not start AutoAI run. See error:\n", e2)
        sys.exit(1)

# ---------------------------
# Poll leaderboard periodically until completion or timeout
# ---------------------------
start = time.time()
finished = False
poll_interval = 30
print("Polling AutoAI run status...")
while time.time() - start < max_wait_seconds:
    try:
        status = autoai.get_status(run_id)
        print("Run status:", status.get('state'))
        if status.get('state') in ('completed', 'finished'):
            finished = True
            break
        if status.get('state') in ('failed', 'error'):
            print("AutoAI run failed or errored:", status)
            break
    except Exception as e:
        print("Could not fetch status yet:", e)
    time.sleep(poll_interval)

if not finished:
    print("AutoAI run did not finish in the allotted time. You can re-run this script with a larger MAX_WAIT_SECONDS or inspect the run in Watson Studio UI.")

# ---------------------------
# Get leaderboard and pick best pipeline
# ---------------------------
try:
    leaderboard = autoai.get_leaderboard(run_id)
    print("Leaderboard (top 10):")
    print(leaderboard.head(10))
    best_pipeline_id = leaderboard.iloc[0]['pipeline_id']
    print("Best pipeline id:", best_pipeline_id)
except Exception as e:
    print("Failed to retrieve leaderboard:", e)
    sys.exit(1)

# ---------------------------
# Save the best pipeline as a model artifact
# ---------------------------
model_name = f"autoai_best_model_{int(time.time())}"
print("Saving best pipeline as a model named:", model_name)
try:
    model_meta = {
        client.repository.ModelMetaNames.NAME: model_name,
        client.repository.ModelMetaNames.TYPE: "automl_0.1",  # AutoAI models use automl runtime types in some regions
        client.repository.ModelMetaNames.RUNTIME_UID: "python-3.8"
    }
    saved_model = autoai.save_pipeline(run_id, best_pipeline_id, meta_props=model_meta)
    model_id = saved_model['metadata']['id']
    print("Saved model id:", model_id)
except Exception as e:
    print("Failed to save pipeline as a model:", e)
    sys.exit(1)

# ---------------------------
# Deploy model (online)
# ---------------------------
print("Creating an online deployment for the model...")
try:
    deployment_props = {
        client.deployments.ConfigurationMetaNames.NAME: f"autoai_deploy_{int(time.time())}",
        client.deployments.ConfigurationMetaNames.DESCRIPTION: "AutoAI best model deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
    deployment = client.deployments.create(model_id, meta_props=deployment_props)
    deployment_id = deployment['metadata']['id']
    scoring_url = deployment['entity']['status'].get('online_url') or deployment['entity']['status'].get('scoring_url')
    print("Deployment id:", deployment_id)
    print("Scoring URL:", scoring_url)
except Exception as e:
    print("Failed to create deployment:", e)
    sys.exit(1)

# ---------------------------
# Score a sample file
# ---------------------------
sample_to_score = dataset_file  # naive: score the same dataset (drop target column)
print("Preparing sample data to send to scoring endpoint...")
try:
    df_full = pd.read_csv(sample_to_score)
    if target_col in df_full.columns:
        score_df = df_full.drop(columns=[target_col])
    else:
        score_df = df_full
    # Use SDK scoring helper
    print("Sending sample to deployment for scoring (first 20 rows)...")
    resp = client.deployments.score(deployment_id, score_df.head(20))
    print("Scoring response (trimmed):")
    # Print structured results concisely
    if isinstance(resp, dict):
        print(json.dumps(resp, indent=2)[:2000])
    else:
        print(resp)
except Exception as e:
    print("Failed to score via SDK:", e)
    print("You can still score using raw REST requests with the deployment's scoring_url and a bearer token from the SDK.")

print("Script finished. Check Watson Studio project for the AutoAI experiment, generated notebook, and the deployment.")
print("If you want, set environment variables WML_PROJECT_ID or WML_SPACE_ID to persist resources into a particular project/space.")

# End of script
