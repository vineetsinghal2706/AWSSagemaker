# src/train.py
import os
import pickle
import boto3
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split


def load_data(s3_uri: str) -> pd.DataFrame:
    """Load CSV dataset from S3 or local path."""
    if s3_uri.startswith("s3://"):
        # Download to temp file
        bucket = s3_uri.split("/")[2]
        key = "/".join(s3_uri.split("/")[3:])
        tmp_file = "/tmp/train.csv"
        boto3.client("s3").download_file(bucket, key, tmp_file)
        return pd.read_csv(tmp_file)
    else:
        return pd.read_csv(s3_uri)


def train_and_log(
    train_s3_uri: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str = "fraud_detection",
    run_name: str = "rf_run",
    registered_model_name: str = "FraudDetectionRF"
):
    """
    Train RandomForest on fraud dataset, log with MLflow, and register model.
    """
    # --- MLflow setup ---
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name) as run:
        # --- Load data ---
        df = load_data(train_s3_uri)
        X = df.drop("Class", axis=1)
        y = df["Class"]

        # --- Train/test split ---
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Model params ---
        params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}
        mlflow.log_params(params)

        # --- Train ---
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- Predict & evaluate ---
        y_pred = model.predict(X_val)
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_val, y_pred)
        }
        mlflow.log_metrics(metrics)

        # --- Save model locally ---
        os.makedirs("/tmp/model", exist_ok=True)
        model_file = "/tmp/model/model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # --- Log + Register model ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        print(f"✅ Model registered as: {registered_model_name}")
        print(f"📊 Metrics: {metrics}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    # Config (can move to env vars later)
    TRAIN_S3_PATH = "s3://creditcardnew/fraud-data/train/train.csv"
    MLFLOW_TRACKING_URI = "http://127.0.0.1:6000"   # 👈 Update if server runs elsewhere
    EXPERIMENT_NAME = "fraud_detection"
    MODEL_NAME = "FraudDetectionRF"

    run_id, metrics = train_and_log(
        train_s3_uri=TRAIN_S3_PATH,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment=EXPERIMENT_NAME,
        registered_model_name=MODEL_NAME
    )

    print("✅ Training complete. Run ID:", run_id)
