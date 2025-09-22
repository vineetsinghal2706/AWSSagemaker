# src/train.py
import os
import pickle
import boto3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sagemaker.experiments.run import Run


def download_from_s3(s3_uri, local_path="/tmp/train.csv"):
    """Download a file from S3 to a local path."""
    if not s3_uri.startswith("s3://"):
        raise ValueError("❌ Expected s3:// URI, got: " + s3_uri)

    # Parse bucket and key
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"✅ Downloaded {s3_uri} -> {local_path}")
    return local_path


def train_and_log(train_s3_uri, experiment_name="fraud-detection", run_name="rf-run"):
    """Train RandomForest on fraud dataset and log metrics to SageMaker Experiments."""

    # --- Download dataset from S3 ---
    local_train = download_from_s3(train_s3_uri)
    df = pd.read_csv(local_train)

    # --- Split features and labels ---
    if "Class" not in df.columns:
        raise ValueError("❌ Dataset must contain a 'Class' column as target.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}

    # ✅ Open SageMaker Experiment run
    with Run(experiment_name=experiment_name, run_name=run_name) as run:
        # Log parameters
        run.log_parameters(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_val, y_pred),
        }

        # ✅ Log metrics one by one
        for k, v in metrics.items():
            run.log_metric(name=k, value=v)

        # Save model locally
        os.makedirs("/tmp/model", exist_ok=True)
        model_file = "/tmp/model/model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print("✅ Training complete. Model saved at:", model_file)
        print("📊 Metrics:", metrics)

        return model_file, metrics


if __name__ == "__main__":
    # 🔥 Update S3 path for your dataset
    TRAIN_S3_PATH = "s3://creditcardnew/fraud-data/train/train.csv"

    model_file, metrics = train_and_log(
        TRAIN_S3_PATH,
        experiment_name="fraud-detection",  # ✅ must follow regex [a-zA-Z0-9](-*[a-zA-Z0-9])
        run_name="rf-run"
    )
