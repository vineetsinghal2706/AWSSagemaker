# src/train.py

import os
import boto3
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sagemaker.experiments.run import Run

# --- Config ---
S3_BUCKET = "creditcardnew"
TRAIN_S3_PATH = f"s3://{S3_BUCKET}/fraud-data/train/train.csv"
EXPERIMENT_NAME = "fraud-detection-exp"
MODEL_REGISTRY_GROUP = "fraud-detection-model-group"

def load_data(s3_uri):
    """Load dataset directly from S3"""
    return pd.read_csv(s3_uri)

def train_and_log():
    # --- Load data ---
    df = load_data(TRAIN_S3_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}

    # --- Start Experiment Run ---
    with Run(experiment_name=EXPERIMENT_NAME) as run:
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }

        # Log params & metrics to SageMaker Experiments
        for k, v in params.items():
            run.log_parameter(k, v)
        for k, v in metrics.items():
            run.log_metric(k, v)

        print("📊 Metrics:", metrics)

        # Save model locally
        os.makedirs("/tmp/model", exist_ok=True)
        model_file = "/tmp/model/model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Upload model to S3
        s3_client = boto3.client("s3")
        s3_key = f"models/rf/{run.run_name}/model.pkl"
        s3_client.upload_file(model_file, S3_BUCKET, s3_key)
        model_s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        print("✅ Model uploaded to:", model_s3_uri)

        # --- Register Model in SageMaker Model Registry ---
        sm_client = boto3.client("sagemaker")
        model_package = sm_client.create_model_package(
            ModelPackageGroupName=MODEL_REGISTRY_GROUP,
            ModelPackageDescription="Fraud detection RF model",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": "382416733822.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:1.0-1",  # sklearn container
                        "ModelDataUrl": model_s3_uri,
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
            ModelApprovalStatus="PendingManualApproval",
        )

        print("✅ Model registered in Model Registry:", model_package["ModelPackageArn"])

if __name__ == "__main__":
    train_and_log()
