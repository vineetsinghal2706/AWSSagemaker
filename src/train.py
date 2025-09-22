# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import boto3
import pickle
import os
import time
from sagemaker import image_uris

# --- Training and Logging ---
def train_and_log(train_s3_uri, experiment_name="fraud-detection-exp", run_name="rf_run"):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # --- Load Data ---
        df = pd.read_csv(train_s3_uri)
        X = df.drop("Class", axis=1)
        y = df["Class"]

        # --- Train/Test Split ---
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Model Parameters ---
        params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}
        mlflow.log_params(params)

        # --- Train Model ---
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- Evaluate Model ---
        y_pred = model.predict(X_val)
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_val, y_pred),
        }
        mlflow.log_metrics(metrics)

        # --- Save Model as Pickle ---
        os.makedirs("/tmp/model", exist_ok=True)
        model_file = "/tmp/model/model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # --- Upload to S3 ---
        s3 = boto3.client("s3")
        bucket = "creditcardnew"
        s3_key = f"models/rf/{run.info.run_id}/model.pkl"
        s3.upload_file(model_file, bucket, s3_key)
        model_s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"✅ Uploaded model to {model_s3_uri}")

        # --- Register Model in SageMaker Model Registry ---
        sm_client = boto3.client("sagemaker")
        model_package_group = "fraud-detection-registry"

        # Create Model Package Group (only if not exists)
        try:
            sm_client.create_model_package_group(
                ModelPackageGroupName=model_package_group,
                ModelPackageGroupDescription="Fraud detection model registry"
            )
            print(f"✅ Created ModelPackageGroup: {model_package_group}")
        except sm_client.exceptions.ResourceInUse:
            print(f"ℹ️ ModelPackageGroup {model_package_group} already exists")

        # ✅ Dynamically get the correct SKLearn image for your region
        region = boto3.Session().region_name
        sklearn_image = image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.0-1",
            image_scope="inference"
        )

        # Register the model
        model_pkg = sm_client.create_model_package(
            ModelPackageGroupName=model_package_group,
            ModelPackageDescription="Fraud detection RF model",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": sklearn_image,
                        "ModelDataUrl": mod
