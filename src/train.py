# src/train.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import boto3
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split

def load_data(s3_uri):
    return pd.read_csv(s3_uri)

def train_and_log(train_s3_uri, test_s3_uri=None, mlflow_experiment='fraud_detection', run_name='rf_run'):
    # ✅ Configure MLflow to use S3
    mlflow.set_tracking_uri("s3://creditcardnew/mlflow/")
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name) as run:
        # load data
        df = pd.read_csv(train_s3_uri)
        X = df.drop('Class', axis=1)
        y = df['Class']

        # train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # params
        params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}
        mlflow.log_params(params)

        # train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # evaluate
        y_pred = model.predict(X_val)
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_val, y_pred)
        }
        mlflow.log_metrics(metrics)

        # save local artifacts
        os.makedirs('/tmp/model', exist_ok=True)
        model_joblib = "/tmp/model/model.joblib"
        model_pkl = "/tmp/model/model.pkl"
        joblib.dump(model, model_joblib)
        with open(model_pkl, "wb") as f:
            pickle.dump(model, f)

        # log model in MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )

        # ✅ upload both .joblib and .pkl to S3
        s3 = boto3.client('s3')
        S3_BUCKET = "creditcardnew"
        base_key = f"models/rf/{run.info.run_id}/"
        
        s3.upload_file(model_joblib, S3_BUCKET, base_key + "model.joblib")
        s3.upload_file(model_pkl, S3_BUCKET, base_key + "model.pkl")

        print("✅ Uploaded model.joblib to", f"s3://{S3_BUCKET}/{base_key}model.joblib")
        print("✅ Uploaded model.pkl to", f"s3://{S3_BUCKET}/{base_key}model.pkl")

        return run.info.run_id, metrics

if __name__ == "__main__":
    # 🔥 Hardcoded paths
    TRAIN_S3_PATH = "s3://creditcardnew/fraud-data/train/train.csv"
    EXPERIMENT_NAME = "fraud_detection"

    run_id, metrics = train_and_log(TRAIN_S3_PATH, mlflow_experiment=EXPERIMENT_NAME)
    print("✅ Training complete. Run ID:", run_id)
    print("📊 Metrics:", metrics)
