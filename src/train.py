# src/train.py
import pandas as pd
import boto3
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sagemaker.experiments.run import Run

def train_and_log(train_s3_uri, experiment_name="fraud_detection", run_name="rf_run"):
    # --- Load data ---
    df = pd.read_csv(train_s3_uri)
    X = df.drop('Class', axis=1)
    y = df['Class']

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
            "mcc": matthews_corrcoef(y_val, y_pred)
        }

        # ✅ Log metrics to SageMaker
        run.log_metrics(metrics)

        # Save model
        os.makedirs("/tmp/model", exist_ok=True)
        model_file = "/tmp/model/model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print("✅ Training done")
        return model_file, metrics
