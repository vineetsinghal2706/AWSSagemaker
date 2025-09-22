import pandas as pd
import boto3
import joblib
from sklearn.ensemble import RandomForestClassifier

def main():
    # ✅ S3 path of your input file
    s3_input_path = "s3://creditcardnew/fraud-data/data/train.csv"

    # Download file from S3 to local (SageMaker training container needs local file access)
    local_file = "/opt/ml/input/data/train/train.csv"
    boto3.client("s3").download_file(
        "creditcardnew",
        "fraud-data/data/train.csv",
        local_file
    )

    # Load data
    df = pd.read_csv(local_file)
    X = df.drop("label", axis=1)
    y = df["label"]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model to /opt/ml/model (SageMaker will upload this to S3 automatically)
    joblib.dump(model, "/opt/ml/model/model.joblib")

if __name__ == "__main__":
    main()
