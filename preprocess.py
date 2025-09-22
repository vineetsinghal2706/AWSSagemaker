# src/preprocess.py
import pandas as pd
import boto3
import os

def split_and_upload():
    # Hardcoded values
    s3_bucket = "creditcardnew"            # your bucket name
    input_key = "creditcard.csv"           # S3 key of your input file
    s3_prefix = "fraud-data"               # destination folder in S3
    train_rows = 150000                    # number of rows for training

    s3 = boto3.client("s3")

    # Download the file from S3 to local temp
    local_input = "/tmp/creditcard.csv"
    s3.download_file(s3_bucket, input_key, local_input)

    # Load data
    df = pd.read_csv(local_input)
    assert len(df) >= train_rows, "Not enough rows for requested train size"

    # Split into train and infer
    df_train = df.sample(n=train_rows, random_state=42)
    df_rest = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    # Save locally
    os.makedirs("/tmp/data", exist_ok=True)
    train_path = "/tmp/data/train.csv"
    infer_path = "/tmp/data/infer.csv"
    df_train.to_csv(train_path, index=False)
    df_rest.to_csv(infer_path, index=False)

    # Upload to S3
    s3_train_key = f"{s3_prefix}/train/train.csv"
    s3_infer_key = f"{s3_prefix}/infer/infer.csv"

    s3.upload_file(train_path, s3_bucket, s3_train_key)
    s3.upload_file(infer_path, s3_bucket, s3_infer_key)

    print(f"✅ Uploaded train -> s3://{s3_bucket}/{s3_train_key}")
    print(f"✅ Uploaded infer -> s3://{s3_bucket}/{s3_infer_key}")

if __name__ == "__main__":
    split_and_upload()
