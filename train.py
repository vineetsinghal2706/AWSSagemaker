# Notebook-friendly version of train.py
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle

# ====== Set S3 paths manually ======
s3_input_bucket = "creditcardnew"
s3_input_key = "train_data.csv"
s3_output_bucket = "creditcardnew"
s3_output_key = "model_output/cc_fraud_model.pkl"
n_estimators = 100

# ====== Read CSV from S3 ======
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=s3_input_bucket, Key=s3_input_key)
df = pd.read_csv(obj['Body'])

# ====== Prepare data ======
X = df.drop(['Class'], axis=1).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== Train Random Forest ======
clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# ====== Evaluate ======
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))

# ====== Save model locally and upload to S3 ======
local_model_path = "cc_fraud_model.pkl"
with open(local_model_path, 'wb') as f:
    pickle.dump(clf, f)

s3.upload_file(local_model_path, s3_output_bucket, s3_output_key)
print(f"Model uploaded to s3://{s3_output_bucket}/{s3_output_key}")
