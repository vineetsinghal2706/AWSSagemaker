# src/register_model.py
import sagemaker
import boto3
import os
from sagemaker.model import Model

# ⚠️ Use SageMaker Execution Role, not IAM user
# Pass through ENV or hardcode
role = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::893169065109:role/datazone_usr_role_6gknc254ej3jfr_5251772u38j19j")

sagemaker_session = sagemaker.Session()
sm_client = boto3.client("sagemaker")


def register_model(model_file, model_package_group_name="FraudDetectionModelGroup"):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found. Run train.py first.")

    # Upload model artifact to S3
    bucket = sagemaker_session.default_bucket()
    model_artifact = sagemaker_session.upload_data(
        path=model_file, bucket=bucket, key_prefix="model-artifacts"
    )
    print(f"✅ Uploaded model to {model_artifact}")

    # Register in Model Registry
    model_package = sm_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription="Fraud detection RandomForest model",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": sagemaker.image_uris.retrieve(
                        framework="sklearn",
                        region=sagemaker_session.boto_region_name,
                        version="1.2-1",
                    ),
                    "ModelDataUrl": model_artifact,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ApprovalStatus="PendingManualApproval",
    )

    print(f"✅ Model registered in {model_package_group_name}")
    return model_package


if __name__ == "__main__":
    model_file = "/tmp/model/model.pkl"
    register_model(model_file)
