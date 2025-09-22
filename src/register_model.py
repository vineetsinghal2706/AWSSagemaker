# src/register_model.py
import sagemaker
import boto3
import os
from sagemaker.model import Model
from sagemaker import get_execution_role

# ✅ Initialize SageMaker session & boto3 client
sagemaker_session = sagemaker.Session()
sm_client = boto3.client("sagemaker")
role = os.getenv("SAGEMAKER_ROLE_ARN", get_execution_role())

def register_model(model_file, model_package_group_name="FraudDetectionModelGroup"):
    """
    Registers the trained model into SageMaker Model Registry
    """
    # Upload model artifact to S3
    bucket = sagemaker_session.default_bucket()
    model_artifact = sagemaker_session.upload_data(path=model_file, bucket=bucket, key_prefix="model-artifacts")

    print(f"✅ Model artifact uploaded: {model_artifact}")

    # Create model package in Model Registry
    model_package = sm_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription="Fraud detection RandomForest model",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": sagemaker.image_uris.retrieve(
                        framework="sklearn",
                        region=sagemaker_session.boto_region_name,
                        version="1.2-1",  # pick sklearn version
                    ),
                    "ModelDataUrl": model_artifact,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ApprovalStatus="PendingManualApproval",
    )

    print(f"✅ Model registered in group {model_package_group_name}")
    return model_package


if __name__ == "__main__":
    model_file = "/tmp/model/model.pkl"  # make sure train.py saved here
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found. Run train.py first.")

    register_model(model_file)
