import sagemaker

from sagemaker.model import Model

def register_model(model_file, model_package_group_name="fraud-detection-models"):
    session = sagemaker.Session()
    role = "arn:aws:iam::893169065109:role/datazone_usr_role_6gknc254ej3jfr_5251772u38j19j"


    # Upload model to S3
    model_s3_uri = session.upload_data(path=model_file, key_prefix="models/rf")

    # Create a SageMaker Model
    sm_model = Model(
        image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:1.2-1-cpu-py3",  # 🔥 Change to your region
        model_data=model_s3_uri,
        role=role
    )

    # Register in Model Registry
    model_package = sm_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name=model_package_group_name,
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        approval_status="PendingManualApproval"
    )

    print(f"✅ Registered model: {model_package.model_package_arn}")
    return model_package.model_package_arn

# ✅ Run when script is executed directly
if __name__ == "__main__":
    model_file = "/tmp/model/model.pkl"  # 🔥 Make sure this file exists (from train.py output)
    arn = register_model(model_file)
    print("🎯 Model registered in Model Registry")
