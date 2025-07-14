from datetime import datetime

from sprout.extensions import db
from sprout.models import Model, ModelFile, TrainingInfo

from sprout.isolation_forest.model import (
    train_isolation_forest_model,
)

from sprout.isolation_forest.model_config import ModelConfig

MINIO_URL = "minio://"
BUCKET_NAME = "health"
BUCKET = MINIO_URL + BUCKET_NAME + "/"


def isolation_forest_service(resource_file, hyperparameter, features, robot_id):

    if not features:
        features = ["torque", "temperature", "current"]

    if not hyperparameter:
        hyperparameter = ModelConfig()
    else:

        hyperparameter = ModelConfig.from_json(hyperparameter)

    result = train_isolation_forest_model(
        BUCKET_NAME,
        resource_file,
        features,
        hyperparameter.n_estimators,
        hyperparameter.contamination,
        hyperparameter.random_state,
    )

    model_filename = result["model_filename"]
    scaler_filename = result["scaler_filename"]
    scores_filename = result["scores_filename"]
    model_size = result["model_size"]
    model_hash = result["model_hash"]
    scaler_size = result["scaler_size"]
    scaler_hash = result["scaler_hash"]
    scores_size = result["scores_size"]
    scores_hash = result["scores_hash"]

    new_model = Model(
        name="IsolationForest",
        robot_id=robot_id,
        description="One more model added",
        created_at=datetime.now(),
    )

    new_training = TrainingInfo(
        model=new_model,
        robot_id=new_model.robot_id,
        hyperparameter=hyperparameter.to_json(),
        training_status="complete",
        created_at=datetime.now(),
    )

    file_path = BUCKET + model_filename
    new_model_file = ModelFile(
        model=new_model,
        file_name=model_filename,
        file_type="model",
        file_path=file_path,
        file_size=model_size,
        file_format="joblib",
        file_hash=model_hash,
        created_at=datetime.now(),
    )

    file_path = BUCKET + scaler_filename
    new_scaler_file = ModelFile(
        model=new_model,
        file_name=scaler_filename,
        file_type="scaler",
        file_path=file_path,
        file_size=scaler_size,
        file_format="joblib",
        file_hash=scaler_hash,
        created_at=datetime.now(),
    )

    file_path = BUCKET + scores_filename
    new_scores_file = ModelFile(
        model=new_model,
        file_name=scores_filename,
        file_type="scores",
        file_path=file_path,
        file_size=scores_size,
        file_format="joblib",
        file_hash=scores_hash,
        created_at=datetime.now(),
    )

    db.session.add(new_model)
    db.session.add(new_training)
    db.session.add(new_model_file)
    db.session.add(new_scaler_file)
    db.session.add(new_scores_file)
    db.session.commit()

    return {
        "train_result": "success",
        "model_id": new_model.model_id,
    }
