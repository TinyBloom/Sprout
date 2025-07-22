import os

import uuid
from datetime import datetime

from flask_restx import Namespace, Resource, fields, reqparse
from flask import request
from minio import S3Error
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from sprout import db
from sprout.isolation_forest.model import (
    predict_with_isolation_forest,
    train_isolation_forest_model,
)
from sprout.isolation_forest.model_config import ModelConfig
from sprout.models import Dataset, Job, Model, ModelFile, TrainingInfo, Case
from sprout.routes import api_ns
from sprout.storage.storage import MinIOModelStorage


UPLOAD_FOLDER = "/tmp/uploads"
ALLOWED_EXTENSIONS = {"csv"}
MINIO_URL = "minio://"
BUCKET_NAME = "health"
BUCKET = MINIO_URL + BUCKET_NAME + "/"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


file_upload_parser = api_ns.parser()
file_upload_parser.add_argument(
    "file", location="files", type=FileStorage, required=True, help="The file to upload"
)
file_upload_parser.add_argument(
    "robot_id", location="form", type=str, required=True, help="ID of the robot"
)
file_upload_parser.add_argument(
    "desc", location="form", type=str, required=True, help="Description of the file"
)


@api_ns.route("/ai/resource/upload")
class FileUploadResource(Resource):
    @api_ns.expect(file_upload_parser)
    def post(self):
        """
        API endpoint to upload a single file to MinIO and store the information in the database.

        Request parameters:
        - robot_id: Form parameter, robot ID
        - file: File object

        Returns:
        - JSON response containing the upload status and dataset_id
        """
        args = file_upload_parser.parse_args()
        # Check if the file is included
        if "file" not in args:
            return {"success": False, "error": "No file uploaded"}, 400
        file = args["file"]

        if file.filename == "":
            return {"success": False, "error": "Empty file name"}, 400

        # Get robot_id parameter
        robot_id = args.get("robot_id")
        if not robot_id:
            return {"success": False, "error": "Missing robot_id parameter"}, 400

        # Get description
        description = args.get("desc")

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                # Generate UUID as a unique filename and dataset_id
                dataset_id = str(uuid.uuid4())
                unique_filename = f"{dataset_id}_{filename}"
                temp_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

                # Save the file to a temporary directory
                file.save(temp_file_path)

                # Upload the file to MinIO using put_object
                file_size = os.stat(temp_file_path).st_size
                storage = MinIOModelStorage()
                with open(temp_file_path, "rb") as file_data:
                    storage.upload_model(file_data, BUCKET_NAME, unique_filename)

                # Delete the temporary file after upload
                os.remove(temp_file_path)

                # File path in MinIO
                file_path = f"{BUCKET}/{unique_filename}"

                # Save to database
                new_dataset = Dataset(
                    dataset_id=dataset_id,
                    file_path=file_path,
                    robot_id=robot_id,
                    description=description,
                    status="Done",
                )

                db.session.add(new_dataset)
                db.session.commit()

                return (
                    {
                        "success": True,
                        "message": "File upload successful",
                        "dataset_id": dataset_id,
                        "file_path": file_path,
                        "robot_id": robot_id,
                    }
                ), 200

            except S3Error as e:
                # Delete the temporary file if an error occurs
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                return {
                    "success": False,
                    "error": f"Error uploading to MinIO: {str(e)}",
                }, 500

            except Exception as e:
                # Ensure the temporary file is cleaned up in case of any error
                if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return {"success": False, "error": f"Server error: {str(e)}"}, 500
        else:
            return {"success": False, "error": "Unsupported file type"}, 400


# Define the request model for Swagger documentation
train_request_model = api_ns.model(
    "TrainRequest",
    {
        "case_id": fields.String(required=True, description="ID of the case"),
        "dataset_id": fields.String(required=True, description="ID of the dataset"),
        "robot_id": fields.String(
            required=True, description="ID of the robot associated with the training"
        ),
        "desc": fields.String(
            required=False, description="Description of the training request"
        ),
        "case_id": fields.String(
            required=True, description="ID of the case associated with the model"
        ),
        "features": fields.List(
            fields.String,
            required=True,
            description="List of features to use for training",
        ),
        "hyperparameter": fields.Raw(
            required=False, description="Optional hyperparameter configuration"
        ),
    },
)


@api_ns.route("/ai/train")
class TrainResource(Resource):
    @api_ns.expect(train_request_model)
    @api_ns.response(200, "Success")
    @api_ns.response(400, "Validation Error")
    def post(self):
        """
        API endpoint to train a model.
        """
        data = request.get_json()

        case_id = data.get("case_id")
        if not case_id:
            return {"success": False, "error": "Missing case_id parameter"}, 400

        dataset_id = data.get("dataset_id")
        if not dataset_id:
            return {"success": False, "error": "Missing dataset_id parameter"}, 400

        # Get robot_id parameter
        robot_id = data.get("robot_id")
        if not robot_id:
            return {"success": False, "error": "Missing robot_id parameter"}, 400

        hyperparameter = data.get("hyperparameter")

        if not hyperparameter:
            hyperparameter = ModelConfig()
        else:
            try:
                hyperparameter = ModelConfig.from_json(hyperparameter)

            except (TypeError, ValueError) as e:
                return {
                    "success": False,
                    "error": f"Invalid model parameters: {str(e)}",
                }, 400

        case = Case.query.get(case_id)
        if not case:
            abort(404, message="Case not found")

        dataset_model = Dataset.query.filter_by(
            dataset_id=dataset_id, robot_id=robot_id
        ).first()
        if not dataset_model:
            return {
                "success": False,
                "error": "Can't find dataset file, please check dataset_id and robot_id",
            }, 400

        file_path = dataset_model.file_path
        features = data.get("features")

        resource_file = file_path.replace(BUCKET, "")
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
            dataset_id=dataset_id,
            case_id=case_id,
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

        return {"train_result": "success", "model_id": new_model.model_id}, 200


# Define the request model for Swagger documentation
predict_request_model = api_ns.model(
    "PredictRequest",
    {
        "model_id": fields.String(required=True, description="ID of the model"),
        "training_data": fields.Raw(
            required=True, description="Optional hyperparameter configuration"
        ),
    },
)


@api_ns.route("/ai/predict")
class PredictResource(Resource):
    @api_ns.expect(predict_request_model)
    @api_ns.response(200, "Success")
    @api_ns.response(400, "Validation Error")
    def post(self):
        """
        API endpoint to detect anomalies in new data.
        """
        data = request.get_json()
        model_id_value = data.get("model_id")
        model = ModelFile.query.filter_by(
            model_id=model_id_value, file_type="model"
        ).first()
        scaler = ModelFile.query.filter_by(
            model_id=model_id_value, file_type="scaler"
        ).first()
        scores = ModelFile.query.filter_by(
            model_id=model_id_value, file_type="scores"
        ).first()
        training_data = data.get("training_data")
        model_path = model.file_path.replace(BUCKET, "")
        scaler_path = scaler.file_path.replace(BUCKET, "")
        scores_path = scores.file_path.replace(BUCKET, "")

        predict_result = predict_with_isolation_forest(
            model_path, training_data, scaler_path, scores_path
        )
        filtered_data = [
            sublist[:3] + sublist[5:] for sublist in predict_result.tolist()
        ]
        return {"predict_result": filtered_data}, 200
