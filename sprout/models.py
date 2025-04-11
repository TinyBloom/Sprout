from sprout.extensions import db
from sqlalchemy.dialects.postgresql import JSON, VARCHAR
import uuid
from datetime import datetime


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(80), unique=True, nullable=False)
    params = db.Column(db.JSON, nullable=True)
    status = db.Column(db.String(50), default="PENDING")

    def __repr__(self):
        return f"<Job {self.job_id}>"


class Dataset(db.Model):
    __tablename__ = "datasets"

    dataset_id = db.Column(
        db.String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    file_path = db.Column(db.Text, nullable=False)
    robot_id = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(48), default="pending")
    created_at = db.Column(db.TIMESTAMP, default=datetime.now)
    description = db.Column(db.Text)

    def __repr__(self):
        return f"<Dataset {self.file_path} - {self.status}>"


class Model(db.Model):
    __tablename__ = "models"

    model_id = db.Column(
        VARCHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name = db.Column(db.String(255), nullable=False)
    robot_id = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.TIMESTAMP, default=datetime.now)
    dataset_id = db.Column(db.String(36))

    training_info = db.relationship(
        "TrainingInfo", back_populates="model", cascade="all, delete-orphan"
    )
    model_files = db.relationship(
        "ModelFile", back_populates="model", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Model {self.name}>"


class TrainingInfo(db.Model):
    __tablename__ = "training_info"

    training_id = db.Column(
        VARCHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id = db.Column(VARCHAR(36), db.ForeignKey("models.model_id"), nullable=False)
    robot_id = db.Column(db.String(255), nullable=False)
    hyperparameter = db.Column(JSON, nullable=False)
    training_status = db.Column(db.String(48), default="pending")
    created_at = db.Column(db.TIMESTAMP, default=datetime.now)
    started_at = db.Column(db.TIMESTAMP)
    completed_at = db.Column(db.TIMESTAMP)

    model = db.relationship("Model", back_populates="training_info")

    def __repr__(self):
        return f"<TrainingInfo {self.training_id} - {self.training_status}>"


class ModelFile(db.Model):
    __tablename__ = "model_files"

    file_id = db.Column(
        VARCHAR(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id = db.Column(VARCHAR(36), db.ForeignKey("models.model_id"), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(48), default="training_data")
    file_path = db.Column(db.Text, nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)
    file_format = db.Column(db.String(32), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.TIMESTAMP, default=datetime.now)

    model = db.relationship("Model", back_populates="model_files")

    def __repr__(self):
        return f"<ModelFile {self.file_name}>"
