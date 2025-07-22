import uuid

from flask import Blueprint, request, jsonify
from flask_restx import Namespace
from werkzeug.utils import secure_filename

from sprout import db
from sprout.models import Job, ModelFile

api_ns = Namespace("api", description="Sprout API namespace")

ai_bp = Blueprint("ai_bp", __name__)


@ai_bp.route("/ai/run", methods=["POST"])
def run_ai_task():
    """
    API Endpoint to trigger an AI task.
    """
    from sprout.celery_worker import run_ai_job

    data = request.get_json()
    params = data.get("params", {})

    # Create a job in the database
    new_job = Job(job_id=str(uuid.uuid4()), params=params, status="PENDING")
    db.session.add(new_job)
    db.session.commit()

    jobs = Job.query.all()
    for job in jobs:
        print(job.id, job.job_id, job.params)

    # Trigger the Celery task
    run_ai_job.apply_async(args=[new_job.id, params])

    return jsonify({"message": "AI job started", "job_id": new_job.id}), 202


@ai_bp.route("/ai/status/<int:job_id>", methods=["GET"])
def get_ai_job_status(job_id):
    """
    API Endpoint to check the status of an AI task.
    """
    job = Job.query.get(job_id)
    if not job:
        return jsonify({"message": "Job not found"}), 404

    return jsonify({"job_id": job.id, "status": job.status, "params": job.params})


# API used for test
@ai_bp.route("/ai/get", methods=["GET"])
def get():
    models = ModelFile.query.all()

    for model in models:
        print(model.model_id)

    return jsonify({"message": "Done"}), 200
