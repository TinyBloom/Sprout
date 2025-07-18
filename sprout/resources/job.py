import time

from flask_restx import abort, reqparse, Resource
from sprout import db
from sprout.models import Job
from sprout.routes import api_ns


# Parser for incoming POST data
job_post_parser = reqparse.RequestParser()
job_post_parser.add_argument("params", type=dict, required=False, help="Params must be a dictionary")

# Parser for incoming PUT data
job_put_parser = reqparse.RequestParser()
job_put_parser.add_argument("job_id", type=str, required=True, help="Job ID is required")
job_put_parser.add_argument("params", type=dict, required=False, help="Params must be a dictionary")


@api_ns.route("/jobs")
class JobListResource(Resource):
    def get(self):
        """Get all jobs"""
        jobs = Job.query.all()
        return [job.to_dict() for job in jobs], 200

    def post(self):
        """Create a new job"""
        from sprout.celery_worker import (
            execute_job,
        )  # Import here to prevent circular import

        args = job_post_parser.parse_args()
        job_id = f"job_{int(time.time())}"
        new_job = Job(job_id=job_id, **args)
        db.session.add(new_job)
        db.session.commit()

        # Trigger Celery task
        execute_job.apply_async(args=[job_id, args])

        return new_job.to_dict(), 201


@api_ns.route("/jobs/<int:job_id>")
class JobResource(Resource):
    def get(self, job_id):
        """Get a single job by ID"""
        job = Job.query.get(job_id)
        if not job:
            abort(404, message="Job not found")
        return job.to_dict(), 200

    def put(self, job_id):
        """Update a job"""
        job = Job.query.get(job_id)
        if not job:
            abort(404, message="Job not found")
        args = job_put_parser.parse_args()
        for key, value in args.items():
            setattr(job, key, value)
        db.session.commit()
        return job.to_dict(), 200

    def delete(self, job_id):
        """Delete a job"""
        job = Job.query.get(job_id)
        if not job:
            abort(404, message="Job not found")
        db.session.delete(job)
        db.session.commit()
        return {"message": "Deleted successfully"}, 200
