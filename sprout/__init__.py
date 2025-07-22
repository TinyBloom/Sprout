from flask import Flask
from flask_migrate import Migrate
from flask_restx import Api

from sprout.config import Config
from sprout.extensions import db, ma, make_celery
from sprout.resources.case import CaseListResource, CaseResource  # noqa: F401
from sprout.resources.job import JobListResource, JobResource  # noqa: F401
from sprout.resources.ai import (
    FileUploadResource,
    PredictResource,
    TrainResource,
)  # noqa: F401
from sprout.routes import ai_bp, api_ns


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["result_backend"] = Config.result_backend
    app.config["broker_url"] = Config.broker_url
    app.config["beat_schedule"] = Config.beat_schedule

    # Initialize extensions
    db.init_app(app)
    ma.init_app(app)
    migrate = Migrate(app, db)
    migrate.init_app(app, db)

    # Register blueprints and routes
    app.register_blueprint(ai_bp, url_prefix="/api")

    # Initialize Flask-RESTx API
    api = Api(
        app,
        version="0.1",
        title="Project Sprout's API",
        description="Sprout APIs",
        doc="/docs",
    )

    # Register the /api namespace
    api.add_namespace(api_ns)

    with app.app_context():
        db.create_all()

    return app


celery = make_celery(create_app())
celery.conf.timezone = "UTC"
