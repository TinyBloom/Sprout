from flask_restx import abort, reqparse, Resource
from sprout import db
from sprout.models import Case, Model
from sprout.routes import api_ns

@api_ns.route("/models")
class ModelListResource(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('case_id', type=str, required=True, help='Case ID to filter models')

    @api_ns.expect(parser)
    def get(self):
        args = self.parser.parse_args()
        if not (models := Model.query.filter_by(case_id=args['case_id']).all()):
            return [], 200
        else:
            return [model.to_dict() for model in models], 200
