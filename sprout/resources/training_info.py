from flask_restx import abort, reqparse, Resource
from sprout import db
from sprout.models import TrainingInfo
from sprout.routes import api_ns

@api_ns.route("/training-infos")
class TrainingInfoListResource(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('case_id', type=str, required=True, help='case_id to filter training infos')

    @api_ns.expect(parser)
    def get(self):
        args = self.parser.parse_args()
        if training_infos := TrainingInfo.query.filter_by(
            case_id=args['case_id']
        ).all():
            return [training_info.to_dict() for training_info in training_infos], 200
        else:
            return [], 200