from flask import Blueprint, request, jsonify

from ml.predict import prediction as model_predict

prediction_bp = Blueprint("prediction", __name__)

@prediction_bp.route("/predict", methods=['POST'])
def predict_route():

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error" : "No input provided."}), 500

        result = model_predict(data)

        return jsonify({
            "status": "success",
            "prediction": result["prediction"].tolist(),
            "probability": result["probability"].tolist()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
