import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route (API name / description)
@app.route("/", methods=["GET"])
def home():
    return """
    <h1>🍷 Power Peta ML API</h1>
    <p>This API serves a trained ML model for wine quality prediction.</p>
    <p>Use the <b>/predict</b> endpoint with JSON data to get predictions.</p>
    <hr>
    <pre>
    Example request:
    curl -X POST http://127.0.0.1:5000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"features":[8.0,0.5,0.46,2.5,0.05,15.0,46.0,1.0,3.3,0.7,10.0]}'
    </pre>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract features
        features = data.get("features")
        if features is None or len(features) != 11:
            return jsonify({"error": "Please provide 11 features in the correct order"}), 400

        # Make prediction
        prediction = model.predict([features])[0]

        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([features]).tolist()[0]
            return jsonify({
                "prediction": int(prediction),
                "probabilities": probabilities
            })
        else:
            return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
