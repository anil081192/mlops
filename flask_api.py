import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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

        # Check if model supports predict_proba (e.g. classifiers like RandomForest, LogisticRegression)
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
