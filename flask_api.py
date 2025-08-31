import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your model (replace 'model.pkl' with your actual file)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Assume input is a list of features
    features = data.get('features')
    if features is None:
        return jsonify({'error': 'No features provided'}), 400
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

