import streamlit as st
import pickle
import pandas as pd

# Load the pickled model
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

st.title("Wine Quality Prediction App")

# Set the model path directly (no input box in UI)
model_path = "model.pkl"
model = None
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the feature names (update these to match your dataset)
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# Define reasonable min/max/default values for sliders (customize as needed)
feature_ranges = {
    "fixed acidity": (4.0, 16.0, 8.0),
    "volatile acidity": (0.1, 1.5, 0.5),
    "citric acid": (0.0, 1.0, 0.3),
    "residual sugar": (0.5, 16.0, 2.5),
    "chlorides": (0.01, 0.2, 0.05),
    "free sulfur dioxide": (1, 72, 15),
    "total sulfur dioxide": (6, 289, 46),
    "density": (0.990, 1.004, 0.996),
    "pH": (2.8, 4.0, 3.3),
    "sulphates": (0.3, 2.0, 0.7),
    "alcohol": (8.0, 15.0, 10.0)
}

# The order must match the training data
all_feature_names = feature_names

if model:
    st.header("Input Wine Features")
    input_data = []
    for feature in feature_names:
        min_val, max_val, default_val = feature_ranges[feature]
        value = st.slider(feature, float(min_val), float(max_val), float(default_val))
        input_data.append(value)

    if st.button("Predict Quality"):
        try:
            # Ensure input DataFrame columns match model training features exactly
            input_df = pd.DataFrame([input_data], columns=all_feature_names)
            prediction = model.predict(input_df)[0]
            label = "Good" if prediction == 1 else "Bad"
            st.success(f"Predicted Wine Quality: {label} ({prediction})")
        except Exception as e:
            st.error(f"Prediction error: {e}")

            

