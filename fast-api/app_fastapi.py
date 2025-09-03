from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle

# Define the POST input structure
class WineFeatures(BaseModel):
    features: list[float] = Field(..., description="List of 11 numerical features")

# Create FastAPI instance
app = FastAPI(title="Power Peta ML API")

# Load your model at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {
        "title": "🍷 Power Peta ML API",
        "description": "This API serves a trained ML model for wine quality prediction.",
        "usage": "POST /predict with JSON {features:[...]} for predictions"
    }

@app.post("/predict")
def predict(data: WineFeatures):
    features = data.features
    if len(features) != 11:
        raise HTTPException(status_code=400, detail="Please provide 11 features in the correct order")
    prediction = model.predict([features])
    result = {"prediction": int(prediction)}
    if hasattr(model, "predict_proba"):
        result["probabilities"] = model.predict_proba([features]).tolist()
    return result

# To run: uvicorn main:app --host 0.0.0.0 --port 5000
