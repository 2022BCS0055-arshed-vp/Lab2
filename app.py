from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")

@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])[0]
    return {
        "name": "Arshed V P",
        "roll_no": "2022BCS0055",
        "wine_quality": int(round(prediction))
    }
