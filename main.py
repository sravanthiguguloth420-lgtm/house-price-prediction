from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# Create FastAPI app
app = FastAPI(title="House Price Prediction")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "house_model.pkl")

# HTML file path
HTML_PATH = os.path.join(
    BASE_DIR, "templates", "index.html"
)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "house_model.pkl not found. "
        "Run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


# Home page
@app.get("/")
def home():
    return FileResponse(HTML_PATH)


# Prediction endpoint
@app.post("/predict")
def predict(
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...)
):
    # Prepare input data
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["area", "bedrooms", "bathrooms"]
    )

    # Predict house price
    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }


# Check API
@app.get("/health")
def health():
    return {"status": "API is running"}