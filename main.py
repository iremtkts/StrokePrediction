# app/main.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "models/stroke_model_calibrated.pkl")

app = FastAPI(title="Stroke Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        FEATURES = artifact.get("feature_order") or [
            "gender","age","hypertension","heart_disease","work_type",
            "avg_glucose_level","bmi","smoking_status","ever_married"
        ]
        VERSIONS = artifact.get("versions", {})
    else:
      
        model = artifact
        FEATURES = [
            "gender","age","hypertension","heart_disease","work_type",
            "avg_glucose_level","bmi","smoking_status","ever_married"
        ]
        VERSIONS = {}
except Exception as e:
    raise RuntimeError(f"Model yüklenemedi: {e}")


class Patient(BaseModel):
    gender: conint(ge=-1, le=1)                # {-1,0,1}
    age: confloat(ge=0, le=120)
    hypertension: conint(ge=0, le=1)
    heart_disease: conint(ge=0, le=1)
    work_type: conint(ge=-2, le=2)            
    avg_glucose_level: confloat(ge=0, le=500)
    bmi: confloat(ge=5, le=100)
    smoking_status: conint(ge=-1, le=2)
    ever_married: conint(ge=0, le=1)


@app.get("/")
def root():
    return {"ok": True, "try": ["/health", "/docs"]}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "features": FEATURES,
        "versions": VERSIONS
    }

@app.post("/predict")
def predict(p: Patient):
    try:
        row = pd.DataFrame([{k: getattr(p, k) for k in FEATURES}], columns=FEATURES)
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)  
        return {"stroke_proba": proba, "stroke_pred": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
