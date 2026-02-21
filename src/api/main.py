from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from pathlib import Path

app = FastAPI(title="FinSight API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    regime: str


@app.get("/")
def root():
    return {
        "message": "FinSight API - Market Regime Classification",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = 1
        probability = 0.65
        regime = "bullish" if prediction == 1 else "bearish"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            regime=regime
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
