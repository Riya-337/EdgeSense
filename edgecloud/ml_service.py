import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(Path(__file__).with_name("model.joblib"))))
SERVICE_LOCATION = os.environ.get("SERVICE_LOCATION", "LOCAL_EDGE")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = list(bundle["features"])
if list(model.feature_names_in_) != feature_names:
    raise ValueError(f"Model feature metadata does not match artifact: {MODEL_PATH}")

app = FastAPI(title="EdgeSense ML Service")


class PredictionRequest(BaseModel):
    features: dict[str, float]
    simulation: dict[str, Any] | None = None


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "location": SERVICE_LOCATION,
        "model": "EdgeSense Random Forest",
    }


@app.post("/predict")
def predict(data: PredictionRequest) -> dict[str, Any]:
    missing = [name for name in feature_names if name not in data.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model features: {', '.join(missing)}")
    if any(not math.isfinite(data.features[name]) for name in feature_names):
        raise HTTPException(status_code=422, detail="Model features must all be finite numbers")

    sample = pd.DataFrame(
        [{name: data.features[name] for name in feature_names}],
        columns=feature_names,
    )
    start_time = time.perf_counter()
    probabilities = model.predict_proba(sample)[0]
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    anomaly_index = list(model.classes_).index(1)
    confidence = float(probabilities[anomaly_index])

    return {
        "processed_by": SERVICE_LOCATION,
        "model": "EdgeSense Random Forest",
        "is_anomaly": confidence >= 0.5,
        "confidence": round(confidence, 4),
        "inference_time_ms": round(inference_time_ms, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
