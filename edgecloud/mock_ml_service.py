import os
import random
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Edge Local ML Service")

class SensorData(BaseModel):
    vibration: float
    motor_current: float
    timestamp: float

@app.get("/")
def health_check():
    return {"status": "ok", "location": "LOCAL_EDGE"}

@app.post("/predict")
def predict(data: SensorData):
    # Simulate low-latency edge inference (~10ms - 20ms execution time)
    start_time = time.time()
    time.sleep(random.uniform(0.01, 0.02))

    # Anomaly calculation heuristic (vibration > 7.5 or current > 12.0)
    is_anomaly = data.vibration > 7.5 or data.motor_current > 12.0
    confidence = random.uniform(0.85, 0.99) if is_anomaly else random.uniform(0.90, 0.99)
    severity = "CRITICAL" if data.vibration > 9.0 else ("WARNING" if is_anomaly else "NORMAL")

    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "processed_by": "LOCAL_EDGE",
        "is_anomaly": is_anomaly,
        "severity": severity,
        "confidence": round(confidence, 4),
        "inference_time_ms": execution_time_ms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)