from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models_pydantic import StudentInput, PredictionOut
from .predict import predict_from_dict
from .db import save_prediction, get_recent_predictions
import time

app = FastAPI(title='Student AI Learning Assistant')

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"status":"ok", "service":"student-ai"}

@app.post("/predict", response_model=PredictionOut)
def predict(student: StudentInput):
    try:
        payload = student.dict()
        result = predict_from_dict(payload)
        # save record with timestamp
        record = {**payload, "prediction": result["prediction"], "probabilities": result.get("probabilities"), "ts": time.time()}
        save_prediction(record)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def recent_predictions(limit: int = 20):
    rows = get_recent_predictions(limit)
    for r in rows:
        r["_id"] = str(r["_id"])
    return rows
