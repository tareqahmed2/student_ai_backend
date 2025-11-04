from pathlib import Path
import joblib
import json
import pandas as pd

ARTIFACTS = Path(__file__).parents[1] / 'artifacts'
MODEL_PATH = ARTIFACTS / 'model.joblib'
META_PATH = ARTIFACTS / 'meta.json'

_model = None
_meta = None

def load_model():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _meta is None:
        with open(META_PATH, 'r') as f:
            _meta = json.load(f)
    return _model, _meta

def predict_from_dict(payload: dict) -> dict:
    model, meta = load_model()
    df = pd.DataFrame([payload])
    preds = model.predict(df)
    probs = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
    out = {
        "prediction": str(preds[0]),
        "probabilities": probs.tolist()[0] if probs is not None else None
    }
    return out
