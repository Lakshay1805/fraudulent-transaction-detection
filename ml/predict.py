import joblib
import pandas as pd
from feature_engineering import full_feature_pipeline

MODEL_FILE = "artifacts/model.pkl"
THRESHOLD_FILE = "artifacts/threshold.pkl"

model = joblib.load(MODEL_FILE)
threshold = joblib.load(THRESHOLD_FILE)

def prediction(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data = full_feature_pipeline(data)

    probs = model.predict_proba(data)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "prediction": preds,
        "probability": probs
    }