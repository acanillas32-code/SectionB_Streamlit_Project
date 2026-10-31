import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

def predict_one(values):
    model = load_model()
    meta = load_metadata()
    row = pd.DataFrame([values], columns=meta["feature_columns"])
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.named_steps["classifier"].classes_)
        if 1 in classes:
            probability = float(model.predict_proba(row)[0][classes.index(1)])
    return prediction, probability
