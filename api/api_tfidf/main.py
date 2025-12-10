from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch.nn.functional as F

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ===========================
# 📌 Charger modèles TF-IDF
# ===========================

MODEL_DIR = "models/tfidf"

vectorizer = joblib.load("/app/model/vectorizer.joblib")
classifier = joblib.load("/app/model/classifier.joblib")
label_encoder = joblib.load("/app/model/label_encoder.joblib")

id2label = {i: lab for i, lab in enumerate(label_encoder.classes_)}

# ===========================
# 📌 API
# ===========================

app = FastAPI(title="TF-IDF Service")

class Ticket(BaseModel):
    text: str

@app.post("/predict")
def predict(ticket: Ticket):

    X = vectorizer.transform([ticket.text])
    pred_id = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()

    label = id2label[pred_id]

    return {
        "model": "tfidf",
        "label": label,
        "confidence": float(proba)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
