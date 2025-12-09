from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import time
import torch.nn.functional as F
from fastapi.responses import Response

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# =============================================================
# 📌 PROMETHEUS METRICS
# =============================================================

REQUEST_COUNT = Counter("transformer_requests_total", "Total requests to transformer API")
REQUEST_LATENCY = Histogram("transformer_latency_seconds", "Latency of transformer predictions")

# =============================================================
# 📌 CHARGEMENT DU MODELE ENTRAÎNÉ LOCAL
# =============================================================

import os
MODEL_DIR = os.getenv("TRANSFORMER_MODEL_DIR", "model")

print(f"🔄 Loading tokenizer from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"🔄 Loading label encoder from {MODEL_DIR}/label_encoder.joblib")
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")

# id2label utilisé pendant la prédiction
id2label = {i: lab for i, lab in enumerate(label_encoder.keys())}

print(f"🔄 Loading transformer model from {MODEL_DIR}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"🚀 Transformer model loaded on device: {device}")

# =============================================================
# 📌 FASTAPI APP
# =============================================================

app = FastAPI(title="Transformer Service")

class Ticket(BaseModel):
    text: str

# =============================================================
# 📌 HEALTH CHECK
# =============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "device": str(device),
    }

# =============================================================
# 📌 PROMETHEUS METRICS
# =============================================================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# =============================================================
# 📌 PREDICTION ENDPOINT
# =============================================================

@app.post("/predict")
def predict(ticket: Ticket):

    REQUEST_COUNT.inc()
    start = time.time()

    # Tokenization
    inputs = tokenizer(
        ticket.text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    confidence, pred_id = torch.max(probs, dim=1)

    pred_id = pred_id.item()
    confidence = float(confidence.item())

    # Map id → label (ex: 0 → "Hardware", 1 → "Network", …)
    label = id2label[pred_id]

    REQUEST_LATENCY.observe(time.time() - start)

    return {
        "model": "transformer",
        "label": label,
        "confidence": confidence,
    }
