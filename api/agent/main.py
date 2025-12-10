import os
import requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langdetect import detect
from fastapi.middleware.cors import CORSMiddleware

# =============================================================
# 📌 CONFIGURATION FASTAPI + CORS
# =============================================================

app = FastAPI(title="Smart Agent Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # autoriser toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# 📌 SERVIR L’INTERFACE HTML / CSS
# =============================================================

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")




@app.get("/")
def root():
    """Affiche ton interface index.html"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# =============================================================
# 📌 CONFIG SERVICES IA
# =============================================================

TFIDF_URL = os.getenv("TFIDF_URL", "http://localhost:8001/predict")
TRANSFORMER_URL = os.getenv("TRANSFORMER_URL", "http://localhost:8002/predict")

class Ticket(BaseModel):
    text: str

# =============================================================
# 📌 DETECTION DE LANGUE
# =============================================================

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

# =============================================================
# 📌 SMART ROUTER (AGENT IA)
# =============================================================

@app.post("/predict")
def smart_predict(ticket: Ticket):

    text = ticket.text
    lang = detect_language(text)

    print(f"🌍 Texte détecté: {lang}")

    # 🌐 1) Français → Transformer
    if lang == "fr":
        try:
            response = requests.post(TRANSFORMER_URL, json={"text": text})
            return {
                "language": "fr",
                "model": "transformer",
                "label": response.json().get("label"),
                "confidence": response.json().get("confidence")
            }
        except:
            return {"error": "❌ Transformer service unavailable"}

    # 🇬🇧 2) Anglais → TF-IDF d'abord
    if lang == "en":
        try:
            response = requests.post(TFIDF_URL, json={"text": text})
            tfidf_result = response.json()

            if "label" in tfidf_result:
                return {
                    "language": "en",
                    "model": "tfidf",
                    "label": tfidf_result["label"],
                    "confidence": tfidf_result["confidence"],
                }
        except:
            pass  # fallback Transformer

        try:
            response = requests.post(TRANSFORMER_URL, json={"text": text})
            return {
                "language": "en",
                "model": "transformer_fallback",
                "label": response.json().get("label"),
                "confidence": response.json().get("confidence"),
            }
        except:
            return {"error": "❌ Both models unavailable"}

    # 🌏 3) Autres langues → Transformer
    try:
        response = requests.post(TRANSFORMER_URL, json={"text": text})
        return {
            "language": lang,
            "model": "transformer",
            "label": response.json().get("label"),
            "confidence": response.json().get("confidence"),
        }
    except:
        return {"error": "❌ Transformer unavailable"}
