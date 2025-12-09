from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
import requests

app = FastAPI(title="Smart Agent Service")

TFIDF_URL = "http://localhost:8001/predict"
TRANSFORMER_URL = "http://localhost:8002/predict"

class Ticket(BaseModel):
    text: str


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"


@app.post("/predict")
def smart_predict(ticket: Ticket):

    text = ticket.text
    lang = detect_language(text)

    # ==============================
    # 1️⃣ SI FRANÇAIS → Transformer
    # ==============================
    if lang == "fr":
        try:
            response = requests.post(TRANSFORMER_URL, json={"text": text})
            return {
                "language": "fr",
                "model_used": "transformer",
                "result": response.json()
            }
        except:
            return {"error": "Transformer service unavailable"}

    # ==============================
    # 2️⃣ SI ANGLAIS → On teste TF-IDF
    # ==============================
    if lang == "en":
        # On essaie TF-IDF
        try:
            response = requests.post(TFIDF_URL, json={"text": text})
            tfidf_result = response.json()

            # Si TF-IDF renvoie une réponse correcte :
            if "label" in tfidf_result:
                return {
                    "language": "en",
                    "model_used": "tfidf",
                    "result": tfidf_result
                }

        except:
            pass  # Si TF-IDF échoue → fallback Transformer

        # Fallback Transformer
        try:
            response = requests.post(TRANSFORMER_URL, json={"text": text})
            return {
                "language": "en",
                "model_used": "transformer (fallback)",
                "result": response.json()
            }
        except:
            return {"error": "Both models unavailable"}

    # ====================================
    # 3️⃣ AUTRES LANGUES → Transformer seul
    # ====================================
    try:
        response = requests.post(TRANSFORMER_URL, json={"text": text})
        return {
            "language": lang,
            "model_used": "transformer",
            "result": response.json()
        }
    except:
        return {"error": "Transformer unavailable"}
