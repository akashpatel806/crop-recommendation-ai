"""
app.py
======
Flask REST API for Crop Recommendation.

Endpoints:
  GET /health          – liveness check
  GET /recommend       – latest MongoDB reading → crop prediction
  GET /history?n=20    – last N readings with predictions
  POST /predict        – manual body {"temperature":25,"humidity":70,"soilMoisture":55}

Run:
    python app.py
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

# ─── Bootstrap ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow Streamlit / any frontend to call us

MODEL_PATH = "crop_model.pkl"

# ─── Auto-train if model missing (needed on first cloud deploy) ───────────────
def auto_train():
    """Train model from CSV if crop_model.pkl doesn't exist."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    CSV = "Crop_recommendation.csv"
    if not os.path.exists(CSV):
        # Try to download it
        print("[INFO] Downloading dataset for training...")
        import urllib.request
        url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
        urllib.request.urlretrieve(url, CSV)

    df = pd.read_csv(CSV)
    X  = df[["temperature", "humidity", "rainfall"]].values
    y  = df["label"].values

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y_enc)

    bundle = {"model": model, "encoder": le,
              "features": ["temperature", "humidity", "rainfall"],
              "classes": list(le.classes_), "version": "1.0"}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[OK] Auto-trained model saved -> {MODEL_PATH}")


# ─── Load model once at startup ───────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model not found. Auto-training now (first deploy)...")
        auto_train()
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    print(f"[OK] Model loaded. Crops: {bundle['classes']}")
    return bundle["model"], bundle["encoder"], bundle["classes"]


# Load model — catch errors so gunicorn doesn't crash silently
_model, _encoder, _classes = None, None, None
try:
    _model, _encoder, _classes = load_model()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

# ─── Root route — shows API status ──────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    status = "ok" if _model is not None else "model_not_loaded"
    return jsonify({
        "app": "Crop Recommendation AI",
        "status": status,
        "endpoints": ["/health", "/recommend", "/history", "/predict"],
        "model_loaded": _model is not None,
    })

# ─── MongoDB connection (lazy, with retry) ───────────────────────────────────
_mongo_client = None

def get_collection():
    global _mongo_client
    import pymongo

    if _mongo_client is None:
        uri = os.getenv("MONGODB_URI")
        _mongo_client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10_000)

    db_name  = os.getenv("DB_NAME", "soil data")
    col_name = os.getenv("COLLECTION_NAME", "sensordatas")
    return _mongo_client[db_name][col_name]


# ─── Core prediction helper ──────────────────────────────────────────────────
def make_prediction(temperature: float, humidity: float, soil_moisture: float) -> dict:
    rainfall_equiv = soil_moisture * 3.0
    x = np.array([[temperature, humidity, rainfall_equiv]])

    pred_enc  = _model.predict(x)[0]
    pred_crop = _encoder.inverse_transform([pred_enc])[0]
    probas    = _model.predict_proba(x)[0]

    ranked = sorted(zip(_classes, probas.tolist()), key=lambda t: -t[1])

    return {
        "recommended_crop": pred_crop,
        "confidence": f"{probas.max()*100:.1f}%",
        "top_crops": [
            {"crop": c, "confidence": f"{p*100:.1f}%", "probability": round(p, 4)}
            for c, p in ranked[:5]
        ],
        "sensor_input": {
            "temperature":  temperature,
            "humidity":     humidity,
            "soilMoisture": soil_moisture,
            "rainfall_equiv": rainfall_equiv,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if _model is not None else "model_not_loaded",
        "model": "RandomForest",
        "model_loaded": _model is not None,
        "crops_supported": len(_classes) if _classes else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/recommend", methods=["GET"])
def recommend():
    """Read latest MongoDB sensor doc and return crop recommendation."""
    if _model is None:
        return jsonify({"error": "Model not loaded yet. Check /health for status."}), 503
    try:
        col    = get_collection()
        latest = col.find_one(sort=[("timestamp", -1)])

        if not latest:
            return jsonify({"error": "No sensor data found in MongoDB"}), 404

        temp     = float(latest.get("temperature",  25.0))
        humidity = float(latest.get("humidity",     70.0))
        moisture = float(latest.get("soilMoisture", 50.0))
        ts       = latest.get("timestamp", "unknown")

        result = make_prediction(temp, humidity, moisture)
        result["data_timestamp"] = str(ts)
        result["source"] = "mongodb_live"

        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/history", methods=["GET"])
def history():
    """Return the last N MongoDB readings with predictions."""
    try:
        n   = int(request.args.get("n", 20))
        n   = min(max(n, 1), 200)  # clamp 1–200
        col = get_collection()

        docs = list(col.find(
            {},
            {"_id": 0, "soilMoisture": 1, "temperature": 1, "humidity": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(n))

        enriched = []
        for doc in docs:
            temp     = float(doc.get("temperature",  25.0))
            humidity = float(doc.get("humidity",     70.0))
            moisture = float(doc.get("soilMoisture", 50.0))

            pred = make_prediction(temp, humidity, moisture)
            enriched.append({
                "timestamp":        str(doc.get("timestamp", "")),
                "temperature":      temp,
                "humidity":         humidity,
                "soilMoisture":     moisture,
                "recommended_crop": pred["recommended_crop"],
                "confidence":       pred["confidence"],
            })

        return jsonify({"count": len(enriched), "readings": enriched})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict_manual():
    """Manual prediction with JSON body — no MongoDB needed."""
    try:
        body = request.get_json(force=True)
        temp     = float(body.get("temperature",  25.0))
        humidity = float(body.get("humidity",     70.0))
        moisture = float(body.get("soilMoisture", 50.0))

        result = make_prediction(temp, humidity, moisture)
        result["source"] = "manual_input"
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n[API] Crop Recommendation API starting on http://localhost:{port}")
    print(f"    Endpoints:")
    print(f"      GET  /health")
    print(f"      GET  /recommend")
    print(f"      GET  /history?n=20")
    print(f"      POST /predict  (body: {{temperature, humidity, soilMoisture}})\n")
    app.run(debug=True, host="0.0.0.0", port=port)
