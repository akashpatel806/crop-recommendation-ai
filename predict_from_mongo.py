"""
predict_from_mongo.py
=====================
Fetch the latest sensor reading from MongoDB Atlas and predict
the best crop to grow, printing a ranked list with confidence %.

Run:
    py predict_from_mongo.py
"""

import os
import pickle
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "crop_model.pkl"

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(path: str):
    if not os.path.exists(path):
        print(f"[ERR] {path} not found. Run `py train_model.py` first.")
        sys.exit(1)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["encoder"], bundle["classes"]


# ─────────────────────────────────────────────
# Connect to MongoDB
# ─────────────────────────────────────────────
def get_latest_reading():
    try:
        import pymongo
    except ImportError:
        print("[ERR] pymongo not installed. Run: py -m pip install pymongo[srv]")
        sys.exit(1)

    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("[ERR] MONGODB_URI not set in .env")
        sys.exit(1)

    db_name  = os.getenv("DB_NAME", "soil data")
    col_name = os.getenv("COLLECTION_NAME", "sensordatas")

    client     = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10_000)
    collection = client[db_name][col_name]

    latest = collection.find_one(sort=[("timestamp", -1)])
    if not latest:
        print("[ERR] No documents found in collection.")
        sys.exit(1)

    print(f"[MONGO] Latest reading from MongoDB:")
    print(f"    Timestamp    : {latest.get('timestamp', 'N/A')}")
    print(f"    Temperature  : {latest.get('temperature', 'N/A')} C")
    print(f"    Humidity     : {latest.get('humidity', 'N/A')} %")
    print(f"    Soil Moisture: {latest.get('soilMoisture', 'N/A')} %\n")

    return {
        "temperature":  float(latest.get("temperature", 25.0)),
        "humidity":     float(latest.get("humidity", 70.0)),
        "soilMoisture": float(latest.get("soilMoisture", 50.0)),
    }


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
def predict(model, encoder, classes, reading: dict) -> None:
    temp     = reading["temperature"]
    humidity = reading["humidity"]
    moisture = reading["soilMoisture"]

    # Map soilMoisture (0-100 %) -> rainfall equivalent (0-300 mm)
    rainfall_equiv = moisture * 3.0

    input_arr = np.array([[temp, humidity, rainfall_equiv]])

    pred_enc  = model.predict(input_arr)[0]
    pred_crop = encoder.inverse_transform([pred_enc])[0]
    probas    = model.predict_proba(input_arr)[0]

    # Rank all crops
    ranked = sorted(
        zip(classes, probas),
        key=lambda x: -x[1]
    )

    print(f"[CROP] Recommended Crop: {pred_crop.upper()}")
    print(f"\n[TOP5] Crop Probabilities:")
    for rank, (crop, prob) in enumerate(ranked[:5], 1):
        bar = "#" * int(prob * 30)
        print(f"    {rank}. {crop:15s} {prob*100:5.1f}%  {bar}")

    print(f"\n[MAP] Sensor -> Model:")
    print(f"    temperature  = {temp}C")
    print(f"    humidity     = {humidity}%")
    print(f"    rainfall     = {rainfall_equiv:.1f} mm  (soilMoisture x 3)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, encoder, classes = load_model(MODEL_PATH)
    reading = get_latest_reading()
    predict(model, encoder, classes, reading)
