# 🌾 Crop Recommendation AI

An end-to-end machine learning system that reads live soil sensor data from ESP32 via MongoDB Atlas and recommends the best crop to grow.

## Architecture

```
ESP32 (DHT11 + Soil Sensor)
         ↓
 MongoDB Atlas (soil data / sensordatas)
         ↓
 Python ML Pipeline (train_model.py)
         ↓
 Flask REST API (app.py)  ←──→  Streamlit Dashboard (dashboard.py)
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Kaggle Dataset
Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset  
Place `Crop_recommendation.csv` in this folder.

### 3. Train the model
```bash
python train_model.py
```
Expected output: ~94–97% accuracy on test set.

### 4. Test with live MongoDB data
```bash
python predict_from_mongo.py
```

### 5. Start the Flask API
```bash
python app.py
```
API runs at http://localhost:5000

### 6. Start the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Dashboard opens at http://localhost:8501

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API liveness check |
| GET | `/recommend` | Latest MongoDB reading → crop prediction |
| GET | `/history?n=50` | Last N readings with predictions |
| POST | `/predict` | Manual input body → prediction |

### Example Response (`/recommend`)
```json
{
  "recommended_crop": "maize",
  "confidence": "67.3%",
  "top_crops": [
    {"crop": "maize",  "confidence": "67.3%", "probability": 0.673},
    {"crop": "cotton", "confidence": "18.1%", "probability": 0.181},
    {"crop": "wheat",  "confidence":  "9.2%", "probability": 0.092}
  ],
  "sensor_input": {
    "temperature": 25.3,
    "humidity": 73.9,
    "soilMoisture": 59,
    "rainfall_equiv": 177.0
  }
}
```

---

## How It Works

| Sensor Field | Dataset Column | Mapping |
|---|---|---|
| `temperature` | `temperature` | Direct (°C) |
| `humidity` | `humidity` | Direct (%) |
| `soilMoisture` | `rainfall` | `soilMoisture × 3.0` → mm |

The model is trained on the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) which contains 2200 records across 22 crop types.

---

## Improving Over Time

Once farmers report which crop they grew, you can add labeled data to MongoDB and retrain:

```python
# Add farmer feedback
collection.update_one(
    {"_id": reading_id},
    {"$set": {"crop_grown": "wheat"}}
)

# Retrain model with your local labeled data
labeled = list(collection.find({"crop_grown": {"$exists": True}}))
# Add to CSV → python train_model.py
```

---

## Files

| File | Purpose |
|------|---------|
| `train_model.py` | Train RandomForest, save as `crop_model.pkl` |
| `predict_from_mongo.py` | CLI: fetch latest reading → predict |
| `app.py` | Flask REST API |
| `dashboard.py` | Streamlit visual dashboard |
| `.env` | MongoDB connection string (keep secret!) |
| `requirements.txt` | Python dependencies |
