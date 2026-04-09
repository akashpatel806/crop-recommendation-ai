"""
train_model.py
==============
Train a RandomForest crop recommendation model using the Kaggle
Crop Recommendation dataset (Crop_recommendation.csv).

Features used:  temperature, humidity, rainfall
  - 'rainfall' maps to your soilMoisture (0-100 %) scaled × 3 → mm

Run:
    python train_model.py
Produces:
    crop_model.pkl   – trained model + metadata
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

CSV_PATH   = "Crop_recommendation.csv"
MODEL_PATH = "crop_model.pkl"

# ─────────────────────────────────────────────
# 1. Load & validate dataset
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"\n[ERR] {csv_path} not found!")
        print("      Download it from:")
        print("      https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print("      and place it in this folder.\n")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[OK]  Loaded dataset: {len(df)} rows, {df['label'].nunique()} crops")
    print(f"      Crops: {sorted(df['label'].unique())}\n")
    return df


# ─────────────────────────────────────────────
# 2. Prepare features
# ─────────────────────────────────────────────
FEATURES = ["temperature", "humidity", "rainfall"]

def prepare_features(df: pd.DataFrame):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"❌  Missing columns in CSV: {missing}")
        sys.exit(1)

    X = df[FEATURES].values
    y = df["label"].values
    return X, y


# ─────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────
def train(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[ACC] Test Accuracy : {acc*100:.2f}%")

    cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy")
    print(f"[CV]  5-fold CV Acc : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature importance
    importances = model.feature_importances_
    for feat, imp in zip(FEATURES, importances):
        print(f"    {feat:20s}: {imp*100:.1f}%")

    return model, le


# ─────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────
def save_model(model, le, path: str):
    bundle = {
        "model":    model,
        "encoder":  le,
        "features": FEATURES,
        "classes":  list(le.classes_),
        "version":  "1.0",
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[OK]  Model saved -> {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df         = load_data(CSV_PATH)
    X, y       = prepare_features(df)
    model, le  = train(X, y)
    save_model(model, le, MODEL_PATH)
    print("\n[DONE] Training complete!\n")
