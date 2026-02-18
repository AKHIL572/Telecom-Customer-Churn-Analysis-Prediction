"""
train.py

Training pipeline for Telecom Churn Prediction.
Retrains model and saves best model to disk.
"""

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_loader import load_data
from preprocessing import clean_data
from feature_engineering import prepare_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "Dataset/churn_dataset.csv"
MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")


# --------------------------------------------------
# Utility: Evaluate Model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return metrics


# --------------------------------------------------
# Training Pipeline
# --------------------------------------------------

def main():

    print("🚀 Starting Training Pipeline...\n")

    # 1️⃣ Load Data
    df = load_data(DATA_PATH)

    # 2️⃣ Clean Data
    df = clean_data(df)

    # 3️⃣ Feature Engineering
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)

    # 4️⃣ Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    results = {}

    # 5️⃣ Train & Evaluate
    for name, model in models.items():
        print(f"\n🔍 Training {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # 6️⃣ Select Best Model (Based on Recall)
    best_model_name = max(results, key=lambda x: results[x]["Recall"])
    print(f"\n🏆 Best Model Selected: {best_model_name}")

    best_model = models[best_model_name]

    # 7️⃣ Save Model & Preprocessor
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\n✅ Model and Preprocessor saved successfully.")
    print("📦 Training pipeline completed.")


if __name__ == "__main__":
    main()
