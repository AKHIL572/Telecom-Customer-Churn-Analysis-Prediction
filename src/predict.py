"""
predict.py

Inference pipeline for Telecom Churn Prediction.

FIX vs. the original version: the original predict.py fed raw input
directly into preprocessor.transform() without ever applying the
engineered features (tenure_group, AutoPayment, HighRiskSegment,
HighCharges) that the model was actually trained on. This would either
error (ColumnTransformer expects those columns) or silently produce
wrong predictions if the underlying sklearn version didn't error out.

Fix: call the SAME engineer_features() function used by train.py, using
the SAME HighCharges threshold saved during training -- guaranteeing
inference-time features always match training-time features exactly.
"""

import joblib
import pandas as pd
import os

from feature_engineering import engineer_features

MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
HIGH_CHARGES_THRESHOLD_PATH = os.path.join(
    MODEL_DIR, "high_charges_threshold.pkl")
DECISION_THRESHOLD_PATH = os.path.join(MODEL_DIR, "decision_threshold.pkl")


def load_artifacts():
    """
    Load the trained pipeline and all supporting artifacts needed to
    reproduce training-time feature engineering exactly.
    """
    for path, label in [
        (MODEL_PATH, "Trained model"),
        (FEATURES_PATH, "Model feature list"),
        (HIGH_CHARGES_THRESHOLD_PATH, "HighCharges threshold"),
        (DECISION_THRESHOLD_PATH, "Decision threshold"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    high_charges_threshold = joblib.load(HIGH_CHARGES_THRESHOLD_PATH)
    decision_threshold = joblib.load(DECISION_THRESHOLD_PATH)

    return model, feature_columns, high_charges_threshold, decision_threshold


def predict_customer(input_data: dict):
    """
    Predict churn for a single customer.

    Parameters
    ----------
    input_data : dict
        Raw customer attributes (same fields as the original dataset,
        minus customerID, TotalCharges, and Churn). Engineered features
        are added automatically -- do not pass them in.

    Returns
    -------
    dict with prediction and probability, at the model's fixed decision
    threshold (not the default 0.5 -- see docs/executive_summary.md for
    why recall is prioritized).
    """
    model, feature_columns, high_charges_threshold, decision_threshold = load_artifacts()

    input_df = pd.DataFrame([input_data])

    # Apply the EXACT same feature engineering used at training time,
    # with the EXACT same HighCharges threshold learned from training data
    input_df = engineer_features(
        input_df, high_charges_threshold=high_charges_threshold)

    # Ensure column order/presence matches what the model was trained on
    missing = set(feature_columns) - set(input_df.columns)
    if missing:
        raise ValueError(f"Input is missing required fields: {missing}")
    input_df = input_df[feature_columns]

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= decision_threshold)

    return {
        "Churn Prediction": prediction,
        "Churn Probability": round(float(probability), 4),
        "Decision Threshold Used": decision_threshold
    }


if __name__ == "__main__":
    # Example customer. Note: TotalCharges is intentionally omitted --
    # the model does not use it (dropped for multicollinearity, see
    # 3_preprocessing_and_modeling.ipynb).
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5
    }

    output = predict_customer(sample_customer)
    print("\nPrediction Result:")
    print(output)
