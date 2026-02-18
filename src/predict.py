"""
predict.py

Inference pipeline for Telecom Churn Prediction.
Loads trained model and preprocessor to make predictions.
"""

import joblib
import pandas as pd
import os


MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")


def load_artifacts():
    """
    Load trained model and preprocessor.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Preprocessor not found. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def predict_customer(input_data: dict):
    """
    Predict churn for a single customer.

    Parameters:
    -----------
    input_data : dict
        Dictionary containing customer features

    Returns:
    --------
    dict with prediction and probability
    """

    model, preprocessor = load_artifacts()

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Transform using saved preprocessor
    input_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    result = {
        "Churn Prediction": int(prediction),
        "Churn Probability": round(float(probability), 4)
    }

    return result


if __name__ == "__main__":

    # Example customer (modify based on dataset columns)
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
        "MonthlyCharges": 85.5,
        "TotalCharges": 420.3
    }

    output = predict_customer(sample_customer)

    print("\n🔮 Prediction Result:")
    print(output)
