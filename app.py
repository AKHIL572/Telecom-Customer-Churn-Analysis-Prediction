import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Robust paths, resolved relative to THIS file, not the launch directory ---
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

from feature_engineering import engineer_features  # noqa: E402

DATA_PATH = APP_DIR / "Dataset" / "churn_dataset.csv"
MODEL_PATH = APP_DIR / "Models" / "best_model.pkl"
FEATURES_PATH = APP_DIR / "Models" / "model_features.pkl"
HIGH_CHARGES_THRESHOLD_PATH = APP_DIR / "Models" / "high_charges_threshold.pkl"
DECISION_THRESHOLD_PATH = APP_DIR / "Models" / "decision_threshold.pkl"

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")
st.title("Telecom Customer Churn Analysis")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


df = load_data()


# -----------------------------
# LOAD MODEL + ARTIFACTS
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    high_charges_threshold = joblib.load(HIGH_CHARGES_THRESHOLD_PATH)
    decision_threshold = joblib.load(DECISION_THRESHOLD_PATH)
    return model, feature_columns, high_charges_threshold, decision_threshold


try:
    model, feature_columns, high_charges_threshold, decision_threshold = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "Model artifacts not found in Models/. Run `python src/train.py` first "
        "to generate best_model.pkl, model_features.pkl, high_charges_threshold.pkl, "
        "and decision_threshold.pkl. The dashboard below will still work; the "
        "prediction tool will not."
    )


# -----------------------------
# BUSINESS KPIs
# -----------------------------
st.subheader("Business Overview")

total_customers = len(df)
churned_customers = int(df['Churn'].sum())
churn_rate = df['Churn'].mean() * 100
revenue_at_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", total_customers)
with col2:
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}")
with col3:
    st.metric("Churned Customers", churned_customers)
with col4:
    st.metric("Revenue at Risk (Monthly)", f"${revenue_at_risk:,.2f}")


# -----------------------------
# FILTERS
# -----------------------------
st.sidebar.header("Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type", df['Contract'].unique(), default=df['Contract'].unique()
)
internet_filter = st.sidebar.multiselect(
    "Internet Service", df['InternetService'].unique(), default=df['InternetService'].unique()
)

filtered_df = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter))
]


# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.subheader("Churn Analysis")

col1, col2 = st.columns(2)

with col1:
    contract_churn = pd.crosstab(
        filtered_df['Contract'], filtered_df['Churn'], normalize='index'
    ) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    contract_churn.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], ax=ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Contract Type vs Churn")
    ax.legend(["No Churn", "Churn"])
    ax.set_xticklabels(contract_churn.index, rotation=0)
    st.pyplot(fig)

with col2:
    avg_monthly = filtered_df.groupby('Churn')['MonthlyCharges'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Churn', y='MonthlyCharges', data=avg_monthly,
                palette=['#2ca02c', '#d62728'], ax=ax)
    ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_ylabel("Average Monthly Charges ($)")
    ax.set_title("Average Monthly Charges by Churn")
    for i, val in enumerate(avg_monthly['MonthlyCharges']):
        ax.text(i, val + 1, f"${val:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)


# -----------------------------
# MODEL PREDICTION -- every field is now real
# -----------------------------
st.subheader("Churn Prediction")

if not model_loaded:
    st.info("Prediction tool disabled until Models/ artifacts exist.")
else:
    with st.form("prediction_form"):
        st.markdown("Fill in all fields for a single customer. Every field below "
                     "is used by the model -- none are silently filled in for you.")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1],
                                           format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with c2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No phone service", "No", "Yes"]
            )
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )

        with c3:
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
        }

        input_df = pd.DataFrame([input_data])
        input_df = engineer_features(input_df, high_charges_threshold=high_charges_threshold)
        input_df = input_df[feature_columns]

        prob = model.predict_proba(input_df)[0][1]

        st.metric("Churn Probability", f"{prob:.2%}")
        st.caption(f"Decision threshold: {decision_threshold} "
                   f"(chosen to prioritize recall -- see docs/executive_summary.md)")

        if prob >= decision_threshold:
            st.error("High Risk Customer -- flagged for retention outreach")
        else:
            st.success("Low Risk Customer -- not flagged")