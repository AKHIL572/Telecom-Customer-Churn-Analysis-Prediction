import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Analysis")

# -----------------------------
# LOAD DATA
# -----------------------------


@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_dataset.csv")

    # ✅ CRITICAL FIX: Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop ID
    df.drop(columns=['customerID'], inplace=True)

    # Feature engineering (MODEL EXPECTS THIS)
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )

    return df


df = load_data()

# -----------------------------
# LOAD MODEL
# -----------------------------


@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()

# -----------------------------
# BUSINESS KPIs
# -----------------------------
st.subheader("📌 Business Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    churn_rate = df['Churn'].mean() * 100
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}")

with col3:
    st.metric("Churned Customers", df['Churn'].sum())

# -----------------------------
# FILTERS
# -----------------------------
st.sidebar.header("🔍 Filters")

contract = st.sidebar.multiselect(
    "Contract Type",
    df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet = st.sidebar.multiselect(
    "Internet Service",
    df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

filtered_df = df[
    (df['Contract'].isin(contract)) &
    (df['InternetService'].isin(internet))
]

# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.subheader("📈 Churn Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=filtered_df, ax=ax)
    ax.set_xticklabels(['No Churn', 'Churn'])
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=filtered_df, ax=ax)
    ax.set_xticklabels(['No Churn', 'Churn'])
    st.pyplot(fig)

# -----------------------------
# MODEL PREDICTION (FIXED)
# -----------------------------
st.subheader("🤖 Churn Prediction")

with st.form("prediction_form"):
    tenure = st.number_input("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 20.0, 150.0, 70.0)
    contract = st.selectbox("Contract", df['Contract'].unique())
    internet = st.selectbox("Internet Service", df['InternetService'].unique())

    submit = st.form_submit_button("Predict")

if submit:
    # ✅ CRITICAL FIX: Create full feature template
    input_df = df.drop(columns=["Churn"]).iloc[0:1].copy()

    # Replace user-selected values
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = tenure * monthly_charges
    input_df["Contract"] = contract
    input_df["InternetService"] = internet

    # tenure_group will auto-adjust correctly
    input_df["tenure_group"] = pd.cut(
        input_df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )

    # Predict
    prob = model.predict_proba(input_df)[0][1]

    st.metric("Churn Probability", f"{prob:.2%}")

    if prob > 0.6:
        st.error("⚠ High Risk Customer")
    elif prob > 0.4:
        st.warning("⚠ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")
