import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")
st.title("📊 Telecom Customer Churn Analysis")

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "Dataset/churn_dataset.csv"
MODEL_PATH = "Models/best_model.pkl"
PREPROCESSOR_PATH = "Models/preprocessor.pkl"

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
# LOAD MODEL + PREPROCESSOR
# -----------------------------


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


model, preprocessor = load_artifacts()

# -----------------------------
# BUSINESS KPIs
# -----------------------------
st.subheader("📌 Business Overview")

# Calculate KPIs
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
    st.metric("💰 Revenue at Risk (Monthly)", f"${revenue_at_risk:,.2f}")

# -----------------------------
# FILTERS
# -----------------------------
st.sidebar.header("🔍 Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

filtered_df = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter))
]

# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.subheader("📈 Churn Analysis")

col1, col2 = st.columns(2)

# 1️⃣ Contract Type vs Churn (Key Business Insight)
with col1:
    contract_churn = pd.crosstab(
        filtered_df['Contract'],
        filtered_df['Churn'],
        normalize='index'
    ) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    contract_churn.plot(
        kind='bar',
        stacked=True,
        color=['#2ca02c', '#d62728'],
        ax=ax
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Contract Type vs Churn")
    ax.legend(["No Churn", "Churn"])
    ax.set_xticklabels(contract_churn.index, rotation=0)

    st.pyplot(fig)

# 2️⃣ Average Monthly Charges by Churn
with col2:
    avg_monthly = filtered_df.groupby(
        'Churn')['MonthlyCharges'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x='Churn',
        y='MonthlyCharges',
        data=avg_monthly,
        palette=['#2ca02c', '#d62728'],
        ax=ax
    )

    ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_ylabel("Average Monthly Charges ($)")
    ax.set_title("Average Monthly Charges by Churn")

    for i, val in enumerate(avg_monthly['MonthlyCharges']):
        ax.text(i, val + 1, f"${val:.2f}",
                ha='center', fontweight='bold')

    st.pyplot(fig)

# -----------------------------
# MODEL PREDICTION
# -----------------------------
st.subheader("🤖 Churn Prediction")

with st.form("prediction_form"):
    tenure = st.number_input("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 20.0, 150.0, 70.0)
    contract = st.selectbox("Contract", df['Contract'].unique())
    internet = st.selectbox("Internet Service", df['InternetService'].unique())

    submit = st.form_submit_button("Predict")

if submit:

    input_data = df.drop(columns=['Churn']).iloc[0:1].copy()

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = tenure * monthly_charges
    input_data["Contract"] = contract
    input_data["InternetService"] = internet

    input_processed = preprocessor.transform(input_data)

    prob = model.predict_proba(input_processed)[0][1]

    st.metric("Churn Probability", f"{prob:.2%}")

    if prob > 0.6:
        st.error("⚠ High Risk Customer")
    elif prob > 0.4:
        st.warning("⚠ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")
