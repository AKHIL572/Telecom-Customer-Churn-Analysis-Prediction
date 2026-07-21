# 📊 Telecom Customer Churn Prediction & Business Intelligence Dashboard

An end-to-end Data Analytics and Machine Learning project that predicts telecom customer churn and provides actionable business insights through an interactive Power BI dashboard and Streamlit web application.

The project follows a complete data analytics workflow—from raw data understanding and exploratory analysis to feature engineering, predictive modeling, dashboard development, and deployment.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers directly impacts recurring revenue and customer lifetime value.

This project aims to identify customers who are likely to churn so that the retention team can proactively engage them before they leave.

The project combines:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- Power BI Dashboard
- Streamlit Deployment

---

# 🎯 Business Problem

The telecom company experiences customer attrition every month.

Instead of reacting after customers leave, the company wants to:

- Identify customers at high risk of churning
- Understand the factors driving churn
- Estimate revenue at risk
- Support targeted retention campaigns
- Enable business users to monitor churn through dashboards

---

# 🎯 Project Objectives

- Analyze customer behavior and churn patterns
- Identify important churn drivers
- Build a predictive churn model
- Maximize Recall to capture as many churning customers as possible
- Create an executive dashboard for business stakeholders
- Deploy an easy-to-use prediction application

---

# 📂 Project Structure

```
Telecom_Churn_Project/
│
├── Dataset/
│   ├── churn_dataset.csv
│   └── cleaned_dataset.csv
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_EDA.ipynb
│   └── 3_preprocessing_and_modeling.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── Models/
│   ├── best_model.pkl
│   ├── model_features.pkl
│   ├── high_charges_threshold.pkl
│   └── decision_threshold.pkl
│
├── dashboard/
│   └── Telecom_Churn_Dashboard.pbix
│
├── docs/
│   ├── executive_summary.md
│   ├── data_dictionary.md
│   └── model_card.md
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 Dataset Information

The project uses the IBM Telecom Customer Churn Dataset.

### Dataset Summary

| Attribute | Value |
|------------|--------|
| Total Customers | 7043 |
| Features | 21 |
| Target Variable | Churn |
| Problem Type | Binary Classification |

Target:

- Yes → Customer Churned
- No → Customer Retained

---

# 🛠 Technologies Used

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn
- Power BI

## Machine Learning

- Scikit-learn

## Deployment

- Streamlit

## Model Persistence

- Joblib

---

# 🔍 Phase 1 — Data Understanding

The raw dataset was thoroughly inspected before any preprocessing.

Key validation steps included:

- Dataset dimensions
- Data types
- Duplicate detection
- Missing value analysis
- Business interpretation of each feature

### Important Finding

`TotalCharges` contained 11 blank values.

Instead of replacing them with a statistical value, they were correctly identified as customers with:

- tenure = 0

Since these customers had not yet accumulated any charges, the missing values were replaced with **0**, which is the business-correct interpretation.

---

# 🧹 Phase 2 — Data Cleaning

Cleaning steps included:

- Removed duplicate records
- Standardized column names
- Converted TotalCharges to numeric
- Filled TotalCharges missing values with 0
- Converted target variable to binary
- Removed customerID
- Saved cleaned dataset

---

# 📈 Phase 3 — Exploratory Data Analysis

The project includes extensive EDA covering:

## Univariate Analysis

- Customer demographics
- Contract distribution
- Internet service usage
- Payment methods
- Monthly charges
- Tenure

## Bivariate Analysis

Relationship between churn and:

- Contract Type
- Payment Method
- Internet Service
- Monthly Charges
- Tech Support
- Online Security
- Tenure

## Multivariate Analysis

- Correlation Heatmap
- Churn patterns
- Revenue impact
- Business segmentation

---

# 💡 Key Business Insights

The analysis revealed that:

- Month-to-month customers churn significantly more than long-term contract customers.
- Fiber optic customers exhibit higher churn rates than DSL users.
- Customers without Tech Support or Online Security plans are more likely to churn.
- Customers with shorter tenure are at the highest risk.
- Electronic check users have the highest churn rate among payment methods.

---

# ⚙ Feature Engineering

The following business-driven features were created:

| Feature | Description |
|----------|-------------|
| tenure_group | Customer tenure buckets |
| AutoPayment | Indicates automatic payment usage |
| HighRiskSegment | Month-to-month + Fiber optic + tenure < 12 |
| HighCharges | MonthlyCharges above training median |

---

# ⚠ Data Leakage Fixes

Three important issues were identified and corrected during model development:

### 1. HighCharges Leakage

The threshold for `HighCharges` is calculated using the **training data only**, preventing information leakage from the test set.

### 2. Model Selection Leakage

The final model is selected using **cross-validated recall** rather than test set performance.

### 3. Threshold Selection Leakage

The decision threshold is optimized using **out-of-fold predictions** from the training data instead of the test data.

---

# 🤖 Machine Learning Models

The following algorithms were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

Hyperparameter tuning was performed using GridSearchCV with **Recall** as the optimization metric.

---

# ✅ Final Model

**Selected Model**

- Logistic Regression

Reason:

- Highest cross-validated Recall
- Excellent generalization
- Fully interpretable coefficients
- Suitable for business decision-making

---

# 📊 Final Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 62.38% |
| Recall | 94.12% |
| Precision | 40.93% |
| F1 Score | 57.05% |
| ROC-AUC | 0.840 |

---

# 📉 Model Validation

Additional validation included:

- Majority-class baseline comparison
- Cross-validation stability analysis
- Overfitting check
- Threshold optimization
- ROC Curve
- Confusion Matrix
- Coefficient interpretation

---

# 📌 Important Predictive Factors

Factors increasing churn risk:

- Fiber optic internet
- Month-to-month contracts
- Short tenure

Factors reducing churn risk:

- Longer customer tenure
- Two-year contracts
- Stable subscription history

---

# 📊 Power BI Dashboard

The project includes a fully interactive Power BI dashboard consisting of two pages.

## Executive Overview

Includes:

- KPI Cards
- Churn Rate
- Revenue at Risk
- Customer Distribution
- Contract Analysis
- Tenure Analysis
- Payment Method Analysis

## Retention Analysis

Includes:

- Tech Support Impact
- Online Security Impact
- Churn Trend by Tenure
- High-Risk Customer Table
- Interactive Filters

---

# 💻 Streamlit Application

The project also includes a Streamlit application that enables users to:

- Enter customer information
- Predict churn probability
- View churn classification
- Use the same preprocessing pipeline as the trained model

---

# 🚀 Running the Project

## Clone Repository

```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
```

## Navigate

```bash
cd telecom-churn-prediction
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python src/train.py
```

## Launch Streamlit

```bash
streamlit run app.py
```

---

# 📈 Business Value

This solution enables businesses to:

- Detect customers likely to churn
- Prioritize retention campaigns
- Reduce recurring revenue loss
- Understand key churn drivers
- Monitor KPIs through interactive dashboards
- Make data-driven retention decisions

---

# ⚠ Limitations

- Trained on a single snapshot of historical customer data.
- Model performance may change as customer behavior evolves.
- High recall results in a higher false-positive rate, which is acceptable for the project's retention-first objective.
- Threshold-based engineered features should be recalculated when retraining with new data.

---

# 🔮 Future Improvements

Potential enhancements include:

- XGBoost and LightGBM models
- SHAP explainability
- Automated retraining pipeline
- Cloud deployment
- Real-time prediction API
- Customer segmentation dashboards
- Model monitoring and drift detection

---

# 👨‍💻 Author

**Akhil T V**

B.Tech Computer Science Graduate

Aspiring Data Analyst | Data Science Enthusiast

---

# ⭐ If you found this project useful

Consider giving the repository a ⭐ to support the project.