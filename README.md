# 📊 Customer Churn Analysis & Prediction System

## 📌 Project Overview
Customer churn is a critical business problem for subscription-based companies.  
This project delivers an **end-to-end customer churn analysis and prediction system** using **Python, Machine Learning, Power BI, and Streamlit**.

The goal is to:
- Understand why customers churn
- Identify high-risk customer segments
- Quantify revenue impact
- Predict churn using a machine learning model
- Present insights through interactive dashboards

---

## 🚀 Key Features
- Complete data science pipeline (EDA → Modeling → Deployment)
- Interactive **Power BI dashboards** (3 pages)
- **Streamlit web app** for real-time churn prediction
- Business-focused insights with revenue impact
- Interview-ready, industry-standard project structure

---

## 🗂️ Project Structure
```
├── Notebook
|   ├── data_understanding.ipynb
|   ├── EDA.ipynb
|   └── preprocessing_and_modeling.ipynb
├── Model
|   └── churn_model.pkl
├── Streamlit_app/
│   ├── app.py
│   └── requirements.txt
├── Power_BI/
│   └── Customer_Churn_Analysis.pbix
├── Dataset/
│   └── churn_data.csv
└── README.md

```


---

## 📁 1️⃣ Data Understanding
- Loaded and explored the raw churn dataset
- Checked:
  - Data types
  - Missing values
  - Target variable distribution
- Built foundational understanding of customer attributes and churn behavior

---

## 📁 2️⃣ Exploratory Data Analysis (EDA)
Performed detailed analysis to uncover churn drivers.

### Key Insights:
- **Month-to-month contract customers churn the most**
- Churn decreases significantly with longer tenure
- Customers without **Tech Support** or **Online Security** have higher churn
- **Electronic check** users show higher churn rates
- Senior citizens churn more compared to non-seniors

### Visuals Used:
- Bar charts & stacked columns
- Line charts (tenure vs churn rate)
- Distribution and comparison plots

---

## 📁 3️⃣ Preprocessing & Modeling
### Data Preprocessing:
- Label Encoding / One-Hot Encoding
- Feature scaling
- Handling categorical and numerical variables

### Modeling:
- Built a churn prediction model
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - ROC-AUC

---

## 📊 Power BI Dashboards (3 Pages)

### 🔹 Page 1: Customer Churn Overview
- Total Customers
- Churned Customers
- Churn Rate (%)
- Churn by Contract Type

**Insight:**  
Month-to-month customers contribute the highest churn.

---

### 🔹 Page 2: Customer Lifecycle & Revenue Impact
- Total Monthly Revenue
- Revenue Lost Due to Churn
- Revenue Retained
- Churn Rate vs Tenure
- Average Monthly Charges (Churn vs Non-Churn)

**Interview Gold Line:**  
> High-value customers are more likely to churn.

---

### 🔹 Page 3: Churn Drivers & Customer Segments
- Service usage vs churn:
  - Internet Service
  - Tech Support
  - Online Security
- Payment method analysis
- Demographic segmentation:
  - Senior Citizen
  - Dependents

**Insight:**  
Customers without tech support churn significantly more.

---

## 🌐 Streamlit Web App
- User-friendly churn prediction dashboard
- Accepts customer inputs
- Predicts churn in real time
- Designed for business users and stakeholders

---

## 🛠️ Tools & Technologies
- **Python** (Pandas, NumPy, Scikit-learn)
- **Power BI**
- **Streamlit**
- **Matplotlib & Seaborn**
- **Machine Learning**

---

## 🎯 Business Value
- Identifies high-risk customers early
- Helps reduce churn through targeted actions
- Quantifies revenue loss due to churn
- Enables data-driven decision-making

---

## 🧾 Resume-Ready Summary
> Built an end-to-end Customer Churn Prediction system using Python, Machine Learning, Power BI dashboards, and a Streamlit web app to deliver actionable business insights and churn predictions.

---

## 📌 Future Improvements
- Deploy Streamlit app to cloud
- Try advanced models (XGBoost, Random Forest)
- Add customer lifetime value (CLV) analysis
- Automate data refresh in Power BI

---

## 👤 Author
**Akhil T V**

---

⭐ If you found this project useful, feel free to star the repository!
