# 📊 Telecom Customer Churn Prediction & Business Dashboard

## 🚀 Project Overview

Customer churn is one of the biggest challenges in the telecom industry.  
This project builds a **Machine Learning model** to predict customer churn and provides an **interactive business dashboard** to analyze churn drivers and revenue risk.

The system helps answer:

- Which customers are likely to leave?
- What factors drive churn?
- How much revenue is at risk?
- What business actions should be taken?

---

## 🎯 Business Objectives

- Predict customer churn using Machine Learning
- Identify key churn drivers
- Estimate revenue at risk
- Provide actionable insights through a dashboard

---

## 🧠 Machine Learning Approach

- Data Cleaning & Preprocessing
- Feature Engineering
- Model Training (Random Forest Classifier)
- Model Evaluation
- Model Saving for Inference
- Deployment using Streamlit

---

## 📂 Project Structure

```
TELECOM_CHURN/
│
├── Dataset/
│ ├── churn_dataset.csv # Raw dataset
│ └── cleaned_dataset.csv # Cleaned dataset
│
├── Models/
│ ├── best_model.pkl # Trained ML model
│ └── preprocessor.pkl # Saved preprocessing pipeline
│
├── Notebook/
│ ├── 1_data_understanding.ipynb
│ ├── 2_EDA.ipynb
│ └── 3_preprocessing_&_modeling.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ └── predict.py
│
├── app.py # Streamlit Dashboard
├── requirements.txt
└── README.md

```


---

## 📊 Dashboard Features

### 🔹 Business KPIs
- Total Customers
- Churn Rate (%)
- Churned Customers
- 💰 Revenue at Risk (Monthly)

### 🔹 Business Insight Visualizations
- Contract Type vs Churn (Key churn driver)
- Average Monthly Charges by Churn

### 🔹 ML Prediction Tool
- Predict churn probability for new customers
- Risk classification (Low / Medium / High)

---

## 💰 Revenue at Risk

Revenue at Risk is calculated as:
Sum of Monthly Charges of churned customers


This gives an estimate of potential monthly revenue loss if no retention strategy is implemented.

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/telecom-churn-project.git
cd telecom-churn-project
```
Install dependencies
```bash
pip install -r requirements.txt
```
Run the application
```bash
streamlit run app.py
```
Model Training (Optional)

If you want to retrain the model:
``` bash
python src/train.py
```
To run prediction script separately:
```
python src/predict.py
```

## 📈 Key Insights from Analysis

- Month-to-month contract customers have the highest churn rate.
- Customers with higher monthly charges tend to churn more.
- Contract type is one of the strongest churn predictors.
- Significant recurring revenue is at risk due to churn.

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 📌 Future Improvements

- Deploy model using cloud services (AWS / GCP / Azure)
- Add SHAP explainability
- Predict revenue at risk using model probabilities
- Add retention strategy simulation
- Connect to live database

## 👨‍💻 Author
Akhil T V

If you found this project helpful, feel free to ⭐ the repository!
