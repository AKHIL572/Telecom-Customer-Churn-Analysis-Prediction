"""
preprocessing.py

Handles data cleaning and preprocessing for Telecom Churn Project.
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full data cleaning pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """

    df = df.copy()

    # -----------------------------------
    # 1️⃣ Remove Duplicates
    # -----------------------------------
    df.drop_duplicates(inplace=True)

    # -----------------------------------
    # 2️⃣ Strip Column Names
    # -----------------------------------
    df.columns = df.columns.str.strip()

    # -----------------------------------
    # 3️⃣ Handle TotalCharges Column
    # (Common issue in Telco churn dataset)
    # -----------------------------------
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # -----------------------------------
    # 4️⃣ Handle Missing Values
    # -----------------------------------
    # Numerical → median
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical → mode
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # -----------------------------------
    # 5️⃣ Convert Target Variable
    # -----------------------------------
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # -----------------------------------
    # 6️⃣ Drop Unnecessary Columns
    # -----------------------------------
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    print("✅ Data preprocessing completed.")
    print(f"Cleaned Shape: {df.shape}")

    return df
