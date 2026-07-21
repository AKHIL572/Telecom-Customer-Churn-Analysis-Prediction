"""
data_loader.py

Loads the raw telecom churn dataset and performs basic sanity checks.
This file's logic was already correct and verified against the raw
dataset (see docs/data_dictionary.md) -- no functional changes needed.
"""

import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from given file path.

    Parameters
    ----------
    file_path : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def basic_data_check(df: pd.DataFrame) -> None:
    """
    Perform basic dataset validation checks. Matches the checks actually
    performed in 1_data_understanding.ipynb (verified against raw data:
    7043 rows, 21 columns, 0 duplicate rows, 0 missing values in every
    column except TotalCharges -- which has 11 blank-string rows, all
    belonging to customers with tenure = 0).
    """
    print("\nBasic Data Information")
    print("-" * 40)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values (raw TotalCharges is a string column -- blank")
    print("entries will NOT show up here until it's coerced to numeric,")
    print("see preprocessing.clean_data):")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
