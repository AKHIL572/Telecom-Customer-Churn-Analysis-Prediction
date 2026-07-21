"""
preprocessing.py

Cleaning logic for the Telecom Churn project. Mirrors
Notebook/1_data_understanding.ipynb exactly.

FIX vs. original: the original version filled ALL numeric nulls with the
column median and ALL categorical nulls with the column mode. This was
never actually needed (verified: no column has missing values except
TotalCharges), and it directly contradicted the notebook's documented,
business-justified reasoning for TotalCharges specifically (all 11 blank
rows have tenure = 0, so 0 is the correct fill value, not an arbitrary
median). That inconsistency is fixed here -- this file now does exactly
what the notebook does, nothing more.
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the full data cleaning pipeline, matching
    1_data_understanding.ipynb exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()

    # 1. Remove duplicates (verified: 0 duplicates exist in the raw data,
    #    but keep this defensive check for future data refreshes)
    df.drop_duplicates(inplace=True)

    # 2. Strip column names
    df.columns = df.columns.str.strip()

    # 3. Coerce TotalCharges to numeric (raw file stores it as a string;
    #    11 rows contain a literal blank " " instead of a number)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # FIX: fill with 0, not median. All 11 affected rows have
        # tenure == 0 (verified against raw data) -- these are brand-new
        # customers who haven't been billed a cumulative total yet, so 0
        # is the business-correct value, not a statistical placeholder.
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 4. Convert target variable to binary
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. Drop customerID -- unique identifier, zero predictive value
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    print("Data preprocessing completed.")
    print(f"Cleaned shape: {df.shape}")

    return df
