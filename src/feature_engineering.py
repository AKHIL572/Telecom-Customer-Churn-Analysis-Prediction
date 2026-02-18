"""
feature_engineering.py

Handles feature transformation, encoding, scaling,
and preparation of training-ready data.
"""

import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Prepare features and target variable.
    Applies encoding and scaling using ColumnTransformer.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
    test_size : float
        Proportion for test split
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test, preprocessor
    """

    # -----------------------------------
    # 1️⃣ Separate Target
    # -----------------------------------
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------------
    # 2️⃣ Identify Column Types
    # -----------------------------------
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # -----------------------------------
    # 3️⃣ Create Transformers
    # -----------------------------------
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # -----------------------------------
    # 4️⃣ Combine Using ColumnTransformer
    # -----------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # -----------------------------------
    # 5️⃣ Train-Test Split (Before Fit)
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # -----------------------------------
    # 6️⃣ Fit Only on Training Data
    # -----------------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("✅ Feature engineering completed.")
    print(f"Training shape: {X_train_processed.shape}")
    print(f"Testing shape: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
