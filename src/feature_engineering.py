"""
feature_engineering.py

Feature engineering, encoding, and train/test preparation for the
Telecom Churn project. Mirrors 3_preprocessing_and_modeling.ipynb.

DESIGN FIX vs. the original project: previously, feature engineering
logic was duplicated between the notebook and this file, and they
silently drifted apart -- the notebook had engineered features
(tenure_group, HighCharges, AutoPayment, HighRiskSegment) that
train.py never implemented. To prevent that from happening again,
`engineer_features()` below is the SINGLE function used by both
train.py (at training time) and predict.py (at inference time). If a
feature changes, it only needs to change here.
"""

import pandas as pd
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame, high_charges_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Add engineered features to a cleaned dataframe. Used identically at
    training time (on train/test splits) and at inference time (on a
    single new customer record), which is why HighCharges takes an
    explicit threshold rather than computing its own median internally
    -- computing a fresh median at inference time on a single row would
    be meaningless, and recomputing it from the full dataset at training
    time would be data leakage (see 3_preprocessing_and_modeling.ipynb
    for the leakage bug this fixes).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (post preprocessing.clean_data)
    high_charges_threshold : float, optional
        The MonthlyCharges cutoff for the HighCharges flag. Must be
        computed from TRAINING data only (see prepare_train_test_split)
        and reused unchanged at inference time. If None, HighCharges is
        not added (used internally before the threshold is known).

    Returns
    -------
    pd.DataFrame
        Dataframe with tenure_group, AutoPayment, HighRiskSegment, and
        (if a threshold is provided) HighCharges added.
    """
    df = df.copy()

    # FIX: include_lowest=True. Without it, pd.cut's first bin is (0, 12],
    # which EXCLUDES tenure == 0 and silently produces NaN for the 11
    # brand-new customers in this dataset (same rows flagged in
    # docs/data_dictionary.md for the TotalCharges issue). Verified via
    # df['tenure_group'].isna().sum() == 0 after this fix.
    df['tenure_group'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'],
        include_lowest=True
    )

    df['AutoPayment'] = df['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in x else 0
    )

    df['HighRiskSegment'] = (
        (df['Contract'] == 'Month-to-month') &
        (df['InternetService'] == 'Fiber optic') &
        (df['tenure'] < 12)
    ).astype(int)

    if high_charges_threshold is not None:
        df['HighCharges'] = (df['MonthlyCharges'] >
                             high_charges_threshold).astype(int)

    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Split into train/test and engineer all features, with HighCharges
    computed leakage-free (median derived from training data only).

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, high_charges_threshold
    """
    # TotalCharges is dropped -- see 3_preprocessing_and_modeling.ipynb
    # for rationale (multicollinearity with tenure * MonthlyCharges)
    if 'TotalCharges' in df.columns:
        df = df.drop(columns=['TotalCharges'])

    # Fixed-threshold features (tenure_group, AutoPayment, HighRiskSegment)
    # are safe to add before the split -- they don't depend on any
    # dataset-wide statistic. HighCharges is intentionally NOT added yet.
    df = engineer_features(df, high_charges_threshold=None)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # FIX (data leakage): compute the HighCharges median from TRAINING
    # data only, then apply that same fixed cutoff to both train and
    # test. The original bug computed this median on the full dataset
    # before splitting, contaminating the test set with training-derived
    # (and vice versa) statistics.
    high_charges_threshold = X_train['MonthlyCharges'].median()

    X_train = engineer_features(
        X_train, high_charges_threshold=high_charges_threshold)
    X_test = engineer_features(
        X_test, high_charges_threshold=high_charges_threshold)

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    print("Feature engineering completed.")
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(
        f"HighCharges threshold (from training median): {high_charges_threshold}")

    return X_train, X_test, y_train, y_test, preprocessor, high_charges_threshold
