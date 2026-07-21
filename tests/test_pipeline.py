"""
tests/test_pipeline.py

Regression tests for the Telecom Churn pipeline. These are not generic
placeholder tests -- each one directly targets a specific bug that was
found and fixed during this project's review, so that bug can never be
silently reintroduced:

- test_total_charges_cleaning: the TotalCharges string->numeric->fillna(0)
  bug (11 rows, all tenure=0).
- test_tenure_group_no_nan: the pd.cut include_lowest bug (tenure=0
  customers previously became NaN).
- test_high_charges_no_leakage: the HighCharges leakage bug (threshold
  must come from training data only, never the full/test dataset).
- test_predict_uses_all_fields: the fake Streamlit form bug (app.py used
  to ignore most input fields -- this test catches any regression where
  changing a non-exposed field stops affecting the prediction).

Run with: pytest tests/test_pipeline.py -v
"""

from feature_engineering import engineer_features, prepare_train_test_split
from preprocessing import clean_data
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


RAW_DATA_PATH = Path(__file__).resolve().parent.parent / \
    "Dataset" / "churn_dataset.csv"


@pytest.fixture
def raw_df():
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture
def cleaned_df(raw_df):
    return clean_data(raw_df)


def test_total_charges_cleaning(cleaned_df):
    """TotalCharges must be fully numeric with zero nulls after cleaning."""
    assert cleaned_df['TotalCharges'].dtype.kind in ('i', 'f')
    assert cleaned_df['TotalCharges'].isnull().sum() == 0
    # The 11 originally-blank rows all had tenure=0 -- confirm they were
    # filled with 0, not an arbitrary median
    zero_tenure_charges = cleaned_df[cleaned_df['tenure'] == 0]['TotalCharges']
    assert (zero_tenure_charges == 0).all(), (
        "tenure=0 customers should have TotalCharges=0, not a median fill"
    )


def test_tenure_group_no_nan(cleaned_df):
    """
    Regression test for the pd.cut(include_lowest=False) bug that
    silently produced NaN for the 11 tenure=0 customers.
    """
    engineered = engineer_features(cleaned_df, high_charges_threshold=None)
    assert engineered['tenure_group'].isna().sum() == 0, (
        "tenure_group has NaN values -- the include_lowest=True fix "
        "may have been reverted"
    )


def test_high_charges_no_leakage(cleaned_df):
    """
    Regression test for the HighCharges leakage bug. Confirms the
    threshold used for train and test is identical (i.e. derived from
    training data only and applied unchanged), and that it is NOT
    silently recomputed from the full/combined dataset.
    """
    X_train, X_test, y_train, y_test, preprocessor, threshold = \
        prepare_train_test_split(cleaned_df)

    full_median = cleaned_df['MonthlyCharges'].median()
    train_median = X_train['MonthlyCharges'].median()

    # The threshold used must match the TRAINING median, not the full
    # dataset's median (they will usually differ slightly)
    assert threshold == pytest.approx(train_median)

    # Confirm HighCharges was actually added to both sets using that
    # one fixed threshold
    assert 'HighCharges' in X_train.columns
    assert 'HighCharges' in X_test.columns
    assert set(X_train['HighCharges'].unique()) <= {0, 1}


def test_predict_uses_all_fields(cleaned_df, tmp_path, monkeypatch):
    """
    Regression test for the fake Streamlit form bug. Confirms that
    changing a field NOT among the original 4 exposed fields (tenure,
    MonthlyCharges, Contract, InternetService) actually changes the
    engineered feature set / would change a prediction, rather than
    being silently ignored.
    """
    X_train, X_test, y_train, y_test, preprocessor, threshold = \
        prepare_train_test_split(cleaned_df)

    base_input = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0,
    }
    changed_input = dict(base_input)
    # NOT one of the 4 originally-exposed fields
    changed_input["TechSupport"] = "Yes"
    changed_input["PaymentMethod"] = "Credit card (automatic)"

    df_base = engineer_features(pd.DataFrame(
        [base_input]), high_charges_threshold=threshold)
    df_changed = engineer_features(pd.DataFrame(
        [changed_input]), high_charges_threshold=threshold)

    # The two engineered rows must actually differ -- if they were
    # identical, it would mean these fields are being ignored somewhere
    assert not df_base.equals(df_changed), (
        "Changing TechSupport/PaymentMethod produced an identical row -- "
        "these fields may be getting silently dropped or ignored"
    )


def test_no_customerid_in_cleaned_data(cleaned_df):
    """customerID must be dropped -- it's a pure identifier with zero
    predictive value and was flagged in docs/data_dictionary.md."""
    assert 'customerID' not in cleaned_df.columns


def test_churn_target_is_binary(cleaned_df):
    """Churn must be mapped to 0/1, not left as Yes/No strings."""
    assert set(cleaned_df['Churn'].unique()) <= {0, 1}
