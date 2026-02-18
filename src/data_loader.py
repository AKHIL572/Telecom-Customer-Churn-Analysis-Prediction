import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from given file path.

    Parameters:
    -----------
    file_path : str
        Path to CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
        print(f"Shape: {df.shape}")
        return df

    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def basic_data_check(df: pd.DataFrame) -> None:
    """
    Perform basic dataset validation checks.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """

    print("\n📌 Basic Data Information")
    print("-" * 40)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())
