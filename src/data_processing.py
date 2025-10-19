import pandas as pd
import numpy as np

def load_excel(data_path, sheet_name=0):
    """
    Load data from an Excel file.
    file_path: Path to the Excel file.
    sheet_name: Sheet index or name (default 0).
    """
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    print(f"Loaded data from {data_path}, shape: {df.shape}")
    return df

def clean_data(df, target_column=None):
    """
    Cleans the dataframe:
      - Removes columns entirely empty.
      - Optionally removes rows missing the target.
      - Fills missing numerics with median.
    """
    # Remove empty columns
    df = df.dropna(axis=1, how='all')

    # Optionally drop rows missing target
    if target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])

    # Fill numeric columns with the median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Strip whitespace from column headers
    df.columns = [c.strip() for c in df.columns]

    print("Cleaned data shape:", df.shape)
    return df

def select_features(df, target_column, features_list=None):
    """
    Splits the dataframe into features and target arrays.
    If features_list is None, uses all numeric columns except the target.
    """
    if features_list is None:
        features_list = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    X = df[features_list]
    y = df[target_column] if target_column in df.columns else None
    print(f"Selected {len(features_list)} features")
    return X, y

def get_processed_data(raw_path, target_column, features_list=None):
    """
    Full pipeline: Load, clean, and select features from a raw Excel input.
    Returns: X (features), y (labels/target)
    """
    df_raw = load_excel(raw_path)
    df_clean = clean_data(df_raw, target_column=target_column)
    X, y = select_features(df_clean, target_column, features_list)
    return X, y
