import pandas as pd
import numpy as np


def add_rolling_features(df, cols, window=3):
    """
    Adds rolling mean and std features for columns in 'cols' with a given window size.
    """
    for col in cols:
        df[f'{col}_rollmean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rollstd_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    return df

def add_ratio_features(df, numerator_cols, denominator_cols):
    """
    Adds ratio features for matching pairs of numerator_cols and denominator_cols.
    """
    for n, d in zip(numerator_cols, denominator_cols):
        feat_name = f'{n}_over_{d}'
        df[feat_name] = df[n] / (df[d] + 1e-6)  # add small epsilon to avoid division by zero
    return df

def feature_engineering_pipeline(df):
    """
    Full feature engineering pipeline that applies all feature additions.
    """
    # Example: Let's add rolling mean and std for selected numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = add_rolling_features(df, numeric_cols, window=3)

    # Example: Add ratio features for specific columns (customize as needed)
    # e.g., df = add_ratio_features(df, ['Net Profit Margin (%)'], ['Return on Equity (%)'])

    return df
