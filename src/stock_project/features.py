"""Feature engineering utilities for stock prediction."""

from __future__ import annotations

import pandas as pd


REQUIRED_INPUT_COLUMNS = ["Date", "Close"]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators used for analysis and prediction.

    Features added:
    - daily_return
    - ma_5
    - ma_20
    - volatility_5
    - target_next_close

    Args:
        df: Input DataFrame with at least Date and Close columns.

    Returns:
        DataFrame with engineered features.
    """
    for col in REQUIRED_INPUT_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    result = df.copy()
    result = result.sort_values("Date").reset_index(drop=True)

    result["daily_return"] = result["Close"].pct_change()
    result["ma_5"] = result["Close"].rolling(window=5).mean()
    result["ma_20"] = result["Close"].rolling(window=20).mean()
    result["volatility_5"] = result["daily_return"].rolling(window=5).std()
    result["target_next_close"] = result["Close"].shift(-1)

    return result


FEATURE_COLUMNS = ["Close", "daily_return", "ma_5", "ma_20", "volatility_5"]


def create_supervised_dataset(df: pd.DataFrame):
    """
    Create X and y for supervised learning.

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Tuple (X, y) where X is features and y is next-day closing price.
    """
    missing = [col for col in FEATURE_COLUMNS + ["target_next_close"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean_df = df.dropna(subset=FEATURE_COLUMNS + ["target_next_close"]).copy()
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["target_next_close"]
    return X, y