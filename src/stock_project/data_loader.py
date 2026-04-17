"""Utilities for downloading stock data from Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.

    Returns:
        A pandas DataFrame containing stock price data.

    Raises:
        ValueError: If the returned DataFrame is empty or missing required columns.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string.")

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError("Downloaded data is empty. Check ticker or date range.")

    # If columns are multi-index like ('Close', 'AAPL'), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df