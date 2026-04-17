"""Risk analysis utilities."""

from __future__ import annotations

import pandas as pd


def calculate_volatility(df: pd.DataFrame) -> float:
    """
    Calculate volatility as the standard deviation of daily returns.
    """
    if "daily_return" not in df.columns:
        raise ValueError("Missing required column: daily_return")

    return float(df["daily_return"].dropna().std())


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """
    Calculate maximum drawdown from closing prices.
    """
    if "Close" not in df.columns:
        raise ValueError("Missing required column: Close")

    close = df["Close"].dropna()
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(drawdown.min())