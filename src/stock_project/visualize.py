"""Visualization utilities for stock data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_and_moving_averages(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot stock closing price and moving averages.
    """
    required = ["Date", "Close", "ma_5", "ma_20"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Close Price")
    plt.plot(df["Date"], df["ma_5"], label="5-Day Moving Average")
    plt.plot(df["Date"], df["ma_20"], label="20-Day Moving Average")
    plt.title("Stock Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()