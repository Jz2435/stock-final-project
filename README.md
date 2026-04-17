# Stock Price Analysis, Prediction, and Risk Assessment

## Project Purpose

This project analyzes historical stock price data from Yahoo Finance, builds a simple machine learning model to predict next-day closing prices, and evaluates financial risk using volatility and maximum drawdown.

## Dataset

The project uses a local historical price CSV file in the workspace:

- HistoricalData_1776395806851.csv

The current pipeline reads this file directly instead of downloading live data at runtime.

Example ticker represented in the dataset:

- TSLA (Tesla, Inc.)

Data fields include:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

## Features Implemented

- Historical price trend analysis
- Daily return calculation
- 5-day and 20-day moving averages
- 5-day rolling volatility
- Next-day closing price prediction using linear regression
- Risk analysis using volatility and maximum drawdown
- Visualization of closing price and moving averages

## Installation

1. Clone the repository:
   ```bash
   git clone <your-github-repo-link>
   cd stock_final_project
   ```

The dataset (TSLA.csv) is included to ensure full reproducibility.
