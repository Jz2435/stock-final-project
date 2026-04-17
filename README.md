# Stock Price Analysis, Prediction, and Risk Assessment

## Project Purpose

This project analyzes historical stock price data from Yahoo Finance, builds a simple machine learning model to predict next-day closing prices, and evaluates financial risk using volatility and maximum drawdown.

## Dataset

The dataset is downloaded directly from Yahoo Finance using the `yfinance` Python package.

Example ticker used in this project:

- AAPL (Apple Inc.)

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
