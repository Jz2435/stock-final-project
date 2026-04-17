# Stock Price Analysis, Prediction, and Risk Assessment

## Project Overview

This project analyzes historical Tesla (TSLA) stock prices, performs feature engineering, builds a machine learning model to predict next-day closing prices, and evaluates financial risk.

The pipeline is designed with a dual-source data loading mechanism:

1. Online-first: download data from Yahoo Finance via `yfinance`
2. Local fallback: load a pre-downloaded Nasdaq CSV file if online fetch fails

This ensures both **robustness** and **full reproducibility**.

---

## Data Sources

### Primary Source (Online)

* Yahoo Finance API via `yfinance`

### Fallback Source (Local)

* Nasdaq historical stock price CSV
* Included in this repository:

  * `data/TSLA.csv`

### Date Range Used

* Start: 2020-01-01
* End: 2025-01-01
* Effective rows after filtering: 1258
* Effective data range: 2020-01-02 to 2024-12-31

---

## Features

The following features are engineered for supervised learning:

* `daily_return`
* `ma_5` (5-day moving average)
* `ma_20` (20-day moving average)
* `volatility_5` (5-day rolling volatility)
* `target_next_close` (next-day closing price)

---

## Model

* Algorithm: Linear Regression
* Train/test split: 80/20 (time-ordered, no shuffle)

---

## Risk Metrics

* **Volatility**: standard deviation of daily returns
* **Maximum Drawdown**: worst peak-to-trough decline

---

## Results Summary

Using the TSLA dataset:

* RMSE: 10.0140
* MAE: 6.8550
* R²: 0.9803
* Volatility: 0.042320
* Maximum Drawdown: -0.736322

### Prediction Summary (Test Set)

* Samples: 248
* Mean: 229.92395
* Median: 210.61775
* Min: 143.07430
* Max: 478.79522

Detailed structured output is saved in:

```
outputs/verification/report_data.json
```

---

## Output Artifacts

* Price & moving averages plot:

  ```
  outputs/price_moving_averages.png
  ```

* Verification logs:

  ```
  outputs/verification/
  ```

---

## Project Structure

```
stock_final_project/
├── src/stock_project
├── tests
├── data
├── outputs
```

---

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Editable Install

```bash
pip install -e .
```

---

## Run

From project root:

```bash
PYTHONPATH=src .venv/bin/python -m stock_project.main
```

The script will automatically indicate which data source is used:

* `online_yahoo`
* `local_csv`

---

## Test

Run all tests:

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Expected output:

```
6 passed
```

---

## Validation Notes

A full verification run has been completed:

* Main pipeline: pass
* Test suite: pass (6 passed)
* Fallback logic: pass

Outputs from Yahoo Finance and local CSV are consistent, with only minor floating-point differences.

---

## Reproducibility

This repository includes:

* `data/TSLA.csv`

so the project can be executed **without any external data dependency**.

---

## Key Takeaways

* TSLA exhibits high volatility and large drawdowns, making prediction more challenging than stable stocks.
* The linear regression model captures overall trends effectively but struggles during high-volatility periods.
* The project demonstrates a complete data science pipeline from data ingestion to modeling and risk evaluation.


