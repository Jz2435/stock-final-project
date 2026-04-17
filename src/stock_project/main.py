"""Main script for stock price analysis project."""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

from stock_project.data_loader import load_stock_data_with_fallback
from stock_project.features import add_technical_features, create_supervised_dataset
from stock_project.model import train_linear_regression_model, evaluate_model
from stock_project.risk import calculate_volatility, calculate_max_drawdown
from stock_project.visualize import plot_price_and_moving_averages


def main() -> None:
    ticker = "TSLA"
    start = "2020-01-01"
    end = "2025-01-01"
    data_file = "HistoricalData_1776395806851.csv"

    print("Loading stock data (online first, local fallback)...")
    df, source = load_stock_data_with_fallback(
        file_path=data_file,
        ticker=ticker,
        start=start,
        end=end,
    )
    print(f"Data source: {source}")

    print("Engineering features...")
    df = add_technical_features(df)

    print("Creating supervised dataset...")
    X, y = create_supervised_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Training model...")
    model = train_linear_regression_model(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R^2: {metrics['r2']:.4f}")

    print("Running risk analysis...")
    volatility = calculate_volatility(df)
    max_drawdown = calculate_max_drawdown(df)
    print(f"Volatility: {volatility:.6f}")
    print(f"Maximum Drawdown: {max_drawdown:.6f}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "price_moving_averages.png"

    print("Saving visualization...")
    plot_price_and_moving_averages(df, str(plot_path))
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()