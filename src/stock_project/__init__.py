"""Stock project package."""

from .data_loader import download_stock_data
from .data_loader import load_stock_data_from_csv
from .data_loader import load_stock_data_with_fallback
from .features import add_technical_features, create_supervised_dataset
from .model import train_linear_regression_model, evaluate_model
from .risk import calculate_volatility, calculate_max_drawdown

__all__ = [
    "download_stock_data",
    "load_stock_data_from_csv",
    "load_stock_data_with_fallback",
    "add_technical_features",
    "create_supervised_dataset",
    "train_linear_regression_model",
    "evaluate_model",
    "calculate_volatility",
    "calculate_max_drawdown",
]