"""Stock project package."""

from .data_loader import download_stock_data
from .features import add_technical_features, create_supervised_dataset
from .model import train_linear_regression_model, evaluate_model
from .risk import calculate_volatility, calculate_max_drawdown

__all__ = [
    "download_stock_data",
    "add_technical_features",
    "create_supervised_dataset",
    "train_linear_regression_model",
    "evaluate_model",
    "calculate_volatility",
    "calculate_max_drawdown",
]