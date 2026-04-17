import pandas as pd

from stock_project.model import train_linear_regression_model, evaluate_model


def test_train_linear_regression_model_has_predict_method():
    X_train = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104],
            "daily_return": [0.01, 0.01, 0.01, 0.01, 0.01],
            "ma_5": [100, 100.5, 101, 101.5, 102],
            "ma_20": [100, 100.2, 100.4, 100.6, 100.8],
            "volatility_5": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    y_train = pd.Series([101, 102, 103, 104, 105])

    model = train_linear_regression_model(X_train, y_train)
    assert hasattr(model, "predict")


def test_evaluate_model_returns_metrics():
    X_train = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104],
            "daily_return": [0.01, 0.01, 0.01, 0.01, 0.01],
            "ma_5": [100, 100.5, 101, 101.5, 102],
            "ma_20": [100, 100.2, 100.4, 100.6, 100.8],
            "volatility_5": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    y_train = pd.Series([101, 102, 103, 104, 105])

    X_test = X_train.copy()
    y_test = y_train.copy()

    model = train_linear_regression_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "predictions" in metrics