import pandas as pd

from stock_project.risk import calculate_volatility, calculate_max_drawdown


def test_calculate_volatility_returns_float():
    df = pd.DataFrame({"daily_return": [0.01, -0.02, 0.03, -0.01, 0.02]})
    result = calculate_volatility(df)
    assert isinstance(result, float)
    assert result >= 0


def test_calculate_max_drawdown_returns_non_positive_float():
    df = pd.DataFrame({"Close": [100, 120, 110, 90, 95]})
    result = calculate_max_drawdown(df)
    assert isinstance(result, float)
    assert result <= 0