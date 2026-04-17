import pandas as pd

from stock_project.features import add_technical_features, create_supervised_dataset


def build_sample_df():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=30),
            "Close": list(range(100, 130)),
        }
    )


def test_add_technical_features_creates_expected_columns():
    df = build_sample_df()
    result = add_technical_features(df)

    expected_columns = {
        "daily_return",
        "ma_5",
        "ma_20",
        "volatility_5",
        "target_next_close",
    }
    assert expected_columns.issubset(set(result.columns))


def test_create_supervised_dataset_returns_non_empty_data():
    df = build_sample_df()
    result = add_technical_features(df)
    X, y = create_supervised_dataset(result)

    assert len(X) > 0
    assert len(y) > 0
    assert len(X) == len(y)