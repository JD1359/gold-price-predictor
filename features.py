import pandas as pd
import numpy as np


def add_rolling_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Add rolling average and standard deviation features.
    These capture short-term and medium-term price trends.
    """
    df["MA_7"]  = df[price_col].rolling(window=7).mean()
    df["MA_14"] = df[price_col].rolling(window=14).mean()
    df["MA_30"] = df[price_col].rolling(window=30).mean()
    df["STD_7"]  = df[price_col].rolling(window=7).std()
    df["STD_30"] = df[price_col].rolling(window=30).std()

    return df


def add_momentum_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Add momentum and rate-of-change features.
    Momentum = today's price vs N days ago. Positive = uptrend.
    """
    df["Momentum_7"]  = df[price_col] - df[price_col].shift(7)
    df["Momentum_30"] = df[price_col] - df[price_col].shift(30)

    # Percentage change (rate of change)
    df["ROC_7"]  = df[price_col].pct_change(periods=7) * 100
    df["ROC_30"] = df[price_col].pct_change(periods=30) * 100

    return df


def add_lag_features(df: pd.DataFrame, price_col: str = "Close", lags: list = None) -> pd.DataFrame:
    """
    Add lagged price features — previous day prices as features.
    These are the most direct predictors for next-day price.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 7]

    for lag in lags:
        df[f"Lag_{lag}"] = df[price_col].shift(lag)

    return df


def add_price_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add High-Low range and Open-Close range features.
    These capture intraday volatility signals.
    Requires 'High', 'Low', 'Open', 'Close' columns.
    """
    if all(col in df.columns for col in ["High", "Low", "Open", "Close"]):
        df["HL_Range"]  = df["High"] - df["Low"]       # Intraday volatility
        df["OC_Range"]  = df["Close"] - df["Open"]     # Intraday direction
        df["HL_Pct"]    = df["HL_Range"] / df["Close"] # Normalized volatility
    return df


def add_target(df: pd.DataFrame, price_col: str = "Close", horizon: int = 1) -> pd.DataFrame:
    """
    Add the prediction target: next N days' closing price.

    Args:
        horizon: How many days ahead to predict (default: 1 = next day).
    """
    df["Target"] = df[price_col].shift(-horizon)
    return df


def build_features(df: pd.DataFrame, price_col: str = "Close", horizon: int = 1) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Applies all feature groups and drops rows with NaN values.

    Args:
        df:        Raw DataFrame with at minimum a price column and DatetimeIndex.
        price_col: Name of the price column to use.
        horizon:   Prediction horizon in days.

    Returns:
        Clean DataFrame ready for model training.
    """
    df = df.copy()
    df = add_rolling_features(df, price_col)
    df = add_momentum_features(df, price_col)
    df = add_lag_features(df, price_col)
    df = add_price_range_features(df)
    df = add_target(df, price_col, horizon)

    # Drop rows with NaN (caused by rolling windows and lags)
    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame, price_col: str = "Close") -> list:
    """
    Return the list of feature column names (excludes raw price and target).
    """
    exclude = {price_col, "Target", "Open", "High", "Low", "Volume", "Date"}
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    # Quick test with dummy data
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    prices = 1800 + np.cumsum(np.random.randn(200) * 5)

    df = pd.DataFrame({
        "Date": dates,
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": np.random.randint(1000, 10000, 200)
    }).set_index("Date")

    df_features = build_features(df)
    feature_cols = get_feature_columns(df_features)

    print(f"  Features built: {len(df_features)} rows × {len(feature_cols)} features")
    print(f"   Feature columns: {feature_cols}")
    print(f"\n{df_features[feature_cols + ['Target']].tail(3)}")
