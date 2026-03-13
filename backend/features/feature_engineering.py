"""
Lag-based and calendar feature engineering for the global XGBoost model.
All lags are shifted to prevent data leakage.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "day_of_week", "month",
    "series_id_enc",
]
LAG_DAYS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 30]


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: date (datetime), series_id (str), views (float).
    Returns enriched DataFrame with lag/rolling features; NaN rows dropped.
    """
    df = df.sort_values(["series_id", "date"]).copy()

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby("series_id")["views"].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            df.groupby("series_id")["views"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    return df.dropna(subset=[f"lag_{d}" for d in LAG_DAYS]).reset_index(drop=True)


def build_single_row(
    history: list[float],
    forecast_date: pd.Timestamp,
    series_enc: int,
) -> dict:
    """
    Build one feature dict for recursive XGBoost forecasting.
    history: list of known/predicted values up to the previous step.
    """
    n = len(history)

    def lag(d: int) -> float:
        idx = n - d
        return float(history[idx]) if idx >= 0 else np.nan

    def rolling(w: int) -> float:
        tail = history[-w:] if len(history) >= w else history
        return float(np.mean(tail)) if tail else np.nan

    return {
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_30": lag(30),
        "rolling_mean_7": rolling(7),
        "rolling_mean_30": rolling(30),
        "day_of_week": forecast_date.dayofweek,
        "month": forecast_date.month,
        "series_id_enc": series_enc,
    }
