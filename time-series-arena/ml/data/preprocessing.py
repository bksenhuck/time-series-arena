"""
Data cleaning and preprocessing utilities.
Applied before feature engineering and model training.
"""
import numpy as np
import pandas as pd


def remove_outliers(series: pd.Series, z_thresh: float = 4.0) -> pd.Series:
    """Replace extreme outliers (|z| > z_thresh) with rolling median."""
    median = series.rolling(7, center=True, min_periods=1).median()
    z = (series - series.mean()) / (series.std() + 1e-8)
    return series.where(z.abs() <= z_thresh, median)


def fill_missing_days(series: pd.Series) -> pd.Series:
    """Fill gaps in the time series index with linear interpolation."""
    full_idx = pd.date_range(
        series.index.min(), series.index.max(), freq="D"
    )
    return series.reindex(full_idx).interpolate(method="time")


def smooth(series: pd.Series, window: int = 7) -> pd.Series:
    """Apply rolling-mean smoothing (used for visualisation only)."""
    return series.rolling(window, min_periods=1).mean()


def preprocess_series(
    series: pd.Series,
    fill: bool = True,
    clip_outliers: bool = True,
) -> pd.Series:
    if fill:
        series = fill_missing_days(series)
    if clip_outliers:
        series = remove_outliers(series)
    return series.clip(lower=0)


def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to every series in a long-format DataFrame."""
    cleaned = []
    for sid, group in df.groupby("series_id"):
        s = group.set_index("date")["views"]
        s = preprocess_series(s)
        tmp = s.reset_index()
        tmp.columns = ["date", "views"]
        tmp["series_id"] = sid
        cleaned.append(tmp)
    return pd.concat(cleaned, ignore_index=True)
