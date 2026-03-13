"""Input validation helpers."""
import pandas as pd

from config import MIN_HORIZON, MAX_HORIZON


def validate_horizon(horizon: int) -> int:
    if not (MIN_HORIZON <= horizon <= MAX_HORIZON):
        raise ValueError(f"horizon must be {MIN_HORIZON}–{MAX_HORIZON}, got {horizon}")
    return horizon


def validate_series(series: pd.Series, min_length: int = 30) -> pd.Series:
    if len(series) < min_length:
        raise ValueError(f"Series too short: {len(series)} < {min_length}")
    if series.isnull().all():
        raise ValueError("Series is entirely null")
    return series.dropna()


def validate_page_name(page: str) -> str:
    if not page or not page.strip():
        raise ValueError("Page name must not be empty")
    return page.strip()
