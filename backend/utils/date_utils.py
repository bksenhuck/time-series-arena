"""Date helpers used across the project."""
import pandas as pd


def to_wiki_date(date_str: str) -> str:
    """Convert '2022-01-01' → '2022010100' (Wikimedia API format)."""
    return date_str.replace("-", "") + "00"


def date_range(start: str, end: str, freq: str = "D") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq)


def future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def strftime_list(dates: pd.DatetimeIndex, fmt: str = "%Y-%m-%d") -> list[str]:
    return [d.strftime(fmt) for d in dates]
