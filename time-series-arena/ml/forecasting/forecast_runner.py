"""
Forecast runner — model-agnostic execution layer.

Accepts any BaseModel and delegates to its predict() method.
Also exposes shared date utilities used by model implementations.
"""
import logging

import pandas as pd

from ml.models.base_model import BaseModel, ForecastResult

logger = logging.getLogger(__name__)


def run_forecast(
    model: BaseModel,
    series: pd.Series,
    series_id: str,
    horizon: int,
) -> ForecastResult:
    """
    Run a forecast through any fitted BaseModel.

    Parameters
    ----------
    model      : a fitted BaseModel instance
    series     : DatetimeIndex-indexed Series of historical views
    series_id  : Wikipedia page name (used for global model lookup)
    horizon    : number of days ahead to forecast

    Returns
    -------
    ForecastResult with dates, mean, lower CI, upper CI
    """
    if not model.is_fitted():
        raise RuntimeError(
            f"Model '{model.name}' has not been fitted yet"
        )
    return model.predict(series, series_id, horizon)


# ── Shared date utilities (used by model implementations) ─────────────────────

def future_dates(
    last_date: pd.Timestamp, horizon: int
) -> pd.DatetimeIndex:
    return pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )


def strftime_list(
    dates: pd.DatetimeIndex, fmt: str = "%Y-%m-%d"
) -> list[str]:
    return [d.strftime(fmt) for d in dates]
