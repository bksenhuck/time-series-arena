"""
Model-agnostic forecasting service.
Works with any BaseForecaster — ARIMA, XGBoost, or any future model.
"""
import logging

import pandas as pd

from backend.models.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


def generate_forecast(
    forecaster: BaseForecaster,
    series: pd.Series,
    series_id: str,
    horizon: int,
) -> ForecastResult:
    """
    Generate a forecast using any fitted BaseForecaster.
    series: DatetimeIndex-indexed Series of views.
    Returns ForecastResult with dates, mean, lower CI, upper CI.
    """
    if not forecaster.is_fitted():
        raise RuntimeError(f"{forecaster.name} has not been fitted yet")
    return forecaster.predict(series, series_id, horizon)
