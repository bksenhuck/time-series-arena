"""
Model-agnostic forecasting service.
Accepts any BaseForecaster — decouples the API layer from model internals.
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
    Run a forecast through any fitted BaseForecaster.

    Parameters
    ----------
    forecaster : fitted BaseForecaster instance
    series     : DatetimeIndex-indexed Series of historical views
    series_id  : Wikipedia page name (used for global model lookup)
    horizon    : number of days ahead to forecast

    Returns
    -------
    ForecastResult with dates, mean, lower CI, upper CI
    """
    if not forecaster.is_fitted():
        raise RuntimeError(
            f"Forecaster '{forecaster.name}' has not been fitted yet"
        )
    return forecaster.predict(series, series_id, horizon)
