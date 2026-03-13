import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

_DEFAULT_ORDER = (2, 1, 2)
_FALLBACK_ORDER = (1, 1, 1)


def train_arima(series: pd.Series, order: tuple = _DEFAULT_ORDER):
    """
    Fit an ARIMA model on a univariate time series.
    Falls back to (1,1,1) if the primary order fails.
    """
    values = series.dropna().values
    if len(values) < 30:
        raise ValueError(f"Series too short for ARIMA: {len(values)} observations")

    for arima_order in (order, _FALLBACK_ORDER):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(values, order=arima_order)
                result = model.fit()
            return result
        except Exception as exc:
            logger.warning("ARIMA%s failed: %s — retrying with fallback", arima_order, exc)

    raise RuntimeError(f"All ARIMA orders failed for series of length {len(values)}")


def forecast_arima(
    result,
    horizon: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean, lower_ci, upper_ci) each of length `horizon`.
    All values are clipped to >= 0 since views cannot be negative.
    """
    fc = result.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    lower = ci[:, 0] if ci.ndim == 2 else ci.iloc[:, 0].values
    upper = ci[:, 1] if ci.ndim == 2 else ci.iloc[:, 1].values
    return (
        np.maximum(0, mean),
        np.maximum(0, lower),
        np.maximum(0, upper),
    )


def arima_in_sample_residuals(result) -> np.ndarray:
    return result.resid
