"""
Local ARIMA forecaster.
Trains one ARIMA model per series on demand inside predict().
No cross-series shared state — each series is modelled independently.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

from backend.models.base import BaseForecaster, ForecastResult
from backend.utils.date_utils import future_dates, strftime_list

_DEFAULT_ORDER = (2, 1, 2)
_FALLBACK_ORDER = (1, 1, 1)


class ARIMAForecaster(BaseForecaster):
    name = "arima"

    def fit(self, df: pd.DataFrame) -> None:
        # ARIMA is local — no shared training step.
        self._fitted = True

    def predict(self, series: pd.Series, series_id: str, horizon: int) -> ForecastResult:
        result = self._fit_series(series)
        mean, lower, upper = self._arima_forecast(result, horizon)
        fdates = future_dates(series.index[-1], horizon)
        return ForecastResult(
            dates=strftime_list(fdates),
            mean=np.maximum(0, mean).tolist(),
            lower=np.maximum(0, lower).tolist(),
            upper=np.maximum(0, upper).tolist(),
            model_name=self.name,
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _fit_series(self, series: pd.Series):
        values = series.dropna().values
        if len(values) < 30:
            raise ValueError(f"Series too short for ARIMA: {len(values)}")
        for order in (_DEFAULT_ORDER, _FALLBACK_ORDER):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return StatsARIMA(values, order=order).fit()
            except Exception:
                continue
        raise RuntimeError("All ARIMA orders failed")

    @staticmethod
    def _arima_forecast(result, horizon: int, alpha: float = 0.05):
        fc = result.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)
        lower = ci[:, 0] if ci.ndim == 2 else ci.iloc[:, 0].values
        upper = ci[:, 1] if ci.ndim == 2 else ci.iloc[:, 1].values
        return mean, lower, upper
