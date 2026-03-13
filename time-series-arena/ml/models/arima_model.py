"""
Local ARIMA forecaster.
Trains one ARIMA model per series on demand inside predict().
No cross-series shared state — each series is modelled independently.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ml.models.base_model import BaseModel, ForecastResult
from ml.forecasting.forecast_runner import future_dates, strftime_list

_DEFAULT_ORDER = (2, 1, 2)
_FALLBACK_ORDER = (1, 1, 1)
_DEFAULT_SEASONAL_ORDER = (1, 0, 1, 7)
_FALLBACK_SEASONAL_ORDER = (0, 0, 0, 0)


class ARIMAModel(BaseModel):
    name = "arima"

    def fit(self, df: pd.DataFrame) -> None:
        # ARIMA is local — no shared training step.
        self._fitted = True

    def predict(
        self, series: pd.Series, series_id: str, horizon: int
    ) -> ForecastResult:
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
        candidates = [
            (_DEFAULT_ORDER, _DEFAULT_SEASONAL_ORDER),
            (_DEFAULT_ORDER, _FALLBACK_SEASONAL_ORDER),
            (_FALLBACK_ORDER, _FALLBACK_SEASONAL_ORDER),
        ]
        for order, seasonal_order in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return SARIMAX(
                        values,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
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
