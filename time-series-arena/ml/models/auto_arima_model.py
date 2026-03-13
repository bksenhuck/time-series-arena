"""
Local AutoARIMA forecaster using pmdarima.
Automatically selects the best (p,d,q) order per series via stepwise AIC search.
No cross-series shared state — each series is modelled independently.
"""
import warnings

import numpy as np
import pandas as pd

from ml.models.base_model import BaseModel, ForecastResult
from ml.forecasting.forecast_runner import future_dates, strftime_list


class AutoARIMAModel(BaseModel):
    name = "auto_arima"

    def fit(self, df: pd.DataFrame) -> None:
        # AutoARIMA is local — no shared global training step.
        self._fitted = True

    def predict(
        self, series: pd.Series, series_id: str, horizon: int
    ) -> ForecastResult:
        from pmdarima import auto_arima  # deferred: keeps startup fast

        values = series.dropna().values
        if len(values) < 30:
            raise ValueError(f"Series too short for AutoARIMA: {len(values)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = auto_arima(
                    values,
                    seasonal=True,
                    m=7,
                    d=0,
                    D=1,
                    stepwise=True,
                    error_action="ignore",
                    suppress_warnings=True,
                    information_criterion="aic",
                    maxiter=30,
                    max_p=2,
                    max_q=2,
                    max_P=2,
                    max_Q=2,
                )
            except Exception:
                # Fallback keeps endpoint responsive if seasonal optimization is unstable.
                model = auto_arima(
                    values,
                    seasonal=False,
                    stepwise=True,
                    error_action="ignore",
                    suppress_warnings=True,
                    information_criterion="aic",
                    maxiter=20,
                    max_p=3,
                    max_q=3,
                    max_d=2,
                )

        fc_mean, fc_ci = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        fdates = future_dates(series.index[-1], horizon)

        return ForecastResult(
            dates=strftime_list(fdates),
            mean=np.maximum(0, fc_mean).tolist(),
            lower=np.maximum(0, fc_ci[:, 0]).tolist(),
            upper=np.maximum(0, fc_ci[:, 1]).tolist(),
            model_name=self.name,
        )
