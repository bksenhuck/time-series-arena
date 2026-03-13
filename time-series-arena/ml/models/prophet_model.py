"""
Local Prophet forecaster.
Uses Meta's Prophet for additive trend + seasonality modelling.
Trains one model per series on demand inside predict() — no shared state.
"""
import logging

import numpy as np
import pandas as pd

from ml.models.base_model import BaseModel, ForecastResult
from ml.forecasting.forecast_runner import strftime_list

# Silence noisy Stan / Prophet loggers
_QUIET = ["cmdstanpy", "prophet"]

log = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    name = "prophet"

    def fit(self, df: pd.DataFrame) -> None:
        # Prophet is local — no shared global training step.
        self._fitted = True

    def predict(
        self, series: pd.Series, series_id: str, horizon: int
    ) -> ForecastResult:
        # Deferred import keeps startup fast when prophet is optional
        from prophet import Prophet  # noqa: PLC0415

        for name in _QUIET:
            logging.getLogger(name).setLevel(logging.WARNING)

        if len(series.dropna()) < 30:
            raise ValueError(
                f"Series too short for Prophet: {len(series.dropna())}"
            )

        df = pd.DataFrame({"ds": series.index, "y": series.values})

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
        )
        m.fit(df)

        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        fc_tail = forecast.tail(horizon)

        return ForecastResult(
            dates=strftime_list(pd.DatetimeIndex(fc_tail["ds"])),
            mean=np.maximum(0, fc_tail["yhat"].values).tolist(),
            lower=np.maximum(0, fc_tail["yhat_lower"].values).tolist(),
            upper=np.maximum(0, fc_tail["yhat_upper"].values).tolist(),
            model_name=self.name,
        )
