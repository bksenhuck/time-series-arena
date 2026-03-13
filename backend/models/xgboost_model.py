"""
Global XGBoost forecaster.
Trains a single model across all series; delegates multi-step prediction
to the recursive_forecast module.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from backend.features.feature_engineering import (
    FEATURE_COLS,
    create_lag_features,
)
from backend.forecasting.recursive_forecast import (
    recursive_xgb_forecast,
)
from backend.models.base import BaseForecaster, ForecastResult

_XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


class XGBoostGlobalForecaster(BaseForecaster):
    name = "xgboost"

    def __init__(self):
        self._model: XGBRegressor | None = None
        self._le: LabelEncoder | None = None
        self._residual_std: float = 0.0
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> None:
        le = LabelEncoder()
        df = df.copy()
        df["series_id_enc"] = le.fit_transform(df["series_id"])

        featured = create_lag_features(df).dropna(subset=FEATURE_COLS)
        X, y = featured[FEATURE_COLS], featured["views"]

        model = XGBRegressor(**_XGB_PARAMS)
        model.fit(X, y, eval_set=[(X, y)], verbose=False)

        self._model = model
        self._le = le
        self._residual_std = float(np.std(y.values - model.predict(X)))
        self._fitted = True

    def predict(
        self, series: pd.Series, series_id: str, horizon: int
    ) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        raw = recursive_xgb_forecast(
            model=self._model,
            le=self._le,
            series=series,
            series_id=series_id,
            horizon=horizon,
            residual_std=self._residual_std,
        )
        return ForecastResult(
            dates=raw["dates"],
            mean=raw["mean"],
            lower=raw["lower"],
            upper=raw["upper"],
            model_name=self.name,
        )

    @property
    def known_series(self) -> list[str]:
        return list(self._le.classes_) if self._le is not None else []

    @property
    def residual_std(self) -> float:
        return self._residual_std
