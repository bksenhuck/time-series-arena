"""Abstract base class shared by all forecasting models."""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import joblib
import pandas as pd


@dataclass
class ForecastResult:
    dates: list[str]
    mean: list[float]
    lower: list[float]
    upper: list[float]
    model_name: str

    def to_dict(self) -> dict:
        return {
            "dates": self.dates,
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "model": self.model_name,
        }


class BaseModel(ABC):
    """
    Common interface for all time-series forecasting models.

    Local models (ARIMA, Prophet) ignore df in fit() and train per-series
    inside predict(). Global models (XGBoost) train once across all series.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit on multi-series data (date, series_id, views)."""

    @abstractmethod
    def predict(
        self, series: pd.Series, series_id: str, horizon: int
    ) -> ForecastResult:
        """Forecast horizon steps ahead for a single series."""

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        return joblib.load(path)

    def is_fitted(self) -> bool:
        return bool(getattr(self, "_fitted", False))
