"""
Recursive multi-step forecasting engine.
Feeds each prediction back as a lag feature for the next step.
Used by the global XGBoost model (and any future lag-based model).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from backend.features.feature_engineering import FEATURE_COLS, build_single_row
from backend.utils.date_utils import future_dates, strftime_list


def recursive_xgb_forecast(
    model: XGBRegressor,
    le: LabelEncoder,
    series: pd.Series,
    series_id: str,
    horizon: int,
    residual_std: float,
) -> dict:
    """
    Generate a horizon-step-ahead forecast using recursive prediction.

    At each step:
      1. Build feature row from history (real values + previous predictions)
      2. Predict next value
      3. Append prediction to history

    Returns a dict with keys: dates, mean, lower, upper.
    """
    try:
        series_enc = int(le.transform([series_id])[0])
    except ValueError:
        return _mean_fallback(series, series_id, horizon, residual_std)

    history = list(series.values)
    fcast_dates = future_dates(series.index[-1], horizon)
    predictions: list[float] = []

    for date in fcast_dates:
        row = build_single_row(history, date, series_enc)
        X = pd.DataFrame([row])[FEATURE_COLS]
        pred = max(0.0, float(model.predict(X)[0]))
        predictions.append(pred)
        history.append(pred)

    # CI widens proportional to sqrt of step (propagated uncertainty)
    steps = np.arange(1, horizon + 1)
    width = 1.96 * residual_std * np.sqrt(steps)

    return {
        "dates": strftime_list(fcast_dates),
        "mean": predictions,
        "lower": [max(0.0, p - w) for p, w in zip(predictions, width)],
        "upper": [p + w for p, w in zip(predictions, width)],
    }


def _mean_fallback(
    series: pd.Series,
    series_id: str,
    horizon: int,
    residual_std: float,
) -> dict:
    mean_val = float(series.mean())
    fcast_dates = future_dates(series.index[-1], horizon)
    preds = [mean_val] * horizon
    width = 1.96 * residual_std
    return {
        "dates": strftime_list(fcast_dates),
        "mean": preds,
        "lower": [max(0.0, mean_val - width)] * horizon,
        "upper": [mean_val + width] * horizon,
    }
