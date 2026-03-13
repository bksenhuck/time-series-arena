"""
Forecasting evaluation metrics: MAE, RMSE, MAPE.
Also provides walk-forward (time-series cross-validation) evaluation.
"""
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(
    actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8
) -> float:
    return float(
        np.mean(np.abs((actual - predicted) / (np.abs(actual) + eps))) * 100
    )


def compute_all(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae":  mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
    }


def walk_forward_evaluate(
    series: pd.Series,
    model_fn,
    horizon: int = 14,
    n_splits: int = 3,
) -> dict:
    """
    Rolling-origin evaluation across n_splits folds.
    model_fn(train: pd.Series, horizon: int) -> np.ndarray of predictions.
    Returns averaged MAE / RMSE / MAPE across folds.
    """
    n = len(series)
    min_train = max(60, n // (n_splits + 1))
    fold_results = []

    for i in range(n_splits):
        split = min_train + i * (horizon * 2)
        if split + horizon > n:
            break
        train = series.iloc[:split]
        test = series.iloc[split: split + horizon].values
        try:
            preds = model_fn(train, horizon)
            fold_results.append(compute_all(test, preds))
        except Exception:
            continue

    if not fold_results:
        return {"mae": None, "rmse": None, "mape": None}

    return {
        "mae":  float(np.mean([r["mae"]  for r in fold_results])),
        "rmse": float(np.mean([r["rmse"] for r in fold_results])),
        "mape": float(np.mean([r["mape"] for r in fold_results])),
    }
