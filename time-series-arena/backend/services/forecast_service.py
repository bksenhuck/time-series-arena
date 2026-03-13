"""Backend orchestration service for forecast requests."""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

from backend.db.database import db_conn
from config import (
    FORECAST_HORIZONS,
    MODEL_PATHS,
    SCENARIO_PROCESSED_DIR,
    TEST_PROCESSED_DIR,
    TRAIN_FIT_WINDOW_DAYS,
    TRAIN_PROCESSED_DIR,
)
from ml.data.loaders import get_series, get_train_series
from ml.features.feature_engineering import FEATURE_COLS, create_lag_features
from ml.forecasting.forecast_runner import run_forecast
from ml.registry.model_registry import build_model, load_model

logger = logging.getLogger(__name__)


def get_available_pages() -> list[str]:
    from ml.data.loaders import get_available_pages as _pages
    return _pages()


def get_series_payload(page: str) -> dict:
    df = get_series(page)
    return {
        "series_id": page,
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "views": df["views"].tolist(),
    }


def get_metrics_for_page(page: str) -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT model_type, mae, rmse, mape "
            "FROM metrics_cache WHERE series_id=?",
            (page,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_metrics() -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT series_id, model_type, mae, rmse, mape "
            "FROM metrics_cache ORDER BY series_id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_scenario_comparison(page: str | None = None) -> dict:
    """Return scenario comparison summary for 3m/6m/9m windows."""
    metrics_path = os.path.join(SCENARIO_PROCESSED_DIR, "scenario_metrics.parquet")
    summary_path = os.path.join(SCENARIO_PROCESSED_DIR, "scenario_summary.parquet")
    if not os.path.exists(metrics_path) or not os.path.exists(summary_path):
        return {"rows": [], "summary": []}

    metrics_df = pd.read_parquet(metrics_path)
    if page:
        metrics_df = metrics_df[metrics_df["series_id"] == page]

    if metrics_df.empty:
        return {"rows": [], "summary": []}

    summary = (
        metrics_df.groupby(["scenario", "scenario_months", "model_type"], as_index=False)[["mae", "rmse", "mape"]]
        .mean()
        .sort_values(["scenario_months", "mae"])
    )

    return {
        "rows": metrics_df.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }


def clear_forecast_cache() -> None:
    with db_conn() as conn:
        conn.execute("DELETE FROM forecast_cache")


def get_train_fit_forecasts(page: str) -> dict:
    """Load or compute in-sample fitted lines for training window."""
    path = _train_fit_path(page)
    if os.path.exists(path):
        return _load_train_fit_from_parquet(path)

    out = _compute_train_fit_forecasts(page)
    _save_train_fit_to_processed(page, out)
    return out


def _compute_train_fit_forecasts(page: str) -> dict:
    """Compute in-sample fitted lines for training window (real + models)."""
    series_df = get_train_series(page)
    if series_df.empty:
        raise ValueError(f"No training data for: {page}")

    series = series_df["views"].astype(float)
    if len(series) < 60:
        raise ValueError(f"Series too short for train-fit chart: {len(series)}")

    start = max(0, len(series) - TRAIN_FIT_WINDOW_DAYS)
    window_idx = series.index[start:]
    window_dates = window_idx.strftime("%Y-%m-%d").tolist()

    out: dict = {
        "actual": {
            "dates": window_dates,
            "mean": np.maximum(0, series.iloc[start:].values).tolist(),
        }
    }

    # ARIMA fitted line
    try:
        values = series.values
        try:
            arima_fit = StatsARIMA(values, order=(2, 1, 2)).fit()
        except Exception:
            arima_fit = StatsARIMA(values, order=(1, 1, 1)).fit()
        arima_pred = np.asarray(arima_fit.fittedvalues, dtype=float)
        out["arima"] = {
            "dates": window_dates,
            "mean": np.maximum(0, arima_pred[start:]).tolist(),
        }
    except Exception as exc:
        logger.warning("train-fit arima failed for %s: %s", page, exc)

    # Prophet fitted line
    try:
        from prophet import Prophet  # noqa: PLC0415

        dfp = pd.DataFrame({"ds": series.index, "y": series.values})
        pm = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
        )
        pm.fit(dfp)
        prophet_pred = pm.predict(dfp[["ds"]])["yhat"].values
        out["prophet"] = {
            "dates": window_dates,
            "mean": np.maximum(0, prophet_pred[start:]).tolist(),
        }
    except Exception as exc:
        logger.warning("train-fit prophet failed for %s: %s", page, exc)

    # AutoARIMA fitted line
    try:
        from pmdarima import auto_arima  # noqa: PLC0415

        am = auto_arima(
            series.values,
            seasonal=False,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            information_criterion="aic",
            max_p=4,
            max_q=4,
            max_d=2,
        )
        auto_pred = np.asarray(am.predict_in_sample(), dtype=float)
        out["auto_arima"] = {
            "dates": window_dates,
            "mean": np.maximum(0, auto_pred[start:]).tolist(),
        }
    except Exception as exc:
        logger.warning("train-fit auto_arima failed for %s: %s", page, exc)

    # XGBoost fitted line (feature-based in-sample)
    try:
        xgb = _get_model_instance("xgboost")
        if page in xgb.known_series:
            base = pd.DataFrame(
                {
                    "date": series.index,
                    "series_id": page,
                    "views": series.values,
                }
            )
            base["series_id_enc"] = xgb._le.transform(base["series_id"])
            featured = create_lag_features(base).dropna(subset=FEATURE_COLS)
            xgb_pred = xgb._model.predict(featured[FEATURE_COLS])
            feat_dates = pd.to_datetime(featured["date"])
            mask = feat_dates >= window_idx[0]
            out["xgboost"] = {
                "dates": feat_dates[mask].dt.strftime("%Y-%m-%d").tolist(),
                "mean": np.maximum(0, np.asarray(xgb_pred)[mask]).tolist(),
            }
    except Exception as exc:
        logger.warning("train-fit xgboost failed for %s: %s", page, exc)

    return out


def get_forecast(page: str, model_name: str, horizon: int) -> dict:
    """
    Backend orchestration entrypoint.

    model_name can be: arima, prophet, xgboost, all
    """
    series_df = get_train_series(page)
    if series_df.empty:
        raise ValueError(f"No data for: {page}")

    series = series_df["views"]
    selected = [model_name] if model_name != "all" else [
        "arima", "prophet", "xgboost", "auto_arima"
    ]

    result: dict = {}
    for key in selected:
        cached = _get_cached(page, key, horizon)
        if cached:
            result[key] = cached
            continue

        model = _get_model_instance(key)
        fc = run_forecast(model, series, page, horizon)
        payload = fc.to_dict()
        _put_cache(page, key, horizon, payload)
        _save_to_processed(page, key, horizon, payload)
        result[key] = payload

    return result


def get_all_horizon_forecasts(page: str, model_name: str) -> dict:
    """
    Load pre-computed forecasts for all FORECAST_HORIZONS from data/processed/test/{page}/.
    Returns: {model: {horizon: forecast_dict, ...}, ...}
    model_name can be: arima, prophet, xgboost, all
    """
    selected = [model_name] if model_name != "all" else ["arima", "prophet", "xgboost", "auto_arima"]
    result: dict = {}
    page_dir = os.path.join(TEST_PROCESSED_DIR, page)

    for model in selected:
        horizons: dict = {}
        for h in FORECAST_HORIZONS:
            path = os.path.join(page_dir, f"{model}__{h}d.parquet")
            if not os.path.exists(path):
                continue
            try:
                horizons[h] = _load_forecast_from_parquet(path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
        if horizons:
            result[model] = horizons

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_to_processed(series_id: str, model_type: str, horizon: int, fc: dict) -> None:
    """Persist test forecast result as Parquet in data/processed/test/{series_id}/."""
    page_dir = os.path.join(TEST_PROCESSED_DIR, series_id)
    os.makedirs(page_dir, exist_ok=True)
    filename = f"{model_type}__{horizon}d.parquet"
    path = os.path.join(page_dir, filename)
    try:
        df = pd.DataFrame(
            {
                "series_id": series_id,
                "model": model_type,
                "horizon": horizon,
                "date": fc["dates"],
                "mean": fc["mean"],
                "lower": fc["lower"],
                "upper": fc["upper"],
            }
        )
        df.to_parquet(path, index=False)
    except Exception as exc:
        logger.warning("Could not save forecast to processed/test: %s", exc)


def _get_model_instance(model_name: str):
    path = MODEL_PATHS.get(model_name)

    # Global model (xgboost) should be loaded from disk if trained.
    if model_name == "xgboost":
        if not path or not os.path.exists(path):
            raise RuntimeError("XGBoost model not trained yet")
        loaded = load_model(path)
        if loaded is None:
            raise RuntimeError("Failed to load XGBoost model")
        return loaded

    # Local models are stateless: instantiate + mark fitted.
    model = build_model(model_name)
    model.fit(pd.DataFrame())
    return model


def _get_cached(series_id: str, model_type: str, horizon: int):
    with db_conn() as conn:
        row = conn.execute(
            "SELECT forecast_json FROM forecast_cache "
            "WHERE series_id=? AND model_type=? AND horizon=?",
            (series_id, model_type, horizon),
        ).fetchone()
    return json.loads(row["forecast_json"]) if row else None


def _put_cache(series_id: str, model_type: str, horizon: int, fc: dict):
    with db_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO forecast_cache
               (series_id, model_type, horizon, forecast_json)
               VALUES (?, ?, ?, ?)""",
            (series_id, model_type, horizon, json.dumps(fc)),
        )


def _load_forecast_from_parquet(path: str) -> dict:
    df = pd.read_parquet(path)
    return {
        "dates": df["date"].astype(str).tolist(),
        "mean": df["mean"].astype(float).tolist(),
        "lower": df["lower"].astype(float).tolist(),
        "upper": df["upper"].astype(float).tolist(),
    }


def _train_fit_path(page: str) -> str:
    return os.path.join(TRAIN_PROCESSED_DIR, page, "train_fit.parquet")


def _save_train_fit_to_processed(page: str, train_fit: dict) -> None:
    page_dir = os.path.join(TRAIN_PROCESSED_DIR, page)
    os.makedirs(page_dir, exist_ok=True)

    actual = train_fit.get("actual") or {}
    df = pd.DataFrame({
        "date": actual.get("dates", []),
        "actual": actual.get("mean", []),
    })
    for model in ("arima", "xgboost", "prophet", "auto_arima"):
        payload = train_fit.get(model) or {}
        if payload.get("dates") and payload.get("mean"):
            model_df = pd.DataFrame({
                "date": payload["dates"],
                model: payload["mean"],
            })
            df = df.merge(model_df, on="date", how="left") if not df.empty else model_df

    if not df.empty:
        df.to_parquet(_train_fit_path(page), index=False)


def _load_train_fit_from_parquet(path: str) -> dict:
    df = pd.read_parquet(path)
    out: dict = {}
    if "actual" in df.columns:
        out["actual"] = {
            "dates": df["date"].astype(str).tolist(),
            "mean": df["actual"].astype(float).tolist(),
        }
    for model in ("arima", "xgboost", "prophet", "auto_arima"):
        if model in df.columns:
            model_df = df[["date", model]].dropna()
            out[model] = {
                "dates": model_df["date"].astype(str).tolist(),
                "mean": model_df[model].astype(float).tolist(),
            }
    return out
