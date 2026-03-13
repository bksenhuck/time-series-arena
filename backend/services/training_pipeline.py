"""
Orchestrates the full pipeline:
1. Ensure data is loaded (downloads Wikipedia pageviews if missing)
2. Fit global XGBoost model (BaseForecaster.fit)
3. Compute and cache walk-forward evaluation metrics per series
"""
import logging
import os

import numpy as np

from backend.config import XGB_MODEL_PATH
from backend.db.session import db_conn
from backend.models import ARIMAForecaster, XGBoostGlobalForecaster
from backend.models.base import BaseForecaster
from backend.services.data_loader import (
    get_all_series,
    get_app_state,
    get_available_pages,
    get_series,
    load_all_data,
    set_app_state,
)
from backend.services.evaluation_metrics import walk_forward_evaluate
from backend.services.forecasting_service import generate_forecast

logger = logging.getLogger(__name__)


def _load_xgb() -> XGBoostGlobalForecaster | None:
    if os.path.exists(XGB_MODEL_PATH):
        return XGBoostGlobalForecaster.load(XGB_MODEL_PATH)
    return None


def run_pipeline(force_retrain: bool = False) -> dict:
    """Run the full training pipeline. Returns status summary."""
    status = {}

    # Step 1: Data
    if get_app_state("data_loaded") != "true":
        logger.info("Downloading Wikipedia pageview data...")
        n = load_all_data()
        set_app_state("data_loaded", "true")
        status["pages_loaded"] = n
        logger.info("Data loaded for %d pages", n)
    else:
        pages = get_available_pages()
        status["pages_loaded"] = len(pages)
        logger.info("Data already cached for %d pages", len(pages))

    # Step 2: Global XGBoost — retrain if forced, missing, or new series added
    xgb = _load_xgb()
    needs_train = force_retrain or xgb is None
    if not needs_train and xgb is not None:
        db_pages = set(get_available_pages())
        if db_pages - set(xgb.known_series):
            logger.info("XGBoost missing %d series — retraining", len(db_pages - set(xgb.known_series)))
            needs_train = True
            set_app_state("metrics_computed", "false")

    if needs_train:
        logger.info("Training global XGBoost model...")
        df = get_all_series()
        if df.empty:
            status["xgb_trained"] = False
            return status
        xgb = XGBoostGlobalForecaster()
        xgb.fit(df)
        xgb.save(XGB_MODEL_PATH)
        status["xgb_trained"] = True
        status["xgb_residual_std"] = xgb.residual_std
    else:
        status["xgb_trained"] = True
        logger.info("XGBoost already trained on %d series", len(xgb.known_series))

    # Step 3: Evaluation metrics
    if get_app_state("metrics_computed") != "true" or force_retrain:
        logger.info("Computing evaluation metrics...")
        _compute_and_cache_metrics(xgb)
        set_app_state("metrics_computed", "true")
        status["metrics_computed"] = True
    else:
        status["metrics_computed"] = True
        logger.info("Metrics already cached")

    return status


def _compute_and_cache_metrics(xgb: XGBoostGlobalForecaster | None):
    arima = ARIMAForecaster()
    arima.fit(None)  # local model — no global fit needed

    for page in get_available_pages():
        try:
            series = get_series(page)["views"]
            if len(series) < 60:
                continue

            if not _has_cached_metrics(page, "arima"):
                def arima_fn(train, h, _m=arima):
                    return np.array(generate_forecast(_m, train, page, h).mean)
                _cache_metrics(page, "arima", walk_forward_evaluate(series, arima_fn))

            if xgb is not None and not _has_cached_metrics(page, "xgboost"):
                def xgb_fn(train, h, _m=xgb):
                    return np.array(generate_forecast(_m, train, page, h).mean)
                _cache_metrics(page, "xgboost", walk_forward_evaluate(series, xgb_fn))

        except Exception as exc:
            logger.warning("Metrics failed for %s: %s", page, exc)


def _has_cached_metrics(series_id: str, model_type: str) -> bool:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM metrics_cache WHERE series_id=? AND model_type=?",
            (series_id, model_type),
        ).fetchone()
        return row is not None


def _cache_metrics(series_id: str, model_type: str, metrics: dict) -> None:
    with db_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO metrics_cache
               (series_id, model_type, mae, rmse, mape)
               VALUES (?, ?, ?, ?, ?)""",
            (series_id, model_type, metrics.get("mae"), metrics.get("rmse"), metrics.get("mape")),
        )


def get_all_metrics() -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT series_id, model_type, mae, rmse, mape FROM metrics_cache ORDER BY series_id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_page_metrics(series_id: str) -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT model_type, mae, rmse, mape FROM metrics_cache WHERE series_id=?",
            (series_id,),
        ).fetchall()
    return [dict(r) for r in rows]
