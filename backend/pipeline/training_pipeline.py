"""
End-to-end training pipeline:
  1. Download Wikipedia pageviews (skips cached pages)
  2. Fit global XGBoost model (retrain if new series detected)
  3. Walk-forward evaluate both models and cache metrics
"""
import logging

import numpy as np

from backend.data.data_loader import (
    get_all_series,
    get_available_pages,
    get_series,
    load_all_pages,
)
from backend.evaluation.metrics import walk_forward_evaluate
from backend.models import ARIMAForecaster, XGBoostGlobalForecaster
from backend.models.model_registry import (
    delete_model,
    is_saved,
    load_model,
    save_model,
)
from backend.services.forecast_service import generate_forecast
from backend.utils.io_utils import db_conn, get_state, set_state
from config import WIKIPEDIA_PAGES, XGB_MODEL_PATH

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(force_retrain: bool = False) -> dict:
    status: dict = {}

    # 1 ── Data
    if get_state("data_loaded") != "true":
        logger.info("Downloading Wikipedia pageview data…")
        n = load_all_pages(WIKIPEDIA_PAGES)
        status["pages_loaded"] = n
    else:
        pages = get_available_pages()
        status["pages_loaded"] = len(pages)
        logger.info("Data cached for %d pages", len(pages))

    # 2 ── XGBoost global model
    xgb = _load_or_train_xgb(force_retrain)
    status["xgb_trained"] = xgb is not None
    if xgb:
        status["xgb_series"] = len(xgb.known_series)

    # 3 ── Evaluation metrics
    if get_state("metrics_computed") != "true" or force_retrain:
        logger.info("Computing walk-forward metrics…")
        _compute_metrics(xgb)
        set_state("metrics_computed", "true")
        status["metrics_computed"] = True
    else:
        status["metrics_computed"] = True
        logger.info("Metrics already cached")

    return status


# ── Internals ─────────────────────────────────────────────────────────────────

def _load_or_train_xgb(
    force: bool,
) -> XGBoostGlobalForecaster | None:
    xgb: XGBoostGlobalForecaster | None = None

    if not force and is_saved(XGB_MODEL_PATH):
        xgb = load_model(XGB_MODEL_PATH)

    needs_train = force or xgb is None
    if not needs_train and xgb is not None:
        missing = set(get_available_pages()) - set(xgb.known_series)
        if missing:
            logger.info(
                "XGBoost missing %d series — retraining", len(missing)
            )
            needs_train = True
            set_state("metrics_computed", "false")

    if needs_train:
        logger.info("Training XGBoost global model…")
        df = get_all_series()
        if df.empty:
            return None
        xgb = XGBoostGlobalForecaster()
        xgb.fit(df)
        save_model(xgb, XGB_MODEL_PATH)
        logger.info(
            "XGBoost trained: %d series, residual_std=%.1f",
            len(xgb.known_series),
            xgb.residual_std,
        )
    else:
        logger.info(
            "XGBoost already trained on %d series", len(xgb.known_series)
        )

    return xgb


def _compute_metrics(xgb: XGBoostGlobalForecaster | None) -> None:
    arima = ARIMAForecaster()
    arima.fit(None)

    for page in get_available_pages():
        try:
            series = get_series(page)["views"]
            if len(series) < 60:
                continue

            if not _has_metrics(page, "arima"):
                def arima_fn(train, h, _m=arima, _p=page):
                    return np.array(generate_forecast(_m, train, _p, h).mean)
                _cache(page, "arima", walk_forward_evaluate(series, arima_fn))

            if xgb is not None and not _has_metrics(page, "xgboost"):
                def xgb_fn(train, h, _m=xgb, _p=page):
                    return np.array(generate_forecast(_m, train, _p, h).mean)
                _cache(page, "xgboost", walk_forward_evaluate(series, xgb_fn))

        except Exception as exc:
            logger.warning("Metrics failed for %s: %s", page, exc)


def _has_metrics(series_id: str, model_type: str) -> bool:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM metrics_cache WHERE series_id=? AND model_type=?",
            (series_id, model_type),
        ).fetchone()
    return row is not None


def _cache(series_id: str, model_type: str, metrics: dict) -> None:
    with db_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO metrics_cache
               (series_id, model_type, mae, rmse, mape)
               VALUES (?, ?, ?, ?, ?)""",
            (
                series_id, model_type,
                metrics.get("mae"), metrics.get("rmse"), metrics.get("mape"),
            ),
        )


# ── Metrics retrieval ─────────────────────────────────────────────────────────

def get_all_metrics() -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT series_id, model_type, mae, rmse, mape "
            "FROM metrics_cache ORDER BY series_id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_page_metrics(series_id: str) -> list[dict]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT model_type, mae, rmse, mape "
            "FROM metrics_cache WHERE series_id=?",
            (series_id,),
        ).fetchall()
    return [dict(r) for r in rows]
