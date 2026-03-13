"""
Training pipeline (ML-only):
  1. Download pageview data (if needed)
  2. Train global XGBoost model
  3. Compute walk-forward metrics for ARIMA/Prophet/XGBoost

No FastAPI or Dash concerns in this module.
"""
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    MODEL_PATHS,
    WIKIPEDIA_PAGES,
    FORECAST_HORIZONS,
    PROCESSED_DIR,
    SCENARIO_PROCESSED_DIR,
    SCENARIO_WINDOWS_MONTHS,
    TEST_PROCESSED_DIR,
    DATA_MAX_DATE,
    DATA_MIN_DATE,
    TEST_WINDOW_MONTHS,
    TRAIN_END_DATE,
    TRAIN_TEST_SPLIT_DATE,
)
from backend.db.database import db_conn
from backend.services.forecast_service import get_train_fit_forecasts
from backend.utils.io_utils import get_state, set_state
from ml.data.loaders import (
    get_all_series,
    get_available_pages,
    get_series,
    get_train_series,
    load_all_pages,
)
from ml.evaluation.metrics import walk_forward_evaluate
from ml.forecasting.forecast_runner import run_forecast
from ml.models.arima_model import ARIMAModel
from ml.models.auto_arima_model import AutoARIMAModel
from ml.models.prophet_model import ProphetModel
from ml.models.xgboost_model import XGBoostModel
from ml.registry.model_registry import is_saved, load_model, save_model

logger = logging.getLogger(__name__)


def run_pipeline(force_retrain: bool = False) -> dict:
    status: dict = {}

    # 0 ── Re-check data range (clear cache when start/end dates change)
    if _reset_if_dates_changed():
        force_retrain = True
        logger.info("Forced full retrain due to date range change")

    # 1 ── Data
    if get_state("data_loaded") != "true":
        logger.info("Downloading Wikipedia pageview data...")
        n = load_all_pages(WIKIPEDIA_PAGES)
        set_state("data_min", DATA_MIN_DATE)
        set_state("data_max", DATA_MAX_DATE)
        set_state("test_window_months", str(TEST_WINDOW_MONTHS))
        set_state("train_test_split", TRAIN_TEST_SPLIT_DATE)
        # Backward-compatible state keys
        set_state("data_start", DATA_MIN_DATE)
        set_state("data_end", DATA_MAX_DATE)
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
        pages_count = len(get_available_pages())
        logger.info("Computing walk-forward metrics for %d pages...", pages_count)
        _compute_metrics(xgb)
        set_state("metrics_computed", "true")
        status["metrics_computed"] = True
    else:
        status["metrics_computed"] = True
        logger.info("Metrics already cached")

    # 4 ── Pre-compute forecasts for all pages × horizons × models
    if get_state("forecasts_precomputed") != "true" or force_retrain:
        pages_count = len(get_available_pages())
        logger.info(
            "Pre-computing forecasts: %d pages × %d horizons × 4 models = %d runs",
            pages_count, len(FORECAST_HORIZONS),
            pages_count * len(FORECAST_HORIZONS) * 4,
        )
        _precompute_forecasts(xgb)
        set_state("forecasts_precomputed", "true")
        status["forecasts_precomputed"] = True
    else:
        status["forecasts_precomputed"] = True
        logger.info("Forecasts already pre-computed")

    # 5 ── Pre-compute train fitted artifacts for dashboard loading
    if get_state("train_fit_precomputed") != "true" or force_retrain:
        pages_count = len(get_available_pages())
        logger.info("Pre-computing train fitted artifacts for %d pages", pages_count)
        _precompute_train_fit_artifacts()
        set_state("train_fit_precomputed", "true")
        status["train_fit_precomputed"] = True
    else:
        status["train_fit_precomputed"] = True
        logger.info("Train fitted artifacts already pre-computed")

    # 6 ── Pre-compute scenario comparison artifacts (3m/6m/9m)
    if get_state("scenario_compare_precomputed") != "true" or force_retrain:
        logger.info("Pre-computing scenario comparison artifacts for %s", SCENARIO_WINDOWS_MONTHS)
        _precompute_scenario_comparison(xgb)
        set_state("scenario_compare_precomputed", "true")
        status["scenario_compare_precomputed"] = True
    else:
        status["scenario_compare_precomputed"] = True
        logger.info("Scenario comparison artifacts already pre-computed")

    return status


# ── Internals ─────────────────────────────────────────────────────────────────

def _reset_if_dates_changed() -> bool:
    """Handle data window/split changes and reset caches as needed."""
    stored_min = get_state("data_min") or get_state("data_start")
    stored_max = get_state("data_max") or get_state("data_end")
    stored_test_months = get_state("test_window_months")
    stored_split = get_state("train_test_split") or get_state("train_end")

    range_changed = stored_min != DATA_MIN_DATE or stored_max != DATA_MAX_DATE
    test_months_changed = stored_test_months != str(TEST_WINDOW_MONTHS)
    split_changed = stored_split != TRAIN_TEST_SPLIT_DATE

    if not range_changed and not split_changed and not test_months_changed:
        return False

    if range_changed:
        logger.info(
            "Data window changed -> clearing pageviews (was %s-%s, now %s-%s)",
            stored_min,
            stored_max,
            DATA_MIN_DATE,
            DATA_MAX_DATE,
        )
        with db_conn() as conn:
            conn.execute("DELETE FROM pageviews")
        set_state("data_loaded", "false")

    if split_changed:
        logger.info(
            "Train/test split changed (was %s, now %s)",
            stored_split,
            TRAIN_TEST_SPLIT_DATE,
        )

    if test_months_changed:
        logger.info(
            "Test window months changed (was %s, now %s)",
            stored_test_months,
            TEST_WINDOW_MONTHS,
        )

    if range_changed:
        set_state("data_loaded", "false")
    set_state("metrics_computed", "false")
    set_state("forecasts_precomputed", "false")
    set_state("train_fit_precomputed", "false")
    set_state("scenario_compare_precomputed", "false")
    set_state("data_min", DATA_MIN_DATE)
    set_state("data_max", DATA_MAX_DATE)
    set_state("test_window_months", str(TEST_WINDOW_MONTHS))
    set_state("train_test_split", TRAIN_TEST_SPLIT_DATE)
    # Backward-compatible state keys
    set_state("data_start", DATA_MIN_DATE)
    set_state("data_end", DATA_MAX_DATE)
    return True


def _load_or_train_xgb(force: bool) -> XGBoostModel | None:
    xgb: XGBoostModel | None = None
    xgb_path = MODEL_PATHS["xgboost"]

    if not force and is_saved(xgb_path):
        loaded = load_model(xgb_path)
        if isinstance(loaded, XGBoostModel):
            xgb = loaded

    needs_train = force or xgb is None
    if not needs_train and xgb is not None:
        missing = set(get_available_pages()) - set(xgb.known_series)
        if missing:
            logger.info("XGBoost missing %d series - retraining", len(missing))
            needs_train = True
            set_state("metrics_computed", "false")

    if needs_train:
        logger.info("Training XGBoost global model...")
        df = get_all_series()
        df = df[df["date"] <= TRAIN_END_DATE]  # only train on pre-cutoff data
        if df.empty:
            return None
        xgb = XGBoostModel()
        xgb.fit(df)
        save_model(xgb, xgb_path)
        logger.info(
            "XGBoost trained: %d series, residual_std=%.1f",
            len(xgb.known_series),
            xgb.residual_std,
        )
    else:
        logger.info("XGBoost already trained on %d series", len(xgb.known_series))

    return xgb


def _compute_metrics(xgb: XGBoostModel | None) -> None:
    arima = ARIMAModel()
    arima.fit(None)

    prophet = ProphetModel()
    prophet.fit(None)

    auto_arima = AutoARIMAModel()
    auto_arima.fit(None)

    pages = get_available_pages()
    bar = tqdm(pages, desc="Computing metrics  ", unit="page", ncols=90, colour="yellow")
    for page in bar:
        bar.set_postfix_str(page[:40])
        try:
            series = get_train_series(page)["views"]
            if len(series) < 60:
                continue

            if not _has_metrics(page, "arima"):
                def arima_fn(train, h, _m=arima, _p=page):
                    return np.array(run_forecast(_m, train, _p, h).mean)
                _cache(page, "arima", walk_forward_evaluate(series, arima_fn))

            if not _has_metrics(page, "prophet"):
                def prophet_fn(train, h, _m=prophet, _p=page):
                    return np.array(run_forecast(_m, train, _p, h).mean)
                _cache(page, "prophet", walk_forward_evaluate(series, prophet_fn))

            if xgb is not None and not _has_metrics(page, "xgboost"):
                def xgb_fn(train, h, _m=xgb, _p=page):
                    return np.array(run_forecast(_m, train, _p, h).mean)
                _cache(page, "xgboost", walk_forward_evaluate(series, xgb_fn))

            if not _has_metrics(page, "auto_arima"):
                def auto_arima_fn(train, h, _m=auto_arima, _p=page):
                    return np.array(run_forecast(_m, train, _p, h).mean)
                _cache(page, "auto_arima", walk_forward_evaluate(series, auto_arima_fn))

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
                series_id,
                model_type,
                metrics.get("mae"),
                metrics.get("rmse"),
                metrics.get("mape"),
            ),
        )


def _precompute_forecasts(xgb: XGBoostModel | None) -> None:
    """Pre-compute and persist test forecasts for all pages × FORECAST_HORIZONS × models."""

    arima = ARIMAModel()
    arima.fit(None)
    prophet = ProphetModel()
    prophet.fit(None)
    auto_arima = AutoARIMAModel()
    auto_arima.fit(None)

    # Save local model instances so models/ folder stays consistent
    for m, path in [
        (arima, MODEL_PATHS["arima"]),
        (prophet, MODEL_PATHS["prophet"]),
        (auto_arima, MODEL_PATHS["auto_arima"]),
    ]:
        if not is_saved(path):
            save_model(m, path)

    models: dict = {"arima": arima, "prophet": prophet, "auto_arima": auto_arima}
    if xgb is not None:
        models["xgboost"] = xgb

    pages = get_available_pages()
    skipped = computed = 0

    bar = tqdm(
        pages,
        desc="Pre-computing forecasts",
        unit="page",
        ncols=90,
        colour="green",
    )
    for page in bar:
        try:
            series = get_train_series(page)["views"]
        except Exception as exc:
            logger.warning("precompute: cannot load series %s: %s", page, exc)
            continue

        page_dir = os.path.join(TEST_PROCESSED_DIR, page)
        os.makedirs(page_dir, exist_ok=True)

        for horizon in FORECAST_HORIZONS:
            for model_name, model in models.items():
                out_path = os.path.join(page_dir, f"{model_name}__{horizon}d.parquet")
                if os.path.exists(out_path):
                    skipped += 1
                    continue
                try:
                    fc = run_forecast(model, series, page, horizon)
                    payload = fc.to_dict()
                    df = pd.DataFrame(
                        {
                            "series_id": page,
                            "model": model_name,
                            "horizon": horizon,
                            "date": payload["dates"],
                            "mean": payload["mean"],
                            "lower": payload["lower"],
                            "upper": payload["upper"],
                        }
                    )
                    df.to_parquet(out_path, index=False)
                    computed += 1
                except Exception as exc:
                    logger.warning(
                        "precompute failed %s/%s/%dd: %s", page, model_name, horizon, exc
                    )
        bar.set_postfix(computed=computed, skipped=skipped)

    logger.info(
        "Forecast pre-computation done: %d computed, %d already existed",
        computed, skipped,
    )


def _precompute_train_fit_artifacts() -> None:
    pages = get_available_pages()
    skipped = computed = 0

    bar = tqdm(
        pages,
        desc="Pre-computing train fit",
        unit="page",
        ncols=90,
        colour="blue",
    )
    for page in bar:
        out_path = os.path.join(PROCESSED_DIR, "train", page, "train_fit.parquet")
        if os.path.exists(out_path):
            skipped += 1
            bar.set_postfix(computed=computed, skipped=skipped)
            continue
        try:
            get_train_fit_forecasts(page)
            computed += 1
        except Exception as exc:
            logger.warning("train-fit precompute failed %s: %s", page, exc)
        bar.set_postfix(computed=computed, skipped=skipped)

    logger.info(
        "Train fit pre-computation done: %d computed, %d already existed",
        computed,
        skipped,
    )


def _precompute_scenario_comparison(xgb: XGBoostModel | None) -> None:
    """Pre-compute metrics for multiple test-window scenarios (e.g., 3m/6m/9m)."""
    arima = ARIMAModel()
    arima.fit(None)
    prophet = ProphetModel()
    prophet.fit(None)
    auto_arima = AutoARIMAModel()
    auto_arima.fit(None)

    pages = get_available_pages()
    max_date = pd.to_datetime(DATA_MAX_DATE)
    rows: list[dict] = []

    for months in SCENARIO_WINDOWS_MONTHS:
        split = max_date - pd.DateOffset(months=int(months))
        scenario = f"{int(months)}m"
        bar = tqdm(
            pages,
            desc=f"Scenario metrics {scenario}",
            unit="page",
            ncols=90,
            colour="magenta",
        )
        for page in bar:
            try:
                full = get_series(page)
                train = full[full.index <= split]["views"]
                if len(train) < 60:
                    continue

                def arima_fn(ts, h, _m=arima, _p=page):
                    return np.array(run_forecast(_m, ts, _p, h).mean)

                def prophet_fn(ts, h, _m=prophet, _p=page):
                    return np.array(run_forecast(_m, ts, _p, h).mean)

                def auto_fn(ts, h, _m=auto_arima, _p=page):
                    return np.array(run_forecast(_m, ts, _p, h).mean)

                metrics_map = {
                    "arima": walk_forward_evaluate(train, arima_fn),
                    "prophet": walk_forward_evaluate(train, prophet_fn),
                    "auto_arima": walk_forward_evaluate(train, auto_fn),
                }

                if xgb is not None:
                    def xgb_fn(ts, h, _m=xgb, _p=page):
                        return np.array(run_forecast(_m, ts, _p, h).mean)

                    metrics_map["xgboost"] = walk_forward_evaluate(train, xgb_fn)

                for model_name, m in metrics_map.items():
                    rows.append(
                        {
                            "scenario": scenario,
                            "scenario_months": int(months),
                            "series_id": page,
                            "model_type": model_name,
                            "mae": m.get("mae"),
                            "rmse": m.get("rmse"),
                            "mape": m.get("mape"),
                        }
                    )
            except Exception as exc:
                logger.warning("scenario %s failed for %s: %s", scenario, page, exc)

    if not rows:
        logger.warning("No rows produced for scenario comparison")
        return

    out_dir = SCENARIO_PROCESSED_DIR
    os.makedirs(out_dir, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(os.path.join(out_dir, "scenario_metrics.parquet"), index=False)

    summary = (
        metrics_df.groupby(["scenario", "scenario_months", "model_type"], as_index=False)[["mae", "rmse", "mape"]]
        .mean()
        .sort_values(["scenario_months", "mae"])
    )
    summary.to_parquet(os.path.join(out_dir, "scenario_summary.parquet"), index=False)

    logger.info("Scenario comparison artifacts saved: %d metric rows", len(metrics_df))
