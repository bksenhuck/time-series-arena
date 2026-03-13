import json
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.data.data_loader import get_series
from backend.models import ARIMAForecaster, XGBoostGlobalForecaster
from backend.services.forecast_service import generate_forecast
from backend.utils.io_utils import db_conn
from config import MAX_HORIZON, MIN_HORIZON, XGB_MODEL_PATH

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level ARIMA instance (stateless — fits per series in predict())
_arima = ARIMAForecaster()
_arima.fit(None)


def _get_xgb() -> XGBoostGlobalForecaster | None:
    if os.path.exists(XGB_MODEL_PATH):
        return XGBoostGlobalForecaster.load(XGB_MODEL_PATH)
    return None


class ForecastRequest(BaseModel):
    page: str
    model: str = Field(pattern="^(arima|xgboost|both)$")
    horizon: int = Field(ge=MIN_HORIZON, le=MAX_HORIZON)


@router.post("/forecast")
def run_forecast(req: ForecastRequest):
    series_df = get_series(req.page)
    if series_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for: {req.page}")

    series = series_df["views"]
    result = {}

    if req.model in ("arima", "both"):
        cached = _get_cached(req.page, "arima", req.horizon)
        if cached:
            result["arima"] = cached
        else:
            try:
                fc = generate_forecast(_arima, series, req.page, req.horizon)
                _put_cache(req.page, "arima", req.horizon, fc.to_dict())
                result["arima"] = fc.to_dict()
            except Exception as exc:
                logger.error("ARIMA failed for %s: %s", req.page, exc)
                raise HTTPException(status_code=500, detail=f"ARIMA failed: {exc}")

    if req.model in ("xgboost", "both"):
        cached = _get_cached(req.page, "xgboost", req.horizon)
        if cached:
            result["xgboost"] = cached
        else:
            xgb = _get_xgb()
            if xgb is None:
                raise HTTPException(status_code=503, detail="XGBoost model not trained yet")
            try:
                fc = generate_forecast(xgb, series, req.page, req.horizon)
                _put_cache(req.page, "xgboost", req.horizon, fc.to_dict())
                result["xgboost"] = fc.to_dict()
            except Exception as exc:
                logger.error("XGBoost failed for %s: %s", req.page, exc)
                raise HTTPException(status_code=500, detail=f"XGBoost failed: {exc}")

    return result


@router.delete("/forecast/cache")
def clear_forecast_cache():
    with db_conn() as conn:
        conn.execute("DELETE FROM forecast_cache")
    return {"status": "cache cleared"}


def _get_cached(series_id: str, model_type: str, horizon: int):
    with db_conn() as conn:
        row = conn.execute(
            "SELECT forecast_json FROM forecast_cache WHERE series_id=? AND model_type=? AND horizon=?",
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
