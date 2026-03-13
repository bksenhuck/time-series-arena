"""FastAPI routes layer (thin HTTP adapter over backend services)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.forecast_service import (
    clear_forecast_cache,
    get_all_horizon_forecasts,
    get_all_metrics,
    get_available_pages,
    get_forecast,
    get_metrics_for_page,
    get_scenario_comparison,
    get_series_payload,
    get_train_fit_forecasts,
)
from config import FORECAST_HORIZONS

_MIN_H = min(FORECAST_HORIZONS)
_MAX_H = max(FORECAST_HORIZONS)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/pages")
def list_pages():
    return {"pages": get_available_pages()}


@router.get("/series/{page}")
def series_data(page: str):
    try:
        payload = get_series_payload(page)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not payload["dates"]:
        raise HTTPException(status_code=404, detail=f"No data for page: {page}")
    return payload


@router.get("/metrics/{page}")
def page_metrics(page: str):
    return {"series_id": page, "metrics": get_metrics_for_page(page)}


@router.get("/metrics")
def all_metrics():
    return {"metrics": get_all_metrics()}


@router.get("/scenarios/compare")
def scenario_compare(page: str | None = None):
    return get_scenario_comparison(page=page)


class ForecastRequest(BaseModel):
    page: str
    model: str = Field(pattern="^(arima|prophet|xgboost|auto_arima|all)$")
    horizon: int = Field(ge=_MIN_H, le=_MAX_H)


@router.get("/forecasts/{page}/{model}")
def all_horizon_forecasts(page: str, model: str):
    """Return pre-computed forecasts for all FORECAST_HORIZONS."""
    if model not in ("arima", "prophet", "xgboost", "auto_arima", "all"):
        raise HTTPException(status_code=400, detail="Invalid model")
    return get_all_horizon_forecasts(page, model)


@router.get("/train-fit/{page}")
def train_fit_forecasts(page: str):
    try:
        return get_train_fit_forecasts(page)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/forecast")
def run_forecast(req: ForecastRequest):
    try:
        return get_forecast(req.page, req.model, req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/forecast/cache")
def clear_cache():
    clear_forecast_cache()
    return {"status": "cache cleared"}
