from fastapi import FastAPI

from backend.api.data import router as data_router
from backend.api.forecasting import router as forecast_router
from backend.api.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="Time Series Arena API", version="1.0.0", docs_url="/api/docs")

    app.include_router(health_router, prefix="/api")
    app.include_router(data_router, prefix="/api")
    app.include_router(forecast_router, prefix="/api")

    return app
