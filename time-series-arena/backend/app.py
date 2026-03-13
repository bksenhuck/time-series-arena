from fastapi import FastAPI

from backend.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Time Series Arena API",
        version="2.0.0",
        docs_url="/api/docs",
    )
    app.include_router(api_router, prefix="/api")
    return app
