"""
Unified entry point: FastAPI + Dash in one process.
FastAPI serves /api/* and Dash is mounted at / via a2wsgi.
"""
import logging
import threading
from contextlib import asynccontextmanager

from a2wsgi import WSGIMiddleware
from fastapi import FastAPI

from app.frontend.app import create_app as create_dash
from backend.app import create_app as create_api
from backend.db.database import init_db
from backend.utils.logging_utils import setup_logging
from config import PORT
from ml.pipeline.training_pipeline import run_pipeline

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


def _run_pipeline() -> None:
    try:
        status = run_pipeline()
        logger.info("Pipeline complete: %s", status)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Initializing database...")
    init_db()
    thread = threading.Thread(target=_run_pipeline, daemon=True, name="pipeline")
    thread.start()
    yield


api = create_api()
api.router.lifespan_context = lifespan

dash_app = create_dash()
api.mount("/", WSGIMiddleware(dash_app.server))

app = api


if __name__ == "__main__":
    import uvicorn

    # Running via import string here ("main:app") while already inside
    # main.py can initialize the Dash app twice and duplicate callbacks.
    uvicorn.run(app, host="127.0.0.1", port=PORT, reload=False)
