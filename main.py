"""
Unified entry point: FastAPI + Dash in a single process.
FastAPI handles /api/* routes; Dash is mounted at / via a2wsgi.
"""
import logging
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from a2wsgi import WSGIMiddleware
from fastapi import FastAPI

from backend.app import create_app as create_api
from backend.utils.io_utils import init_db
from frontend.app import create_app as create_dash


def _run_pipeline():
    try:
        from backend.pipeline.training_pipeline import run_pipeline
        status = run_pipeline()
        logger.info("Pipeline complete: %s", status)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Initialising database…")
    init_db()
    thread = threading.Thread(target=_run_pipeline, daemon=True, name="pipeline")
    thread.start()
    yield


# ── Build apps ───────────────────────────────────────────────────────────────

api = create_api()
api.router.lifespan_context = lifespan

dash_app = create_dash()
api.mount("/", WSGIMiddleware(dash_app.server))

# The ASGI app exported for gunicorn/uvicorn
app = api


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8050))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
