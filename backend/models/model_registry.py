"""
Central registry for saving, loading, and querying trained models.
Decouples model persistence from model logic.
"""
import logging
import os
from typing import Optional, Type

from backend.models.base import BaseForecaster
from backend.utils.io_utils import save_object, load_object

logger = logging.getLogger(__name__)


def save_model(forecaster: BaseForecaster, path: str) -> None:
    save_object(forecaster, path)
    logger.info("Saved %s → %s", forecaster.name, path)


def load_model(path: str) -> Optional[BaseForecaster]:
    if not os.path.exists(path):
        return None
    obj = load_object(path)
    logger.info("Loaded model from %s (type=%s)", path, type(obj).__name__)
    return obj


def is_saved(path: str) -> bool:
    return os.path.exists(path)


def delete_model(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        logger.info("Deleted model file: %s", path)


def model_info(path: str) -> dict:
    """Return basic metadata about a saved model file."""
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    stat = os.stat(path)
    return {
        "exists": True,
        "path": path,
        "size_kb": round(stat.st_size / 1024, 1),
    }
