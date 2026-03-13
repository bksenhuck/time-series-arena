"""
Model registry — maps string keys to model classes and provides
dynamic instantiation. Add new models here; nothing else needs to change.
"""
from __future__ import annotations

import logging
import os
from typing import Type

import joblib

from ml.models.arima_model import ARIMAModel
from ml.models.auto_arima_model import AutoARIMAModel
from ml.models.base_model import BaseModel
from ml.models.prophet_model import ProphetModel
from ml.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# ── Registry mapping ──────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "arima":      ARIMAModel,
    "prophet":    ProphetModel,
    "xgboost":    XGBoostModel,
    "auto_arima": AutoARIMAModel,
}


def get_model_class(name: str) -> Type[BaseModel]:
    """Return the model class for a given registry key."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def build_model(name: str) -> BaseModel:
    """Instantiate a fresh model by registry key."""
    cls = get_model_class(name)
    return cls()


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_model(model: BaseModel, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved %s → %s", model.name, path)


def load_model(path: str) -> BaseModel | None:
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    logger.info("Loaded model from %s (type=%s)", path, type(obj).__name__)
    return obj


def is_saved(path: str) -> bool:
    return os.path.exists(path)


def delete_model(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        logger.info("Deleted model file: %s", path)
