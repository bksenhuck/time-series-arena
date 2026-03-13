import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from backend.config import XGB_MODEL_PATH
from backend.services.feature_engineering import FEATURE_COLS, create_lag_features

logger = logging.getLogger(__name__)

_XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def train_xgboost(df: pd.DataFrame) -> tuple[XGBRegressor, LabelEncoder, float]:
    """
    Train a global XGBoost model on all series.
    df must contain: date, series_id, views.
    Returns (model, label_encoder, train_residual_std).
    """
    df = df.copy()
    le = LabelEncoder()
    df["series_id_enc"] = le.fit_transform(df["series_id"])

    featured = create_lag_features(df)
    featured = featured.dropna(subset=FEATURE_COLS)

    X = featured[FEATURE_COLS]
    y = featured["views"]

    model = XGBRegressor(**_XGB_PARAMS)
    model.fit(X, y, eval_set=[(X, y)], verbose=False)

    residual_std = float(np.std(y.values - model.predict(X)))
    logger.info("XGBoost trained. Residual std=%.2f rows=%d", residual_std, len(X))

    _save_model(model, le, residual_std)
    return model, le, residual_std


def _save_model(model: XGBRegressor, le: LabelEncoder, residual_std: float) -> None:
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "le": le, "residual_std": residual_std}, XGB_MODEL_PATH)
    logger.info("XGBoost model saved to %s", XGB_MODEL_PATH)


def load_xgboost() -> tuple[XGBRegressor, LabelEncoder, float] | None:
    if not os.path.exists(XGB_MODEL_PATH):
        return None
    payload = joblib.load(XGB_MODEL_PATH)
    return payload["model"], payload["le"], payload["residual_std"]


def is_model_trained() -> bool:
    return os.path.exists(XGB_MODEL_PATH)
