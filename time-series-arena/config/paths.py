import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(_ROOT, "models"))
DB_PATH = os.getenv("DB_PATH", os.path.join(_ROOT, "data", "raw", "timeseries.db"))
PROCESSED_DIR = os.getenv("PROCESSED_DIR", os.path.join(_ROOT, "data", "processed"))
TRAIN_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "train")
TEST_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "test")
SCENARIO_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "scenarios")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TRAIN_PROCESSED_DIR, exist_ok=True)
os.makedirs(TEST_PROCESSED_DIR, exist_ok=True)
os.makedirs(SCENARIO_PROCESSED_DIR, exist_ok=True)

MODEL_PATHS = {
    "arima": os.path.join(MODEL_DIR, "arima_models.joblib"),
    "prophet": os.path.join(MODEL_DIR, "prophet_models.joblib"),
    "xgboost": os.path.join(MODEL_DIR, "xgboost_model.joblib"),
    "auto_arima": os.path.join(MODEL_DIR, "auto_arima_models.joblib"),
}
