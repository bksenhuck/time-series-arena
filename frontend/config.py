import os
from dotenv import load_dotenv

load_dotenv()

from config import MIN_HORIZON, MAX_HORIZON, FORECAST_HORIZON_DEFAULT

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
API_BASE = os.getenv("API_BASE", "http://localhost:8050")
DEFAULT_HORIZON = FORECAST_HORIZON_DEFAULT

THEME = {
    "bg": "#0f1117",
    "surface": "#1a1d2e",
    "surface2": "#252840",
    "border": "#2e3250",
    "text": "#e2e8f0",
    "text_muted": "#8892a4",
    "arima_color": "#60a5fa",    # blue
    "xgb_color": "#f472b6",      # pink
    "history_color": "#94a3b8",  # slate
    "ci_opacity": 0.15,
}
