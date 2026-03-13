import os
from datetime import date
from calendar import monthrange

from dotenv import load_dotenv

load_dotenv()


def _parse_ymd(value: str) -> date:
	y, m, d = (int(x) for x in value.split("-"))
	return date(y, m, d)


def _shift_months(base: date, months: int) -> date:
	"""Shift date by N months, clamping day to the target month length."""
	total = base.year * 12 + (base.month - 1) + months
	year = total // 12
	month = (total % 12) + 1
	day = min(base.day, monthrange(year, month)[1])
	return date(year, month, day)


def _parse_int_csv(value: str) -> list[int]:
	return [int(x.strip()) for x in value.split(",") if x.strip()]

# ── Server ────────────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", 8050))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
API_BASE = os.getenv("API_BASE", "http://localhost:8050")

# ── Data ──────────────────────────────────────────────────────────────────────
# Explicit data window and train/test split
DATA_MIN_DATE = os.getenv("DATA_MIN_DATE", os.getenv("DATA_START_DATE", "2020-01-01"))
DATA_MAX_DATE = os.getenv("DATA_MAX_DATE", os.getenv("DATA_END_DATE", "2025-12-31"))

# Moving split: last N months are test, the rest is train.
# If TEST_WINDOW_MONTHS <= 0, fallback to explicit TRAIN_TEST_SPLIT_DATE.
TEST_WINDOW_MONTHS = int(os.getenv("TEST_WINDOW_MONTHS", "5"))
_explicit_split = os.getenv("TRAIN_TEST_SPLIT_DATE", os.getenv("TRAIN_END_DATE", "2024-12-31"))
if TEST_WINDOW_MONTHS > 0:
	_max_dt = _parse_ymd(DATA_MAX_DATE)
	TRAIN_TEST_SPLIT_DATE = _shift_months(_max_dt, -TEST_WINDOW_MONTHS).isoformat()
else:
	TRAIN_TEST_SPLIT_DATE = _explicit_split

# Backward-compatible aliases used across the codebase
DATA_START_DATE = DATA_MIN_DATE
DATA_END_DATE = DATA_MAX_DATE
TRAIN_END_DATE = TRAIN_TEST_SPLIT_DATE

# ── Forecasting ───────────────────────────────────────────────────────────────
TEST_WINDOW_DAYS = max(
	1,
	(_parse_ymd(DATA_MAX_DATE) - _parse_ymd(TRAIN_TEST_SPLIT_DATE)).days,
)

# Keep short-term horizons and always include full test-window horizon.
_BASE_FORECAST_HORIZONS = [7, 14, 20, 25, 30, 50]
FORECAST_HORIZONS = sorted(set(_BASE_FORECAST_HORIZONS + [TEST_WINDOW_DAYS]))

# Scenario comparison windows (months of test split)
SCENARIO_WINDOWS_MONTHS = _parse_int_csv(os.getenv("SCENARIO_WINDOWS_MONTHS", "3,6,9"))

# ── Chart ─────────────────────────────────────────────────────────────────────
# Forecast line width: thickest = shortest horizon, thinnest = longest horizon
MAX_LINE_WIDTH = float(os.getenv("MAX_LINE_WIDTH", "3.0"))
MIN_LINE_WIDTH = float(os.getenv("MIN_LINE_WIDTH", "1.0"))

# Global line width policy for dashboards
DEFAULT_LINE_WIDTH = float(os.getenv("DEFAULT_LINE_WIDTH", "1.0"))
REAL_DASHED_LINE_WIDTH = float(os.getenv("REAL_DASHED_LINE_WIDTH", "3.0"))

# Training history shown on forecast chart:
# 0 = show full training history; N > 0 = show last N days only
FORECAST_TRAIN_HISTORY_DAYS = int(os.getenv("FORECAST_TRAIN_HISTORY_DAYS", "0"))

# Number of training days to render for in-sample fitted model lines
TRAIN_FIT_WINDOW_DAYS = int(os.getenv("TRAIN_FIT_WINDOW_DAYS", "180"))
