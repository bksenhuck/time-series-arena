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
DATA_MIN_DATE = "2020-01-01"
DATA_MAX_DATE = "2025-12-31"

# Last N months are test; the rest is train.
TEST_WINDOW_MONTHS = 5

_max_dt = _parse_ymd(DATA_MAX_DATE)
TRAIN_TEST_SPLIT_DATE = _shift_months(_max_dt, -TEST_WINDOW_MONTHS).isoformat()

# Aliases used across the codebase
DATA_START_DATE = DATA_MIN_DATE
DATA_END_DATE = DATA_MAX_DATE
TRAIN_END_DATE = TRAIN_TEST_SPLIT_DATE

# ── Forecasting ───────────────────────────────────────────────────────────────
TEST_WINDOW_DAYS = max(1, (_parse_ymd(DATA_MAX_DATE) - _parse_ymd(TRAIN_TEST_SPLIT_DATE)).days)

_BASE_FORECAST_HORIZONS = [7, 14, 20, 25, 30, 50]
FORECAST_HORIZONS = sorted(set(_BASE_FORECAST_HORIZONS + [TEST_WINDOW_DAYS]))

SCENARIO_WINDOWS_MONTHS = [3, 6, 9]

# ── Chart ─────────────────────────────────────────────────────────────────────
MAX_LINE_WIDTH = 3.0
MIN_LINE_WIDTH = 1.0
DEFAULT_LINE_WIDTH = 1.0
REAL_DASHED_LINE_WIDTH = 3.0
FORECAST_TRAIN_HISTORY_DAYS = 0
TRAIN_FIT_WINDOW_DAYS = 180
