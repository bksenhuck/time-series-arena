"""
Shared callback helpers for forecast interactions.
Imported and called by individual page modules.
"""
from __future__ import annotations

import logging

import requests
from dash import Input, Output, callback

from config import API_BASE

_API = API_BASE
log = logging.getLogger(__name__)


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_series(page: str) -> dict | None:
    """GET /api/series/{page} → {dates, views} or None on error."""
    try:
        url = f"{_API}/api/series/{requests.utils.quote(page, safe='')}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("fetch_series(%s) failed: %s", page, exc)
        return None


def post_forecast(page: str, model: str, horizon: int) -> tuple[dict, str]:
    """POST /api/forecast → (result_dict, status_message)."""
    try:
        resp = requests.post(
            f"{_API}/api/forecast",
            json={"page": page, "model": model, "horizon": horizon},
            timeout=60,
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.text
            return {}, f"⚠ API error {resp.status_code}: {detail}"
        label = f"{page.replace('_', ' ')} | {horizon}d | {model}"
        return resp.json(), f"✓ Forecast ready — {label}"
    except Exception as exc:
        log.error("post_forecast(%s) failed: %s", page, exc)
        return {}, f"⚠ Request failed: {exc}"


def fetch_all_horizon_forecasts(page: str, model: str) -> dict:
    """GET /api/forecasts/{page}/{model} → {model: {horizon: fc_dict, ...}, ...}"""
    try:
        url = f"{_API}/api/forecasts/{requests.utils.quote(page, safe='')}/{model}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("fetch_all_horizon_forecasts(%s, %s) failed: %s", page, model, exc)
        return {}


def fetch_train_fit_forecasts(page: str) -> dict:
    """GET /api/train-fit/{page} -> {actual, arima, xgboost, prophet, auto_arima}."""
    try:
        url = f"{_API}/api/train-fit/{requests.utils.quote(page, safe='')}"
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("fetch_train_fit_forecasts(%s) failed: %s", page, exc)
        return {}


def fetch_page_metrics(page: str) -> list[dict]:
    """GET /api/metrics/{page} → list of metric dicts."""
    try:
        url = f"{_API}/api/metrics/{requests.utils.quote(page, safe='')}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("metrics", [])
    except Exception as exc:
        log.warning("fetch_page_metrics(%s) failed: %s", page, exc)
        return []


def fetch_scenario_comparison(page: str | None = None) -> dict:
    """GET /api/scenarios/compare?page=... -> scenario comparison payload."""
    try:
        url = f"{_API}/api/scenarios/compare"
        params = {"page": page} if page else None
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("fetch_scenario_comparison(%s) failed: %s", page, exc)
        return {"rows": [], "summary": []}


# ── Reusable Dash callback factory ────────────────────────────────────────────

def register_page_loader(page_dropdown_id: str, pages_store_id: str) -> None:
    """
    Register a Dash callback that populates a page-selector dropdown
    with all available series from the backend.
    """
    @callback(
        Output(page_dropdown_id, "options"),
        Output(page_dropdown_id, "value"),
        Output(pages_store_id, "data"),
        Input(page_dropdown_id, "id"),
    )
    def _load(dropdown_id):
        try:
            resp = requests.get(f"{_API}/api/pages", timeout=10)
            resp.raise_for_status()
            pages = resp.json().get("pages", [])
        except Exception as exc:
            log.warning("register_page_loader failed: %s", exc)
            pages = []
        options = [{"label": p.replace("_", " "), "value": p} for p in pages]
        return options, (pages[0] if pages else None), pages
