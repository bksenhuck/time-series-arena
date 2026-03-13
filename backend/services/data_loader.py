import time
import logging
from typing import Optional

import pandas as pd
import requests

from backend.config import DATA_START_DATE, DATA_END_DATE, WIKIPEDIA_PAGES
from backend.db.session import db_conn

logger = logging.getLogger(__name__)

_WIKI_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{page}/daily/{start}/{end}"
)
_HEADERS = {"User-Agent": "TimeSeriesArena/1.0 (educational project)"}


def _date_to_wiki(date_str: str) -> str:
    return date_str.replace("-", "") + "00"


def fetch_page_views(page: str, start: str = DATA_START_DATE, end: str = DATA_END_DATE) -> Optional[pd.DataFrame]:
    url = _WIKI_API.format(
        page=requests.utils.quote(page, safe=""),
        start=_date_to_wiki(start),
        end=_date_to_wiki(end),
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code == 404:
            logger.warning("Page not found: %s", page)
            return None
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return None
        df = pd.DataFrame(items)[["timestamp", "views"]]
        df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        df["series_id"] = page
        df["views"] = df["views"].astype(int)
        return df[["date", "series_id", "views"]]
    except Exception as exc:
        logger.error("Error fetching %s: %s", page, exc)
        return None


def store_pageviews(df: pd.DataFrame) -> None:
    with db_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO pageviews (date, series_id, views) VALUES (?, ?, ?)",
            df[["date", "series_id", "views"]].itertuples(index=False, name=None),
        )


def load_all_data(pages: list[str] = WIKIPEDIA_PAGES, delay: float = 0.5) -> int:
    total = 0
    for i, page in enumerate(pages):
        if has_page_data(page):
            logger.info("[%d/%d] Cached: %s", i + 1, len(pages), page)
            total += 1
            continue
        logger.info("[%d/%d] Fetching: %s", i + 1, len(pages), page)
        df = fetch_page_views(page)
        if df is not None and not df.empty:
            store_pageviews(df)
            total += 1
        time.sleep(delay)
    return total


def has_page_data(page: str) -> bool:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM pageviews WHERE series_id = ?", (page,)
        ).fetchone()
        return row["cnt"] > 0


def get_available_pages() -> list[str]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT series_id FROM pageviews ORDER BY series_id"
        ).fetchall()
        return [r["series_id"] for r in rows]


def get_series(page: str) -> pd.DataFrame:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT date, views FROM pageviews WHERE series_id = ? ORDER BY date",
            (page,),
        ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "views"])
    df["date"] = pd.to_datetime(df["date"])
    df["views"] = df["views"].astype(float)
    return df.set_index("date")


def get_all_series() -> pd.DataFrame:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT date, series_id, views FROM pageviews ORDER BY series_id, date"
        ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "series_id", "views"])
    df["date"] = pd.to_datetime(df["date"])
    df["views"] = df["views"].astype(float)
    return df


def get_app_state(key: str) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_app_state(key: str, value: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)", (key, value)
        )
