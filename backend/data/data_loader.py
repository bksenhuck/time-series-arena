"""
Reads and writes Wikipedia pageview data to/from SQLite.
High-level data access layer — no HTTP calls here.
"""
import logging
from typing import Optional

import pandas as pd

from backend.utils.io_utils import db_conn, get_state, set_state

logger = logging.getLogger(__name__)


# ── Write ─────────────────────────────────────────────────────────────────────

def store_pageviews(df: pd.DataFrame) -> None:
    with db_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO pageviews (date, series_id, views) VALUES (?, ?, ?)",
            df[["date", "series_id", "views"]].itertuples(index=False, name=None),
        )


def load_all_pages(pages: list[str], delay: float = 0.3) -> int:
    """Download and store data for all pages. Returns number of pages stored."""
    from backend.data.wiki_collector import fetch_many
    batches = fetch_many(pages, delay=delay, skip_existing=has_data)
    for page, df in batches.items():
        store_pageviews(df)
    set_state("data_loaded", "true")
    return len(batches) + len([p for p in pages if has_data(p)])


# ── Read ──────────────────────────────────────────────────────────────────────

def has_data(page: str) -> bool:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pageviews WHERE series_id=?", (page,)
        ).fetchone()
        return row["cnt"] > 0


def get_available_pages() -> list[str]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT series_id FROM pageviews ORDER BY series_id"
        ).fetchall()
    return [r["series_id"] for r in rows]


def get_series(page: str) -> pd.DataFrame:
    """Returns a DatetimeIndex-indexed DataFrame with column 'views'."""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT date, views FROM pageviews WHERE series_id=? ORDER BY date", (page,)
        ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "views"])
    df["date"] = pd.to_datetime(df["date"])
    df["views"] = df["views"].astype(float)
    return df.set_index("date")


def get_all_series() -> pd.DataFrame:
    """Returns long-format DataFrame with columns [date, series_id, views]."""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT date, series_id, views FROM pageviews ORDER BY series_id, date"
        ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "series_id", "views"])
    df["date"] = pd.to_datetime(df["date"])
    df["views"] = df["views"].astype(float)
    return df
