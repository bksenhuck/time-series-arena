"""
Reads and writes Wikipedia pageview data.
Includes SQLite access and Wikimedia API fetch helpers.
"""
import logging
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from backend.db.database import db_conn
from backend.utils.io_utils import get_state, set_state
from config import DATA_END_DATE, DATA_START_DATE, TRAIN_END_DATE

logger = logging.getLogger(__name__)

_API_TEMPLATE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{page}/daily/{start}/{end}"
)
_HEADERS = {"User-Agent": "TimeSeriesArena/1.0 (clean-arch)"}


# ── Write ─────────────────────────────────────────────────────────────────────

def store_pageviews(df: pd.DataFrame) -> None:
    with db_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO pageviews "
            "(date, series_id, views) VALUES (?, ?, ?)",
            df[["date", "series_id", "views"]].itertuples(
                index=False, name=None
            ),
        )


def load_all_pages(pages: list[str], delay: float = 0.3) -> int:
    """Download and store data for all pages. Returns number stored."""
    already = [p for p in pages if has_data(p)]
    to_fetch = [p for p in pages if not has_data(p)]
    logger.info(
        "Pageview data: %d pages already cached, %d to download",
        len(already), len(to_fetch),
    )
    batches = fetch_many(to_fetch, delay=delay)
    for page, df in batches.items():
        store_pageviews(df)
    set_state("data_loaded", "true")
    return len(batches) + len(already)


# ── Read ──────────────────────────────────────────────────────────────────────

def has_data(page: str) -> bool:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pageviews WHERE series_id=?",
            (page,),
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
            "SELECT date, views FROM pageviews "
            "WHERE series_id=? ORDER BY date",
            (page,),
        ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "views"])
    df["date"] = pd.to_datetime(df["date"])
    df["views"] = df["views"].astype(float)
    return df.set_index("date")


def get_train_series(page: str) -> pd.DataFrame:
    """Like get_series but capped at TRAIN_END_DATE — safe for model fitting.
    Data after TRAIN_END_DATE is kept only as out-of-sample "real values" for plotting."""
    df = get_series(page)
    return df[df.index <= TRAIN_END_DATE]


def get_all_series() -> pd.DataFrame:
    """Returns long-format DataFrame: [date, series_id, views]."""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT date, series_id, views FROM pageviews "
            "ORDER BY series_id, date"
        ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "series_id", "views"])
    df["date"] = pd.to_datetime(df["date"])
    df["views"] = df["views"].astype(float)
    return df


# ── Wikimedia fetch helpers ───────────────────────────────────────────────────

def to_wiki_date(date_str: str) -> str:
    """Convert '2022-01-01' -> '2022010100' (Wikimedia format)."""
    return date_str.replace("-", "") + "00"


def fetch_pageviews(
    page: str,
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
) -> Optional[pd.DataFrame]:
    """Fetch daily pageviews for one article (or None on failure)."""
    url = _API_TEMPLATE.format(
        page=requests.utils.quote(page, safe=""),
        start=to_wiki_date(start),
        end=to_wiki_date(end),
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code == 404:
            logger.warning("No data for page: %s", page)
            return None
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return None
        return pd.DataFrame(
            {
                "date": [x["timestamp"][:8] for x in items],
                "series_id": [page] * len(items),
                "views": [x["views"] for x in items],
            }
        )
    except Exception as exc:
        logger.error("fetch_pageviews(%s) failed: %s", page, exc)
        return None


def fetch_many(
    pages: list[str],
    delay: float = 0.3,
    skip_existing=None,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple pages sequentially with a polite request delay."""
    out: dict[str, pd.DataFrame] = {}
    bar = tqdm(
        pages,
        desc="Downloading pageviews",
        unit="page",
        ncols=90,
        colour="cyan",
    )
    for i, page in enumerate(bar, start=1):
        if skip_existing and skip_existing(page):
            bar.set_postfix_str(f"{page[:30]} [cached]")
            continue
        bar.set_postfix_str(page[:40])
        df = fetch_pageviews(page)
        if df is not None and not df.empty:
            out[page] = df
            bar.set_postfix_str(f"{page[:30]} ✓ {len(df)}d")
        else:
            bar.set_postfix_str(f"{page[:30]} ⚠ no data")
        if i < len(pages):
            time.sleep(delay)
    return out
