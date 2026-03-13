"""
Fetches daily Wikipedia pageview data from the Wikimedia REST API.
Responsible only for HTTP requests — storage is handled by data_loader.
"""
import logging
import time
from typing import Optional

import pandas as pd
import requests

from backend.utils.date_utils import to_wiki_date
from config import DATA_START_DATE, DATA_END_DATE

logger = logging.getLogger(__name__)

_API_TEMPLATE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{page}/daily/{start}/{end}"
)
_HEADERS = {"User-Agent": "TimeSeriesArena/1.0 (educational project)"}


def fetch_pageviews(
    page: str,
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily pageviews for one Wikipedia article.
    Returns a DataFrame with columns [date, series_id, views] or None on failure.
    """
    url = _API_TEMPLATE.format(
        page=requests.utils.quote(page, safe=""),
        start=to_wiki_date(start),
        end=to_wiki_date(end),
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


def fetch_many(
    pages: list[str],
    delay: float = 0.3,
    skip_existing: callable = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch pageviews for a list of pages.
    skip_existing(page) -> bool: callback to skip already-cached pages.
    Returns {page: DataFrame} for successfully fetched pages.
    """
    results = {}
    total = len(pages)
    for i, page in enumerate(pages):
        if skip_existing and skip_existing(page):
            logger.info("[%d/%d] Cached: %s", i + 1, total, page)
            continue
        logger.info("[%d/%d] Fetching: %s", i + 1, total, page)
        df = fetch_pageviews(page)
        if df is not None and not df.empty:
            results[page] = df
        time.sleep(delay)
    return results
