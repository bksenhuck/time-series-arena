"""
scripts/collect_data.py
───────────────────────
Download Wikipedia pageview data for all configured pages and store them
in the SQLite database.

Usage:
    python -m scripts.collect_data [--pages PAGE1 PAGE2 ...] [--force]

Examples:
    # Collect all pages defined in config.py
    python -m scripts.collect_data

    # Collect a specific subset
    python -m scripts.collect_data --pages Python JavaScript Machine_learning

    # Re-download even if data already exists
    python -m scripts.collect_data --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure the project root is on the Python path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.data_loader import has_data, load_all_pages, store_pageviews
from backend.data.wiki_collector import fetch_pageviews
from backend.utils.io_utils import init_db
from config import WIKIPEDIA_PAGES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("collect_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Wikipedia pageview data.")
    p.add_argument(
        "--pages", nargs="+", default=None,
        help="Specific page names to collect (default: all pages in config.py).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download pages that are already in the database.",
    )
    p.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between API requests (default: 0.3).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pages = args.pages or WIKIPEDIA_PAGES

    init_db()

    if args.force:
        log.info("Force mode: re-downloading all %d pages", len(pages))
        to_fetch = pages
    else:
        to_fetch = [p for p in pages if not has_data(p)]
        already  = len(pages) - len(to_fetch)
        if already:
            log.info("%d pages already cached — skipping", already)

    if not to_fetch:
        log.info("Nothing to fetch. Use --force to re-download.")
        return

    log.info("Fetching %d pages (delay=%.1fs) …", len(to_fetch), args.delay)
    ok = skipped = errors = 0

    for i, page in enumerate(to_fetch, 1):
        df = fetch_pageviews(page)
        if df is None:
            log.warning("[%d/%d] SKIP  %s", i, len(to_fetch), page)
            skipped += 1
        else:
            store_pageviews(df)
            log.info("[%d/%d] OK    %s  (%d rows)", i, len(to_fetch), page, len(df))
            ok += 1

    import time
    time.sleep(args.delay)

    log.info("Done — ok=%d  skipped=%d  errors=%d", ok, skipped, errors)


if __name__ == "__main__":
    main()
