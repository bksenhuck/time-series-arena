"""Download and cache Wikipedia pageview data into SQLite."""

from ml.data.loaders import load_all_pages
from config import WIKIPEDIA_PAGES


if __name__ == "__main__":
    n = load_all_pages(WIKIPEDIA_PAGES)
    print(f"Loaded/cached data for {n} pages")
