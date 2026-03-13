"""
scripts/reset_db.py
───────────────────
Utility to wipe and re-initialise the SQLite database.
Useful during development when the schema changes.

Usage:
    python -m scripts.reset_db [--confirm]

WARNING: This permanently deletes all cached data, metrics and forecasts.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.io_utils import init_db
from config import DB_PATH

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reset_db")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wipe and re-create the Time Series Arena database."
    )
    p.add_argument("--confirm", action="store_true",
                   help="Required flag to confirm destructive action.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.confirm:
        print("This will DELETE the entire database.")
        print(f"  Path: {DB_PATH}")
        print("Re-run with --confirm to proceed.")
        sys.exit(0)

    db_file = Path(DB_PATH)
    if db_file.exists():
        log.info("Removing %s …", db_file)
        os.remove(db_file)
    else:
        log.info("Database does not exist yet — nothing to remove.")

    # Also clear any XGBoost model artefacts and state flags
    from config import XGB_MODEL_PATH
    for path in [XGB_MODEL_PATH]:
        if Path(path).exists():
            os.remove(path)
            log.info("Removed %s", path)

    log.info("Re-initialising schema …")
    init_db()
    log.info("Done. Fresh database created at %s", DB_PATH)


if __name__ == "__main__":
    main()
