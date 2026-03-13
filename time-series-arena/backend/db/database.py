"""Database session and schema initialization."""
import contextlib
import os
import sqlite3

from config import DB_PATH


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


@contextlib.contextmanager
def db_conn():
    ensure_dir(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pageviews (
                date      TEXT NOT NULL,
                series_id TEXT NOT NULL,
                views     INTEGER NOT NULL,
                PRIMARY KEY (date, series_id)
            );
            CREATE TABLE IF NOT EXISTS forecast_cache (
                series_id     TEXT NOT NULL,
                model_type    TEXT NOT NULL,
                horizon       INTEGER NOT NULL,
                forecast_json TEXT NOT NULL,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (series_id, model_type, horizon)
            );
            CREATE TABLE IF NOT EXISTS metrics_cache (
                series_id   TEXT NOT NULL,
                model_type  TEXT NOT NULL,
                mae         REAL,
                rmse        REAL,
                mape        REAL,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (series_id, model_type)
            );
            CREATE TABLE IF NOT EXISTS app_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
