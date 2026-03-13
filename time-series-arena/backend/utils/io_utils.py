"""Generic object/state persistence helpers for backend orchestration."""
import joblib

from backend.db.database import db_conn, ensure_dir


def save_object(obj, path: str) -> None:
    ensure_dir(path)
    joblib.dump(obj, path)


def load_object(path: str):
    return joblib.load(path)


def get_state(key: str) -> str | None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key=?",
            (key,),
        ).fetchone()
        return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)",
            (key, value),
        )
