"""
Upload local artifacts to Google Cloud Storage.

Usage:
    python -m backend.pipelines.deploy.upload_to_gcs
    python -m backend.pipelines.deploy.upload_to_gcs --dry-run

Artifacts uploaded:
    - data/raw/timeseries.db  → gs://BUCKET/data/raw/timeseries.db
    - models/                 → gs://BUCKET/models/
    - data/processed/         → gs://BUCKET/data/processed/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[4]  # project root

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GCS_DB_BLOB_PATH = os.environ.get("GCS_DB_BLOB_PATH", "data/raw/timeseries.db")
GCS_MODELS_PREFIX = os.environ.get("GCS_MODELS_PREFIX", "models")
GCS_PROCESSED_PREFIX = os.environ.get("GCS_PROCESSED_PREFIX", "data/processed")

DB_LOCAL = ROOT / os.environ.get("DB_PATH", "data/raw/timeseries.db")
MODELS_LOCAL = ROOT / os.environ.get("MODEL_DIR", "models")
PROCESSED_LOCAL = ROOT / "data" / "processed"


def _upload_file(bucket, local_path: Path, blob_path: str, dry_run: bool) -> None:
    size_mb = local_path.stat().st_size / 1_048_576
    if dry_run:
        print(f"  [dry-run] {local_path.name} → gs://{bucket.name}/{blob_path}  ({size_mb:.2f} MB)")
        return
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    print(f"  uploaded  {local_path.name} → gs://{bucket.name}/{blob_path}  ({size_mb:.2f} MB)")


def _upload_dir(bucket, local_dir: Path, gcs_prefix: str, dry_run: bool) -> int:
    count = 0
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_dir)
        blob_path = f"{gcs_prefix}/{relative.as_posix()}"
        _upload_file(bucket, local_file, blob_path, dry_run)
        count += 1
    return count


def run(dry_run: bool = False) -> None:
    if not BUCKET_NAME:
        raise EnvironmentError(
            "GCS_BUCKET_NAME not set. Please fill .env with your bucket name."
        )

    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "google-cloud-storage not installed. Run: pip install google-cloud-storage"
        )

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    print(f"\nTarget bucket: gs://{BUCKET_NAME}")
    print("=" * 50)

    # 1. SQLite database
    print("\n[1/3] Database")
    if DB_LOCAL.exists():
        _upload_file(bucket, DB_LOCAL, GCS_DB_BLOB_PATH, dry_run)
    else:
        print(f"  WARNING: database not found at {DB_LOCAL}")

    # 2. Trained models
    print("\n[2/3] Models")
    if MODELS_LOCAL.exists():
        count = _upload_dir(bucket, MODELS_LOCAL, GCS_MODELS_PREFIX, dry_run)
        print(f"  {count} model file(s) processed")
    else:
        print(f"  WARNING: models dir not found at {MODELS_LOCAL}")

    # 3. Processed artifacts (parquet)
    print("\n[3/3] Processed artifacts")
    if PROCESSED_LOCAL.exists():
        count = _upload_dir(bucket, PROCESSED_LOCAL, GCS_PROCESSED_PREFIX, dry_run)
        print(f"  {count} artifact file(s) processed")
    else:
        print(f"  WARNING: processed dir not found at {PROCESSED_LOCAL}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload artifacts to GCS")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
