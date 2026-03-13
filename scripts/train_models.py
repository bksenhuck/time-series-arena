"""
scripts/train_models.py
───────────────────────
Train the global XGBoost model and (optionally) compute walk-forward metrics
for all stored Wikipedia series.

Usage:
    python -m scripts.train_models [--force] [--no-metrics]

Options:
    --force       Force retraining even if a saved model already exists.
    --no-metrics  Skip the walk-forward evaluation step (faster).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.data_loader import get_all_series, get_available_pages
from backend.models import XGBoostGlobalForecaster
from backend.models.model_registry import is_saved, load_model, save_model
from backend.utils.io_utils import init_db, get_state, set_state
from config import XGB_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("train_models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost global model.")
    p.add_argument("--force", action="store_true",
                   help="Force retraining even if model already exists.")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip walk-forward evaluation after training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    pages = get_available_pages()
    if not pages:
        log.error("No series in database. Run collect_data.py first.")
        sys.exit(1)

    log.info("%d series available in database.", len(pages))

    # ── Load or train XGBoost ──────────────────────────────────────────────
    needs_train = args.force

    if not needs_train and is_saved(XGB_MODEL_PATH):
        log.info("Loading existing model from %s …", XGB_MODEL_PATH)
        xgb = load_model(XGB_MODEL_PATH)
        missing = set(pages) - set(xgb.known_series)
        if missing:
            log.info("%d new series detected — retraining.", len(missing))
            needs_train = True
            set_state("metrics_computed", "false")
    else:
        needs_train = True

    if needs_train:
        log.info("Training XGBoost global model on %d series …", len(pages))
        t0 = time.time()
        df = get_all_series()
        xgb = XGBoostGlobalForecaster()
        xgb.fit(df)
        save_model(xgb, XGB_MODEL_PATH)
        elapsed = time.time() - t0
        log.info(
            "Training complete in %.1fs — series=%d  residual_std=%.1f",
            elapsed, len(xgb.known_series), xgb.residual_std,
        )
    else:
        log.info("Model already up-to-date (use --force to retrain).")

    # ── Walk-forward evaluation ────────────────────────────────────────────
    if args.no_metrics:
        log.info("Skipping metrics (--no-metrics).")
        return

    if get_state("metrics_computed") == "true" and not args.force:
        log.info("Metrics already cached (use --force to recompute).")
        return

    log.info("Computing walk-forward evaluation metrics …")
    from backend.pipeline.training_pipeline import _compute_metrics  # noqa: PLC2701
    _compute_metrics(xgb)
    set_state("metrics_computed", "true")
    log.info("Metrics computed and cached.")


if __name__ == "__main__":
    main()
