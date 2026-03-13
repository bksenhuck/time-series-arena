"""
scripts/evaluate_models.py
──────────────────────────
Run walk-forward evaluation for ARIMA and / or XGBoost on all (or selected)
series, then print a summary table to stdout.

Usage:
    python -m scripts.evaluate_models [--pages PAGE1 PAGE2 ...] [--model arima|xgboost|both] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.data_loader import get_available_pages, get_series
from backend.evaluation.metrics import walk_forward_evaluate
from backend.models import ARIMAForecaster, XGBoostGlobalForecaster
from backend.models.model_registry import is_saved, load_model
from backend.pipeline.training_pipeline import _cache, _has_metrics  # noqa: PLC2701
from backend.services.forecast_service import generate_forecast
from backend.utils.io_utils import init_db
from config import XGB_MODEL_PATH

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("evaluate_models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ARIMA / XGBoost walk-forward.")
    p.add_argument("--pages",  nargs="+", default=None)
    p.add_argument("--model",  choices=["arima", "xgboost", "both"], default="both")
    p.add_argument("--force",  action="store_true",
                   help="Recompute even if metrics are already cached.")
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--splits",  type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    pages = args.pages or get_available_pages()
    if not pages:
        log.error("No series in database. Run collect_data.py first.")
        sys.exit(1)

    # Prepare models
    arima = xgb = None
    if args.model in ("arima", "both"):
        arima = ARIMAForecaster()
        arima.fit(None)

    if args.model in ("xgboost", "both"):
        if not is_saved(XGB_MODEL_PATH):
            log.error("XGBoost model not found at %s. Run train_models.py first.", XGB_MODEL_PATH)
            sys.exit(1)
        xgb = load_model(XGB_MODEL_PATH)

    results: list[dict] = []

    for page in pages:
        try:
            series = get_series(page)["views"]
            if len(series) < 60:
                log.warning("Series too short (%d rows) for %s — skipping.", len(series), page)
                continue

            for model_name, model_obj in [("arima", arima), ("xgboost", xgb)]:
                if model_obj is None:
                    continue
                if not args.force and _has_metrics(page, model_name):
                    log.debug("Cached metrics found for %s / %s", page, model_name)
                    continue

                def _fn(train, h, _m=model_obj, _p=page):
                    return np.array(generate_forecast(_m, train, _p, h).mean)

                m = walk_forward_evaluate(series, _fn,
                                          horizon=args.horizon,
                                          n_splits=args.splits)
                _cache(page, model_name, m)
                results.append({"page": page, "model": model_name, **m})
                log.info("%-35s  %-8s  MAE=%-8.1f RMSE=%-8.1f MAPE=%.1f%%",
                         page[:35], model_name,
                         m["mae"] or 0, m["rmse"] or 0, m["mape"] or 0)

        except Exception as exc:
            log.warning("Failed for %s: %s", page, exc)

    if results:
        _print_summary(results)
    else:
        log.info("All metrics were already cached (use --force to recompute).")


def _print_summary(results: list[dict]) -> None:
    def _avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "─" * 60)
    print(f"{'Model':<12}  {'Pages':>6}  {'MAE':>8}  {'RMSE':>8}  {'MAPE %':>8}")
    print("─" * 60)

    for model_name in ("arima", "xgboost"):
        rows = [r for r in results if r["model"] == model_name]
        if not rows:
            continue
        print(f"{model_name.upper():<12}  {len(rows):>6}  "
              f"{_avg(rows, 'mae') or 0:>8.1f}  "
              f"{_avg(rows, 'rmse') or 0:>8.1f}  "
              f"{_avg(rows, 'mape') or 0:>8.1f}")

    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
