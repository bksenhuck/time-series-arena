"""Train ML models and compute cached metrics."""

from ml.pipeline.training_pipeline import run_pipeline


if __name__ == "__main__":
    status = run_pipeline(force_retrain=True)
    print(status)
