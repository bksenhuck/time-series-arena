from fastapi import APIRouter, HTTPException

from backend.data.data_loader import get_available_pages, get_series
from backend.pipeline.training_pipeline import get_page_metrics

router = APIRouter()


@router.get("/pages")
def list_pages():
    return {"pages": get_available_pages()}


@router.get("/series/{page}")
def series_data(page: str):
    try:
        df = get_series(page)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for page: {page}")
    return {
        "series_id": page,
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "views": df["views"].tolist(),
    }


@router.get("/metrics/{page}")
def page_metrics(page: str):
    metrics = get_page_metrics(page)
    return {"series_id": page, "metrics": metrics}


@router.get("/metrics")
def all_metrics():
    from backend.pipeline.training_pipeline import get_all_metrics
    return {"metrics": get_all_metrics()}
