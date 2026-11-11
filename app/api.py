"""FastAPI entry point for the local NEXUS 5G dashboard."""

import argparse
from datetime import date
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.service import (
    get_models,
    get_overview,
    history,
    prediction_detail,
    run_prediction,
)
from src.preprocessing.data import CATEGORICAL_KPIS, KPI_COLUMNS


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str | None = Field(default=None, max_length=120)
    KPIs: dict[str, list[Any]]

    @field_validator("KPIs")
    @classmethod
    def validate_kpis(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        missing = sorted(set(KPI_COLUMNS) - set(value))
        if missing:
            raise ValueError(f"Missing KPI channels: {', '.join(missing)}")
        for channel in KPI_COLUMNS:
            values = value[channel]
            if len(values) != 128:
                raise ValueError(f"{channel} must contain exactly 128 timestep values.")
            if channel in CATEGORICAL_KPIS:
                if any(item is not None and not isinstance(item, str) for item in values):
                    raise ValueError(f"{channel} values must be strings or null.")
            else:
                for index, item in enumerate(values):
                    if item is None:
                        continue
                    try:
                        float(item)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValueError(
                            f"{channel}[{index}] must be numeric or null."
                        ) from error
        return value


app = FastAPI(
    title="NEXUS 5G Network Intelligence API",
    version="0.1.0",
    description=(
        "Local REST API for batch model inference, model status, and prediction history."
    ),
)
MAX_REQUEST_BYTES = 2 * 1024 * 1024


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="JSON request exceeds 2 MiB.")
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    return get_models()


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    return get_overview()


@app.post("/api/predictions", status_code=201)
def create_prediction(request: PredictionRequest) -> dict[str, Any]:
    try:
        return run_prediction(request.sample_id, request.KPIs)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{error}. Train/select the models locally or set INA_MODELS_PATH "
                "to the directory containing results/models."
            ),
        ) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/predictions")
def predictions(
    limit: int = Query(default=100, ge=1, le=500),
    search: str | None = None,
    status: str | None = Query(default=None, pattern="^(normal|anomaly)$"),
    predicted_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict[str, Any]]:
    records = history(limit, search, status, predicted_type)
    if date_from:
        records = [
            row for row in records
            if row["created_at"][:10] >= date_from.isoformat()
        ]
    if date_to:
        records = [
            row for row in records
            if row["created_at"][:10] <= date_to.isoformat()
        ]
    return records


@app.get("/api/predictions/{prediction_id}")
def prediction(prediction_id: str) -> dict[str, Any]:
    record = prediction_detail(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return record


WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        requested = (WEB_DIST / path).resolve()
        if path and requested.is_relative_to(WEB_DIST.resolve()) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(WEB_DIST / "index.html")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("INA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INA_PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
