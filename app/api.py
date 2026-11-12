"""FastAPI entry point for the local NEXUS 5G dashboard."""

import argparse
from collections import deque
from datetime import date
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.service import (
    get_models,
    get_overview,
    history,
    prediction_detail,
    random_test_sample,
    run_prediction,
)
from app.settings import public_demo_enabled
from src.preprocessing.data import CATEGORICAL_KPIS, KPI_COLUMNS

logger = logging.getLogger(__name__)

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
        unexpected = sorted(set(value) - set(KPI_COLUMNS))
        if unexpected:
            raise ValueError(f"Unexpected KPI channels: {', '.join(unexpected)}")
        for channel in KPI_COLUMNS:
            values = value[channel]
            if len(values) != 128:
                raise ValueError(f"{channel} must contain exactly 128 timestep values.")
            if channel in CATEGORICAL_KPIS:
                if any(
                    item is not None
                    and (not isinstance(item, str) or len(item) > 128)
                    for item in values
                ):
                    raise ValueError(f"{channel} values must be strings or null.")
            else:
                for index, item in enumerate(values):
                    if item is None:
                        continue
                    if isinstance(item, bool):
                        raise ValueError(f"{channel}[{index}] must be numeric or null.")
                    try:
                        number = float(item)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValueError(
                            f"{channel}[{index}] must be numeric or null."
                        ) from error
                    if not math.isfinite(number):
                        raise ValueError(
                            f"{channel}[{index}] must be finite or null."
                        )
        return value


app = FastAPI(
    title="NEXUS 5G Network Intelligence API",
    version="0.1.0",
    description=(
        "Local REST API for batch model inference, model status, and prediction history."
    ),
    docs_url=None if public_demo_enabled() else "/docs",
    redoc_url=None if public_demo_enabled() else "/redoc",
    openapi_url=None if public_demo_enabled() else "/openapi.json",
)
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class RequestBodySizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/api/predictions"
        ):
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value.decode("latin1")
                for name, value in scope["headers"]
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length."},
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_length < 0:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length."},
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_length > self.max_bytes:
                response = JSONResponse(
                    {"detail": "JSON request exceeds 2 MiB."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                response = JSONResponse(
                    {"detail": "JSON request exceeds 2 MiB."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_body() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_body, send)


app.add_middleware(
    RequestBodySizeLimitMiddleware,
    max_bytes=MAX_REQUEST_BYTES,
)

_public_demo_rate_windows = {
    "prediction": (10, 60),
    "sample": (6, 60),
}
_public_demo_events: dict[str, deque[float]] = {
    name: deque() for name in _public_demo_rate_windows
}
_public_demo_rate_lock = Lock()
_inference_slots = BoundedSemaphore(1)


def _allow_public_demo_request(bucket: str) -> tuple[bool, int]:
    maximum, window_seconds = _public_demo_rate_windows[bucket]
    now = monotonic()
    with _public_demo_rate_lock:
        events = _public_demo_events[bucket]
        while events and events[0] <= now - window_seconds:
            events.popleft()
        if len(events) >= maximum:
            retry_after = math.ceil(events[0] + window_seconds - now)
            return False, max(retry_after, 1)
        events.append(now)
    return True, 0


@app.middleware("http")
async def protect_public_demo(request: Request, call_next):
    inference_request = (
        request.method == "POST" and request.url.path == "/api/predictions"
    )
    bucket = (
        "prediction"
        if inference_request
        else "sample"
        if request.method == "GET" and request.url.path == "/api/samples/random"
        else None
    )
    if public_demo_enabled() and bucket:
        allowed, retry_after = _allow_public_demo_request(bucket)
        if not allowed:
            return JSONResponse(
                {"detail": "Public demo request limit reached. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        if inference_request and not _inference_slots.acquire(blocking=False):
            return JSONResponse(
                {"detail": "Inference capacity is busy. Try again shortly."},
                status_code=503,
                headers={"Retry-After": "2"},
            )
        try:
            return await call_next(request)
        finally:
            if inference_request:
                _inference_slots.release()
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ready")
def ready() -> dict[str, str]:
    unavailable = [model["task"] for model in get_models() if not model["available"]]
    if unavailable:
        raise HTTPException(
            status_code=503,
            detail="One or more required models are unavailable.",
        )
    return {"status": "ready"}


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    return get_models()


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    return get_overview()


@app.get("/api/samples/random")
def sample(
    status: str = Query(default="any", pattern="^(any|normal|anomaly)$"),
) -> dict[str, Any]:
    try:
        return random_test_sample(status)
    except ImportError as error:
        detail = (
            "Real sample data is temporarily unavailable."
            if public_demo_enabled()
            else str(error)
        )
        logger.warning("Could not load a random sample: %s", error)
        raise HTTPException(status_code=503, detail=detail) from error
    except (FileNotFoundError, OSError, ConnectionError, RuntimeError, ValueError) as error:
        detail = (
            "Real sample data is temporarily unavailable."
            if public_demo_enabled()
            else (
                f"Could not load a sample from the held-out TelecomTS test "
                f"partition: {error}"
            )
        )
        logger.warning("Could not load a random sample: %s", error)
        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from error


@app.post("/api/predictions", status_code=201)
def create_prediction(request: PredictionRequest) -> dict[str, Any]:
    try:
        return run_prediction(request.sample_id, request.KPIs)
    except FileNotFoundError as error:
        if public_demo_enabled():
            logger.exception("Required inference model is unavailable.")
            raise HTTPException(
                status_code=503,
                detail="Inference is temporarily unavailable.",
            ) from error
        raise HTTPException(
            status_code=503,
            detail=(
                f"{error}. Train/select the models locally or set INA_MODELS_PATH "
                "to the directory containing results/models."
            ),
        ) from error
    except (ValueError, RuntimeError) as error:
        if public_demo_enabled() and isinstance(error, RuntimeError):
            logger.exception("Public demo inference failed.")
            raise HTTPException(
                status_code=503,
                detail="Inference failed. Please try again later.",
            ) from error
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
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("INA_PORT", "8000"))),
    )
    args = parser.parse_args()
    try:
        is_loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        is_loopback = args.host.lower() == "localhost"
    if not is_loopback and not public_demo_enabled():
        parser.error(
            "Binding to a non-local interface requires INA_PUBLIC_DEMO=true."
        )
    uvicorn.run(app, host=args.host, port=args.port)
