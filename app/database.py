"""SQLite persistence for local prediction history."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "results" / "app" / "predictions.sqlite3"


def database_path() -> Path:
    return Path(os.environ.get("INA_DATABASE_PATH", DEFAULT_DATABASE_PATH))


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                anomaly_status TEXT NOT NULL,
                predicted_type TEXT NOT NULL,
                detection_score REAL NOT NULL,
                confidence REAL NOT NULL,
                affected_kpis_json TEXT NOT NULL,
                details_json TEXT NOT NULL,
                input_json TEXT NOT NULL,
                inference_ms REAL NOT NULL
            )
            """
        )
        with connection:
            yield connection
    finally:
        connection.close()


def save_prediction(record: dict[str, Any]) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                id, created_at, anomaly_status, predicted_type, detection_score,
                confidence, affected_kpis_json, details_json, input_json, inference_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["created_at"],
                record["anomaly_status"],
                record["predicted_type"],
                record["detection_score"],
                record["confidence"],
                json.dumps(record["affected_kpis"]),
                json.dumps(record["details"]),
                json.dumps(record["input"]),
                record["inference_ms"],
            ),
        )


def list_predictions(
    limit: int = 100,
    search: str | None = None,
    anomaly_status: str | None = None,
    predicted_type: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if search:
        clauses.append("(id LIKE ? OR predicted_type LIKE ?)")
        parameters.extend((f"%{search}%", f"%{search}%"))
    if anomaly_status:
        clauses.append("anomaly_status = ?")
        parameters.append(anomaly_status)
    if predicted_type:
        clauses.append("predicted_type = ?")
        parameters.append(predicted_type)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    parameters.append(limit)
    with _connect() as connection:
        rows = connection.execute(
            f"""
            SELECT id, created_at, anomaly_status, predicted_type, detection_score,
                   confidence, affected_kpis_json, inference_ms
            FROM predictions {where}
            ORDER BY created_at DESC LIMIT ?
            """,
            parameters,
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "anomaly_status": row["anomaly_status"],
            "predicted_type": row["predicted_type"],
            "detection_score": row["detection_score"],
            "confidence": row["confidence"],
            "affected_kpis": json.loads(row["affected_kpis_json"]),
            "inference_ms": row["inference_ms"],
        }
        for row in rows
    ]


def get_prediction(prediction_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "anomaly_status": row["anomaly_status"],
        "predicted_type": row["predicted_type"],
        "detection_score": row["detection_score"],
        "confidence": row["confidence"],
        "affected_kpis": json.loads(row["affected_kpis_json"]),
        "details": json.loads(row["details_json"]),
        "input": json.loads(row["input_json"]),
        "inference_ms": row["inference_ms"],
    }


def prediction_counts() -> dict[str, Any]:
    with _connect() as connection:
        total, anomaly_count, average_ms = connection.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(anomaly_status = 'anomaly'), 0),
                   COALESCE(AVG(inference_ms), 0)
            FROM predictions
            """
        ).fetchone()
        categories = connection.execute(
            """
            SELECT predicted_type, COUNT(*) AS count
            FROM predictions
            WHERE anomaly_status = 'anomaly'
            GROUP BY predicted_type
            ORDER BY count DESC
            """
        ).fetchall()
    return {
        "total": total,
        "anomalies": anomaly_count,
        "average_inference_ms": average_ms,
        "categories": [
            {"name": row["predicted_type"], "count": row["count"]}
            for row in categories
        ],
    }
