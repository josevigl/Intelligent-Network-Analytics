"""Tests for the local dashboard API and prediction-history persistence."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import api, database
from src.preprocessing.data import CATEGORICAL_KPIS, KPI_COLUMNS


def valid_kpis() -> dict[str, list[float | str]]:
    return {
        channel: (
            ["TCP"] * 128
            if channel in CATEGORICAL_KPIS
            else [0.25] * 128
        )
        for channel in KPI_COLUMNS
    }


class AppTests(unittest.TestCase):
    def test_api_health_and_kpi_window_validation(self) -> None:
        client = TestClient(api.app)

        self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
        with patch("app.api.run_prediction", side_effect=FileNotFoundError("missing model")) as predict:
            response = client.post("/api/predictions", json={"KPIs": valid_kpis()})
        self.assertEqual(response.status_code, 503)
        self.assertIn("INA_MODELS_PATH", response.json()["detail"])
        predict.assert_called_once()

        malformed = valid_kpis()
        malformed[KPI_COLUMNS[0]] = [0.25]
        response = client.post("/api/predictions", json={"KPIs": malformed})
        self.assertEqual(response.status_code, 422)

    def test_prediction_endpoint_returns_and_persists_model_record(self) -> None:
        record = {
            "id": "NX-TEST000001",
            "created_at": "2026-01-01T12:00:00+00:00",
            "anomaly_status": "anomaly",
            "predicted_type": "handover_failure",
            "detection_score": 0.91,
            "confidence": 0.91,
            "affected_kpis": ["RSRP"],
            "inference_ms": 14.0,
            "details": {
                "tasks": {},
                "localization": {
                    "timeline": [0] * 128,
                    "first_anomalous_timestep": None,
                    "last_anomalous_timestep": None,
                    "anomalous_timesteps": 0,
                    "segments": [],
                },
                "diagnostic": {
                    "summary": "Test record",
                    "hypotheses": [],
                    "observed": [],
                    "notice": "Test only",
                },
                "input_channels": list(KPI_COLUMNS),
            },
            "input": {"KPIs": valid_kpis()},
        }
        client = TestClient(api.app)
        with patch("app.api.run_prediction", return_value=record) as predict:
            response = client.post("/api/predictions", json={"KPIs": valid_kpis()})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["id"], record["id"])
        predict.assert_called_once()

    def test_history_round_trip_and_filters(self) -> None:
        record = {
            "id": "NX-DB000001",
            "created_at": "2026-01-01T12:00:00+00:00",
            "anomaly_status": "anomaly",
            "predicted_type": "handover_failure",
            "detection_score": 0.91,
            "confidence": 0.91,
            "affected_kpis": ["RSRP"],
            "inference_ms": 14.0,
            "details": {"diagnostic": {"summary": "Test"}},
            "input": {"KPIs": valid_kpis()},
        }
        with tempfile.TemporaryDirectory() as directory:
            database_file = Path(directory) / "predictions.sqlite3"
            with patch.dict(os.environ, {"INA_DATABASE_PATH": str(database_file)}):
                database.save_prediction(record)
                listed = database.list_predictions(
                    search="handover",
                    anomaly_status="anomaly",
                    predicted_type="handover_failure",
                )
                detail = database.get_prediction(record["id"])
                counts = database.prediction_counts()

        self.assertEqual([item["id"] for item in listed], [record["id"]])
        self.assertEqual(detail["input"], record["input"])
        self.assertEqual(counts["total"], 1)
        self.assertEqual(counts["anomalies"], 1)

    def test_prediction_request_rejects_missing_channels(self) -> None:
        sample = valid_kpis()
        sample.pop(KPI_COLUMNS[0])

        with self.assertRaises(ValidationError):
            api.PredictionRequest(KPIs=sample)


if __name__ == "__main__":
    unittest.main()
