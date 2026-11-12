"""Tests for the local dashboard API and prediction-history persistence."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError

from app import api, database
from app import service
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

    def test_prediction_request_rejects_nonfinite_and_unknown_values(self) -> None:
        sample = valid_kpis()
        sample["RSRP"][0] = float("inf")
        with self.assertRaises(ValidationError):
            api.PredictionRequest(KPIs=sample)

        sample = valid_kpis()
        sample["RSRP"][0] = True
        with self.assertRaises(ValidationError):
            api.PredictionRequest(KPIs=sample)

        sample = valid_kpis()
        sample["unexpected"] = [0.0] * 128
        with self.assertRaises(ValidationError):
            api.PredictionRequest(KPIs=sample)

    def test_random_sample_comes_from_held_out_data_and_cleans_values(self) -> None:
        class FakeDataset:
            def __getitem__(self, position: int) -> dict[str, object]:
                self.assert_position = position
                annotation = {
                    "exists": True,
                    "type": "Jamming",
                    "affected_kpis": np.array(["RSRP", "UL_SNR"], dtype=object),
                }
                sample = valid_kpis()
                sample["RSRP"][0] = float("nan")
                return {"KPIs": sample, "anomalies": annotation}

        dataset = FakeDataset()
        provenance = {
            "dataset_name": "AliMaatouk/TelecomTS",
            "source_split": "train",
            "dataset_sha256": "a" * 64,
        }
        with patch.object(
            service, "_test_sample_source", return_value=(dataset, (42,), provenance)
        ):
            result = service.random_test_sample("anomaly")

        self.assertEqual(dataset.assert_position, 42)
        self.assertEqual(result["source"]["partition"], "test")
        self.assertEqual(result["reference"]["anomaly_type"], "Jamming")
        self.assertEqual(result["reference"]["affected_kpis"], ["RSRP", "UL_SNR"])
        self.assertIsNone(result["KPIs"]["RSRP"][0])
        self.assertEqual(len(result["KPIs"]), len(KPI_COLUMNS))
        self.assertEqual(len(result["KPIs"]["RSRP"]), 128)

    def test_test_sample_source_loads_saved_holdout_positions(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 100

        dataset = FakeDataset()
        provenance = {
            "dataset_name": "AliMaatouk/TelecomTS",
            "source_split": "train",
            "dataset_sha256": "a" * 64,
            "project_config_sha256": "config-hash",
            "test_positions": [12, 42],
            "test_sample_count": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            models_path = Path(directory) / "models"
            comparison_path = models_path / "binary_detection" / "comparison.json"
            comparison_path.parent.mkdir(parents=True)
            comparison_path.write_text(
                json.dumps({"holdout_provenance": provenance}),
                encoding="utf-8",
            )
            service._test_sample_source.cache_clear()
            try:
                with (
                    patch.object(service, "_models_root", return_value=models_path),
                    patch.object(
                        service,
                        "load_pipeline_config",
                        return_value={"test": True},
                    ),
                    patch.object(service, "config_fingerprint", return_value="config-hash"),
                    patch("datasets.load_dataset", return_value=dataset) as load_dataset,
                ):
                    loaded_dataset, positions, loaded_provenance = (
                        service._test_sample_source()
                    )
            finally:
                service._test_sample_source.cache_clear()

        self.assertIs(loaded_dataset, dataset)
        self.assertEqual(positions, (12, 42))
        self.assertEqual(loaded_provenance, provenance)
        load_dataset.assert_called_once_with("AliMaatouk/TelecomTS", split="train")

    def test_random_sample_status_filter_and_endpoint_errors(self) -> None:
        class FakeDataset:
            rows = [
                {"KPIs": valid_kpis(), "anomalies": {"exists": False}},
                {
                    "KPIs": valid_kpis(),
                    "anomalies": {"exists": True, "type": "Jamming"},
                },
            ]

            def __getitem__(self, position: int) -> dict[str, object]:
                return self.rows[position]

        provenance = {
            "dataset_name": "AliMaatouk/TelecomTS",
            "source_split": "train",
            "dataset_sha256": "a" * 64,
        }
        with patch.object(
            service,
            "_test_sample_source",
            return_value=(FakeDataset(), (0, 1), provenance),
        ):
            result = service.random_test_sample("normal")
        self.assertEqual(result["reference"]["anomaly_status"], "normal")

        client = TestClient(api.app)
        invalid_filter = client.get("/api/samples/random?status=unknown")
        self.assertEqual(invalid_filter.status_code, 422)
        with patch("app.api.random_test_sample", side_effect=ImportError("data extra missing")):
            unavailable = client.get("/api/samples/random")
        self.assertEqual(unavailable.status_code, 503)

    def test_public_demo_rate_limit_and_security_headers(self) -> None:
        api._public_demo_events["sample"].clear()
        sample = {
            "KPIs": valid_kpis(),
            "source": {
                "dataset": "TelecomTS",
                "source_split": "train",
                "partition": "test",
                "row_position": 1,
            },
            "reference": {
                "anomaly_status": "normal",
                "anomaly_type": "Normal",
                "affected_kpis": [],
            },
        }
        client = TestClient(api.app)
        try:
            with patch.dict(os.environ, {"INA_PUBLIC_DEMO": "true"}):
                with patch("app.api.random_test_sample", return_value=sample):
                    for _ in range(6):
                        response = client.get("/api/samples/random")
                        self.assertEqual(response.status_code, 200)
                    limited = client.get("/api/samples/random")
        finally:
            api._public_demo_events["sample"].clear()

        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_public_demo_disables_persistence_and_history(self) -> None:
        client = TestClient(api.app)
        with patch.dict(os.environ, {"INA_PUBLIC_DEMO": "true"}):
            with patch("app.database._connect") as connect:
                with self.assertRaisesRegex(RuntimeError, "disabled"):
                    database.save_prediction({})
                self.assertEqual(service.history(), [])
                self.assertIsNone(service.prediction_detail("NX-PRIVATE"))
                with patch("app.service.get_models", return_value=[]):
                    overview = service.get_overview()
                history_response = client.get("/api/predictions")
                detail_response = client.get("/api/predictions/NX-PRIVATE")

        self.assertEqual(overview["total_predictions"], 0)
        self.assertEqual(overview["recent_predictions"], [])
        connect.assert_not_called()
        self.assertEqual(history_response.json(), [])
        self.assertEqual(detail_response.status_code, 404)

    def test_prediction_request_body_is_limited(self) -> None:
        response = TestClient(api.app).post(
            "/api/predictions",
            content=b"x" * (api.MAX_REQUEST_BYTES + 1),
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)

    def test_external_bind_requires_public_demo_mode_and_uses_platform_port(self) -> None:
        with patch.dict(
            os.environ,
            {"INA_PUBLIC_DEMO": "false", "INA_HOST": "0.0.0.0"},
        ):
            with patch.object(sys, "argv", ["ina-app"]):
                with self.assertRaises(SystemExit):
                    api.main()

        with patch.dict(
            os.environ,
            {
                "INA_PUBLIC_DEMO": "true",
                "INA_HOST": "0.0.0.0",
                "PORT": "9123",
            },
        ):
            with patch.object(sys, "argv", ["ina-app"]):
                with patch("uvicorn.run") as run:
                    api.main()

        run.assert_called_once_with(api.app, host="0.0.0.0", port=9123)


if __name__ == "__main__":
    unittest.main()
