"""Model discovery and batch prediction service for the NEXUS API."""

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.database import get_prediction, list_predictions, prediction_counts, save_prediction
from app.settings import public_demo_enabled
from src.artifacts import TaskModelArtifact, load_task_artifact
from src.config import load_pipeline_config
from src.evaluation.holdout import config_fingerprint
from src.inference import predict_task_component
from src.models.training import summarize_windows
from src.preprocessing.data import (
    KPI_COLUMNS,
    TASK_NAMES,
    extract_inference_windows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_TITLES = {
    "binary_detection": "Anomaly Detection",
    "anomaly_type": "Anomaly Classification",
    "temporal_localization": "Temporal Localization",
    "affected_kpis": "Affected KPI Identification",
}


def _models_root() -> Path:
    return Path(os.environ.get("INA_MODELS_PATH", PROJECT_ROOT / "results" / "models"))


@lru_cache(maxsize=1)
def _test_sample_source() -> tuple[Any, tuple[int, ...], dict[str, Any]]:
    comparison_path = _models_root() / "binary_detection" / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Test-split provenance is missing: {comparison_path}"
        )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    provenance = comparison.get("holdout_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Binary-detection comparison has no holdout provenance.")
    dataset_name = provenance.get("dataset_name")
    source_split = provenance.get("source_split")
    dataset_sha256 = provenance.get("dataset_sha256")
    positions = provenance.get("test_positions")
    if dataset_name != "AliMaatouk/TelecomTS":
        raise ValueError(
            "The saved test samples are not from AliMaatouk/TelecomTS."
        )
    if source_split != "train":
        raise ValueError("Expected the TelecomTS train source split in model provenance.")
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha256)
    ):
        raise ValueError("A valid source dataset fingerprint is missing.")
    if not isinstance(positions, list) or not positions:
        raise ValueError("Held-out test positions are missing from model provenance.")
    if provenance.get("test_sample_count") != len(positions):
        raise ValueError("Held-out test positions do not match their recorded count.")
    if any(not isinstance(position, int) or position < 0 for position in positions):
        raise ValueError("Held-out test positions must be non-negative integers.")
    if len(set(positions)) != len(positions):
        raise ValueError("Held-out test positions contain duplicates.")

    config = load_pipeline_config()
    if provenance.get("project_config_sha256") != config_fingerprint(config):
        raise ValueError(
            "The current project config differs from the config used for the "
            "saved test partition."
        )
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Downloading a real test sample requires the optional 'data' "
            "dependencies. Install with: python -m pip install -e \".[app,data]\""
        ) from error

    dataset = load_dataset(dataset_name, split=source_split)
    if max(positions) >= len(dataset):
        raise ValueError(
            "The cached source dataset is shorter than the saved test positions."
        )
    return dataset, tuple(positions), provenance


def _sample_kpis(raw_kpis: Any) -> dict[str, list[Any]] | None:
    if not isinstance(raw_kpis, dict) or set(KPI_COLUMNS) - set(raw_kpis):
        return None
    normalized: dict[str, list[Any]] = {}
    for channel in KPI_COLUMNS:
        values = raw_kpis[channel]
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 128:
            return None
        channel_values = []
        for value in values:
            try:
                missing = pd.isna(value)
            except (TypeError, ValueError):
                missing = False
            if isinstance(missing, (bool, np.bool_)) and missing:
                channel_values.append(None)
            elif channel in {"DL_Protocol", "UL_Protocol"}:
                if not isinstance(value, str):
                    return None
                channel_values.append(value)
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    return None
                channel_values.append(number if math.isfinite(number) else None)
        normalized[channel] = channel_values
    return normalized


def random_test_sample(status: str = "any") -> dict[str, Any]:
    if status not in {"any", "normal", "anomaly"}:
        raise ValueError("Sample status must be 'any', 'normal', or 'anomaly'.")
    dataset, positions, provenance = _test_sample_source()
    candidates = list(positions)
    random.SystemRandom().shuffle(candidates)
    for position in candidates:
        row = dataset[position]
        annotation = row.get("anomalies")
        if not isinstance(annotation, dict):
            continue
        exists = annotation.get("exists")
        is_anomaly = isinstance(exists, (bool, np.bool_)) and bool(exists)
        if status != "any" and (status == "anomaly") != is_anomaly:
            continue
        kpis = _sample_kpis(row.get("KPIs"))
        if kpis is None:
            continue
        anomaly_type = annotation.get("type")
        if not is_anomaly or not isinstance(anomaly_type, str) or not anomaly_type:
            anomaly_type = "Normal"
        affected = annotation.get("affected_kpis", ())
        if isinstance(affected, np.ndarray):
            affected = affected.tolist()
        affected_kpis = sorted(
            {
                item
                for item in affected
                if isinstance(item, str) and item in KPI_COLUMNS
            }
        ) if isinstance(affected, (list, tuple, set)) else []
        return {
            "KPIs": kpis,
            "source": {
                "dataset": provenance["dataset_name"],
                "source_split": provenance["source_split"],
                "partition": "test",
                "row_position": position,
                "dataset_sha256": provenance["dataset_sha256"],
            },
            "reference": {
                "anomaly_status": "anomaly" if is_anomaly else "normal",
                "anomaly_type": anomaly_type,
                "affected_kpis": affected_kpis,
            },
        }
    raise ValueError(
        f"No valid held-out test sample is available for status {status!r}."
    )


@lru_cache(maxsize=4)
def load_selected_model(task_name: str) -> TaskModelArtifact:
    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown model task: {task_name}")
    task_path = _models_root() / task_name
    comparison_path = task_path / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Model comparison is missing: {comparison_path}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    selected = [
        model for model in comparison.get("models", []) if model.get("selected")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one selected model for {task_name}; found {len(selected)}."
        )
    model_name = selected[0].get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Selected model name is missing for {task_name}.")
    artifact = load_task_artifact(task_path / model_name / "model_bundle.joblib")
    if artifact.task_name != task_name or artifact.model_name != model_name:
        raise ValueError(f"Selected artifact does not match the comparison for {task_name}.")
    return artifact


def _model_input(frame: pd.DataFrame, artifact: TaskModelArtifact) -> np.ndarray:
    windows, _ = extract_inference_windows(
        frame,
        feature_channels=artifact.feature_channels,
        sequence_length=artifact.sequence_length,
    )
    if not len(windows):
        raise ValueError("The submitted KPI sample contains no valid inference window.")
    transformed = artifact.preprocessor.transform(windows)
    if artifact.input_mode == "window_statistics":
        model_input, feature_names = summarize_windows(
            transformed,
            artifact.transformed_feature_names,
            getattr(artifact, "summary_statistics", ()) or None,
        )
        if feature_names != artifact.model_feature_names:
            raise ValueError("Inference feature order differs from the saved model.")
    elif artifact.input_mode == "per_timestep":
        model_input = transformed.reshape(-1, transformed.shape[-1])
        if tuple(artifact.preprocessor.get_feature_names_out()) != artifact.model_feature_names:
            raise ValueError("Inference feature order differs from the saved model.")
    else:
        raise ValueError(f"Unknown model input mode: {artifact.input_mode!r}.")
    return model_input


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _probability_scores(
    frame: pd.DataFrame,
    artifact: TaskModelArtifact,
    prediction: Any,
) -> dict[str, Any]:
    model_input = _model_input(frame, artifact)
    model = artifact.model
    if not hasattr(model, "predict_proba"):
        return {"confidence": 1.0, "scores": []}
    probabilities = model.predict_proba(model_input)
    if isinstance(probabilities, list):
        probabilities = np.column_stack(
            [values[:, 1] if values.ndim == 2 else values for values in probabilities]
        )
    probabilities = np.asarray(probabilities)
    if artifact.task_name == "binary_detection":
        score = float(probabilities[0, 1])
        return {
            "confidence": score if int(prediction) == 1 else 1.0 - score,
            "score": score,
            "scores": [
                {"label": "Normal", "score": float(probabilities[0, 0])},
                {"label": "Anomaly", "score": score},
            ],
        }
    if artifact.task_name == "anomaly_type":
        labels = getattr(artifact.target_encoder, "classes_", None)
        if labels is None:
            labels = getattr(model, "classes_", np.arange(probabilities.shape[1]))
        scores = [
            {"label": str(label), "score": float(value)}
            for label, value in zip(labels, probabilities[0])
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        top = next((row["score"] for row in scores if row["label"] == str(prediction)), 0.0)
        return {"confidence": top, "scores": scores}
    if artifact.task_name == "affected_kpis":
        labels = getattr(artifact.target_encoder, "classes_", ())
        scores = [
            {"label": str(label), "score": float(value)}
            for label, value in zip(labels, probabilities[0])
        ]
        return {
            "confidence": float(np.mean([row["score"] for row in scores])) if scores else 0.0,
            "scores": sorted(scores, key=lambda item: item["score"], reverse=True),
        }
    return {"confidence": float(np.max(probabilities)), "scores": []}


def _localization(predictions: Any) -> dict[str, Any]:
    labels = np.asarray(predictions, dtype=np.int8).reshape(-1)
    anomalous = np.flatnonzero(labels)
    segments: list[dict[str, int]] = []
    if len(anomalous):
        start = previous = int(anomalous[0])
        for timestep in anomalous[1:]:
            current = int(timestep)
            if current > previous + 1:
                segments.append({"start": start, "end": previous})
                start = current
            previous = current
        segments.append({"start": start, "end": previous})
    return {
        "timeline": labels.astype(int).tolist(),
        "first_anomalous_timestep": int(anomalous[0]) if len(anomalous) else None,
        "last_anomalous_timestep": int(anomalous[-1]) if len(anomalous) else None,
        "anomalous_timesteps": int(len(anomalous)),
        "segments": segments,
    }


def _diagnostic(
    anomaly_status: str,
    predicted_type: str,
    affected_kpis: list[str],
    input_kpis: dict[str, list[Any]],
) -> dict[str, Any]:
    if anomaly_status == "normal":
        return {
            "summary": "The selected models did not identify an anomaly in this window.",
            "hypotheses": [],
            "observed": ["The binary detector classified the supplied window as normal."],
            "notice": "This is a model prediction, not a guarantee of network health.",
        }
    hypotheses = [
        "Review the affected KPI traces around the localized interval and compare them with the same cell's baseline.",
        "Correlate the event with radio conditions, traffic load, and recent configuration or handover changes.",
        "Confirm the prediction against independent counters and operator procedures before assigning root cause.",
    ]
    observed = [
        f"Classification model output: {predicted_type}.",
        (
            "Affected-KPI model output: " + ", ".join(affected_kpis)
            if affected_kpis
            else "Affected-KPI model returned no KPI above its learned decision thresholds."
        ),
    ]
    if "RSRP" in input_kpis:
        values = pd.to_numeric(pd.Series(input_kpis["RSRP"]), errors="coerce").dropna()
        if not values.empty:
            observed.append(
                f"Observed RSRP range in supplied sample: {values.min():.2f} to {values.max():.2f}."
            )
    return {
        "summary": (
            "An anomaly was predicted. The items below are investigation hypotheses, "
            "not a confirmed root cause."
        ),
        "hypotheses": hypotheses,
        "observed": observed,
        "notice": "Model output and observed input statistics are not root-cause confirmation.",
    }


def run_prediction(sample_id: str | None, input_kpis: dict[str, list[Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    frame = pd.DataFrame({"KPIs": [input_kpis]})
    task_results: dict[str, Any] = {}
    for task_name in TASK_NAMES:
        artifact = load_selected_model(task_name)
        result = predict_task_component(frame, artifact)
        if len(result.predictions) != 1:
            raise RuntimeError(
                f"Expected one prediction for {task_name}; received "
                f"{len(result.predictions)}."
            )
        prediction = result.predictions[0]
        score_data = _probability_scores(frame, artifact, prediction)
        task_results[task_name] = {
            "model_name": artifact.model_name,
            "model_version": artifact.library_versions.get("scikit_learn", "unknown"),
            "prediction": _json_value(prediction),
            **score_data,
        }

    detection = task_results["binary_detection"]
    is_anomaly = bool(detection["prediction"])
    anomaly_status = "anomaly" if is_anomaly else "normal"
    predicted_type = (
        str(task_results["anomaly_type"]["prediction"]) if is_anomaly else "Normal"
    )
    affected_kpis = list(task_results["affected_kpis"]["prediction"]) if is_anomaly else []
    localization = _localization(task_results["temporal_localization"]["prediction"])
    elapsed_ms = (time.perf_counter() - started) * 1000
    record = {
        "id": sample_id or f"NX-{uuid.uuid4().hex[:10].upper()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anomaly_status": anomaly_status,
        "predicted_type": predicted_type,
        "detection_score": float(detection["score"]),
        "confidence": float(detection["confidence"]),
        "affected_kpis": affected_kpis,
        "inference_ms": elapsed_ms,
        "details": {
            "tasks": task_results,
            "localization": localization,
            "diagnostic": _diagnostic(anomaly_status, predicted_type, affected_kpis, input_kpis),
            "input_channels": list(input_kpis),
        },
        "input": {"KPIs": input_kpis},
    }
    if not public_demo_enabled():
        save_prediction(record)
    return record


def get_overview() -> dict[str, Any]:
    counts = (
        {
            "total": 0,
            "anomalies": 0,
            "average_inference_ms": 0,
            "categories": [],
        }
        if public_demo_enabled()
        else prediction_counts()
    )
    models = get_models()
    return {
        "data_source": "Local model artifacts and analyses run through this app",
        "live_telemetry": False,
        "total_predictions": counts["total"],
        "anomalies": counts["anomalies"],
        "average_inference_ms": counts["average_inference_ms"],
        "models_available": sum(item["available"] for item in models),
        "models": models,
        "categories": counts["categories"],
        "recent_predictions": [] if public_demo_enabled() else list_predictions(limit=8),
        "dataset_baseline": {
            "name": "AliMaatouk/TelecomTS",
            "split": "train",
            "rows": 32000,
            "note": "Baseline dataset size; dashboard activity counts are local app analyses, not live traffic.",
        },
    }


def get_models() -> list[dict[str, Any]]:
    models = []
    for task_name in TASK_NAMES:
        title = TASK_TITLES[task_name]
        try:
            artifact = load_selected_model(task_name)
            comparison = json.loads(
                (_models_root() / task_name / "comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            selected = next(
                model for model in comparison["models"] if model.get("selected")
            )
            metrics = selected.get("test_metrics") or selected.get("validation_metrics") or {}
            models.append(
                {
                    "task": task_name,
                    "title": title,
                    "model_name": artifact.model_name,
                    "version": artifact.library_versions.get("scikit_learn", "unknown"),
                    "available": True,
                    "metrics": metrics,
                    "metric_source": "test" if selected.get("test_metrics") else "validation",
                }
            )
        except (FileNotFoundError, ValueError, KeyError) as error:
            models.append(
                {
                    "task": task_name,
                    "title": title,
                    "model_name": None,
                    "version": None,
                    "available": False,
                    "metrics": {},
                    "error": (
                        "Model unavailable."
                        if public_demo_enabled()
                        else str(error)
                    ),
                }
            )
    return models


def history(
    limit: int = 100,
    search: str | None = None,
    anomaly_status: str | None = None,
    predicted_type: str | None = None,
) -> list[dict[str, Any]]:
    if public_demo_enabled():
        return []
    return list_predictions(limit, search, anomaly_status, predicted_type)


def prediction_detail(prediction_id: str) -> dict[str, Any] | None:
    if public_demo_enabled():
        return None
    return get_prediction(prediction_id)
