"""MLRun registry, inference, and evaluation handlers."""

import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from pipelines.train import load_training_frame
from src.artifacts import (
    TaskModelArtifact,
    load_task_artifact,
)
from src.evaluation.holdout import config_fingerprint, validate_holdout_manifest
from src.evaluation.metrics import summarize_task_metrics
from src.inference import predict_task_component


def _decode_json_value(value: Any, name: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a JSON string, object, or array.")
    try:
        return json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} must contain valid JSON.") from error


def _load_inference_frame(dataset_name: str, split: str) -> pd.DataFrame:
    frame = load_training_frame(dataset_name, split)
    if "KPIs" not in frame.columns:
        raise ValueError("The input dataset must contain a 'KPIs' column.")
    return frame


def _resolve_model_artifact(model_uri: Any) -> TaskModelArtifact:
    if isinstance(model_uri, TaskModelArtifact):
        return model_uri
    if isinstance(model_uri, (str, Path)) and Path(model_uri).exists():
        return load_task_artifact(model_uri)
    if not isinstance(model_uri, str) or not model_uri:
        raise TypeError("model_uri must be a non-empty local path or MLRun URI.")

    from mlrun.artifacts import get_model

    model_file, _, _ = get_model(model_uri, suffix=".joblib")
    return load_task_artifact(model_file)


def _resolve_json_artifact(artifact_uri: str) -> dict[str, Any]:
    if not isinstance(artifact_uri, str) or not artifact_uri:
        raise TypeError("artifact_uri must be a non-empty MLRun URI.")
    import mlrun

    payload = mlrun.get_dataitem(artifact_uri).get()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("The registered JSON artifact did not return text.")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("The registered JSON artifact must contain an object.")
    return value


def _artifact_uri(artifact: Any, key: str) -> str:
    uri = getattr(artifact, "uri", None)
    if not isinstance(uri, str) or not uri:
        raise RuntimeError(f"MLRun did not return a URI for artifact {key!r}.")
    return uri


def register_model_component(
    context: Any,
    model_path: str,
    comparison_path: str,
    holdout_manifest_json: Any,
    dataset_name: str,
    source_split: str,
    code_revision: str,
    registration_tag: str,
) -> dict[str, str]:
    """Register a locally trained selected model and its provenance in MLRun."""
    bundle_path = Path(model_path)
    comparison_file = Path(comparison_path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle does not exist: {bundle_path}")
    if not comparison_file.is_file():
        raise FileNotFoundError(f"Model comparison does not exist: {comparison_file}")

    artifact = load_task_artifact(bundle_path)
    if not artifact.selected:
        raise ValueError("Only the selected model bundle may be registered.")
    manifest = _decode_json_value(holdout_manifest_json, "holdout_manifest_json")
    if not isinstance(manifest, dict):
        raise ValueError("The holdout manifest must be a JSON object.")
    if manifest.get("task_name") != artifact.task_name:
        raise ValueError("The holdout manifest belongs to a different task.")
    if manifest.get("dataset_name") != dataset_name:
        raise ValueError("The holdout manifest names a different source dataset.")
    if manifest.get("source_split") != source_split:
        raise ValueError("The holdout manifest names a different source split.")
    if manifest.get("code_revision") != code_revision:
        raise ValueError("The holdout manifest has a different code revision.")
    artifact_project_config = artifact.config.get("project")
    if not isinstance(artifact_project_config, dict):
        raise ValueError("The model bundle has no project configuration provenance.")
    if manifest.get("project_config_sha256") != config_fingerprint(
        artifact_project_config
    ):
        raise ValueError(
            "The holdout manifest configuration differs from the trained model."
        )

    with comparison_file.open("r", encoding="utf-8") as file:
        comparison = json.load(file)
    if not isinstance(comparison, dict) or comparison.get("task_name") != artifact.task_name:
        raise ValueError("The comparison file does not match the model task.")
    selected_models = [
        candidate
        for candidate in comparison.get("models", [])
        if candidate.get("selected")
    ]
    if (
        len(selected_models) != 1
        or selected_models[0].get("model_name") != artifact.model_name
    ):
        raise ValueError(
            "The bundle must match the single model selected in comparison.json."
        )

    model_metrics: dict[str, float] = {}
    metric_sets = {
        "cv": artifact.cv_metrics or {},
        "validation": artifact.validation_metrics,
        "test": artifact.test_metrics or {},
    }
    for metric_set, metrics in metric_sets.items():
        for name, value in metrics.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Model has a non-finite {metric_set} metric {name!r}."
                )
            model_metrics[f"{metric_set}_{name}"] = numeric_value

    model_record = context.log_model(
        key=artifact.task_name,
        tag=registration_tag,
        model_file=str(bundle_path),
        framework="scikit-learn",
        algorithm=artifact.model_name,
        metrics=model_metrics,
        parameters=artifact.model_parameters or {},
        labels={
            "task": artifact.task_name,
            "algorithm": artifact.model_name,
            "dataset": dataset_name.replace("/", "-"),
            "code_revision": code_revision[:12],
        },
    )
    model_uri = _artifact_uri(model_record, artifact.task_name)
    manifest_record = context.log_artifact(
        f"{artifact.task_name}_holdout_manifest",
        body=json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        ),
        format="json",
        labels={"task": artifact.task_name},
    )
    manifest_uri = _artifact_uri(
        manifest_record,
        f"{artifact.task_name}_holdout_manifest",
    )
    comparison_record = context.log_artifact(
        f"{artifact.task_name}_comparison",
        local_path=str(comparison_file),
        format="json",
        labels={"task": artifact.task_name},
    )
    comparison_uri = _artifact_uri(
        comparison_record,
        f"{artifact.task_name}_comparison",
    )

    context.set_label("task", artifact.task_name)
    context.set_label("model", artifact.model_name)
    context.log_result("model_uri", model_uri)
    context.log_result("holdout_manifest_uri", manifest_uri)
    context.log_result("comparison_uri", comparison_uri)
    context.log_result("dataset_sha256", manifest["dataset_sha256"])
    context.log_result("code_revision", code_revision)
    return {
        "task_name": artifact.task_name,
        "model_uri": model_uri,
        "holdout_manifest_uri": manifest_uri,
        "comparison_uri": comparison_uri,
    }


def _jsonable_prediction(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def batch_inference_component(
    context: Any,
    dataset_name: str,
    split: str,
    model_uri: str,
) -> dict[str, Any]:
    """Run a registered model over an unlabeled batch and log predictions."""
    frame = _load_inference_frame(dataset_name, split)
    artifact = _resolve_model_artifact(model_uri)
    result = predict_task_component(frame, artifact)
    records = [
        {
            "position": int(position),
            "prediction": _jsonable_prediction(prediction),
        }
        for position, prediction in zip(result.positions, result.predictions)
    ]
    context.set_label("task", artifact.task_name)
    context.set_label("model", artifact.model_name)
    context.log_result("prediction_count", len(records))
    with tempfile.TemporaryDirectory(
        prefix=f"ina-{artifact.task_name}-predictions-"
    ) as temporary:
        predictions_path = Path(temporary) / "predictions.json"
        with predictions_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "task_name": artifact.task_name,
                    "model_name": artifact.model_name,
                    "predictions": records,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
        prediction_artifact = context.log_artifact(
            "predictions",
            local_path=str(predictions_path),
            format="json",
            labels={"task": artifact.task_name, "model": artifact.model_name},
        )
    return {
        "task_name": artifact.task_name,
        "model_name": artifact.model_name,
        "prediction_count": len(records),
        "predictions_uri": _artifact_uri(prediction_artifact, "predictions"),
    }


def evaluate_registered_model_component(
    context: Any,
    dataset_name: str,
    split: str,
    model_uri: str,
    holdout_manifest_uri: str,
    project_config_json: Any,
) -> dict[str, Any]:
    """Evaluate a registered model on its data-verified held-out partition."""
    frame = _load_inference_frame(dataset_name, split)
    artifact = _resolve_model_artifact(model_uri)
    manifest = _resolve_json_artifact(holdout_manifest_uri)
    project_config = _decode_json_value(project_config_json, "project_config_json")
    if not isinstance(project_config, dict):
        raise ValueError("project_config_json must be a JSON object.")
    if manifest.get("dataset_name") is None:
        raise ValueError("The holdout manifest has no dataset provenance.")
    if manifest["dataset_name"] != dataset_name:
        raise ValueError(
            "The requested source dataset differs from the registered holdout."
        )
    if manifest.get("source_split") != split:
        raise ValueError(
            "The requested source split differs from the registered holdout."
        )

    splits = validate_holdout_manifest(
        frame=frame,
        manifest=manifest,
        task_name=artifact.task_name,
        config=project_config,
    )
    test_frame = frame.iloc[splits.positions_test]
    prediction_result = predict_task_component(test_frame, artifact)
    if len(prediction_result.predictions) != len(splits.y_test):
        raise RuntimeError(
            "The registered model produced a different number of predictions "
            "than the held-out test targets."
        )

    y_true = splits.y_test
    y_pred = prediction_result.predictions
    if artifact.task_name == "affected_kpis":
        if artifact.target_encoder is None:
            raise ValueError("Affected-KPI artifact is missing its label encoder.")
        known_labels = set(str(value) for value in artifact.target_encoder.classes_)
        unknown = {
            str(label)
            for sample in y_true
            for label in sample
            if str(label) not in known_labels
        }
        if unknown:
            raise ValueError(
                "Test data contains affected-KPI labels absent from the model: "
                f"{sorted(unknown)}"
            )
        y_true = artifact.target_encoder.transform(y_true)
        y_pred = artifact.target_encoder.transform(y_pred.tolist())
    elif artifact.task_name == "anomaly_type":
        if artifact.target_encoder is None:
            raise ValueError("Anomaly-type artifact is missing its label encoder.")
        unknown = set(y_true) - set(artifact.target_encoder.classes_)
        if unknown:
            raise ValueError(
                "Test data contains anomaly types absent from the model: "
                f"{sorted(unknown)}"
            )

    metrics = summarize_task_metrics(artifact.task_name, y_true, y_pred)
    if artifact.task_name == "temporal_localization":
        if not artifact.decision_thresholds:
            raise ValueError("Temporal-localization artifact has no decision threshold.")
        metrics["threshold"] = float(artifact.decision_thresholds[0])

    for name, value in metrics.items():
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Cannot log non-finite evaluation metric {name!r}.")
        context.log_result(f"test_{name}", numeric_value)

    evaluation = {
        "task_name": artifact.task_name,
        "model_name": artifact.model_name,
        "model_uri": model_uri,
        "holdout_manifest_uri": holdout_manifest_uri,
        "dataset_sha256": manifest["dataset_sha256"],
        "code_revision": manifest["code_revision"],
        "sample_count": int(len(y_true)),
        "group_count": (
            int(len(np.unique(splits.groups_test)))
            if splits.groups_test is not None
            else None
        ),
        "grouping_strategy": manifest["grouping_strategy"],
        "metrics": metrics,
    }
    context.set_label("task", artifact.task_name)
    context.set_label("model", artifact.model_name)
    context.log_result("test_samples", evaluation["sample_count"])
    with tempfile.TemporaryDirectory(
        prefix=f"ina-{artifact.task_name}-evaluation-"
    ) as temporary:
        evaluation_path = Path(temporary) / "evaluation.json"
        with evaluation_path.open("w", encoding="utf-8") as file:
            json.dump(evaluation, file, ensure_ascii=False, indent=2)
            file.write("\n")
        evaluation_artifact = context.log_artifact(
            "evaluation",
            local_path=str(evaluation_path),
            format="json",
            labels={"task": artifact.task_name, "model": artifact.model_name},
        )
    evaluation["evaluation_uri"] = _artifact_uri(
        evaluation_artifact,
        "evaluation",
    )
    return evaluation
