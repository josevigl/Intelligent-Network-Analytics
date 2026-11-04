"""MLRun handlers for TelecomTS model training and experiment tracking."""

import json
import math
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from pipelines.train import load_training_frame
from src.models.experiments import run_task_model_comparison
from src.preprocessing.data import TASK_NAMES
from src.preprocessing.pipeline import prepare_task_data_component


def load_telecomts_component(
    context: Any,
    dataset_name: str = "AliMaatouk/TelecomTS",
    split: str = "train",
    selected_kpis_json: str = "[]",
) -> None:
    """Load and register only the fields needed by the selected task branches."""
    if split != "train":
        raise ValueError(
            "MLRun model training only accepts the source 'train' split; "
            "validation and test are created from it locally."
        )

    selected_kpis = json.loads(selected_kpis_json)
    if (
        not isinstance(selected_kpis, list)
        or not selected_kpis
        or not all(isinstance(kpi, str) for kpi in selected_kpis)
    ):
        raise ValueError("selected_kpis_json must be a non-empty JSON list of strings.")

    frame = load_training_frame(dataset_name, split)
    required_columns = {"KPIs", "anomalies"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "The source training split is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    kept_columns = [
        column
        for column in ("start_time", "end_time", "KPIs", "anomalies")
        if column in frame.columns
    ]
    frame = frame.loc[:, kept_columns].copy()
    selected_kpi_set = set(selected_kpis)
    frame["KPIs"] = frame["KPIs"].map(
        lambda values: (
            {
                key: value
                for key, value in values.items()
                if key in selected_kpi_set
            }
            if isinstance(values, dict)
            else values
        )
    )

    context.log_dataset(
        key="train_dataset",
        df=frame,
        format="parquet",
        stats=False,
    )
    context.log_result("rows", int(len(frame)))
    context.log_result("columns", int(frame.shape[1]))
    context.set_label("dataset", dataset_name.replace("/", "-"))
    context.set_label("source_split", split)
    context.set_label("selected_kpi_count", str(len(selected_kpis)))


def _resolve_frame(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset

    as_dataframe = getattr(dataset, "as_df", None)
    if callable(as_dataframe):
        frame = as_dataframe()
        if isinstance(frame, pd.DataFrame):
            return frame

    raise TypeError(
        "The MLRun dataset input must be a pandas DataFrame or expose as_df()."
    )


def _flatten_metrics(result: dict[str, Any]) -> dict[str, float]:
    selected = next(
        (model for model in result["models"] if model["selected"]),
        None,
    )
    if selected is None:
        raise RuntimeError("The model comparison did not select a candidate.")

    metric_sets = {
        "cv": selected["cv_metrics"],
        "validation": selected["validation_metrics"],
        "test": selected["test_metrics"] or {},
    }
    flattened: dict[str, float] = {}
    for metric_set, metrics in metric_sets.items():
        for name, value in metrics.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Cannot log non-finite {metric_set} metric {name!r}."
                )
            flattened[f"{metric_set}_{name}"] = numeric_value
    return flattened


def _candidate_metrics(result: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for candidate in result["models"]:
        metric_sets = {
            "cv": candidate["cv_metrics"],
            "validation": candidate["validation_metrics"],
            "test": candidate["test_metrics"] or {},
        }
        for metric_set, metrics in metric_sets.items():
            for name, value in metrics.items():
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Cannot log non-finite {metric_set} metric {name!r} "
                        f"for {candidate['model_name']}."
                    )
                flattened[
                    f"{candidate['model_name']}_{metric_set}_{name}"
                ] = numeric_value
    return flattened


def _artifact_uri(artifact: Any, key: str) -> str:
    uri = getattr(artifact, "uri", None)
    if not isinstance(uri, str) or not uri:
        raise RuntimeError(f"MLRun did not return a URI for artifact {key!r}.")
    return uri


def train_model_comparison_component(
    context: Any,
    dataset,
    task_name: str,
    project_config_json: str,
    model_config_json: str,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Compare estimators and register metrics, parameters, and model bundles.

    The untyped dataset input lets MLRun provide a DataItem, which is converted
    to a DataFrame by _resolve_frame.
    """
    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown task name: {task_name!r}.")

    frame = _resolve_frame(dataset)
    project_config = json.loads(project_config_json)
    model_config = json.loads(model_config_json)
    if not isinstance(project_config, dict) or not isinstance(model_config, dict):
        raise ValueError("Serialized model and project configs must be JSON objects.")

    splits = prepare_task_data_component(frame, task_name, project_config)
    with tempfile.TemporaryDirectory(prefix=f"ina-{task_name}-") as temporary:
        artifact_directory = Path(temporary) / "models"
        result = run_task_model_comparison(
            task_name,
            splits,
            project_config,
            model_config,
            artifact_directory,
            evaluate_test=evaluate_test,
        )
        selected = next(
            model for model in result["models"] if model["selected"]
        )
        selected_bundle = Path(selected["artifact_path"])
        if not selected_bundle.is_file():
            raise FileNotFoundError(
                f"Selected model bundle was not created: {selected_bundle}"
            )

        context.set_label("task", task_name)
        context.set_label("selected_model", selected["model_name"])
        context.log_result("selected_model", selected["model_name"])
        context.log_result(
            "cross_validation_folds", int(result["cross_validation_folds"])
        )
        context.log_result(
            "train_samples", int(result["sample_counts"]["train"])
        )
        context.log_result(
            "validation_samples", int(result["sample_counts"]["validation"])
        )
        context.log_result("test_samples", int(result["sample_counts"]["test"]))

        metrics = _flatten_metrics(result)
        logged_metrics = _candidate_metrics(result)
        logged_metrics.update(metrics)
        for name, value in logged_metrics.items():
            context.log_result(name, value)

        for candidate in result["models"]:
            context.log_result(
                f"{candidate['model_name']}_parameters",
                json.dumps(candidate["parameters"], sort_keys=True),
            )
            bundle_path = Path(candidate["artifact_path"])
            labels = {
                "task": task_name,
                "model": candidate["model_name"],
                "selected": str(candidate["selected"]).lower(),
            }
            bundle_artifact = context.log_artifact(
                key=f"{candidate['model_name']}_bundle",
                local_path=str(bundle_path),
                labels=labels,
            )
            candidate["artifact_path"] = _artifact_uri(
                bundle_artifact,
                f"{candidate['model_name']}_bundle",
            )
            manifest_path = bundle_path.with_name("metadata.json")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Model metadata was not created: {manifest_path}"
                )
            context.log_artifact(
                key=f"{candidate['model_name']}_metadata",
                local_path=str(manifest_path),
                labels=labels,
            )

        comparison_path = artifact_directory / task_name / "comparison.json"
        with comparison_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        comparison_artifact = context.log_artifact(
            key="comparison",
            local_path=str(comparison_path),
            labels={"task": task_name},
        )
        comparison_uri = _artifact_uri(comparison_artifact, "comparison")
        selected_model_artifact = context.log_model(
            key="selected_model",
            model_file=str(selected_bundle),
            metrics=metrics,
            parameters=json.loads(json.dumps(selected["parameters"])),
            labels={
                "task": task_name,
                "algorithm": selected["model_name"],
            },
        )
        selected_model_uri = _artifact_uri(
            selected_model_artifact,
            "selected_model",
        )

    return {
        "task_name": task_name,
        "selected_model": selected["model_name"],
        "metrics": metrics,
        "parameters": selected["parameters"],
        "comparison_uri": comparison_uri,
        "selected_model_uri": selected_model_uri,
    }
