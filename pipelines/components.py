"""Pure-Python stages that can be wrapped as independent MLRun functions."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.artifacts import build_task_artifact, save_task_artifact
from src.models.training import train_model_component
from src.preprocessing.pipeline import (
    PreprocessedTask,
    prepare_task_data_component,
    preprocess_task_component,
)


def train_task_component(
    frame: pd.DataFrame,
    task_name: str,
    config: dict[str, Any],
    artifact_directory: str | Path,
) -> dict[str, Any]:
    """Run one task branch and persist its model plus train-fitted transforms."""
    splits = prepare_task_data_component(frame, task_name, config)
    preprocessed = preprocess_task_component(splits, config)
    result = train_model_component(preprocessed, config)
    artifact = build_task_artifact(preprocessed, result, config)
    artifact_path = save_task_artifact(artifact, artifact_directory)
    return {
        "task_name": task_name,
        "artifact_path": str(artifact_path),
        "validation_metrics": result.validation_metrics,
        "sample_counts": {
            "train": result.train_sample_count,
            "validation": result.validation_sample_count,
            "test": result.test_sample_count,
        },
        "feature_channels": list(splits.feature_channels),
        "model_feature_names": list(result.model_feature_names),
    }


def preprocess_split_component(
    splits: Any,
    config: dict[str, Any],
) -> PreprocessedTask:
    """Standalone preprocessing stage for MLRun or another pipeline runner."""
    return preprocess_task_component(splits, config)
