"""Independent data preparation and preprocessing stages for pipeline runners."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import validate_pipeline_config
from src.preprocessing.data import TASK_FEATURES, build_task_dataset
from src.preprocessing.sequence import SequencePreprocessor
from src.preprocessing.splitting import TaskSplits, split_task_dataset


@dataclass(frozen=True)
class PreprocessedTask:
    """Train-fitted transformations and all transformed task partitions."""

    splits: TaskSplits
    preprocessor: SequencePreprocessor
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray


def prepare_task_data_component(
    frame: Any,
    task_name: str,
    config: dict[str, Any],
) -> TaskSplits:
    """Extract one task and split it before fitting any preprocessing."""
    validate_pipeline_config(config)
    feature_map = config["preprocessing"]["selected_kpis"]
    feature_channels = feature_map.get(task_name)
    if feature_channels is None:
        raise ValueError(f"No selected KPI channels configured for {task_name!r}.")
    if task_name not in TASK_FEATURES:
        raise ValueError(f"Unknown task: {task_name!r}.")

    dataset = build_task_dataset(
        frame,
        task_name,
        feature_channels=feature_channels,
        sequence_length=int(config["preprocessing"].get("sequence_length", 128)),
    )
    splits = split_task_dataset(
        dataset,
        test_size=float(config["project"]["test_size"]),
        validation_size=float(config["training"]["validation_size"]),
        seed=int(config["project"]["seed"]),
    )
    return splits


def preprocess_task_component(
    splits: TaskSplits,
    config: dict[str, Any],
) -> PreprocessedTask:
    """Fit imputers, scaler, and categorical encoder on train windows only."""
    validate_pipeline_config(config)
    preprocessor = SequencePreprocessor(
        feature_channels=splits.feature_channels,
        categorical_missing_value=config["preprocessing"].get(
            "categorical_missing_value", "__MISSING__"
        ),
    )
    X_train = preprocessor.fit_transform(splits.X_train)
    X_validation = preprocessor.transform(splits.X_validation)
    X_test = preprocessor.transform(splits.X_test)
    return PreprocessedTask(
        splits=splits,
        preprocessor=preprocessor,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
    )
