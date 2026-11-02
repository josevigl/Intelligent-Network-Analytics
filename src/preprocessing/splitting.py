"""Deterministic train/validation/test splits for the four task targets."""

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing.data import TaskDataset


@dataclass(frozen=True)
class TaskSplits:
    """Indices and raw samples for one task's disjoint data partitions."""

    task_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    positions_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    positions_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    positions_test: np.ndarray
    feature_channels: tuple[str, ...]
    label_classes: tuple[str, ...] = ()


def _multilabel_matrix(labels: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    classes = tuple(sorted({label for sample in labels for label in sample}))
    if not classes:
        raise ValueError("Multilabel targets contain no labels.")
    positions = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(labels), len(classes)), dtype=np.int8)
    for row_index, sample in enumerate(labels):
        for label in sample:
            matrix[row_index, positions[label]] = 1
    return matrix, classes


def _iterative_split(
    indices: np.ndarray,
    targets: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as error:
        raise ImportError(
            "The affected-KPI pipeline requires iterative-stratification."
        ) from error

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_positions, holdout_positions = next(
        splitter.split(np.zeros((len(indices), 1)), targets)
    )
    return indices[train_positions], indices[holdout_positions]


def _split_indices(
    dataset: TaskDataset,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one.")
    if not 0 < validation_size < 1 - test_size:
        raise ValueError(
            "validation_size must be positive and smaller than 1 - test_size."
        )

    indices = np.arange(len(dataset), dtype=np.int64)
    label_classes: tuple[str, ...] = ()

    if dataset.task_name == "affected_kpis":
        multilabel_targets, label_classes = _multilabel_matrix(dataset.y)
        train_validation, test = _iterative_split(
            indices, multilabel_targets, test_size, seed
        )
        validation_fraction = validation_size / (1 - test_size)
        train_targets = multilabel_targets[train_validation]
        train, validation = _iterative_split(
            train_validation,
            train_targets,
            validation_fraction,
            seed + 1,
        )
    else:
        if dataset.task_name == "temporal_localization":
            stratify = dataset.stratify_groups
        else:
            stratify = dataset.y
        train_validation, test = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

        validation_fraction = validation_size / (1 - test_size)
        validation_stratify = (
            dataset.stratify_groups[train_validation]
            if dataset.task_name == "temporal_localization"
            else dataset.y[train_validation]
        )
        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction,
            random_state=seed + 1,
            stratify=validation_stratify,
        )

    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(validation, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
        label_classes,
    )


def split_task_dataset(
    dataset: TaskDataset,
    test_size: float,
    validation_size: float,
    seed: int,
) -> TaskSplits:
    """Split raw task data before fitting any input or target encoder."""
    train, validation, test, label_classes = _split_indices(
        dataset, test_size, validation_size, seed
    )
    all_indices = (train, validation, test)
    if any(len(partition) == 0 for partition in all_indices):
        raise ValueError("Train, validation, and test partitions must be non-empty.")
    if (
        np.intersect1d(train, validation).size
        or np.intersect1d(train, test).size
        or np.intersect1d(validation, test).size
    ):
        raise RuntimeError("The generated data partitions overlap.")

    return TaskSplits(
        task_name=dataset.task_name,
        X_train=dataset.X[train],
        y_train=dataset.y[train],
        positions_train=dataset.positions[train],
        X_validation=dataset.X[validation],
        y_validation=dataset.y[validation],
        positions_validation=dataset.positions[validation],
        X_test=dataset.X[test],
        y_test=dataset.y[test],
        positions_test=dataset.positions[test],
        feature_channels=dataset.feature_channels,
        label_classes=label_classes,
    )
