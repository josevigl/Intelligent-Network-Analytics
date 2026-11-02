"""Deterministic train/validation/test splits for the four task targets."""

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    groups_train: np.ndarray | None = None
    groups_validation: np.ndarray | None = None
    groups_test: np.ndarray | None = None
    stratify_groups_train: np.ndarray | None = None
    stratify_groups_validation: np.ndarray | None = None
    stratify_groups_test: np.ndarray | None = None


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


def _group_target_matrix(dataset: TaskDataset) -> np.ndarray:
    if dataset.task_name == "affected_kpis":
        return _multilabel_matrix(dataset.y)[0]
    if dataset.task_name == "temporal_localization":
        labels = dataset.stratify_groups
    else:
        labels = dataset.y
    if labels is None:
        raise ValueError("Group-aware splitting requires target labels.")
    classes, encoded = np.unique(labels, return_inverse=True)
    matrix = np.zeros((len(labels), len(classes)), dtype=np.int8)
    matrix[np.arange(len(labels)), encoded] = 1
    return matrix


def _group_holdout(
    indices: np.ndarray,
    groups: np.ndarray,
    target_matrix: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups[indices])
    if len(unique_groups) < 3:
        raise ValueError(
            "At least three independent time-overlap groups are required for "
            "train/validation/test splitting."
        )

    best: tuple[np.ndarray, np.ndarray] | None = None
    best_score = float("inf")
    for offset in range(64):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed + offset,
        )
        train_local, holdout_local = next(
            splitter.split(indices, groups=groups[indices])
        )
        train_indices = indices[train_local]
        holdout_indices = indices[holdout_local]
        observed_size = len(holdout_indices) / len(indices)
        target_error = np.abs(
            target_matrix[holdout_indices].mean(axis=0)
            - target_matrix[indices].mean(axis=0)
        ).mean()
        score = abs(observed_size - test_size) + 2.0 * target_error
        if score < best_score:
            best = (train_indices, holdout_indices)
            best_score = score
    if best is None:
        raise RuntimeError("Unable to create a group-aware holdout split.")
    return best


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

    if dataset.groups is not None:
        target_matrix = _group_target_matrix(dataset)
        train_validation, test = _group_holdout(
            indices,
            dataset.groups,
            target_matrix,
            test_size,
            seed,
        )
        validation_fraction = validation_size / (1 - test_size)
        train, validation = _group_holdout(
            train_validation,
            dataset.groups,
            target_matrix,
            validation_fraction,
            seed + 64,
        )
        if dataset.task_name == "affected_kpis":
            label_classes = _multilabel_matrix(dataset.y)[1]
    elif dataset.task_name == "affected_kpis":
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
        groups_train=dataset.groups[train] if dataset.groups is not None else None,
        groups_validation=(
            dataset.groups[validation] if dataset.groups is not None else None
        ),
        groups_test=dataset.groups[test] if dataset.groups is not None else None,
        stratify_groups_train=(
            dataset.stratify_groups[train]
            if dataset.stratify_groups is not None
            else None
        ),
        stratify_groups_validation=(
            dataset.stratify_groups[validation]
            if dataset.stratify_groups is not None
            else None
        ),
        stratify_groups_test=(
            dataset.stratify_groups[test]
            if dataset.stratify_groups is not None
            else None
        ),
    )
