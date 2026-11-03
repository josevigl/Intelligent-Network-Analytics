"""Train validation-scored Random Forest baselines on preprocessed task data."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import validate_pipeline_config
from src.evaluation.metrics import summarize_task_metrics
from src.preprocessing.pipeline import PreprocessedTask


WINDOW_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "range",
    "slope",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "first",
    "last",
    "delta",
    "iqr",
)

DEFAULT_WINDOW_STATISTICS = WINDOW_STATISTICS[:6]


@dataclass(frozen=True)
class ModelTrainingResult:
    """Fitted task model and validation-only metrics."""

    task_name: str
    model: RandomForestClassifier
    input_mode: str
    model_feature_names: tuple[str, ...]
    target_encoder: MultiLabelBinarizer | None
    target_classes: tuple[str, ...]
    validation_metrics: dict[str, float]
    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int


def summarize_windows(
    X: np.ndarray,
    channel_names: tuple[str, ...] | list[str] | None = None,
    statistics: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create the same six per-window summaries used by the EDA forests."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("X must have shape (samples, timesteps, features).")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Window summaries require samples and at least two timesteps.")
    if not np.isfinite(values).all():
        raise ValueError("KPI preprocessing must remove all NaN and infinite values.")
    if channel_names is None:
        channel_names = tuple(
            f"feature_{index}" for index in range(values.shape[2])
        )
    else:
        channel_names = tuple(channel_names)
    if len(channel_names) != values.shape[2]:
        raise ValueError("channel_names must match the number of input features.")
    selected_statistics = tuple(
        DEFAULT_WINDOW_STATISTICS if statistics is None else statistics
    )
    if not selected_statistics:
        raise ValueError("At least one window statistic must be selected.")
    if len(set(selected_statistics)) != len(selected_statistics):
        raise ValueError("Window statistics must be unique.")
    unknown_statistics = set(selected_statistics) - set(WINDOW_STATISTICS)
    if unknown_statistics:
        raise ValueError(
            f"Unknown window statistics: {sorted(unknown_statistics)}."
        )

    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1)
    minima = values.min(axis=1)
    maxima = values.max(axis=1)
    timeline = np.arange(values.shape[1], dtype=np.float64)
    centered_timeline = timeline - timeline.mean()
    slopes = np.einsum(
        "t,ntf->nf", centered_timeline, values - means[:, None, :]
    ) / np.square(centered_timeline).sum()
    summaries = {
        "mean": means,
        "std": standard_deviations,
        "min": minima,
        "max": maxima,
        "range": maxima - minima,
        "slope": slopes,
        "q10": np.quantile(values, 0.10, axis=1),
        "q25": np.quantile(values, 0.25, axis=1),
        "median": np.median(values, axis=1),
        "q75": np.quantile(values, 0.75, axis=1),
        "q90": np.quantile(values, 0.90, axis=1),
        "first": values[:, 0, :],
        "last": values[:, -1, :],
        "delta": values[:, -1, :] - values[:, 0, :],
        "iqr": np.quantile(values, 0.75, axis=1)
        - np.quantile(values, 0.25, axis=1),
    }
    matrix = np.concatenate(
        [summaries[name] for name in selected_statistics],
        axis=1,
    ).astype(np.float32)
    feature_names = tuple(
        f"{channel}__{statistic}"
        for statistic in selected_statistics
        for channel in channel_names
    )
    return matrix, feature_names


def _validate_target_coverage(
    y: np.ndarray, known_classes: tuple[str, ...], partition_name: str
) -> None:
    unseen = {
        label
        for sample in y
        for label in sample
        if label not in known_classes
    }
    if unseen:
        raise ValueError(
            f"{partition_name} contains affected-KPI labels absent from the "
            f"training partition: {sorted(unseen)}"
        )


def _make_model(config: dict[str, Any], task_name: str) -> RandomForestClassifier:
    settings = config["training"].get("random_forest", {})
    return RandomForestClassifier(
        n_estimators=int(settings.get("n_estimators", 100)),
        max_depth=settings.get("max_depth", 14),
        min_samples_leaf=int(settings.get("min_samples_leaf", 2)),
        class_weight=(
            "balanced_subsample"
            if task_name in {"binary_detection", "anomaly_type", "temporal_localization"}
            else None
        ),
        random_state=int(config["project"]["seed"]),
        n_jobs=1,
    )


def _temporal_training_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows = X.reshape(-1, X.shape[-1])
    y_rows = y.reshape(-1)
    if max_rows is None or max_rows >= len(y_rows):
        return X_rows, y_rows
    if max_rows < 2:
        raise ValueError("max_temporal_training_timesteps must be at least two.")

    rng = np.random.default_rng(seed)
    classes, class_counts = np.unique(y_rows, return_counts=True)
    if len(classes) != 2 or np.any(class_counts == 0):
        raise ValueError("Temporal training targets must contain both classes.")
    requested = np.rint(max_rows * class_counts / class_counts.sum()).astype(int)
    requested = np.maximum(requested, 1)
    while requested.sum() > max_rows:
        requested[np.argmax(requested)] -= 1
    while requested.sum() < max_rows:
        requested[np.argmax(class_counts - requested)] += 1

    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(y_rows == label), size=count, replace=False)
            for label, count in zip(classes, requested)
        ]
    )
    rng.shuffle(selected)
    return X_rows[selected], y_rows[selected]


def train_model_component(
    preprocessed: PreprocessedTask,
    config: dict[str, Any],
) -> ModelTrainingResult:
    """Train one estimator; use validation data for metrics, never for fitting."""
    validate_pipeline_config(config)
    task_name = preprocessed.splits.task_name
    target_encoder = None
    target_classes: tuple[str, ...] = ()

    if task_name == "temporal_localization":
        X_train = preprocessed.X_train.reshape(-1, preprocessed.X_train.shape[-1])
        y_train = preprocessed.splits.y_train.reshape(-1)
        max_rows = config["training"].get("random_forest", {}).get(
            "max_temporal_training_timesteps"
        )
        X_train, y_train = _temporal_training_rows(
            preprocessed.X_train,
            preprocessed.splits.y_train,
            int(max_rows) if max_rows is not None else None,
            int(config["project"]["seed"]),
        )
        model = _make_model(config, task_name)
        model.fit(X_train, y_train)
        X_validation = preprocessed.X_validation.reshape(
            -1, preprocessed.X_validation.shape[-1]
        )
        y_validation = preprocessed.splits.y_validation.reshape(-1)
        y_pred = model.predict(X_validation)
        validation_metrics = summarize_task_metrics(
            task_name, y_validation, y_pred
        )
        input_mode = "per_timestep"
        feature_names = tuple(preprocessed.preprocessor.get_feature_names_out())
    elif task_name == "affected_kpis":
        target_encoder = MultiLabelBinarizer()
        y_train = target_encoder.fit_transform(preprocessed.splits.y_train)
        _validate_target_coverage(
            preprocessed.splits.y_validation,
            tuple(str(value) for value in target_encoder.classes_),
            "Validation data",
        )
        y_validation = target_encoder.transform(
            preprocessed.splits.y_validation
        )
        target_classes = tuple(str(value) for value in target_encoder.classes_)
        channel_names = tuple(preprocessed.preprocessor.get_feature_names_out())
        X_train, feature_names = summarize_windows(
            preprocessed.X_train, channel_names
        )
        X_validation, validation_feature_names = summarize_windows(
            preprocessed.X_validation, channel_names
        )
        if feature_names != validation_feature_names:
            raise RuntimeError("Training and validation feature orders differ.")
        model = _make_model(config, task_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_validation)
        validation_metrics = summarize_task_metrics(
            task_name, y_validation, y_pred
        )
        input_mode = "window_statistics"
    else:
        y_train = preprocessed.splits.y_train
        y_validation = preprocessed.splits.y_validation
        channel_names = tuple(preprocessed.preprocessor.get_feature_names_out())
        X_train, feature_names = summarize_windows(
            preprocessed.X_train, channel_names
        )
        X_validation, validation_feature_names = summarize_windows(
            preprocessed.X_validation, channel_names
        )
        if feature_names != validation_feature_names:
            raise RuntimeError("Training and validation feature orders differ.")
        model = _make_model(config, task_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_validation)
        validation_metrics = summarize_task_metrics(
            task_name, y_validation, y_pred
        )
        input_mode = "window_statistics"
        target_classes = tuple(str(value) for value in model.classes_)

    return ModelTrainingResult(
        task_name=task_name,
        model=model,
        input_mode=input_mode,
        model_feature_names=tuple(feature_names),
        target_encoder=target_encoder,
        target_classes=target_classes,
        validation_metrics=validation_metrics,
        train_sample_count=len(preprocessed.splits.y_train),
        validation_sample_count=len(preprocessed.splits.y_validation),
        test_sample_count=len(preprocessed.splits.y_test),
    )
