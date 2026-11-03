"""Cross-validated comparison and final evaluation for task model families."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ParameterGrid,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    TaskModelArtifact,
    save_task_artifact,
)
from src.evaluation.metrics import summarize_task_metrics
from src.models.estimators import SafeBinaryLogisticRegression
from src.models.training import (
    DEFAULT_WINDOW_STATISTICS,
    WINDOW_STATISTICS,
    _temporal_training_rows,
    summarize_windows,
)
from src.preprocessing.data import TASK_NAMES
from src.preprocessing.sequence import SequencePreprocessor
from src.preprocessing.splitting import TaskSplits


@dataclass
class FittedCandidate:
    """Best cross-validated configuration fitted on the outer training split."""

    model_name: str
    estimator_name: str
    parameters: dict[str, Any]
    model: Any
    preprocessor: SequencePreprocessor
    target_encoder: Any | None
    model_feature_names: tuple[str, ...]
    target_classes: tuple[str, ...]
    thresholds: tuple[float, ...]
    cv_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    summary_statistics: tuple[str, ...]


def _read_model_config(
    task_name: str,
    model_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    models = model_config.get("models", {}).get(task_name)
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No model candidates configured for {task_name!r}.")
    for model_name, specification in models.items():
        if not isinstance(specification, dict) or not isinstance(
            specification.get("parameter_grid"), dict
        ):
            raise ValueError(
                f"Model {task_name}.{model_name} requires a parameter_grid mapping."
            )
    return models


def _build_estimator(
    estimator_name: str,
    parameters: dict[str, Any],
    task_name: str,
    seed: int,
) -> Any:
    if estimator_name in {
        "logistic_regression",
        "one_vs_rest_logistic_regression",
        "classifier_chain",
    }:
        logistic_parameters = dict(parameters)
        class_weight = logistic_parameters.pop("class_weight", None)
        estimator = (
            SafeBinaryLogisticRegression(
                max_iter=2000,
                random_state=seed,
                class_weight=class_weight,
                **logistic_parameters,
            )
            if estimator_name in {
                "one_vs_rest_logistic_regression",
                "classifier_chain",
            }
            else LogisticRegression(
                max_iter=2000,
                random_state=seed,
                class_weight=class_weight,
                **logistic_parameters,
            )
        )
        if estimator_name == "one_vs_rest_logistic_regression":
            return OneVsRestClassifier(estimator, n_jobs=1)
        if estimator_name == "classifier_chain":
            return ClassifierChain(
                estimator,
                order="random",
                random_state=seed,
            )
        return estimator

    if estimator_name == "random_forest":
        forest_parameters = dict(parameters)
        forest_parameters.setdefault("class_weight", "balanced")
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
            **forest_parameters,
        )

    if estimator_name == "svm":
        return SVC(
            random_state=seed,
            **parameters,
        )

    if estimator_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost candidates require the optional model dependencies. "
                "Install them with `python -m pip install -e \".[models]\"`."
            ) from error
        objective = (
            "multi:softprob"
            if task_name == "anomaly_type"
            else "binary:logistic"
        )
        xgboost_parameters = dict(parameters)
        xgboost_parameters.pop("sample_weight", None)
        return XGBClassifier(
            objective=objective,
            eval_metric="mlogloss" if task_name == "anomaly_type" else "logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            **xgboost_parameters,
        )

    raise ValueError(f"Unknown estimator family: {estimator_name!r}.")


def _fit_estimator(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    if isinstance(model, OneVsRestClassifier):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Label .* is present in all training examples\.",
                category=UserWarning,
            )
            return model.fit(X, y)
    return model.fit(X, y)


def _fit_candidate_estimator(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight_mode: str | None,
) -> Any:
    if sample_weight_mode is None or sample_weight_mode == "none":
        return _fit_estimator(model, X, y)
    if y.ndim != 1:
        raise ValueError(
            "Sample weighting is currently supported for single-label targets."
        )
    classes, class_counts = np.unique(y, return_counts=True)
    if sample_weight_mode not in {"balanced", "sqrt_balanced"}:
        raise ValueError(f"Unknown sample weight mode: {sample_weight_mode!r}.")
    power = 1.0 if sample_weight_mode == "balanced" else 0.5
    class_weights = np.power(
        len(y) / (len(classes) * class_counts),
        power,
    )
    class_weights /= np.average(
        class_weights,
        weights=class_counts,
    )
    weights = np.asarray(
        [class_weights[np.searchsorted(classes, label)] for label in y],
        dtype=np.float64,
    )
    return model.fit(X, y, sample_weight=weights)


def _task_window_statistics(
    task_name: str,
    model_config: dict[str, Any],
) -> tuple[str, ...]:
    configured = (
        model_config.get("features", {})
        .get(task_name, {})
        .get("window_statistics")
    )
    statistics = tuple(configured or DEFAULT_WINDOW_STATISTICS)
    if not set(statistics).issubset(WINDOW_STATISTICS):
        unknown = set(statistics) - set(WINDOW_STATISTICS)
        raise ValueError(f"Unknown window statistics: {sorted(unknown)}.")
    if len(set(statistics)) != len(statistics):
        raise ValueError("Configured window statistics must be unique.")
    return statistics


def _probabilities(model: Any, X: np.ndarray, multilabel: bool) -> np.ndarray:
    raw = model.predict_proba(X)
    if isinstance(raw, list):
        if not raw:
            raise ValueError("The estimator returned no per-label probabilities.")
        model_classes = getattr(model, "classes_", None)
        columns = []
        for index, values in enumerate(raw):
            probabilities = np.asarray(values)
            classes = (
                np.asarray(model_classes[index])
                if isinstance(model_classes, list)
                else np.asarray([0, 1])
            )
            positive_indices = np.flatnonzero(classes == 1)
            if positive_indices.size == 0:
                columns.append(np.zeros(len(X), dtype=np.float64))
            elif probabilities.ndim == 1:
                columns.append(probabilities)
            else:
                columns.append(probabilities[:, positive_indices[0]])
        return np.column_stack(columns)
    probabilities = np.asarray(raw)
    if multilabel:
        if probabilities.ndim != 2:
            raise ValueError("Multi-label probabilities must be a 2D matrix.")
        return probabilities
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return probabilities


def _choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fallback: float = 0.5,
) -> float:
    if len(np.unique(y_true)) < 2:
        return fallback
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    if len(thresholds) == 0:
        return fallback
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    best = np.flatnonzero(f1 == np.max(f1))
    best_index = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[best_index])


def _tune_thresholds(
    task_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fallback: float = 0.5,
) -> tuple[float, ...]:
    if task_name in {"binary_detection", "temporal_localization"}:
        return (
            _choose_threshold(
                y_true.reshape(-1), probabilities.reshape(-1), fallback
            ),
        )
    if task_name == "affected_kpis":
        return tuple(
            _choose_threshold(
                y_true[:, index], probabilities[:, index], fallback
            )
            for index in range(y_true.shape[1])
        )
    return ()


def _predict_from_probabilities(
    task_name: str,
    model: Any,
    X: np.ndarray,
    thresholds: tuple[float, ...],
    target_encoder: MultiLabelBinarizer | None,
) -> np.ndarray:
    if task_name == "anomaly_type":
        return model.predict(X)

    probabilities = _probabilities(
        model,
        X,
        multilabel=task_name == "affected_kpis",
    )
    if task_name in {"binary_detection", "temporal_localization"}:
        threshold = thresholds[0] if thresholds else 0.5
        return (probabilities >= threshold).astype(np.int64)
    predictions = (
        probabilities >= np.asarray(thresholds, dtype=np.float64)
    ).astype(np.int8)
    if target_encoder is None:
        raise ValueError("Multi-label predictions require their fitted label encoder.")
    return predictions


def _default_predictions(
    task_name: str,
    model: Any,
    X: np.ndarray,
    target_encoder: MultiLabelBinarizer | None,
) -> np.ndarray:
    if task_name == "affected_kpis":
        if target_encoder is None:
            raise ValueError("Multi-label training requires a label encoder.")
        return np.asarray(model.predict(X), dtype=np.int8)
    predictions = np.asarray(model.predict(X))
    if task_name == "anomaly_type":
        if not isinstance(target_encoder, LabelEncoder):
            raise ValueError("Anomaly type models require a fitted label encoder.")
        return target_encoder.inverse_transform(predictions.astype(np.int64))
    return predictions


def _metrics(
    task_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return summarize_task_metrics(task_name, y_true, y_pred)


def _cv_splits(
    task_name: str,
    splits: TaskSplits,
    requested_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    sample_count = len(splits.y_train)
    groups = splits.groups_train
    if task_name == "affected_kpis":
        stratification = None
    elif task_name == "temporal_localization":
        stratification = splits.stratify_groups_train
    else:
        stratification = splits.y_train

    fold_count = requested_folds
    if groups is not None:
        fold_count = min(fold_count, len(np.unique(groups)))
        if stratification is not None:
            fold_count = min(
                fold_count,
                min(
                    len(np.unique(groups[stratification == label]))
                    for label in np.unique(stratification)
                ),
            )
    elif stratification is not None:
        fold_count = min(
            fold_count,
            int(np.unique(stratification, return_counts=True)[1].min()),
        )
    if fold_count < 2:
        raise ValueError(
            "Cross-validation requires at least two folds with independent "
            "groups and target examples."
        )

    if groups is not None and stratification is not None:
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        indices = np.arange(sample_count)
        folds = list(splitter.split(indices, stratification, groups))
    elif groups is not None:
        folds = list(
            GroupKFold(n_splits=fold_count).split(
                np.arange(sample_count), groups=groups
            )
        )
    elif stratification is not None:
        splitter = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        folds = list(splitter.split(np.arange(sample_count), stratification))
    else:
        splitter = KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        folds = list(splitter.split(np.arange(sample_count)))

    if groups is not None:
        for train_indices, validation_indices in folds:
            if np.intersect1d(
                groups[train_indices], groups[validation_indices]
            ).size:
                raise RuntimeError("A cross-validation fold contains group leakage.")
    return folds


def _fit_fold_targets(
    task_name: str,
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any | None]:
    if task_name == "anomaly_type":
        encoder = LabelEncoder()
        encoded_train = encoder.fit_transform(y_train)
        unseen = set(y_validation) - set(encoder.classes_)
        if unseen:
            raise ValueError(
                "A cross-validation holdout contains anomaly types absent "
                f"from its training fold: {sorted(unseen)}"
            )
        return encoded_train, encoder.transform(y_validation), encoder
    if task_name != "affected_kpis":
        return y_train, y_validation, None
    encoder = MultiLabelBinarizer()
    encoded_train = encoder.fit_transform(y_train)
    known_labels = set(str(value) for value in encoder.classes_)
    unseen = {
        label
        for sample in y_validation
        for label in sample
        if label not in known_labels
    }
    if unseen:
        raise ValueError(
            "A cross-validation holdout contains labels absent from its "
            f"training fold: {sorted(unseen)}"
        )
    return encoded_train, encoder.transform(y_validation), encoder


def _collect_cv_scores(
    task_name: str,
    splits: TaskSplits,
    model_specs: dict[str, dict[str, Any]],
    model_config: dict[str, Any],
    model_config_context: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    fold_count = int(model_config["cross_validation"].get("folds", 5))
    folds = _cv_splits(task_name, splits, fold_count, seed)
    candidates: list[dict[str, Any]] = []
    summary_statistics = _task_window_statistics(task_name, model_config)
    for model_name, specification in model_specs.items():
        estimator_name = str(specification["estimator"])
        for parameters in ParameterGrid(specification["parameter_grid"]):
            candidates.append(
                {
                    "model_name": model_name,
                    "estimator_name": estimator_name,
                    "parameters": dict(parameters),
                    "fold_metrics": [],
                }
            )

    for fold_number, (train_indices, validation_indices) in enumerate(folds):
        preprocessor = SequencePreprocessor(
            feature_channels=splits.feature_channels,
            categorical_missing_value=model_config_context["preprocessing"].get(
                "categorical_missing_value", "__MISSING__"
            ),
        )
        X_train = preprocessor.fit_transform(splits.X_train[train_indices])
        X_validation = preprocessor.transform(splits.X_train[validation_indices])
        if task_name == "temporal_localization":
            X_train = X_train.reshape(-1, X_train.shape[-1])
            X_validation = X_validation.reshape(-1, X_validation.shape[-1])
            y_train = splits.y_train[train_indices].reshape(-1)
            y_validation = splits.y_train[validation_indices].reshape(-1)
        else:
            channel_names = tuple(preprocessor.get_feature_names_out())
            X_train, train_features = summarize_windows(
                X_train, channel_names, summary_statistics
            )
            X_validation, validation_features = summarize_windows(
                X_validation, channel_names, summary_statistics
            )
            if train_features != validation_features:
                raise RuntimeError("Fold feature order changed after preprocessing.")
            y_train = splits.y_train[train_indices]
            y_validation = splits.y_train[validation_indices]

        y_train_encoded, y_validation_encoded, encoder = _fit_fold_targets(
            task_name,
            y_train,
            y_validation,
        )
        for candidate in candidates:
            model = _build_estimator(
                candidate["estimator_name"],
                candidate["parameters"],
                task_name,
                seed + fold_number,
            )
            if task_name == "temporal_localization":
                max_rows = model_config_context["training"]["random_forest"].get(
                    "max_temporal_training_timesteps"
                )
                if max_rows is not None:
                    X_fit, y_fit = _temporal_training_rows(
                        X_train.reshape(
                            len(train_indices),
                            -1,
                            X_train.shape[-1],
                        )
                        if X_train.ndim == 2
                        else X_train,
                        splits.y_train[train_indices],
                        int(max_rows),
                        seed + fold_number,
                    )
                else:
                    X_fit, y_fit = X_train, y_train_encoded
            else:
                X_fit, y_fit = X_train, y_train_encoded
            _fit_candidate_estimator(
                model,
                X_fit,
                y_fit,
                candidate["parameters"].get("sample_weight"),
            )
            prediction = _default_predictions(
                task_name, model, X_validation, encoder
            )
            metric_targets = (
                encoder.inverse_transform(y_validation_encoded)
                if task_name == "anomaly_type"
                else y_validation_encoded
            )
            candidate["fold_metrics"].append(
                _metrics(task_name, metric_targets, prediction)
            )

    for candidate in candidates:
        metrics = candidate["fold_metrics"]
        keys = sorted({key for fold in metrics for key in fold})
        candidate["cv_metrics"] = {
            f"{key}_mean": float(np.mean([fold[key] for fold in metrics]))
            for key in keys
        }
        candidate["cv_metrics"].update(
            {
                f"{key}_std": float(np.std([fold[key] for fold in metrics]))
                for key in keys
            }
        )
    return candidates, folds


def _fit_candidate(
    task_name: str,
    splits: TaskSplits,
    candidate: dict[str, Any],
    model_config_context: dict[str, Any],
    summary_statistics: tuple[str, ...],
    threshold_fallback: float,
    seed: int,
) -> FittedCandidate:
    preprocessor = SequencePreprocessor(
        feature_channels=splits.feature_channels,
        categorical_missing_value=model_config_context["preprocessing"].get(
            "categorical_missing_value", "__MISSING__"
        ),
    )
    X_train = preprocessor.fit_transform(splits.X_train)
    X_validation = preprocessor.transform(splits.X_validation)
    if task_name == "temporal_localization":
        X_train_model = X_train.reshape(-1, X_train.shape[-1])
        X_validation_model = X_validation.reshape(-1, X_validation.shape[-1])
        y_train = splits.y_train.reshape(-1)
        y_validation = splits.y_validation.reshape(-1)
        feature_names = tuple(preprocessor.get_feature_names_out())
    else:
        channel_names = tuple(preprocessor.get_feature_names_out())
        X_train_model, feature_names = summarize_windows(
            X_train, channel_names, summary_statistics
        )
        X_validation_model, validation_names = summarize_windows(
            X_validation, channel_names, summary_statistics
        )
        if feature_names != validation_names:
            raise RuntimeError("Train and validation feature orders differ.")
        y_train = splits.y_train
        y_validation = splits.y_validation

    y_train_encoded, y_validation_encoded, encoder = _fit_fold_targets(
        task_name, y_train, y_validation
    )
    if task_name == "affected_kpis":
        y_validation_encoded = encoder.transform(splits.y_validation)
    model = _build_estimator(
        candidate["estimator_name"],
        candidate["parameters"],
        task_name,
        seed,
    )
    if task_name == "temporal_localization":
        max_rows = model_config_context["training"]["random_forest"].get(
            "max_temporal_training_timesteps"
        )
        if max_rows is not None:
            X_train_model, y_train_encoded = _temporal_training_rows(
                X_train,
                splits.y_train,
                int(max_rows),
                seed,
            )
    _fit_candidate_estimator(
        model,
        X_train_model,
        y_train_encoded,
        candidate["parameters"].get("sample_weight"),
    )

    if task_name == "anomaly_type":
        validation_probabilities = np.empty((0,))
        thresholds = ()
        if not isinstance(encoder, LabelEncoder):
            raise ValueError("Anomaly type model is missing its label encoder.")
        validation_predictions = encoder.inverse_transform(
            model.predict(X_validation_model).astype(np.int64)
        )
    else:
        validation_probabilities = _probabilities(
            model,
            X_validation_model,
            multilabel=task_name == "affected_kpis",
        )
        thresholds = _tune_thresholds(
            task_name,
            y_validation_encoded,
            validation_probabilities,
            threshold_fallback,
        )
        validation_predictions = _predict_from_probabilities(
            task_name,
            model,
            X_validation_model,
            thresholds,
            encoder,
        )
    validation_metric_targets = (
        encoder.inverse_transform(y_validation_encoded)
        if task_name == "anomaly_type"
        else y_validation_encoded
    )
    validation_metrics = _metrics(
        task_name, validation_metric_targets, validation_predictions
    )
    if task_name == "temporal_localization":
        validation_metrics["threshold"] = thresholds[0]

    target_classes = (
        tuple(str(value) for value in encoder.classes_)
        if encoder is not None
        else tuple(str(value) for value in model.classes_)
    )
    return FittedCandidate(
        model_name=candidate["model_name"],
        estimator_name=candidate["estimator_name"],
        parameters=candidate["parameters"],
        model=model,
        preprocessor=preprocessor,
        target_encoder=encoder,
        model_feature_names=feature_names,
        target_classes=target_classes,
        thresholds=thresholds,
        cv_metrics=candidate["cv_metrics"],
        validation_metrics=validation_metrics,
        summary_statistics=summary_statistics,
    )


def _test_metrics(
    task_name: str,
    fitted: FittedCandidate,
    splits: TaskSplits,
) -> dict[str, float]:
    X_test_transformed = fitted.preprocessor.transform(splits.X_test)
    if task_name == "temporal_localization":
        X_test = X_test_transformed.reshape(-1, X_test_transformed.shape[-1])
        y_test = splits.y_test.reshape(-1)
    else:
        X_test, feature_names = summarize_windows(
            X_test_transformed,
            tuple(fitted.preprocessor.get_feature_names_out()),
            fitted.summary_statistics,
        )
        if feature_names != fitted.model_feature_names:
            raise RuntimeError("Test feature order differs from the saved model.")
        y_test = splits.y_test
    if task_name == "affected_kpis":
        if fitted.target_encoder is None:
            raise ValueError("The selected multi-label model has no label encoder.")
        known_labels = set(str(value) for value in fitted.target_encoder.classes_)
        unknown = {
            label
            for sample in y_test
            for label in sample
            if label not in known_labels
        }
        if unknown:
            raise ValueError(
                "Test data contains affected-KPI labels absent from training: "
                f"{sorted(unknown)}"
            )
        y_test = fitted.target_encoder.transform(y_test)

    if task_name == "anomaly_type":
        if not isinstance(fitted.target_encoder, LabelEncoder):
            raise ValueError("Anomaly type model is missing its label encoder.")
        unknown = set(y_test) - set(fitted.target_encoder.classes_)
        if unknown:
            raise ValueError(
                "Test data contains anomaly types absent from training: "
                f"{sorted(unknown)}"
            )
        predictions = fitted.target_encoder.inverse_transform(
            fitted.model.predict(X_test).astype(np.int64)
        )
    else:
        predictions = _predict_from_probabilities(
            task_name,
            fitted.model,
            X_test,
            fitted.thresholds,
            fitted.target_encoder,
        )
    metrics = _metrics(task_name, y_test, predictions)
    if task_name == "temporal_localization":
        metrics["threshold"] = fitted.thresholds[0]
    return metrics


def run_task_model_comparison(
    task_name: str,
    splits: TaskSplits,
    project_config: dict[str, Any],
    model_config: dict[str, Any],
    artifact_directory: str | Path,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """CV-search each configured family, compare on validation, test once."""
    if task_name not in TASK_NAMES or task_name != splits.task_name:
        raise ValueError(f"Task and data split do not match: {task_name!r}.")
    specifications = _read_model_config(task_name, model_config)
    seed = int(model_config["cross_validation"].get("shuffle_seed", 42))
    raw_candidates, folds = _collect_cv_scores(
        task_name,
        splits,
        specifications,
        model_config,
        project_config,
        seed,
    )

    best_by_family: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        family = candidate["model_name"]
        previous = best_by_family.get(family)
        metric = candidate["cv_metrics"]["f1_macro_mean"]
        if previous is None or metric > previous["cv_metrics"]["f1_macro_mean"]:
            best_by_family[family] = candidate

    threshold_objective = model_config.get("selection", {}).get(
        "threshold_objective", "f1"
    )
    if threshold_objective != "f1":
        raise ValueError(
            f"Unsupported threshold objective: {threshold_objective!r}."
        )
    threshold_fallback = float(
        model_config.get("selection", {}).get("threshold_fallback", 0.5)
    )
    if not 0 < threshold_fallback < 1:
        raise ValueError("selection.threshold_fallback must be between zero and one.")
    summary_statistics = _task_window_statistics(task_name, model_config)
    fitted_candidates = [
        _fit_candidate(
            task_name,
            splits,
            candidate,
            project_config,
            summary_statistics,
            threshold_fallback,
            seed,
        )
        for candidate in best_by_family.values()
    ]
    primary_metric = str(
        model_config.get("selection", {}).get("primary_metric", "f1_macro")
    )
    if primary_metric not in {"f1_macro", "f1_micro", "f1_weighted"}:
        raise ValueError(f"Unsupported model selection metric: {primary_metric!r}.")
    selection_source = model_config.get("selection", {}).get(
        "model_source_by_task", {}
    ).get(task_name, "validation")
    if selection_source == "cross_validation":
        selected = max(
            fitted_candidates,
            key=lambda fitted: fitted.cv_metrics[f"{primary_metric}_mean"],
        )
    elif selection_source == "validation":
        selected = max(
            fitted_candidates,
            key=lambda fitted: (
                fitted.validation_metrics[primary_metric],
                fitted.cv_metrics[f"{primary_metric}_mean"],
            ),
        )
    else:
        raise ValueError(
            "Model selection source must be 'validation' or 'cross_validation'."
        )
    selected_test_metrics = (
        _test_metrics(task_name, selected, splits)
        if evaluate_test
        else None
    )

    result_rows: list[dict[str, Any]] = []
    candidate_root = Path(artifact_directory) / task_name
    for fitted in fitted_candidates:
        is_selected = fitted.model_name == selected.model_name
        test_metrics = selected_test_metrics if is_selected else None
        artifact = TaskModelArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            task_name=task_name,
            sequence_length=fitted.preprocessor.sequence_length_,
            feature_channels=splits.feature_channels,
            transformed_feature_names=tuple(
                fitted.preprocessor.get_feature_names_out()
            ),
            model_feature_names=fitted.model_feature_names,
            input_mode=(
                "per_timestep"
                if task_name == "temporal_localization"
                else "window_statistics"
            ),
            preprocessor=fitted.preprocessor,
            model=fitted.model,
            target_encoder=fitted.target_encoder,
            target_classes=fitted.target_classes,
            config={
                "project": project_config,
                "model_selection": model_config,
            },
            validation_metrics=fitted.validation_metrics,
            library_versions={
                "python": __import__("platform").python_version(),
                "scikit_learn": __import__("sklearn").__version__,
            },
            model_name=fitted.model_name,
            model_parameters=fitted.parameters,
            decision_thresholds=fitted.thresholds,
            cv_metrics=fitted.cv_metrics,
            test_metrics=test_metrics,
            selected=is_selected,
            summary_statistics=fitted.summary_statistics,
        )
        artifact_path = save_task_artifact(
            artifact,
            candidate_root / fitted.model_name,
        )
        result_rows.append(
            {
                "task_name": task_name,
                "model_name": fitted.model_name,
                "estimator": fitted.estimator_name,
                "parameters": fitted.parameters,
                "cv_metrics": fitted.cv_metrics,
                "validation_metrics": fitted.validation_metrics,
                "test_metrics": test_metrics,
                "decision_thresholds": list(fitted.thresholds),
                "selected": is_selected,
                "artifact_path": str(artifact_path),
            }
        )

    payload = {
        "task_name": task_name,
        "primary_metric": primary_metric,
        "selection_source": selection_source,
        "test_evaluated": evaluate_test,
        "group_split": splits.groups_train is not None,
        "grouping_strategy": (
            "overlapping_time_intervals"
            if splits.groups_train is not None
            else "stratified_samples"
        ),
        "cross_validation_folds": len(folds),
        "sample_counts": {
            "train": len(splits.y_train),
            "validation": len(splits.y_validation),
            "test": len(splits.y_test),
        },
        "group_count": (
            int(
                len(
                    np.unique(
                        np.concatenate(
                            [
                                splits.groups_train,
                                splits.groups_validation,
                                splits.groups_test,
                            ]
                        )
                    )
                )
            )
            if splits.groups_train is not None
            else None
        ),
        "cv_candidate_results": [
            {
                "model_name": candidate["model_name"],
                "estimator": candidate["estimator_name"],
                "parameters": candidate["parameters"],
                "cv_metrics": candidate["cv_metrics"],
            }
            for candidate in raw_candidates
        ],
        "models": result_rows,
        "selected_model": selected.model_name,
        "test_metrics": selected_test_metrics,
    }
    candidate_root.mkdir(parents=True, exist_ok=True)
    with (candidate_root / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return payload
