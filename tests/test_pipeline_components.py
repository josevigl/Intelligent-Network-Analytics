"""Focused tests for train-only preprocessing and artifact-based inference."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.artifacts import (
    build_task_artifact,
    load_task_artifact,
    save_task_artifact,
)
from src.config import load_pipeline_config
from src.inference import predict_task_component
from src.models.training import summarize_windows, train_model_component
from src.preprocessing.pipeline import (
    prepare_task_data_component,
    preprocess_task_component,
)
from src.preprocessing.splitting import TaskSplits
from pipelines.train import run_training_pipeline
from src.models.experiments import run_task_model_comparison
from pipelines.train_models import run_model_comparison_pipeline
from src.models.estimators import SafeBinaryLogisticRegression


class PipelineComponentTests(unittest.TestCase):
    def test_preprocessing_fits_only_train_and_ignores_unknown_category(self) -> None:
        splits = TaskSplits(
            task_name="binary_detection",
            X_train=np.array(
                [
                    [[1.0, "A"], [3.0, "A"]],
                    [[3.0, "A"], [5.0, "A"]],
                ],
                dtype=object,
            ),
            y_train=np.array([0, 1]),
            positions_train=np.array([0, 1]),
            X_validation=np.array([[[1000.0, "B"], [1000.0, "B"]]], dtype=object),
            y_validation=np.array([1]),
            positions_validation=np.array([2]),
            X_test=np.array([[[2000.0, "C"], [2000.0, "C"]]], dtype=object),
            y_test=np.array([0]),
            positions_test=np.array([3]),
            feature_channels=("RSRP", "DL_Protocol"),
        )
        config = load_pipeline_config()
        processed = preprocess_task_component(splits, config)

        numeric_pipeline = processed.preprocessor.transformer_.named_transformers_[
            "numeric"
        ]
        np.testing.assert_allclose(
            numeric_pipeline.named_steps["scaler"].mean_, [3.0]
        )
        self.assertEqual(
            processed.preprocessor.transformer_.named_transformers_[
                "categorical"
            ].named_steps["encoder"].categories_[0].tolist(),
            ["A"],
        )
        self.assertEqual(processed.X_validation.shape, (1, 2, 2))
        self.assertTrue(np.all(processed.X_validation[:, :, 1] == 0))
        self.assertTrue(np.isfinite(processed.X_test).all())

    def test_artifact_round_trip_inference_does_not_refit(self) -> None:
        train_windows = np.array(
            [
                [[0.0], [1.0]],
                [[0.1], [1.1]],
                [[4.0], [5.0]],
                [[4.1], [5.1]],
                [[0.2], [1.2]],
                [[3.9], [4.9]],
                [[0.3], [1.3]],
                [[4.2], [5.2]],
            ],
            dtype=object,
        )
        splits = TaskSplits(
            task_name="binary_detection",
            X_train=train_windows,
            y_train=np.array([0, 0, 1, 1, 0, 1, 0, 1]),
            positions_train=np.arange(8),
            X_validation=train_windows[:4],
            y_validation=np.array([0, 0, 1, 1]),
            positions_validation=np.arange(8, 12),
            X_test=train_windows[4:],
            y_test=np.array([0, 1, 0, 1]),
            positions_test=np.arange(12, 16),
            feature_channels=("RSRP",),
        )
        config = load_pipeline_config()
        config["training"]["random_forest"].update(
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=1,
        )
        processed = preprocess_task_component(splits, config)
        trained = train_model_component(processed, config)
        artifact = build_task_artifact(processed, trained, config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = save_task_artifact(artifact, temporary_directory)
            loaded = load_task_artifact(artifact_path)
            frame = pd.DataFrame(
                {
                    "KPIs": [
                        {"RSRP": [0.2, 1.2]},
                        {"RSRP": [4.1, 5.1]},
                    ]
                }
            )
            with patch.object(
                loaded.preprocessor,
                "fit",
                side_effect=AssertionError("Inference attempted to fit."),
            ):
                predictions = predict_task_component(frame, loaded)

        self.assertEqual(loaded.task_name, "binary_detection")
        self.assertEqual(Path(artifact_path).name, "model_bundle.joblib")
        self.assertEqual(predictions.positions.tolist(), [0, 1])
        self.assertEqual(predictions.predictions.shape, (2,))

    def test_window_summary_names_match_feature_order(self) -> None:
        values = np.array([[[1.0, 4.0], [3.0, 8.0]]])
        matrix, feature_names = summarize_windows(values, ("RSRP", "UL_SNR"))

        self.assertEqual(matrix.shape, (1, 12))
        self.assertEqual(
            feature_names[:4],
            ("RSRP__mean", "UL_SNR__mean", "RSRP__std", "UL_SNR__std"),
        )

    def test_window_summary_supports_robust_anomaly_type_statistics(self) -> None:
        values = np.array([[[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]]])
        selected_statistics = (
            "mean",
            "median",
            "q10",
            "q90",
            "first",
            "last",
            "delta",
            "iqr",
        )
        matrix, feature_names = summarize_windows(
            values,
            ("RSRP", "UL_SNR"),
            selected_statistics,
        )

        self.assertEqual(matrix.shape, (1, 16))
        self.assertEqual(feature_names[:2], ("RSRP__mean", "UL_SNR__mean"))
        self.assertEqual(feature_names[-2:], ("RSRP__iqr", "UL_SNR__iqr"))
        self.assertTrue(np.isfinite(matrix).all())

    def test_safe_binary_logistic_regression_handles_constant_label(self) -> None:
        features = np.array([[0.0], [1.0], [2.0]])
        for label in (0, 1):
            estimator = SafeBinaryLogisticRegression().fit(
                features,
                np.full(3, label),
            )
            np.testing.assert_array_equal(
                estimator.predict(features),
                np.full(3, label),
            )
            np.testing.assert_allclose(
                estimator.predict_proba(features)[:, label],
                np.ones(3),
            )

    def test_all_four_training_branches_save_independent_artifacts(self) -> None:
        rows = []
        for index in range(120):
            anomalous = index % 2 == 0
            anomaly_type = f"type_{(index // 2) % 3}"
            annotation = (
                {
                    "exists": True,
                    "type": anomaly_type,
                    "anomaly_duration": {"start": 1, "end": 2},
                    "affected_kpis": [
                        "RSRP",
                        "UL_SNR" if (index // 2) % 2 == 0 else "DL_MCS",
                    ],
                }
                if anomalous
                else {"exists": False}
            )
            signal = float(index % 17)
            rows.append(
                {
                    "KPIs": {
                        "RSRP": [signal, signal + 1, signal + 2, signal + 3]
                    },
                    "anomalies": annotation,
                }
            )
        frame = pd.DataFrame(rows)
        config = load_pipeline_config()
        config["preprocessing"]["sequence_length"] = 4
        config["preprocessing"]["selected_kpis"] = {
            task: ["RSRP"]
            for task in (
                "binary_detection",
                "anomaly_type",
                "temporal_localization",
                "affected_kpis",
            )
        }
        config["training"]["max_parallel_tasks"] = 1
        config["training"]["random_forest"].update(
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=1,
            max_temporal_training_timesteps=100,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            results = run_training_pipeline(
                frame,
                config,
                artifact_directory=temporary_directory,
            )

            self.assertEqual(len(results), 4)
            for task_name, result in results.items():
                artifact = load_task_artifact(result["artifact_path"])
                self.assertEqual(artifact.task_name, task_name)
                self.assertEqual(
                    artifact.feature_channels,
                    ("RSRP",),
                )
                self.assertTrue(result["validation_metrics"])

    def test_group_split_has_no_time_overlap_between_partitions(self) -> None:
        rows = []
        base_time = pd.Timestamp("2025-01-01T00:00:00")
        for index in range(60):
            anomalous = index % 2 == 0
            rows.append(
                {
                    "start_time": base_time
                    + pd.Timedelta(seconds=(index // 2) * 50 + (index % 2) * 10),
                    "end_time": base_time
                    + pd.Timedelta(seconds=(index // 2) * 50 + (index % 2) * 10 + 12),
                    "KPIs": {"RSRP": [float(index)] * 4},
                    "anomalies": (
                        {
                            "exists": True,
                            "type": "type_1",
                            "anomaly_duration": {"start": 1, "end": 2},
                            "affected_kpis": ["RSRP"],
                        }
                        if anomalous
                        else {"exists": False}
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        config = load_pipeline_config()
        config["preprocessing"]["sequence_length"] = 4
        config["preprocessing"]["selected_kpis"]["binary_detection"] = ["RSRP"]
        splits = prepare_task_data_component(frame, "binary_detection", config)

        self.assertIsNotNone(splits.groups_train)
        self.assertFalse(
            set(splits.groups_train) & set(splits.groups_validation)
        )
        self.assertFalse(set(splits.groups_train) & set(splits.groups_test))
        self.assertFalse(
            set(splits.groups_validation) & set(splits.groups_test)
        )
        self.assertTrue(len(np.unique(splits.groups_train)) < len(splits.y_train))

    def test_training_entrypoint_writes_comparison_results(self) -> None:
        rows = []
        base_time = pd.Timestamp("2025-01-01T00:00:00")
        for index in range(60):
            signal = float(index % 2) * 5 + index / 100
            rows.append(
                {
                    "start_time": base_time + pd.Timedelta(seconds=index * 20),
                    "end_time": base_time
                    + pd.Timedelta(seconds=index * 20 + 12),
                    "KPIs": {"RSRP": [signal, signal + 1, signal + 2, signal + 3]},
                    "anomalies": {"exists": bool(index % 2)},
                }
            )
        project_config = load_pipeline_config()
        project_config["preprocessing"]["sequence_length"] = 4
        project_config["preprocessing"]["selected_kpis"]["binary_detection"] = [
            "RSRP"
        ]
        model_config = {
            "cross_validation": {"folds": 2, "shuffle_seed": 42},
            "selection": {"primary_metric": "f1_macro"},
            "models": {
                "binary_detection": {
                    "logistic_regression": {
                        "estimator": "logistic_regression",
                        "parameter_grid": {
                            "C": [1.0],
                            "class_weight": ["balanced"],
                        },
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_task_directory = (
                Path(temporary_directory) / "anomaly_type"
            )
            previous_task_directory.mkdir()
            (previous_task_directory / "comparison.json").write_text(
                json.dumps(
                    {
                        "task_name": "anomaly_type",
                        "selected_model": "svm",
                        "group_split": True,
                        "grouping_strategy": "overlapping_time_intervals",
                        "cross_validation_folds": 2,
                        "group_count": 12,
                        "models": [
                            {
                                "model_name": "svm",
                                "estimator": "svm",
                                "parameters": {},
                                "cv_metrics": {
                                    "f1_macro_mean": 0.5,
                                    "f1_macro_std": 0.0,
                                    "precision_macro_mean": 0.5,
                                    "recall_macro_mean": 0.5,
                                },
                                "validation_metrics": {
                                    "f1_macro": 0.5,
                                    "f1_micro": 0.5,
                                    "f1_weighted": 0.5,
                                    "precision_macro": 0.5,
                                    "recall_macro": 0.5,
                                },
                                "test_metrics": {"f1_macro": 0.5},
                                "decision_thresholds": [],
                                "selected": True,
                                "artifact_path": "anomaly_type/svm/model_bundle.joblib",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            results = run_model_comparison_pipeline(
                pd.DataFrame(rows),
                project_config,
                model_config,
                temporary_directory,
                ["binary_detection"],
            )
            comparison_path = Path(temporary_directory) / "comparison.csv"

            self.assertTrue(comparison_path.is_file())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["task_name"], "binary_detection")
            self.assertTrue(results[0]["group_split"])
            comparison = pd.read_csv(comparison_path)
            self.assertIn("cv_precision_macro_mean", comparison.columns)
            self.assertIn("cv_recall_macro_mean", comparison.columns)
            self.assertGreaterEqual(
                float(comparison.loc[0, "cv_precision_macro_mean"]), 0
            )
            self.assertGreaterEqual(
                float(comparison.loc[0, "cv_recall_macro_mean"]), 0
            )
            self.assertEqual(
                set(comparison["task_name"]),
                {"binary_detection", "anomaly_type"},
            )
            consolidated = json.loads(
                (Path(temporary_directory) / "comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(consolidated), 2)

    def test_cv_model_selection_saves_test_metrics_only_for_selected_model(self) -> None:
        def make_task_splits(task_name: str) -> TaskSplits:
            n_train, n_holdout = 24, 8
            train_X = np.empty((n_train, 4, 1), dtype=object)
            holdout_X = np.empty((n_holdout, 4, 1), dtype=object)
            for index in range(n_train):
                signal = float(index % 2) * 4 + index / 100
                train_X[index, :, 0] = [signal, signal + 1, signal + 2, signal + 3]
            for index in range(n_holdout):
                signal = float(index % 2) * 4 + index / 100
                holdout_X[index, :, 0] = [signal, signal + 1, signal + 2, signal + 3]
            y_train = np.asarray([index % 2 for index in range(n_train)])
            y_holdout = np.asarray([index % 2 for index in range(n_holdout)])
            if task_name == "anomaly_type":
                y_train = np.asarray(
                    [f"type_{index % 3}" for index in range(n_train)]
                )
                y_holdout = np.asarray(
                    [f"type_{index % 3}" for index in range(n_holdout)]
                )
            elif task_name == "temporal_localization":
                y_train = np.tile(np.array([0, 0, 1, 1]), (n_train, 1))
                y_holdout = np.tile(np.array([0, 0, 1, 1]), (n_holdout, 1))
            elif task_name == "affected_kpis":
                labels = [
                    ("RSRP", "RX_Bytes", "TX_Bytes"),
                    ("DL_MCS", "RX_Bytes", "TX_Bytes"),
                    ("RSRP", "DL_MCS", "RX_Bytes", "TX_Bytes"),
                ]
                y_train = np.empty(n_train, dtype=object)
                y_holdout = np.empty(n_holdout, dtype=object)
                y_train[:] = [labels[index % 3] for index in range(n_train)]
                y_holdout[:] = [labels[index % 3] for index in range(n_holdout)]
            return TaskSplits(
                task_name=task_name,
                X_train=train_X,
                y_train=y_train,
                positions_train=np.arange(n_train),
                X_validation=holdout_X,
                y_validation=y_holdout,
                positions_validation=np.arange(n_train, n_train + n_holdout),
                X_test=holdout_X.copy(),
                y_test=y_holdout.copy(),
                positions_test=np.arange(n_train + n_holdout, n_train + 2 * n_holdout),
                feature_channels=("RSRP",),
                groups_train=np.arange(n_train),
                groups_validation=np.arange(n_train, n_train + n_holdout),
                groups_test=np.arange(n_train + n_holdout, n_train + 2 * n_holdout),
                stratify_groups_train=(
                    np.asarray([f"type_{i % 3}" for i in range(n_train)])
                    if task_name == "temporal_localization"
                    else None
                ),
            )

        project_config = load_pipeline_config()
        project_config["training"]["random_forest"][
            "max_temporal_training_timesteps"
        ] = 100
        model_config = {
            "cross_validation": {"folds": 2, "shuffle_seed": 42},
            "selection": {"primary_metric": "f1_macro"},
            "models": {},
        }
        algorithm_mapping = {
            "binary_detection": ["logistic_regression", "random_forest", "xgboost"],
            "anomaly_type": ["logistic_regression", "random_forest", "svm", "xgboost"],
            "temporal_localization": [
                "logistic_regression",
                "random_forest",
                "xgboost",
            ],
            "affected_kpis": [
                "one_vs_rest_logistic_regression",
                "random_forest",
                "classifier_chain",
            ],
        }
        estimator_mapping = {
            "one_vs_rest_logistic_regression": "one_vs_rest_logistic_regression",
            "classifier_chain": "classifier_chain",
        }
        for task_name, algorithms in algorithm_mapping.items():
            model_config["models"][task_name] = {}
            for algorithm in algorithms:
                estimator = estimator_mapping.get(algorithm, algorithm)
                if estimator == "logistic_regression" and algorithm.startswith(
                    "one_vs_rest"
                ):
                    estimator = "one_vs_rest_logistic_regression"
                parameters = (
                    {"n_estimators": [3], "max_depth": [3]}
                    if estimator in {"random_forest", "xgboost"}
                    else {"C": [1.0], "class_weight": ["balanced"]}
                )
                if estimator == "xgboost":
                    parameters["learning_rate"] = [0.2]
                if estimator == "svm":
                    parameters = {
                        "C": [1.0],
                        "kernel": ["linear"],
                        "class_weight": ["balanced"],
                    }
                model_config["models"][task_name][algorithm] = {
                    "estimator": estimator,
                    "parameter_grid": parameters,
                }

        with tempfile.TemporaryDirectory() as temporary_directory:
            for task_name in algorithm_mapping:
                result = run_task_model_comparison(
                    task_name,
                    make_task_splits(task_name),
                    project_config,
                    model_config,
                    temporary_directory,
                )
                self.assertEqual(len(result["models"]), len(algorithm_mapping[task_name]))
                self.assertEqual(
                    sum(model["selected"] for model in result["models"]), 1
                )
                self.assertTrue(result["test_metrics"])
                self.assertTrue(
                    all(
                        model["test_metrics"] is None
                        for model in result["models"]
                        if not model["selected"]
                    )
                )
                if task_name == "binary_detection":
                    selected_result = next(
                        model for model in result["models"] if model["selected"]
                    )
                    artifact = load_task_artifact(
                        selected_result["artifact_path"]
                    )
                    inference_frame = pd.DataFrame(
                        {
                            "KPIs": [
                                {"RSRP": [0.0, 1.0, 2.0, 3.0]},
                                {"RSRP": [4.0, 5.0, 6.0, 7.0]},
                            ]
                        }
                    )
                    predictions = predict_task_component(
                        inference_frame, artifact
                    )
                    self.assertEqual(predictions.predictions.shape, (2,))
                    self.assertTrue(artifact.decision_thresholds)


if __name__ == "__main__":
    unittest.main()
